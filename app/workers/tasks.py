import io
import json
import uuid
import logging
import numpy as np
import pandas as pd
import boto3
import psycopg2
import ssl
from app.workers.celery_app import celery
from psycopg2.extras import RealDictCursor
from datetime import datetime
from celery import shared_task
from app.config import Config

logger = logging.getLogger(__name__)

s3 = boto3.client(
    "s3",
    region_name=Config.AWS_REGION,
    aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
)

def get_db():
    return psycopg2.connect(Config.DATABASE_URL, cursor_factory=RealDictCursor)

@celery.task(bind=True, max_retries=2)
def run_attack_job(self, job_id: str):
    conn = get_db()
    cur = conn.cursor()

    try:
        # 1. Load job
        cur.execute("SELECT * FROM evaluation_jobs WHERE id = %s", (job_id,))
        job = dict(cur.fetchone())

        # Mark as running
        cur.execute("""
            UPDATE evaluation_jobs
            SET status = 'running', started_at = %s
            WHERE id = %s
        """, (datetime.utcnow(), job_id))
        conn.commit()

        config = job["attack_config"]
        user_id = str(job["user_id"])

        # 2. Load model from S3
        cur.execute("SELECT * FROM ml_models WHERE id = %s", (str(job["model_id"]),))
        model_record = dict(cur.fetchone())
        model_obj = s3.get_object(Bucket=Config.S3_BUCKET, Key=model_record["s3_key"])
        model_bytes = model_obj["Body"].read()

        # 3. Load dataset from S3
        cur.execute("SELECT * FROM datasets WHERE id = %s", (str(job["dataset_id"]),))
        dataset_record = dict(cur.fetchone())
        dataset_obj = s3.get_object(Bucket=Config.S3_BUCKET, Key=dataset_record["s3_key"])
        df = pd.read_csv(io.BytesIO(dataset_obj["Body"].read()))

        # Last column = label, rest = features
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(int)

        # Cap at 500 samples for speed
        if len(X) > 500:
            X, y = X[:500], y[:500]

        # NOTE: we intentionally do NOT re-scale X here. Re-fitting a
        # MinMaxScaler on just this uploaded sample computes a different
        # min/max than whatever scaling the model was actually trained
        # with, which silently corrupts every prediction (this caused
        # clean_accuracy to come out far below chance level in testing).
        # Uploaded data is expected to already be in the scale the model
        # expects; clip_values=(0.0, 1.0) on the ART classifier below
        # still keeps attacks from pushing values out of a sane range.
        X = X.astype(np.float32)

        n_classes = len(np.unique(y))

        # 4. Build ART classifier
        classifier = build_classifier(
            model_bytes=model_bytes,
            framework=model_record["framework"],
            input_shape=(X.shape[1],),
            n_classes=n_classes,
        )

        # 5. Clean accuracy baseline
        clean_preds = np.argmax(classifier.predict(X), axis=1)
        clean_accuracy = float(np.mean(clean_preds == y))

        # 6. Run attacks
        attacks_to_run = config.get("attacks", ["fgsm"])
        epsilon = float(config.get("epsilon", 0.03))
        results = []

        for attack_name in attacks_to_run:
            X_adv = run_attack(classifier, X, y, attack_name, epsilon)

            adv_preds = np.argmax(classifier.predict(X_adv), axis=1)
            robust_acc = float(np.mean(adv_preds == y))
            n_flipped = int(np.sum(adv_preds != clean_preds))
            art_metrics = compute_art_metrics(classifier, X, y, attack_name, epsilon)

            # Per-sample analysis
            perturbation = X_adv - X
            sample_results = []
            for i in range(len(X)):
                delta = perturbation[i]
                abs_delta = np.abs(delta)
                top_indices = np.argsort(abs_delta)[::-1][:5]
                top_features = [
                    {"feature_idx": int(idx), "delta": float(delta[idx])}
                    for idx in top_indices
                ]
                sample_results.append({
                    "sample_index": i,
                    "original_label": int(y[i]),
                    "clean_prediction": int(clean_preds[i]),
                    "adv_prediction": int(adv_preds[i]),
                    "was_flipped": bool(adv_preds[i] != clean_preds[i]),
                    "perturbation_l2": float(np.linalg.norm(delta)),
                    "top_features": top_features,
                })

            # Risk score for this attack.
            # accuracy_drop can legitimately be negative or near-zero if an
            # attack barely affects the model. Clamp risk_score to [0, 100]
            # so a negative drop can't produce a negative "risk" or push
            # the overall score above 100%.
            accuracy_drop = clean_accuracy - robust_acc
            risk_score = round(min(max(accuracy_drop * 100 * 1.5, 0.0), 100.0), 2)

            # Save adversarial examples to S3
            adv_buffer = io.BytesIO()
            np.save(adv_buffer, X_adv)
            adv_s3_key = f"users/{user_id}/results/{job_id}/{attack_name}_adv.npy"
            s3.put_object(
                Bucket=Config.S3_BUCKET,
                Key=adv_s3_key,
                Body=adv_buffer.getvalue(),
                ServerSideEncryption="AES256",
            )

            # Save safe values (samples never flipped)
            never_flipped = np.array([
                s["was_flipped"] is False for s in sample_results
            ])
            safe_X = X[never_flipped]
            safe_y = y[never_flipped]
            safe_buffer = io.BytesIO()
            np.savez(safe_buffer, X=safe_X, y=safe_y)
            safe_s3_key = f"users/{user_id}/results/{job_id}/{attack_name}_safe.npz"
            s3.put_object(
                Bucket=Config.S3_BUCKET,
                Key=safe_s3_key,
                Body=safe_buffer.getvalue(),
                ServerSideEncryption="AES256",
            )

            # Save result to DB
            result_id = str(uuid.uuid4())
            cur.execute("""
                INSERT INTO evaluation_results (
                    id, job_id, attack_type,
                    clean_accuracy, robust_accuracy, accuracy_drop,
                    risk_score, n_samples_total, n_samples_flipped,
                    report_s3_key, safe_values_s3_key,
                    loss_sensitivity, clever_score, empirical_robustness_score
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                result_id, job_id, attack_name,
                clean_accuracy, robust_acc, accuracy_drop,
                risk_score, len(X), n_flipped,
                adv_s3_key, safe_s3_key,
                art_metrics["loss_sensitivity"],
                art_metrics["clever_score"],
                art_metrics["empirical_robustness_score"],
            ))

            # Save sample-level results
            for s in sample_results:
                cur.execute("""
                    INSERT INTO sample_level_results (
                        result_id, sample_index, original_label,
                        clean_prediction, adv_prediction,
                        was_flipped, perturbation_l2, most_perturbed_features
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                """, (
                    result_id, s["sample_index"], s["original_label"],
                    s["clean_prediction"], s["adv_prediction"],
                    s["was_flipped"], s["perturbation_l2"],
                    json.dumps(s["top_features"])
                ))

            conn.commit()
            results.append({
                "attack": attack_name,
                "clean_accuracy": clean_accuracy,
                "robust_accuracy": robust_acc,
                "accuracy_drop": accuracy_drop,
                "risk_score": risk_score,
                "n_flipped": n_flipped,
            })

        # 7. Mark job done
        cur.execute("""
            UPDATE evaluation_jobs
            SET status = 'done', completed_at = %s
            WHERE id = %s
        """, (datetime.utcnow(), job_id))
        conn.commit()
        return {"job_id": job_id, "status": "done", "results": results}

    except Exception as e:
        cur.execute("""
            UPDATE evaluation_jobs
            SET status = 'failed', error_message = %s
            WHERE id = %s
        """, (str(e), job_id))
        conn.commit()
        raise
    finally:
        cur.close()
        conn.close()


def build_classifier(model_bytes, framework, input_shape, n_classes):
    import torch
    import torch.nn as nn
    import numpy as np
    from art.estimators.classification import PyTorchClassifier, SklearnClassifier

    if framework == "torchscript":
        # TorchScript — full gradients, no class reference needed
        buffer = io.BytesIO(model_bytes)
        model = torch.jit.load(buffer, map_location="cpu")
        model.eval()
        return PyTorchClassifier(
            model=model,
            loss=nn.CrossEntropyLoss(),
            input_shape=input_shape,
            nb_classes=n_classes,
            clip_values=(0.0, 1.0),
        )

    elif framework == "onnx":
        import onnxruntime as rt

        session = rt.InferenceSession(model_bytes)
        input_name = session.get_inputs()[0].name

        # Finite differences for gradient estimation
        class ONNXForward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, sess, inp_name):
                ctx.save_for_backward(x)
                ctx.sess = sess
                ctx.inp_name = inp_name
                x_np = x.detach().cpu().numpy().astype(np.float32)
                output = sess.run(None, {inp_name: x_np})[0]
                return torch.tensor(output, dtype=torch.float32)

            @staticmethod
            def backward(ctx, grad_output):
                x, = ctx.saved_tensors
                eps = 1e-4
                x_np = x.detach().cpu().numpy().astype(np.float32)
                grad = np.zeros_like(x_np)
                for i in range(x_np.shape[1]):
                    x_plus = x_np.copy(); x_plus[:, i] += eps
                    x_minus = x_np.copy(); x_minus[:, i] -= eps
                    out_plus  = ctx.sess.run(None, {ctx.inp_name: x_plus})[0]
                    out_minus = ctx.sess.run(None, {ctx.inp_name: x_minus})[0]
                    grad[:, i] = np.sum(
                        (out_plus - out_minus) / (2 * eps) * grad_output.cpu().numpy(),
                        axis=1
                    )
                return torch.tensor(grad, dtype=torch.float32), None, None

        class ONNXWrapper(nn.Module):
            def __init__(self, sess, inp_name):
                super().__init__()
                self.sess = sess
                self.inp_name = inp_name
            def forward(self, x):
                return ONNXForward.apply(x, self.sess, self.inp_name)

        wrapper = ONNXWrapper(session, input_name)
        wrapper.eval()
        return PyTorchClassifier(
            model=wrapper,
            loss=nn.CrossEntropyLoss(),
            input_shape=input_shape,
            nb_classes=n_classes,
            clip_values=(0.0, 1.0),
        )

    elif framework == "sklearn":
        import pickle
        model = pickle.loads(model_bytes)
        return SklearnClassifier(model=model, clip_values=(0.0, 1.0))

    else:
        raise ValueError(f"Unsupported framework: {framework}")


def run_attack(classifier, X, y, attack_name, epsilon):
    from art.attacks.evasion import (
        FastGradientMethod,
        ProjectedGradientDescent,
        CarliniL2Method,
    )

    if attack_name == "fgsm":
        attack = FastGradientMethod(estimator=classifier, eps=epsilon)
    elif attack_name == "pgd":
        attack = ProjectedGradientDescent(
            estimator=classifier, eps=epsilon,
            eps_step=epsilon / 4, max_iter=40,
        )
    elif attack_name == "cw":
        attack = CarliniL2Method(
            classifier=classifier, max_iter=50
        )
    else:
        raise ValueError(f"Unknown attack: {attack_name}")

    return attack.generate(X)

def compute_art_metrics(classifier, X, y, attack_name, epsilon, max_samples=20):
    """
    ART's built-in robustness metrics (art.metrics), separate from the
    hand-rolled accuracy_drop/risk_score. Capped to a small sample since
    CLEVER and loss_sensitivity are gradient-heavy and slow.
    """
    from art.metrics import empirical_robustness, clever_u, loss_sensitivity

    n = min(max_samples, len(X))
    idx = np.random.choice(len(X), size=n, replace=False)
    X_sample, y_sample = X[idx], y[idx]

    metrics = {"loss_sensitivity": None, "clever_score": None, "empirical_robustness_score": None}

    try:
        ls = loss_sensitivity(classifier, X_sample, y_sample)
        metrics["loss_sensitivity"] = float(np.mean(ls))
    except Exception as e:
        logger.warning(f"loss_sensitivity failed for {attack_name}: {e}")

    try:
        attack_params = {"eps": epsilon}
        if attack_name == "pgd":
            attack_params = {"eps": epsilon, "eps_step": epsilon / 4, "max_iter": 40}
        er = empirical_robustness(classifier, X_sample, attack_name=attack_name, attack_params=attack_params)
        metrics["empirical_robustness_score"] = float(er)
    except Exception as e:
        logger.warning(f"empirical_robustness failed for {attack_name}: {e}")

    try:
        scores = []
        for i in range(min(5, n)):
            score = clever_u(classifier, X_sample[i:i+1][0], nb_batches=10, batch_size=16, radius=epsilon * 2, norm=2)
            scores.append(score)
        metrics["clever_score"] = float(np.mean(scores)) if scores else None
    except Exception as e:
        logger.warning(f"clever_u failed for {attack_name}: {e}")

    return metrics

def build_trainable_classifier(model_bytes, framework, input_shape, n_classes, lr=1e-4):
    """
    Like build_classifier(), but attaches a real PyTorch optimizer —
    required for ART's AdversarialTrainer.fit(), which calls the
    underlying classifier's .fit() internally.

    Only torchscript is supported here: the onnx path in build_classifier()
    wraps an onnxruntime session with no trainable nn.Parameters (gradients
    there are finite-difference estimates, not learnable weights), so there
    is nothing for an optimizer to update.
    """
    import torch
    import torch.nn as nn
    from art.estimators.classification import PyTorchClassifier

    if framework != "torchscript":
        raise ValueError(
            f"Hardening requires trainable parameters and is only supported "
            f"for 'torchscript' models, not '{framework}'."
        )

    buffer = io.BytesIO(model_bytes)
    model = torch.jit.load(buffer, map_location="cpu")
    model.train()  # enable training mode (dropout/batchnorm behave differently than .eval())

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        input_shape=input_shape,
        nb_classes=n_classes,
        clip_values=(0.0, 1.0),
    )


def harden_model(classifier, X, y, epsilon, nb_epochs=10, ratio=0.5):
    """
    Adversarial training via ART's AdversarialTrainer. `classifier` must
    have been built with build_trainable_classifier() (i.e. has an
    optimizer attached) — a classifier from build_classifier() will raise
    "An optimizer is needed to train the model, but none for provided."
    """
    from art.attacks.evasion import ProjectedGradientDescent
    from art.defences.trainer import AdversarialTrainer

    attack = ProjectedGradientDescent(
        estimator=classifier, eps=epsilon,
        eps_step=epsilon / 4, max_iter=40,
    )
    trainer = AdversarialTrainer(classifier, attacks=attack, ratio=ratio)
    trainer.fit(X, y, nb_epochs=nb_epochs, batch_size=32)
    return classifier


def serialize_hardened_model(classifier, framework):
    """
    Serializes a hardened torchscript classifier back to bytes so it can
    be stored in S3 and re-downloaded later.
    """
    import torch

    if framework == "torchscript":
        scripted = torch.jit.script(classifier.model)
        buffer = io.BytesIO()
        torch.jit.save(scripted, buffer)
        return buffer.getvalue()

    else:
        raise ValueError(f"Hardening/serialization not supported for framework '{framework}'")


@celery.task(bind=True, max_retries=1)
def run_hardening_job(self, job_id: str):
    """
    Separate task from run_attack_job: loads the same model/dataset,
    trains a hardened copy, re-runs the same attacks on it, and stores
    a before/after comparison plus the hardened model itself.

    Only supports torchscript models — see build_trainable_classifier()
    for why onnx and sklearn are excluded.
    """
    conn = get_db()
    cur = conn.cursor()

    try:
        cur.execute("SELECT * FROM evaluation_jobs WHERE id = %s", (job_id,))
        job = dict(cur.fetchone())

        cur.execute("""
            UPDATE evaluation_jobs
            SET hardening_status = 'running'
            WHERE id = %s
        """, (job_id,))
        conn.commit()

        config = job["attack_config"]
        user_id = str(job["user_id"])

        cur.execute("SELECT * FROM ml_models WHERE id = %s", (str(job["model_id"]),))
        model_record = dict(cur.fetchone())

        # Hardening needs real trainable parameters — only torchscript qualifies.
        # (onnx is excluded here even though it's allowed for attacks/certification,
        # because ONNXWrapper has no nn.Parameters for an optimizer to update.)
        if model_record["framework"] != "torchscript":
            cur.execute("""
                UPDATE evaluation_jobs
                SET hardening_status = 'unsupported',
                    hardening_error = %s
                WHERE id = %s
            """, (
                f"Hardening requires a trainable model and is only supported "
                f"for 'torchscript', not '{model_record['framework']}'.",
                job_id
            ))
            conn.commit()
            return {"job_id": job_id, "status": "unsupported"}

        model_obj = s3.get_object(Bucket=Config.S3_BUCKET, Key=model_record["s3_key"])
        model_bytes = model_obj["Body"].read()

        cur.execute("SELECT * FROM datasets WHERE id = %s", (str(job["dataset_id"]),))
        dataset_record = dict(cur.fetchone())
        dataset_obj = s3.get_object(Bucket=Config.S3_BUCKET, Key=dataset_record["s3_key"])
        df = pd.read_csv(io.BytesIO(dataset_obj["Body"].read()))

        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(int)

        # Hardening is expensive — cap sample size lower than the standard 500
        if len(X) > 300:
            X, y = X[:300], y[:300]

        # Same rationale as run_attack_job: do NOT re-scale uploaded data.
        X = X.astype(np.float32)
        n_classes = len(np.unique(y))
        epsilon = float(config.get("epsilon", 0.03))
        attacks_to_run = config.get("attacks", ["fgsm"])

        # --- Baseline (before hardening) — plain inference classifier is fine ---
        baseline_classifier = build_classifier(
            model_bytes=model_bytes,
            framework=model_record["framework"],
            input_shape=(X.shape[1],),
            n_classes=n_classes,
        )
        clean_preds_before = np.argmax(baseline_classifier.predict(X), axis=1)
        clean_acc_before = float(np.mean(clean_preds_before == y))

        robust_acc_before = {}
        for attack_name in attacks_to_run:
            X_adv = run_attack(baseline_classifier, X, y, attack_name, epsilon)
            adv_preds = np.argmax(baseline_classifier.predict(X_adv), axis=1)
            robust_acc_before[attack_name] = float(np.mean(adv_preds == y))

        # --- Hardening — needs the trainable classifier with an optimizer ---
        hardened_classifier = build_trainable_classifier(
            model_bytes=model_bytes,
            framework=model_record["framework"],
            input_shape=(X.shape[1],),
            n_classes=n_classes,
        )
        harden_model(hardened_classifier, X, y, epsilon, nb_epochs=10)
        hardened_classifier.model.eval()  # switch back to eval mode before predicting

        # --- Post-hardening evaluation ---
        clean_preds_after = np.argmax(hardened_classifier.predict(X), axis=1)
        clean_acc_after = float(np.mean(clean_preds_after == y))

        for attack_name in attacks_to_run:
            X_adv = run_attack(hardened_classifier, X, y, attack_name, epsilon)
            adv_preds = np.argmax(hardened_classifier.predict(X_adv), axis=1)
            robust_acc_after = float(np.mean(adv_preds == y))

            improvement = robust_acc_after - robust_acc_before[attack_name]

            result_id = str(uuid.uuid4())
            cur.execute("""
                INSERT INTO hardening_results (
                    id, job_id, attack_type,
                    clean_accuracy_before, robust_accuracy_before,
                    clean_accuracy_after, robust_accuracy_after,
                    robustness_improvement
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                result_id, job_id, attack_name,
                clean_acc_before, robust_acc_before[attack_name],
                clean_acc_after, robust_acc_after,
                improvement,
            ))
            conn.commit()

        # --- Save hardened model to S3 ---
        try:
            hardened_bytes = serialize_hardened_model(hardened_classifier, model_record["framework"])
            hardened_s3_key = f"users/{user_id}/results/{job_id}/hardened_model.pt"
            s3.put_object(
                Bucket=Config.S3_BUCKET,
                Key=hardened_s3_key,
                Body=hardened_bytes,
                ServerSideEncryption="AES256",
            )
            cur.execute("""
                UPDATE evaluation_jobs
                SET hardened_model_s3_key = %s
                WHERE id = %s
            """, (hardened_s3_key, job_id))
        except Exception:
            # Serialization is best-effort — the before/after metrics are the
            # main deliverable, so don't fail the job if export fails.
            pass

        cur.execute("""
            UPDATE evaluation_jobs
            SET hardening_status = 'done'
            WHERE id = %s
        """, (job_id,))
        conn.commit()

        return {"job_id": job_id, "status": "done"}

    except Exception as e:
        cur.execute("""
            UPDATE evaluation_jobs
            SET hardening_status = 'failed', hardening_error = %s
            WHERE id = %s
        """, (str(e), job_id))
        conn.commit()
        raise
    finally:
        cur.close()
        conn.close()