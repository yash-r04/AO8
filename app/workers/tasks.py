# app/workers/tasks.py
import io
import json
import uuid
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

        # Normalize to [0,1]
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X).astype(np.float32)

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

            # Risk score for this attack
            accuracy_drop = clean_accuracy - robust_acc
            risk_score = round(min(accuracy_drop * 100 * 1.5, 100), 2)

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
                    report_s3_key, safe_values_s3_key
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                result_id, job_id, attack_name,
                clean_accuracy, robust_acc, accuracy_drop,
                risk_score, len(X), n_flipped,
                adv_s3_key, safe_s3_key
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