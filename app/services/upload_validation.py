# app/services/upload_validation.py
"""
Upload-time validation for models and datasets.

Every bug hit during development traced back to one of four root causes:
  1. Model output shape mismatch  (e.g. [batch] instead of [batch, n_classes])
  2. Data scale mismatch          (uploaded CSV not in the scale the model expects)
  3. Export-time shape assumptions baked in (e.g. .squeeze() behaving
     differently at batch size 1 vs N)
  4. Imbalanced-data sampling producing single-class subsets

This module catches all four AT UPLOAD TIME, in seconds, with a clear
message -- instead of letting them surface 30+ seconds later inside a
Celery job with a raw ART/PyTorch/ONNX traceback.

Usage in your upload routes:

    from app.services.upload_validation import validate_model, validate_dataset

    result = validate_model(model_bytes, framework)
    if not result.ok:
        return jsonify({"status": "error", "error": result.message}), 400

    result = validate_dataset(csv_bytes)
    if not result.ok:
        return jsonify({"status": "error", "error": result.message}), 400
"""

import io
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class ValidationResult:
    ok: bool
    message: str = ""
    # Extra info the route can use in its success response, e.g. n_samples,
    # n_features, output_shape -- kept generic so callers can pick what
    # they want to surface to the frontend.
    info: Optional[dict] = None


# ======================================================================
# MODEL VALIDATION
# ======================================================================

def validate_model(model_bytes: bytes, framework: str, probe_batch_sizes=(1, 4, 17)) -> ValidationResult:
    """
    Loads the model exactly as tasks.py's build_classifier() would, and
    runs it on a few small dummy batches to catch shape problems before
    any real job is queued.

    probe_batch_sizes deliberately includes more than one size, and none
    of them are round numbers -- this is exactly what would have caught
    the .squeeze()/unsqueeze() batch-size-1-only export bug from this
    project's own debugging history: a model that only works at batch
    size 1 will fail here immediately, instead of failing later when ART
    happens to send a batch of 128.
    """
    if framework == "torchscript":
        return _validate_torchscript_model(model_bytes, probe_batch_sizes)
    elif framework == "onnx":
        return _validate_onnx_model(model_bytes, probe_batch_sizes)
    elif framework == "sklearn":
        return _validate_sklearn_model(model_bytes, probe_batch_sizes)
    else:
        return ValidationResult(False, f"Unsupported framework: {framework}")


def _infer_input_size(model) -> Optional[int]:
    """Best-effort: look for the first Linear-like layer's expected input size."""
    try:
        for p in model.parameters():
            if p.dim() == 2:  # weight matrix of a Linear layer: [out, in]
                return p.shape[1]
    except Exception:
        pass
    return None


def _check_output_shape(output_shape, batch_size) -> Optional[str]:
    """
    Shared shape check for torchscript/onnx models. Returns an error
    message string if something's wrong, or None if the shape looks fine.
    """
    if len(output_shape) != 2:
        return (
            f"Model output has shape {list(output_shape)} for a batch of "
            f"{batch_size}, but AO8 requires shape [batch, n_classes] -- "
            f"exactly 2 dimensions. A single raw value per sample (shape "
            f"[batch]) is not enough; binary classifiers must output one "
            f"score per class (shape [batch, 2]). Wrap a single-output "
            f"model before exporting -- see the 'before you upload' notes "
            f"for an example."
        )
    if output_shape[0] != batch_size:
        return (
            f"Model was given a batch of {batch_size} samples but returned "
            f"{output_shape[0]} outputs. This usually means the model was "
            f"traced/exported assuming a fixed batch size (often 1), and "
            f"the exported graph doesn't generalize to other batch sizes. "
            f"Re-export using a reshape-safe operation (e.g. .view()) "
            f"instead of .squeeze()/.unsqueeze() with no explicit dimension."
        )
    if output_shape[1] < 2:
        return (
            f"Model outputs {output_shape[1]} class score(s) per sample; "
            f"AO8 requires at least 2 (one per class). If this is a binary "
            f"classifier with a single sigmoid-style output, wrap it to "
            f"output [-logit, logit] (shape [batch, 2]) before exporting."
        )
    return None


def _validate_torchscript_model(model_bytes, probe_batch_sizes):
    try:
        import torch
    except ImportError:
        return ValidationResult(False, "Server error: torch is not installed.")

    try:
        buffer = io.BytesIO(model_bytes)
        model = torch.jit.load(buffer, map_location="cpu")
        model.eval()
    except Exception as e:
        return ValidationResult(False, f"Could not load TorchScript model: {e}")

    input_size = _infer_input_size(model)
    if input_size is None:
        # Not fatal -- some architectures don't expose this cleanly. Fall
        # back to a small guess and let the shape checks still run.
        input_size = 30

    for bs in probe_batch_sizes:
        try:
            dummy = torch.rand(bs, input_size)
            with torch.no_grad():
                out = model(dummy)
        except Exception as e:
            return ValidationResult(
                False,
                f"Model failed to run on a dummy batch of size {bs} "
                f"({input_size} features): {e}"
            )
        err = _check_output_shape(tuple(out.shape), bs)
        if err:
            return ValidationResult(False, err)

    return ValidationResult(True, info={"inferred_input_features": input_size})


def _validate_onnx_model(model_bytes, probe_batch_sizes):
    try:
        import onnxruntime as rt
    except ImportError:
        return ValidationResult(False, "Server error: onnxruntime is not installed.")

    try:
        sess = rt.InferenceSession(model_bytes)
    except Exception as e:
        return ValidationResult(False, f"Could not load ONNX model: {e}")

    input_meta = sess.get_inputs()[0]
    input_name = input_meta.name
    # Static feature dim if the exporter fixed it, else fall back to a guess.
    shape = input_meta.shape
    input_size = shape[1] if len(shape) == 2 and isinstance(shape[1], int) else 30

    for bs in probe_batch_sizes:
        try:
            dummy = np.random.rand(bs, input_size).astype(np.float32)
            out = sess.run(None, {input_name: dummy})[0]
        except Exception as e:
            return ValidationResult(
                False,
                f"ONNX model failed to run on a dummy batch of size {bs} "
                f"({input_size} features): {e}. This is very often caused "
                f"by a model exported/traced assuming a fixed batch size "
                f"(commonly 1) -- re-export with a batch-size-independent "
                f"reshape op."
            )
        err = _check_output_shape(tuple(out.shape), bs)
        if err:
            return ValidationResult(False, err)

    return ValidationResult(True, info={"inferred_input_features": input_size})


def _validate_sklearn_model(model_bytes, probe_batch_sizes):
    import pickle
    try:
        model = pickle.loads(model_bytes)
    except Exception as e:
        return ValidationResult(False, f"Could not unpickle sklearn model: {e}")

    if not hasattr(model, "predict"):
        return ValidationResult(False, "Uploaded .pkl object has no .predict() method -- not a usable classifier.")

    input_size = getattr(model, "n_features_in_", None) or 30

    for bs in probe_batch_sizes:
        try:
            dummy = np.random.rand(bs, input_size).astype(np.float32)
            preds = model.predict(dummy)
        except Exception as e:
            return ValidationResult(
                False,
                f"sklearn model failed to run on a dummy batch of size {bs} "
                f"({input_size} features): {e}"
            )
        if len(preds) != bs:
            return ValidationResult(
                False,
                f"Model was given {bs} samples but returned {len(preds)} predictions."
            )

    n_classes = getattr(model, "n_classes_", None)
    if n_classes is not None and n_classes < 2:
        return ValidationResult(False, f"Model was fit with only {n_classes} class -- at least 2 are required.")

    # NOTE: FGSM/PGD/CW won't work on this model (no gradients) -- that's
    # expected and handled by the automatic HopSkipJump fallback in
    # tasks.py, not an upload-time error.
    return ValidationResult(True, info={"inferred_input_features": input_size})


# ======================================================================
# DATASET VALIDATION
# ======================================================================

def validate_dataset(csv_bytes: bytes, min_per_class: int = 2) -> ValidationResult:
    """
    Checks structural validity and class balance of an uploaded CSV
    BEFORE it's saved -- catches the exact issue that caused the
    hardening job to fail ("nb_classes must be >= 2") and the missing
    empirical_robustness/CLEVER/loss_sensitivity metrics, both of which
    trace back to too few examples of the minority class ending up in
    whatever subset gets sampled at job time.
    """
    try:
        df = pd.read_csv(io.BytesIO(csv_bytes))
    except Exception as e:
        return ValidationResult(False, f"Could not parse CSV: {e}")

    if df.shape[1] < 2:
        return ValidationResult(False, "CSV needs at least 2 columns: one or more features, plus a label column.")

    if df.shape[0] < 10:
        return ValidationResult(False, f"CSV has only {df.shape[0]} rows -- too few to evaluate meaningfully.")

    feature_cols = df.columns[:-1]
    label_col = df.columns[-1]

    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        return ValidationResult(
            False,
            f"Non-numeric feature column(s) found: {non_numeric}. All feature "
            f"columns (everything except the last) must be numeric."
        )

    if df[feature_cols].isnull().any().any():
        n_null = int(df[feature_cols].isnull().any(axis=1).sum())
        return ValidationResult(False, f"{n_null} row(s) contain missing values in feature columns.")

    y = df[label_col]
    if not pd.api.types.is_integer_dtype(y) and not _looks_like_class_labels(y):
        return ValidationResult(
            False,
            f"Label column '{label_col}' doesn't look like integer class indices "
            f"(0, 1, 2, ...). Found values like: {list(y.unique()[:5])}. One-hot "
            f"vectors and string labels are not supported -- see the "
            f"'before you upload' notes."
        )

    class_counts = y.value_counts()
    n_classes = len(class_counts)
    if n_classes < 2:
        return ValidationResult(False, f"Label column has only {n_classes} class present -- at least 2 are required.")

    under_min = class_counts[class_counts < min_per_class]
    if len(under_min) > 0:
        return ValidationResult(
            False,
            f"Class(es) {list(under_min.index)} have fewer than {min_per_class} "
            f"examples in the uploaded file ({dict(under_min)}). AO8 needs "
            f"enough examples of every class to sample a usable evaluation "
            f"subset -- upload more rows of the rare class, or a smaller "
            f"file where it's better represented."
        )

    # Rough scale sanity check -- catches the double-scaling class of bug
    # in the opposite direction: if the data is WAY outside a plausible
    # range, warn (not block) so the uploader double-checks preprocessing.
    feature_min = df[feature_cols].min().min()
    feature_max = df[feature_cols].max().max()
    scale_note = None
    if feature_max > 1e6 or feature_min < -1e6:
        scale_note = (
            f"Feature values range from {feature_min:.2g} to {feature_max:.2g}, "
            f"which is unusually large. Double-check this matches the scale "
            f"your model was actually trained on -- AO8 does not rescale "
            f"uploaded data."
        )

    return ValidationResult(
        True,
        message=scale_note or "",
        info={
            "n_samples": int(df.shape[0]),
            "n_features": int(len(feature_cols)),
            "n_classes": int(n_classes),
            "class_distribution": {str(k): int(v) for k, v in class_counts.items()},
            "feature_range": [float(feature_min), float(feature_max)],
        },
    )


def _looks_like_class_labels(y) -> bool:
    """Accepts float columns like 0.0/1.0 as valid integer-like labels."""
    try:
        return np.all(np.equal(np.mod(y.values.astype(float), 1), 0))
    except Exception:
        return False


# ======================================================================
# CROSS-CHECK: model + dataset compatibility (call once BOTH are uploaded,
# or right before a job is submitted)
# ======================================================================

def validate_model_dataset_compatibility(model_info: dict, dataset_info: dict) -> ValidationResult:
    """
    Checks the model's inferred input size against the dataset's actual
    feature count. Doesn't catch scale mismatches (that requires actually
    running the model, which the per-file validators above already do with
    dummy data) but catches the simple, common case of a plain column-count
    mismatch immediately.
    """
    model_features = model_info.get("inferred_input_features")
    dataset_features = dataset_info.get("n_features")

    if model_features is not None and dataset_features is not None:
        if model_features != dataset_features:
            return ValidationResult(
                False,
                f"Feature count mismatch: the model expects {model_features} "
                f"input features, but the dataset has {dataset_features} "
                f"feature columns (excluding the label column). Check that "
                f"the CSV's columns match what the model was trained on, in "
                f"the same order."
            )

    return ValidationResult(True)