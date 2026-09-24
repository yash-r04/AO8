# app/routes/models.py
import uuid
import io
import boto3
from flask import Blueprint, request, jsonify, session
from app.db import get_db
from app.config import Config
from app.services.auth_helpers import login_required
from app.services.upload_validation import validate_model, validate_dataset

models_bp = Blueprint("models", __name__)

s3 = boto3.client(
    "s3",
    region_name=Config.AWS_REGION,
    aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
)

@models_bp.route("/upload", methods=["POST"])
@login_required
def upload_model():
    file = request.files.get("model_file")
    framework = request.form.get("framework")

    if not file or not framework:
        return jsonify({"error": "file and framework required"}), 400

    if framework not in ("torchscript", "onnx", "sklearn"):
        return jsonify({"error": "framework must be torchscript, onnx, or sklearn"}), 400

    user_id = session["user"]["id"]
    model_id = str(uuid.uuid4())
    s3_key = f"users/{user_id}/models/{model_id}/model.pt"

    file_bytes = file.read()

    # Full validation: loads the model, runs it on dummy batches of
    # several different sizes, checks output shape is [batch, n_classes].
    # Catches the shape/export bugs from this project's history in
    # seconds, before anything touches S3 or a Celery job.
    result = validate_model(file_bytes, framework)
    if not result.ok:
        return jsonify({"error": result.message}), 400

    # Upload to S3
    try:
        s3.put_object(
            Bucket=Config.S3_BUCKET,
            Key=s3_key,
            Body=file_bytes,
            ServerSideEncryption="AES256",
        )
    except Exception as e:
        return jsonify({"error": f"S3 upload failed: {str(e)}"}), 500

    # Save to DB
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO ml_models
            (id, user_id, name, framework, s3_key, upload_status, file_size_bytes)
        VALUES (%s, %s, %s, %s, %s, 'ready', %s)
    """, (model_id, user_id, file.filename, framework, s3_key, len(file_bytes)))
    conn.commit()
    cur.close()
    conn.close()

    response = {
        "status": "ready",
        "model_id": model_id,
        "filename": file.filename,
        "size_bytes": len(file_bytes),
    }
    if result.info:
        response["inferred_input_features"] = result.info.get("inferred_input_features")

    return jsonify(response)


@models_bp.route("/list", methods=["GET"])
@login_required
def list_models():
    user_id = session["user"]["id"]
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, name, framework, upload_status, file_size_bytes, created_at
        FROM ml_models
        WHERE user_id = %s AND upload_status = 'ready'
        ORDER BY created_at DESC
    """, (user_id,))
    models = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify([dict(m) for m in models])


@models_bp.route("/delete/<model_id>", methods=["DELETE"])
@login_required
def delete_model(model_id):
    user_id = session["user"]["id"]
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT s3_key FROM ml_models WHERE id = %s AND user_id = %s
    """, (model_id, user_id))
    model = cur.fetchone()

    if not model:
        return jsonify({"error": "Not found"}), 404

    s3.delete_object(Bucket=Config.S3_BUCKET, Key=model["s3_key"])
    cur.execute("DELETE FROM ml_models WHERE id = %s", (model_id,))
    conn.commit()
    cur.close()
    conn.close()
    return jsonify({"status": "deleted"})


@models_bp.route("/dataset/upload", methods=["POST"])
@login_required
def upload_dataset():
    """
    Accepts a CSV file. Validates structure, numeric feature columns,
    integer label column, and per-class sample counts (catches the
    single-class-subset bug from this project's history) BEFORE
    saving anything to S3 or the DB.
    """
    file = request.files.get("dataset_file")
    if not file:
        return jsonify({"error": "No file provided"}), 400

    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files supported"}), 400

    user_id = session["user"]["id"]
    dataset_id = str(uuid.uuid4())
    s3_key = f"users/{user_id}/datasets/{dataset_id}/data.csv"

    file_bytes = file.read()

    result = validate_dataset(file_bytes)
    if not result.ok:
        return jsonify({"error": result.message}), 400

    n_samples = result.info["n_samples"]
    n_features = result.info["n_features"]

    # Upload to S3
    try:
        s3.put_object(
            Bucket=Config.S3_BUCKET,
            Key=s3_key,
            Body=file_bytes,
            ServerSideEncryption="AES256",
        )
    except Exception as e:
        return jsonify({"error": f"S3 upload failed: {str(e)}"}), 500

    # Save to DB
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO datasets (id, user_id, name, s3_key, n_samples, n_features)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (dataset_id, user_id, file.filename, s3_key, n_samples, n_features))
    conn.commit()
    cur.close()
    conn.close()

    response = {
        "status": "ready",
        "dataset_id": dataset_id,
        "filename": file.filename,
        "n_samples": n_samples,
        "n_features": n_features,
    }
    # Non-fatal warning (e.g. unusual feature scale) still ready to use,
    # but worth surfacing to the uploader.
    if result.message:
        response["warning"] = result.message

    return jsonify(response)


@models_bp.route("/dataset/list", methods=["GET"])
@login_required
def list_datasets():
    user_id = session["user"]["id"]
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, name, n_samples, n_features, created_at
        FROM datasets
        WHERE user_id = %s
        ORDER BY created_at DESC
    """, (user_id,))
    datasets = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify([dict(d) for d in datasets])