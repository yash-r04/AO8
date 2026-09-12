# app/routes/evaluate.py
import uuid
import json
import ssl
from flask import Blueprint, request, jsonify, session
from app.db import get_db
from app.services.auth_helpers import login_required

evaluate_bp = Blueprint("evaluate", __name__)

@evaluate_bp.route("/submit", methods=["POST"])
@login_required
def submit_job():
    data = request.get_json()
    model_id   = data.get("model_id")
    dataset_id = data.get("dataset_id")
    attacks    = data.get("attacks", ["fgsm"])
    epsilon    = data.get("epsilon", 0.03)

    if not model_id or not dataset_id:
        return jsonify({"error": "model_id and dataset_id required"}), 400

    user_id = session["user"]["id"]
    job_id  = str(uuid.uuid4())

    conn = get_db()
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO evaluation_jobs
            (id, user_id, model_id, dataset_id, status, attack_config)
        VALUES (%s, %s, %s, %s, 'queued', %s)
    """, (job_id, user_id, model_id, dataset_id,
          json.dumps({"attacks": attacks, "epsilon": epsilon})))
    conn.commit()
    cur.close()
    conn.close()

    from app.workers.celery_app import celery
    celery.send_task("app.workers.tasks.run_attack_job", args=[job_id])

    return jsonify({"job_id": job_id, "status": "queued"})


@evaluate_bp.route("/status/<job_id>", methods=["GET"])
@login_required
def job_status(job_id):
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("""
        SELECT id, status, started_at, completed_at, error_message,
               hardening_requested, hardening_status, hardening_error
        FROM evaluation_jobs WHERE id = %s
    """, (job_id,))
    job = cur.fetchone()
    cur.close()
    conn.close()

    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(dict(job))


@evaluate_bp.route("/harden/<job_id>", methods=["POST"])
@login_required
def harden_job(job_id):
    """
    Triggers adversarial training on an already-completed job's model.
    Requires the original attack job to be 'done' first, since hardening
    re-uses the same model/dataset/attack config.
    """
    user_id = session["user"]["id"]
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, status, hardening_status FROM evaluation_jobs
        WHERE id = %s AND user_id = %s
    """, (job_id, user_id))
    job = cur.fetchone()

    if not job:
        cur.close()
        conn.close()
        return jsonify({"error": "Not found"}), 404

    if job["status"] != "done":
        cur.close()
        conn.close()
        return jsonify({"error": "Original evaluation must complete before hardening"}), 400

    if job["hardening_status"] in ("running", "done"):
        cur.close()
        conn.close()
        return jsonify({"error": f"Hardening already {job['hardening_status']}"}), 400

    cur.execute("""
        UPDATE evaluation_jobs
        SET hardening_requested = TRUE, hardening_status = 'queued', hardening_error = NULL
        WHERE id = %s
    """, (job_id,))
    conn.commit()
    cur.close()
    conn.close()

    from app.workers.celery_app import celery
    celery.send_task("app.workers.tasks.run_hardening_job", args=[job_id])

    return jsonify({"job_id": job_id, "hardening_status": "queued"})