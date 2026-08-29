# app/routes/results.py
from flask import Blueprint, jsonify, session, render_template
from app.db import get_db
from app.services.auth_helpers import login_required
from app.services.storage import generate_download_url
import json

results_bp = Blueprint("results", __name__)

@results_bp.route("/history")
@login_required
def history():
    """Render the history page."""
    return render_template("history.html", user=session["user"])

@results_bp.route("/api/history")
@login_required
def api_history():
    """Return all jobs for the logged-in user as JSON."""
    user_id = session["user"]["id"]
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT
            j.id, j.status, j.queued_at, j.completed_at,
            j.attack_config, j.error_message,
            m.name as model_name,
            d.name as dataset_name
        FROM evaluation_jobs j
        LEFT JOIN ml_models m ON j.model_id = m.id
        LEFT JOIN datasets d ON j.dataset_id = d.id
        WHERE j.user_id = %s
        ORDER BY j.queued_at DESC
    """, (user_id,))
    jobs = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify([dict(j) for j in jobs])

@results_bp.route("/api/job/<job_id>")
@login_required
def api_job_detail(job_id):
    """Full result for one job including per-attack breakdown."""
    user_id = session["user"]["id"]
    conn = get_db()
    cur = conn.cursor()

    # Verify job belongs to user
    cur.execute("""
        SELECT * FROM evaluation_jobs
        WHERE id = %s AND user_id = %s
    """, (job_id, user_id))
    job = cur.fetchone()
    if not job:
        return jsonify({"error": "Not found"}), 404

    # Get attack results
    cur.execute("""
        SELECT attack_type, clean_accuracy, robust_accuracy,
               accuracy_drop, risk_score, n_samples_total,
               n_samples_flipped, safe_values_s3_key
        FROM evaluation_results
        WHERE job_id = %s
        ORDER BY risk_score DESC
    """, (job_id,))
    attack_results = cur.fetchall()

    # Get sample-level flipped rows for each attack
    cur.execute("""
        SELECT s.sample_index, s.original_label, s.clean_prediction,
               s.adv_prediction, s.perturbation_l2, s.most_perturbed_features,
               r.attack_type
        FROM sample_level_results s
        JOIN evaluation_results r ON s.result_id = r.id
        WHERE r.job_id = %s AND s.was_flipped = true
        ORDER BY s.perturbation_l2 ASC
        LIMIT 50
    """, (job_id,))
    flipped = cur.fetchall()

    cur.close()
    conn.close()

    return jsonify({
        "job": dict(job),
        "attack_results": [dict(r) for r in attack_results],
        "flipped_samples": [dict(f) for f in flipped],
    })

@results_bp.route("/api/job/<job_id>/download-safe-values/<attack_type>")
@login_required
def download_safe_values(job_id, attack_type):
    """Generate a 5-minute presigned download URL for safe values."""
    user_id = session["user"]["id"]
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT r.safe_values_s3_key
        FROM evaluation_results r
        JOIN evaluation_jobs j ON r.job_id = j.id
        WHERE r.job_id = %s AND r.attack_type = %s AND j.user_id = %s
    """, (job_id, attack_type, user_id))
    result = cur.fetchone()
    cur.close()
    conn.close()

    if not result or not result["safe_values_s3_key"]:
        return jsonify({"error": "Safe values not found"}), 404

    url = generate_download_url(result["safe_values_s3_key"], expires_in=300)
    return jsonify({"download_url": url, "expires_in": 300})

@results_bp.route("/results/<job_id>")
@login_required
def view_results(job_id):
    user_id = session["user"]["id"]
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT id FROM evaluation_jobs
        WHERE id = %s AND user_id = %s
    """, (job_id, user_id))
    job = cur.fetchone()
    cur.close()
    conn.close()

    if not job:
        return render_template("404.html"), 404  # or abort(404)

    return render_template("results.html", job_id=job_id)