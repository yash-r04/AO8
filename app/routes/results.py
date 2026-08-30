# app/routes/results.py
import io
from flask import Blueprint, jsonify, session, render_template, make_response
from app.db import get_db
from app.services.auth_helpers import login_required
from app.services.storage import generate_download_url
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import json

results_bp = Blueprint("results", __name__)

@results_bp.route("/history")
@login_required
def history():
    """Render the history page."""
    return render_template("history.html", user=session.get("user"))

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
            j.certified_radius, j.certified_accuracy, j.certification_method,
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

    cur.execute("""
        SELECT * FROM evaluation_jobs
        WHERE id = %s AND user_id = %s
    """, (job_id, user_id))
    job = cur.fetchone()
    if not job:
        return jsonify({"error": "Not found"}), 404

    cur.execute("""
        SELECT attack_type, clean_accuracy, robust_accuracy,
               accuracy_drop, risk_score, n_samples_total,
               n_samples_flipped, safe_values_s3_key,
               loss_sensitivity, clever_score, empirical_robustness_score
        FROM evaluation_results
        WHERE job_id = %s
        ORDER BY risk_score DESC
    """, (job_id,))
    attack_results = cur.fetchall()

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

@results_bp.route("/<job_id>/download.pdf")
@login_required
def download_pdf(job_id):
    """
    Generates a detailed PDF report server-side with reportlab.
    Mirrors the in-app results page: overall score, per-attack
    breakdown, ART metrics, certification, and top flipped samples.
    Requires: pip install reportlab
    """
    user_id = session["user"]["id"]
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, status, queued_at, completed_at,
               certified_radius, certified_accuracy, certification_method
        FROM evaluation_jobs WHERE id = %s AND user_id = %s
    """, (job_id, user_id))
    job = cur.fetchone()
    if not job:
        cur.close()
        conn.close()
        return jsonify({"error": "Not found"}), 404

    cur.execute("""
        SELECT attack_type, clean_accuracy, robust_accuracy,
               accuracy_drop, risk_score, n_samples_total, n_samples_flipped,
               loss_sensitivity, clever_score, empirical_robustness_score
        FROM evaluation_results
        WHERE job_id = %s
        ORDER BY risk_score DESC
    """, (job_id,))
    attack_results = cur.fetchall()

    cur.execute("""
        SELECT s.sample_index, s.original_label, s.clean_prediction,
               s.adv_prediction, s.perturbation_l2, r.attack_type
        FROM sample_level_results s
        JOIN evaluation_results r ON s.result_id = r.id
        WHERE r.job_id = %s AND s.was_flipped = true
        ORDER BY s.perturbation_l2 ASC
        LIMIT 15
    """, (job_id,))
    flipped_samples = cur.fetchall()

    cur.close()
    conn.close()

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    margin = inch
    y = height - margin
    page_num = [1]  # mutable so the footer helper can update it

    def draw_footer():
        c.setFont("Helvetica", 8)
        c.setFillColorRGB(0.45, 0.45, 0.45)
        c.drawString(margin, 0.5 * inch, "AO8  |  Yashaswini C Rao")
        c.drawRightString(width - margin, 0.5 * inch, f"Page {page_num[0]}")
        c.setFillColorRGB(0, 0, 0)

    def new_page():
        draw_footer()
        c.showPage()
        page_num[0] += 1
        c.setFont("Helvetica", 9)
        return height - margin

    def check_space(current_y, needed=0.3 * inch):
        if current_y < margin + needed:
            return new_page()
        return current_y

    # ---------- Header ----------
    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, y, "AO8 Robustness Report")
    y -= 0.32 * inch

    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Job ID: {job_id}")
    y -= 0.2 * inch
    c.drawString(margin, y, f"Status: {job['status']}")
    y -= 0.2 * inch
    if job.get("queued_at"):
        c.drawString(margin, y, f"Queued: {job['queued_at']}")
        y -= 0.2 * inch
    if job.get("completed_at"):
        c.drawString(margin, y, f"Completed: {job['completed_at']}")
        y -= 0.2 * inch
    y -= 0.15 * inch

    # ---------- Overall robustness score ----------
    if attack_results:
        avg_drop = sum(r["accuracy_drop"] for r in attack_results) / len(attack_results)
        overall_score = round(100 - avg_drop * 100, 1)
        tag = "ROBUST" if overall_score >= 75 else "MODERATE RISK" if overall_score >= 50 else "HIGH RISK"
    else:
        overall_score = None
        tag = "N/A"

    y = check_space(y, 0.5 * inch)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(margin, y, "Overall Robustness Score")
    y -= 0.26 * inch
    c.setFont("Helvetica", 11)
    score_text = f"{overall_score}%  —  {tag}" if overall_score is not None else "N/A"
    c.drawString(margin, y, score_text)
    y -= 0.35 * inch

    # ---------- Certification ----------
    y = check_space(y, 0.4 * inch)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(margin, y, "Certified Robustness")
    y -= 0.24 * inch
    c.setFont("Helvetica", 9)
    if job.get("certification_method"):
        c.drawString(
            margin, y,
            f"Method: {job['certification_method']}   "
            f"Certified radius: {job['certified_radius']:.4f}   "
            f"Certified accuracy: {job['certified_accuracy']*100:.1f}%"
        )
        y -= 0.2 * inch
        c.setFont("Helvetica-Oblique", 8)
        c.drawString(margin, y, "No perturbation within this radius can change a certified prediction.")
        y -= 0.3 * inch
    else:
        c.drawString(margin, y, "Not available for this model's framework.")
        y -= 0.3 * inch

    # ---------- Per-attack results ----------
    y = check_space(y, 0.4 * inch)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(margin, y, "Attack Results")
    y -= 0.28 * inch

    c.setFont("Helvetica", 9)
    for r in attack_results:
        y = check_space(y, 0.55 * inch)

        c.setFont("Helvetica-Bold", 10)
        c.drawString(margin, y, r["attack_type"].upper())
        y -= 0.2 * inch

        c.setFont("Helvetica", 9)
        line = (
            f"clean: {r['clean_accuracy']*100:.1f}%   "
            f"robust: {r['robust_accuracy']*100:.1f}%   "
            f"drop: {r['accuracy_drop']*100:.1f}%   "
            f"risk score: {r['risk_score']:.1f}   "
            f"flipped: {r['n_samples_flipped']}/{r['n_samples_total']}"
        )
        c.drawString(margin + 0.15 * inch, y, line)
        y -= 0.2 * inch

        parts = []
        if r.get("empirical_robustness_score") is not None:
            parts.append(f"empirical_robustness: {r['empirical_robustness_score']:.4f}")
        if r.get("clever_score") is not None:
            parts.append(f"clever: {r['clever_score']:.4f}")
        if r.get("loss_sensitivity") is not None:
            parts.append(f"loss_sensitivity: {r['loss_sensitivity']:.4f}")

        if parts:
            c.setFont("Helvetica-Oblique", 8)
            c.drawString(margin + 0.15 * inch, y, "   ".join(parts))
            c.setFont("Helvetica", 9)
            y -= 0.2 * inch

        y -= 0.12 * inch  # spacing before next attack block

        # ---------- Flipped samples ----------
    y = check_space(y, 0.4 * inch)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(margin, y, "Most Fragile Samples (smallest perturbation to flip)")
    y -= 0.3 * inch

    # Fixed x-positions per column instead of string padding —
    # Helvetica isn't monospaced, so padded strings don't line up.
    col_x = {
        "attack": margin,
        "sample": margin + 0.9 * inch,
        "orig":   margin + 1.9 * inch,
        "clean":  margin + 2.6 * inch,
        "adv":    margin + 3.3 * inch,
        "pert":   margin + 4.0 * inch,
    }

    if flipped_samples:
        y = check_space(y, 0.3 * inch)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(col_x["attack"], y, "ATTACK")
        c.drawString(col_x["sample"], y, "SAMPLE")
        c.drawString(col_x["orig"],   y, "ORIG")
        c.drawString(col_x["clean"],  y, "CLEAN")
        c.drawString(col_x["adv"],    y, "ADV")
        c.drawString(col_x["pert"],   y, "PERTURBATION L2")
        y -= 0.06 * inch
        c.line(margin, y, width - margin, y)
        y -= 0.16 * inch

        c.setFont("Helvetica", 8)
        for i, s in enumerate(flipped_samples):
            y = check_space(y, 0.2 * inch)
            if i % 2 == 1:
                c.setFillColorRGB(0.95, 0.95, 0.95)
                c.rect(margin - 0.05 * inch, y - 0.05 * inch, width - 2 * margin + 0.1 * inch, 0.18 * inch, fill=1, stroke=0)
                c.setFillColorRGB(0, 0, 0)

            c.drawString(col_x["attack"], y, s["attack_type"].upper())
            c.drawString(col_x["sample"], y, f"#{s['sample_index']}")
            c.drawString(col_x["orig"],   y, str(s["original_label"]))
            c.drawString(col_x["clean"],  y, str(s["clean_prediction"]))
            c.drawString(col_x["adv"],    y, str(s["adv_prediction"]))
            c.drawString(col_x["pert"],   y, f"{s['perturbation_l2']:.4f}")
            y -= 0.19 * inch
    else:
        c.setFont("Helvetica", 9)
        c.drawString(margin, y, "No flipped samples recorded.")
        y -= 0.2 * inch

    # ---------- Finalize ----------
    draw_footer()
    c.save()
    buf.seek(0)

    response = make_response(buf.read())
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = f"attachment; filename=ao8_report_{job_id}.pdf"
    return response


@results_bp.route("/<job_id>")
@login_required
def view_results(job_id):
    """
    Fixed: was "/results/<job_id>", which double-prefixed to
    /results/results/<job_id> under url_prefix="/results".
    Also now passes user= explicitly so the navbar's Test/History
    links don't disappear on this page if the context processor
    isn't set up yet.
    """
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
        return render_template("404.html"), 404

    return render_template("results.html", job_id=job_id, user=session.get("user"))