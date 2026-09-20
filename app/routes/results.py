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

REPORT_TITLE = "O8 \u2014 Ocean's 8 \u2014 Adversarial Robustness Evaluator"
REPORT_FOOTER_CREDIT = "made by yashaswini"


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

    cur.execute("""
        SELECT attack_type, clean_accuracy_before, robust_accuracy_before,
               clean_accuracy_after, robust_accuracy_after, robustness_improvement
        FROM hardening_results
        WHERE job_id = %s
        ORDER BY attack_type
    """, (job_id,))
    hardening_results = cur.fetchall()

    cur.close()
    conn.close()

    return jsonify({
        "job": dict(job),
        "attack_results": [dict(r) for r in attack_results],
        "flipped_samples": [dict(f) for f in flipped],
        "hardening_results": [dict(h) for h in hardening_results],
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


@results_bp.route("/api/job/<job_id>/download-hardened-model")
@login_required
def download_hardened_model(job_id):
    """Generate a 5-minute presigned download URL for the hardened model."""
    user_id = session["user"]["id"]
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT hardened_model_s3_key FROM evaluation_jobs
        WHERE id = %s AND user_id = %s
    """, (job_id, user_id))
    result = cur.fetchone()
    cur.close()
    conn.close()

    if not result or not result["hardened_model_s3_key"]:
        return jsonify({"error": "Hardened model not found"}), 404

    url = generate_download_url(result["hardened_model_s3_key"], expires_in=300)
    return jsonify({"download_url": url, "expires_in": 300})


# ======================================================================
# Reference-range classification helpers
#
# Mirrors how a lab/diagnostic report flags each value against a normal
# range: every metric gets a (label, color) verdict based on fixed,
# documented thresholds, not just a raw number. Thresholds are intentionally
# conservative and are exactly the same ones used for the on-screen report,
# so the PDF and web report never disagree.
# ======================================================================

_FLAG_GREEN = (0.10, 0.55, 0.30)
_FLAG_AMBER = (0.75, 0.50, 0.05)
_FLAG_RED = (0.70, 0.10, 0.15)


def classify_accuracy(value):
    """For clean_accuracy / robust_accuracy (fraction 0-1). Higher = better."""
    if value is None:
        return ("N/A", (0.4, 0.4, 0.4))
    if value >= 0.90:
        return ("NORMAL", _FLAG_GREEN)
    elif value >= 0.70:
        return ("BORDERLINE", _FLAG_AMBER)
    else:
        return ("LOW", _FLAG_RED)


def classify_drop(value):
    """For accuracy_drop (fraction, can be negative). Lower = better."""
    if value is None:
        return ("N/A", (0.4, 0.4, 0.4))
    v = max(value, 0.0)  # a negative drop just means the attack didn't hurt at all
    if v <= 0.10:
        return ("NORMAL", _FLAG_GREEN)
    elif v <= 0.30:
        return ("ELEVATED", _FLAG_AMBER)
    else:
        return ("HIGH RISK", _FLAG_RED)


def classify_risk_score(value):
    """For risk_score (0-100). Lower = better."""
    if value is None:
        return ("N/A", (0.4, 0.4, 0.4))
    v = max(value, 0.0)
    if v <= 30:
        return ("NORMAL", _FLAG_GREEN)
    elif v <= 60:
        return ("ELEVATED", _FLAG_AMBER)
    else:
        return ("HIGH RISK", _FLAG_RED)


def classify_flip_rate(n_flipped, n_total):
    """For fraction of samples whose prediction flipped. Lower = better."""
    if not n_total:
        return ("N/A", (0.4, 0.4, 0.4))
    v = n_flipped / n_total
    if v <= 0.10:
        return ("NORMAL", _FLAG_GREEN)
    elif v <= 0.40:
        return ("ELEVATED", _FLAG_AMBER)
    else:
        return ("HIGH RISK", _FLAG_RED)


def overall_impression(attack_results):
    """
    Produces a short, rule-based written summary from the actual computed
    metrics -- not a fabricated verdict. Picks the worst-flagged attack and
    explains why, the same way a lab report's "impression" line summarizes
    the panel rather than restating every number.
    """
    if not attack_results:
        return "No attacks were run for this job."

    worst = max(attack_results, key=lambda r: max(r["risk_score"] or 0, 0))
    label, _ = classify_risk_score(worst["risk_score"])
    drop_pct = (worst["accuracy_drop"] or 0) * 100

    if label == "HIGH RISK":
        return (
            f"The model showed a significant vulnerability under {worst['attack_type'].upper()}, "
            f"with accuracy dropping by {drop_pct:.1f} percentage points. This indicates the model's "
            f"decision boundary is easily crossed by small, deliberate perturbations and should not be "
            f"considered reliable in an adversarial setting without hardening."
        )
    elif label == "ELEVATED":
        return (
            f"The model showed moderate sensitivity under {worst['attack_type'].upper()}, "
            f"with accuracy dropping by {drop_pct:.1f} percentage points. This is within a range worth "
            f"monitoring; consider adversarial hardening if this model will face adversarial input in "
            f"production."
        )
    else:
        return (
            f"The model maintained accuracy within a normal range across all attacks tested "
            f"(worst case: {worst['attack_type'].upper()}, {drop_pct:.1f} point drop). No immediate "
            f"robustness concerns were identified for the attacks and epsilon budget evaluated."
        )


@results_bp.route("/<job_id>/download.pdf")
@login_required
def download_pdf(job_id):
    """
    Detailed, lab-report-style PDF: custom letterhead/footer, a reference-range
    legend, per-attack results flagged against those ranges (like abnormal
    values on a diagnostic panel), ART robustness metrics, a written overall
    impression, hardening before/after comparison (if a hardening job was
    run), and the full flipped-samples breakdown.
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
               safe_values_s3_key,
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
        LIMIT 30
    """, (job_id,))
    flipped_samples = cur.fetchall()

    cur.execute("""
        SELECT attack_type, clean_accuracy_before, robust_accuracy_before,
               clean_accuracy_after, robust_accuracy_after, robustness_improvement
        FROM hardening_results
        WHERE job_id = %s
        ORDER BY attack_type
    """, (job_id,))
    hardening_results = cur.fetchall()

    cur.close()
    conn.close()

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    margin = inch
    content_width = width - 2 * margin
    page_num = [1]

    # ---------- Letterhead / footer helpers ----------

    def draw_letterhead(first_page):
        """Full branded header on page 1; a slim running header afterward."""
        nonlocal_y = height - margin
        if first_page:
            # Accent bar
            c.setFillColorRGB(0.0, 0.55, 0.45)
            c.rect(0, height - 0.14 * inch, width, 0.14 * inch, fill=1, stroke=0)
            c.setFillColorRGB(0, 0, 0)

            c.setFont("Helvetica-Bold", 17)
            c.drawString(margin, nonlocal_y - 0.1 * inch, REPORT_TITLE)

            c.setFont("Helvetica-Oblique", 9)
            c.setFillColorRGB(0.35, 0.35, 0.35)
            c.drawString(margin, nonlocal_y - 0.32 * inch, "Robustness Diagnostic Report")
            c.setFillColorRGB(0, 0, 0)

            c.setLineWidth(1)
            c.line(margin, nonlocal_y - 0.42 * inch, width - margin, nonlocal_y - 0.42 * inch)
            return nonlocal_y - 0.65 * inch
        else:
            c.setFont("Helvetica-Bold", 8)
            c.setFillColorRGB(0.4, 0.4, 0.4)
            c.drawString(margin, nonlocal_y - 0.05 * inch, REPORT_TITLE)
            c.drawRightString(width - margin, nonlocal_y - 0.05 * inch, f"Job {job_id[:8]}")
            c.setFillColorRGB(0, 0, 0)
            c.setLineWidth(0.5)
            c.line(margin, nonlocal_y - 0.12 * inch, width - margin, nonlocal_y - 0.12 * inch)
            return nonlocal_y - 0.3 * inch

    def draw_footer():
        c.setFont("Helvetica", 8)
        c.setFillColorRGB(0.45, 0.45, 0.45)
        c.drawString(margin, 0.5 * inch, REPORT_FOOTER_CREDIT)
        c.drawCentredString(width / 2, 0.5 * inch, "AO8 \u2014 Adversarial Robustness Evaluator")
        c.drawRightString(width - margin, 0.5 * inch, f"Page {page_num[0]}")
        c.setFillColorRGB(0, 0, 0)

    def new_page():
        draw_footer()
        c.showPage()
        page_num[0] += 1
        y2 = draw_letterhead(first_page=False)
        c.setFont("Helvetica", 9)
        return y2

    def check_space(current_y, needed=0.3 * inch):
        if current_y < margin + needed:
            return new_page()
        return current_y

    def draw_flag_badge(x, y, label, color):
        """Small colored pill, like an abnormal-value flag on a lab report."""
        c.setFont("Helvetica-Bold", 7.5)
        text_width = c.stringWidth(label, "Helvetica-Bold", 7.5)
        pad = 5
        badge_w = text_width + pad * 2
        badge_h = 12
        c.setFillColorRGB(*color)
        c.roundRect(x, y - 2, badge_w, badge_h, 3, fill=1, stroke=0)
        c.setFillColorRGB(1, 1, 1)
        c.drawString(x + pad, y + 1.5, label)
        c.setFillColorRGB(0, 0, 0)
        return badge_w

    # ---------- Page 1: letterhead + job meta ----------
    y = draw_letterhead(first_page=True)

    c.setFont("Helvetica", 9)
    c.drawString(margin, y, f"Job ID:")
    c.setFont("Helvetica-Bold", 9)
    c.drawString(margin + 0.7 * inch, y, job_id)
    y -= 0.2 * inch

    c.setFont("Helvetica", 9)
    c.drawString(margin, y, f"Status: {job['status']}")
    y -= 0.2 * inch
    if job.get("queued_at"):
        c.drawString(margin, y, f"Queued: {job['queued_at']}")
        y -= 0.2 * inch
    if job.get("completed_at"):
        c.drawString(margin, y, f"Completed: {job['completed_at']}")
        y -= 0.2 * inch
    y -= 0.1 * inch

    # ---------- Overall robustness score ----------
    if attack_results:
        avg_drop = sum(r["accuracy_drop"] for r in attack_results) / len(attack_results)
        overall_score = round(max(min(100 - avg_drop * 100, 100), 0), 1)
        tag = "ROBUST" if overall_score >= 75 else "MODERATE RISK" if overall_score >= 50 else "HIGH RISK"
        tag_color = _FLAG_GREEN if overall_score >= 75 else _FLAG_AMBER if overall_score >= 50 else _FLAG_RED
    else:
        overall_score = None
        tag = "N/A"
        tag_color = (0.4, 0.4, 0.4)

    y = check_space(y, 0.6 * inch)
    c.setFillColorRGB(0.96, 0.96, 0.96)
    c.rect(margin, y - 0.55 * inch, content_width, 0.55 * inch, fill=1, stroke=0)
    c.setFillColorRGB(0, 0, 0)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin + 0.15 * inch, y - 0.22 * inch, "Overall Robustness Score")
    c.setFont("Helvetica-Bold", 20)
    score_text = f"{overall_score}%" if overall_score is not None else "N/A"
    c.drawString(margin + 0.15 * inch, y - 0.46 * inch, score_text)
    draw_flag_badge(margin + 1.6 * inch, y - 0.44 * inch, tag, tag_color)
    y -= 0.75 * inch

    # ---------- Reference range legend ----------
    y = check_space(y, 0.9 * inch)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(margin, y, "Reference Ranges")
    y -= 0.18 * inch
    c.setFont("Helvetica-Oblique", 7.5)
    c.setFillColorRGB(0.35, 0.35, 0.35)
    c.drawString(margin, y, "Every metric below is flagged against these fixed thresholds.")
    c.setFillColorRGB(0, 0, 0)
    y -= 0.22 * inch

    legend_rows = [
        ("Clean / robust accuracy", "\u2265 90% normal", "70\u201390% borderline", "< 70% low"),
        ("Accuracy drop", "\u2264 10 pts normal", "10\u201330 pts elevated", "> 30 pts high risk"),
        ("Risk score (0\u2013100)", "\u2264 30 normal", "30\u201360 elevated", "> 60 high risk"),
        ("Samples flipped", "\u2264 10% normal", "10\u201340% elevated", "> 40% high risk"),
    ]
    c.setFont("Helvetica", 7.5)
    for metric, g, a, r in legend_rows:
        y = check_space(y, 0.18 * inch)
        c.drawString(margin, y, metric)
        c.setFillColorRGB(*_FLAG_GREEN)
        c.drawString(margin + 2.0 * inch, y, g)
        c.setFillColorRGB(*_FLAG_AMBER)
        c.drawString(margin + 3.6 * inch, y, a)
        c.setFillColorRGB(*_FLAG_RED)
        c.drawString(margin + 5.2 * inch, y, r)
        c.setFillColorRGB(0, 0, 0)
        y -= 0.17 * inch
    y -= 0.15 * inch

    # ---------- Overall impression ----------
    y = check_space(y, 0.7 * inch)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Impression")
    y -= 0.2 * inch
    c.setFont("Helvetica", 9)
    impression_text = overall_impression([dict(r) for r in attack_results])
    y = draw_wrapped_text(c, impression_text, margin, y, content_width, 9, 0.15 * inch, check_space)
    y -= 0.15 * inch

    # ---------- Certification ----------
    y = check_space(y, 0.4 * inch)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Certified Robustness")
    y -= 0.22 * inch
    c.setFont("Helvetica", 9)
    if job.get("certification_method"):
        c.drawString(
            margin, y,
            f"Method: {job['certification_method']}   "
            f"Certified radius: {job['certified_radius']:.4f}   "
            f"Certified accuracy: {job['certified_accuracy']*100:.1f}%"
        )
        y -= 0.18 * inch
        c.setFont("Helvetica-Oblique", 8)
        c.setFillColorRGB(0.4, 0.4, 0.4)
        c.drawString(margin, y, "No perturbation within this radius can change a certified prediction.")
        c.setFillColorRGB(0, 0, 0)
        y -= 0.25 * inch
    else:
        c.setFillColorRGB(0.4, 0.4, 0.4)
        c.drawString(margin, y, "Not available for this model's framework.")
        c.setFillColorRGB(0, 0, 0)
        y -= 0.25 * inch

    # ---------- Per-attack results panel (flagged, like a lab panel) ----------
    y = check_space(y, 0.5 * inch)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Attack Results Panel")
    y -= 0.26 * inch

    for r in attack_results:
        y = check_space(y, 1.3 * inch)

        # Attack name + overall flag for this attack
        risk_label, risk_color = classify_risk_score(r["risk_score"])
        c.setFillColorRGB(0.97, 0.97, 0.97)
        c.rect(margin, y - 0.02 * inch, content_width, 0.24 * inch, fill=1, stroke=0)
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(margin + 0.1 * inch, y + 0.04 * inch, r["attack_type"].upper())
        draw_flag_badge(margin + content_width - 1.1 * inch, y + 0.03 * inch, risk_label, risk_color)
        y -= 0.32 * inch

        # Metric rows: label / value / flag
        metric_rows = [
            ("Clean accuracy", f"{r['clean_accuracy']*100:.1f}%", classify_accuracy(r["clean_accuracy"])),
            ("Robust accuracy", f"{r['robust_accuracy']*100:.1f}%", classify_accuracy(r["robust_accuracy"])),
            ("Accuracy drop", f"{r['accuracy_drop']*100:.1f} pts", classify_drop(r["accuracy_drop"])),
            ("Risk score", f"{r['risk_score']:.1f} / 100", classify_risk_score(r["risk_score"])),
            (
                "Samples flipped",
                f"{r['n_samples_flipped']} / {r['n_samples_total']} "
                f"({(r['n_samples_flipped']/r['n_samples_total']*100) if r['n_samples_total'] else 0:.1f}%)",
                classify_flip_rate(r["n_samples_flipped"], r["n_samples_total"]),
            ),
        ]

        c.setFont("Helvetica", 8.5)
        for label, value, (flag_label, flag_color) in metric_rows:
            y = check_space(y, 0.2 * inch)
            c.drawString(margin + 0.2 * inch, y, label)
            c.drawString(margin + 2.3 * inch, y, value)
            draw_flag_badge(margin + 4.2 * inch, y - 0.03 * inch, flag_label, flag_color)
            y -= 0.19 * inch

        # ART metrics (secondary, no flag — reference-only figures)
        art_parts = []
        if r.get("empirical_robustness_score") is not None:
            art_parts.append(f"empirical_robustness: {r['empirical_robustness_score']:.4f}")
        if r.get("clever_score") is not None:
            art_parts.append(f"CLEVER: {r['clever_score']:.4f}")
        if r.get("loss_sensitivity") is not None:
            art_parts.append(f"loss_sensitivity: {r['loss_sensitivity']:.4f}")
        if art_parts:
            y = check_space(y, 0.2 * inch)
            c.setFont("Helvetica-Oblique", 7.5)
            c.setFillColorRGB(0.4, 0.4, 0.4)
            c.drawString(margin + 0.2 * inch, y, "  \u00b7  ".join(art_parts))
            c.setFillColorRGB(0, 0, 0)
            y -= 0.2 * inch

        if r.get("safe_values_s3_key"):
            n_safe = r["n_samples_total"] - r["n_samples_flipped"]
            y = check_space(y, 0.18 * inch)
            c.setFont("Helvetica-Oblique", 7.5)
            c.setFillColorRGB(0.4, 0.4, 0.4)
            c.drawString(margin + 0.2 * inch, y, f"{n_safe} samples never flipped \u2014 safe-values export available in-app.")
            c.setFillColorRGB(0, 0, 0)
            y -= 0.2 * inch

        y -= 0.15 * inch

    # ---------- Hardening comparison (before/after), if run ----------
    if hardening_results:
        y = check_space(y, 0.5 * inch)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Hardening Comparison (Before / After)")
        y -= 0.22 * inch
        c.setFont("Helvetica-Oblique", 7.5)
        c.setFillColorRGB(0.4, 0.4, 0.4)
        c.drawString(margin, y, "Effect of adversarial training on the same attacks, holding data and epsilon fixed.")
        c.setFillColorRGB(0, 0, 0)
        y -= 0.26 * inch

        col_x = {
            "attack": margin,
            "clean_b": margin + 1.0 * inch,
            "robust_b": margin + 2.0 * inch,
            "clean_a": margin + 3.1 * inch,
            "robust_a": margin + 4.1 * inch,
            "improve": margin + 5.2 * inch,
        }
        y = check_space(y, 0.3 * inch)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(col_x["attack"], y, "ATTACK")
        c.drawString(col_x["clean_b"], y, "CLEAN (BEFORE)")
        c.drawString(col_x["robust_b"], y, "ROBUST (BEFORE)")
        c.drawString(col_x["clean_a"], y, "CLEAN (AFTER)")
        c.drawString(col_x["robust_a"], y, "ROBUST (AFTER)")
        c.drawString(col_x["improve"], y, "\u0394 ROBUST")
        y -= 0.06 * inch
        c.line(margin, y, width - margin, y)
        y -= 0.18 * inch

        c.setFont("Helvetica", 8)
        for h in hardening_results:
            y = check_space(y, 0.2 * inch)
            improvement_pct = (h["robustness_improvement"] or 0) * 100
            c.drawString(col_x["attack"], y, h["attack_type"].upper())
            c.drawString(col_x["clean_b"], y, f"{h['clean_accuracy_before']*100:.1f}%")
            c.drawString(col_x["robust_b"], y, f"{h['robust_accuracy_before']*100:.1f}%")
            c.drawString(col_x["clean_a"], y, f"{h['clean_accuracy_after']*100:.1f}%")
            c.drawString(col_x["robust_a"], y, f"{h['robust_accuracy_after']*100:.1f}%")
            if improvement_pct >= 0:
                c.setFillColorRGB(*_FLAG_GREEN)
                c.drawString(col_x["improve"], y, f"+{improvement_pct:.1f} pts")
            else:
                c.setFillColorRGB(*_FLAG_RED)
                c.drawString(col_x["improve"], y, f"{improvement_pct:.1f} pts")
            c.setFillColorRGB(0, 0, 0)
            y -= 0.19 * inch
        y -= 0.15 * inch

    # ---------- Flipped samples ----------
    y = check_space(y, 0.5 * inch)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Most Fragile Samples (smallest perturbation to flip)")
    y -= 0.15 * inch
    c.setFont("Helvetica-Oblique", 7.5)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.setFillColorRGB(0, 0, 0)
    y -= 0.24 * inch

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
                c.rect(margin - 0.05 * inch, y - 0.05 * inch, content_width + 0.1 * inch, 0.18 * inch, fill=1, stroke=0)
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

    # ---------- Disclaimer ----------
    y = check_space(y, 0.5 * inch)
    c.setFont("Helvetica-Oblique", 7.5)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    disclaimer = (
        "This report reflects results for the model and dataset as uploaded, evaluated at the "
        "epsilon budget configured for this job. Reference ranges are general guidance, not a "
        "certification of production-readiness. Results assume the uploaded dataset was scaled "
        "and formatted to match what the model was trained on."
    )
    y = draw_wrapped_text(c, disclaimer, margin, y, content_width, 7.5, 0.14 * inch, check_space)

    # ---------- Finalize ----------
    draw_footer()
    c.save()
    buf.seek(0)
    c.setFillColorRGB(0, 0, 0)

    response = make_response(buf.read())
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = f"attachment; filename=ao8_report_{job_id}.pdf"
    return response


def draw_wrapped_text(c, text, x, y, max_width, font_size, line_height, check_space_fn):
    """
    Simple word-wrap for reportlab (which has no built-in paragraph flow on
    a bare Canvas). Wraps `text` to `max_width` at `font_size`, drawing each
    line and advancing y, calling check_space_fn before each line so long
    blocks (like the impression or disclaimer) page-break cleanly.
    """
    words = text.split()
    line = ""
    for word in words:
        trial = f"{line} {word}".strip()
        if c.stringWidth(trial, "Helvetica", font_size) <= max_width:
            line = trial
        else:
            y = check_space_fn(y, line_height)
            c.drawString(x, y, line)
            y -= line_height
            line = word
    if line:
        y = check_space_fn(y, line_height)
        c.drawString(x, y, line)
        y -= line_height
    c.setFillColorRGB(0, 0, 0)
    return y


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