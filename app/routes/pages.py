# app/routes/pages.py
from flask import Blueprint, render_template, session
from app.services.auth_helpers import login_required

pages_bp = Blueprint("pages", __name__)


@pages_bp.route("/app")
@login_required
def app_index():
    """The Test page: upload model + dataset, run an evaluation."""
    return render_template("app.html", user=session.get("user"))


@pages_bp.route("/learn")
def learn():
    """Public page explaining FGSM / PGD / C&W. No login required."""
    return render_template("learn.html", user=session.get("user"))