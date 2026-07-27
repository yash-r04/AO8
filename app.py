import os
from functools import wraps
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, session
from flask_dance.contrib.google import make_google_blueprint, google
from flask_dance.contrib.github import make_github_blueprint, github

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me-in-production")

# Allow HTTP in dev — remove this line before deploying to production
os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")

# ── Google OAuth ──────────────────────────────────────────────────────────────
google_bp = make_google_blueprint(
    client_id=os.environ.get("GOOGLE_CLIENT_ID"),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
    scope=["openid", "https://www.googleapis.com/auth/userinfo.email",
           "https://www.googleapis.com/auth/userinfo.profile"],
    redirect_to="google_done",
)
app.register_blueprint(google_bp, url_prefix="/auth")

# ── GitHub OAuth ──────────────────────────────────────────────────────────────
github_bp = make_github_blueprint(
    client_id=os.environ.get("GITHUB_CLIENT_ID"),
    client_secret=os.environ.get("GITHUB_CLIENT_SECRET"),
    scope="read:user,user:email",
    redirect_to="github_done",
)
app.register_blueprint(github_bp, url_prefix="/auth")


# ── Auth helpers ──────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ── OAuth callbacks ───────────────────────────────────────────────────────────
@app.route("/auth/google/done")
def google_done():
    if not google.authorized:
        return redirect(url_for("login"))
    resp = google.get("/oauth2/v2/userinfo")
    if not resp.ok:
        return redirect(url_for("login"))
    info = resp.json()
    session["user"] = {
        "name":     info.get("name", "User"),
        "email":    info.get("email", ""),
        "avatar":   info.get("picture", ""),
        "provider": "google",
    }
    return redirect(url_for("index"))


@app.route("/auth/github/done")
def github_done():
    if not github.authorized:
        return redirect(url_for("login"))
    resp = github.get("/user")
    if not resp.ok:
        return redirect(url_for("login"))
    info = resp.json()
    # GitHub may not expose email publicly — fall back to login handle
    name = info.get("name") or info.get("login", "User")
    session["user"] = {
        "name":     name,
        "email":    info.get("email", ""),
        "avatar":   info.get("avatar_url", ""),
        "provider": "github",
    }
    return redirect(url_for("index"))


# ── Auth routes ───────────────────────────────────────────────────────────────
@app.route("/login")
def login():
    if "user" in session:
        return redirect(url_for("index"))
    return render_template("login.html")


@app.route("/logout")
def logout():
    # Also clear Flask-Dance tokens from session
    for key in list(session.keys()):
        if key.startswith(("google_oauth_", "github_oauth_")):
            session.pop(key)
    session.pop("user", None)
    return redirect(url_for("login"))


# ── App routes (all protected) ────────────────────────────────────────────────
@app.route("/")
@login_required
def index():
    return render_template("index.html")


@app.route("/upload")
@login_required
def upload():
    return render_template("upload.html")


@app.route("/benchmark")
@login_required
def benchmark():
    return render_template("benchmark.html")


@app.route("/report")
@login_required
def report():
    return render_template("report.html")


if __name__ == "__main__":
    app.run(debug=True)