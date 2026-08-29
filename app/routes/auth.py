# app/routes/auth.py
import requests
from flask import (
    Blueprint, redirect, request,
    session, url_for, jsonify
)
from app.config import Config
from app.services.auth_helpers import get_or_create_user
import urllib.parse

auth_bp = Blueprint("auth", __name__)


# ── GOOGLE via COGNITO ────────────────────────────────────────────

@auth_bp.route("/google")
def google_login():
    """Redirect user to Cognito hosted UI (which shows Google button)."""
    params = {
        "client_id":     Config.COGNITO_CLIENT_ID,
        "response_type": "code",
        "scope":         "email openid",
        "redirect_uri":  Config.COGNITO_REDIRECT_URI,
    }
    query = urllib.parse.urlencode(params)
    cognito_url = f"https://{Config.COGNITO_DOMAIN}/oauth2/authorize?{query}"
    return redirect(cognito_url)


@auth_bp.route("/callback")
def google_callback():
    """
    Cognito redirects here after Google login.
    We get a 'code' in the URL, exchange it for user info.
    """
    code = request.args.get("code")
    if not code:
        return "Login failed — no code returned", 400

    # Exchange code for tokens
    token_response = requests.post(
        f"https://{Config.COGNITO_DOMAIN}/oauth2/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type":   "authorization_code",
            "client_id":    Config.COGNITO_CLIENT_ID,
            "client_secret": Config.COGNITO_CLIENT_SECRET,
            "redirect_uri": Config.COGNITO_REDIRECT_URI,
            "code":         code,
        }
    )

    if token_response.status_code != 200:
        return f"Token exchange failed: {token_response.text}", 400

    tokens = token_response.json()
    access_token = tokens["access_token"]

    # Use access token to get user info from Cognito
    userinfo_response = requests.get(
        f"https://{Config.COGNITO_DOMAIN}/oauth2/userInfo",
        headers={"Authorization": f"Bearer {access_token}"}
    )
    userinfo = userinfo_response.json()

    # userinfo looks like:
    # {"sub": "uuid", "email": "user@gmail.com", "name": "Yashaswini"}

    user = get_or_create_user(
        provider="google",
        provider_id=userinfo["sub"],
        email=userinfo["email"],
        name=userinfo.get("name", ""),
    )

    # Save to session
    session["user"] = {
        "id":       str(user["id"]),
        "email":    user["email"],
        "name":     user["name"],
        "provider": "google",
    }

    return redirect(url_for("pages.app_index"))


# ── GITHUB DIRECT OAUTH ───────────────────────────────────────────

@auth_bp.route("/github")
def github_login():
    """Redirect user to GitHub OAuth page."""
    params = {
        "client_id":    Config.GITHUB_CLIENT_ID,
        "redirect_uri": Config.GITHUB_REDIRECT_URI,
        "scope":        "user:email",
    }
    query = urllib.parse.urlencode(params)
    return redirect(f"https://github.com/login/oauth/authorize?{query}")


@auth_bp.route("/github/callback")
def github_callback():
    """
    GitHub redirects here after user approves.
    Exchange code for token, then get user email.
    """
    code = request.args.get("code")
    if not code:
        return "Login failed — no code from GitHub", 400

    # Exchange code for access token
    token_response = requests.post(
        "https://github.com/login/oauth/access_token",
        headers={"Accept": "application/json"},
        data={
            "client_id":     Config.GITHUB_CLIENT_ID,
            "client_secret": Config.GITHUB_CLIENT_SECRET,
            "code":          code,
        }
    )
    tokens = token_response.json()
    access_token = tokens.get("access_token")

    if not access_token:
        return "GitHub token exchange failed", 400

    headers = {"Authorization": f"Bearer {access_token}"}

    # Get basic profile
    profile = requests.get("https://api.github.com/user", headers=headers).json()

    # Get verified email — profile email can be null if user hid it
    emails = requests.get("https://api.github.com/user/emails", headers=headers).json()
    primary_email = next(
        (e["email"] for e in emails if e["primary"] and e["verified"]),
        None
    )

    if not primary_email:
        return "No verified email on your GitHub account", 400

    user = get_or_create_user(
        provider="github",
        provider_id=str(profile["id"]),
        email=primary_email,
        name=profile.get("name") or profile.get("login", ""),
    )

    session["user"] = {
        "id":       str(user["id"]),
        "email":    user["email"],
        "name":     user["name"],
        "provider": "github",
    }

    return redirect(url_for("pages.app_index"))


# ── LOGOUT ────────────────────────────────────────────────────────

@auth_bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


# ── CURRENT USER (useful for your frontend JS) ────────────────────

@auth_bp.route("/me")
def me():
    user = session.get("user")
    if not user:
        return jsonify({"error": "Not logged in"}), 401
    return jsonify(user)