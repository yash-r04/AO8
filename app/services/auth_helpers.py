# app/services/auth_helpers.py
import psycopg2
from psycopg2.extras import RealDictCursor
from app.db import get_db
from datetime import datetime

def get_or_create_user(provider: str, provider_id: str, email: str, name: str) -> dict:
    """
    Look up user by provider + provider_id.
    If they don't exist, create them.
    Either way, update last_login and return the user dict.
    """
    conn = get_db()
    cur = conn.cursor()

    # Check if user exists
    cur.execute("""
        SELECT * FROM users
        WHERE provider = %s AND provider_id = %s
    """, (provider, provider_id))
    user = cur.fetchone()

    if user:
        # Update last login time
        cur.execute("""
            UPDATE users SET last_login = %s WHERE id = %s
        """, (datetime.utcnow(), user["id"]))
    else:
        # Create new user
        cur.execute("""
            INSERT INTO users (email, name, provider, provider_id, last_login)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING *
        """, (email, name, provider, provider_id, datetime.utcnow()))
        user = cur.fetchone()

    conn.commit()
    cur.close()
    conn.close()

    return dict(user)

def login_required(f):
    """Decorator — protects routes that need a logged-in user."""
    from functools import wraps
    from flask import session, redirect, url_for
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("auth.login_page"))
        return f(*args, **kwargs)
    return decorated