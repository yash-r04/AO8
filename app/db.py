# app/db.py
import psycopg2
from psycopg2.extras import RealDictCursor
from app.config import Config

def get_db():
    conn = psycopg2.connect(Config.DATABASE_URL, cursor_factory=RealDictCursor)
    return conn