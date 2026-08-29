# check_ids.py
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import os

load_dotenv()
conn = psycopg2.connect(os.getenv("DATABASE_URL"), cursor_factory=RealDictCursor)
cur = conn.cursor()

print("--- YOUR MODELS ---")
cur.execute("SELECT id, name, framework FROM ml_models WHERE upload_status = 'ready'")
for row in cur.fetchall():
    print(dict(row))

print("\n--- YOUR DATASETS ---")
cur.execute("SELECT id, name, n_samples FROM datasets")
for row in cur.fetchall():
    print(dict(row))

cur.close()
conn.close()