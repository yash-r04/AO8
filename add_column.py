# add_hardening_columns.py
import psycopg2
from app.config import Config

conn = psycopg2.connect(Config.DATABASE_URL)
cur = conn.cursor()

cur.execute("""
    ALTER TABLE evaluation_jobs
        ADD COLUMN IF NOT EXISTS hardening_requested BOOLEAN DEFAULT FALSE,
        ADD COLUMN IF NOT EXISTS hardening_status VARCHAR(20),
        ADD COLUMN IF NOT EXISTS hardening_error TEXT,
        ADD COLUMN IF NOT EXISTS hardened_model_s3_key TEXT;
""")

cur.execute("""
    CREATE TABLE IF NOT EXISTS hardening_results (
        id UUID PRIMARY KEY,
        job_id UUID NOT NULL REFERENCES evaluation_jobs(id) ON DELETE CASCADE,
        attack_type VARCHAR(20) NOT NULL,
        clean_accuracy_before FLOAT,
        robust_accuracy_before FLOAT,
        clean_accuracy_after FLOAT,
        robust_accuracy_after FLOAT,
        robustness_improvement FLOAT,
        created_at TIMESTAMP DEFAULT NOW()
    );
""")

conn.commit()
cur.close()
conn.close()
print("Hardening columns and table added.")