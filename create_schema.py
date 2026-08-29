# create_schema.py
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

conn = psycopg2.connect(os.getenv("DATABASE_URL"))
cur = conn.cursor()

#cur.executescript = cur.execute  # psycopg2 uses execute

schema = """
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    provider VARCHAR(50) NOT NULL,
    provider_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ,
    UNIQUE(provider, provider_id)
);

CREATE TABLE IF NOT EXISTS ml_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    framework VARCHAR(50),
    s3_key VARCHAR(512) NOT NULL,
    file_size_bytes BIGINT,
    upload_status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS evaluation_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    model_id UUID REFERENCES ml_models(id),
    status VARCHAR(50) DEFAULT 'queued',
    attack_config JSONB NOT NULL,
    celery_task_id VARCHAR(255),
    queued_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES evaluation_jobs(id) ON DELETE CASCADE,
    attack_type VARCHAR(50),
    clean_accuracy FLOAT,
    robust_accuracy FLOAT,
    accuracy_drop FLOAT,
    risk_score FLOAT,
    n_samples_total INTEGER,
    n_samples_flipped INTEGER,
    report_s3_key VARCHAR(512),
    safe_values_s3_key VARCHAR(512),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sample_level_results (
    id BIGSERIAL PRIMARY KEY,
    result_id UUID REFERENCES evaluation_results(id) ON DELETE CASCADE,
    sample_index INTEGER,
    original_label INTEGER,
    clean_prediction INTEGER,
    adv_prediction INTEGER,
    was_flipped BOOLEAN,
    perturbation_l2 FLOAT,
    most_perturbed_features JSONB
);

CREATE INDEX IF NOT EXISTS idx_jobs_user ON evaluation_jobs(user_id, queued_at DESC);
CREATE INDEX IF NOT EXISTS idx_results_job ON evaluation_results(job_id);
CREATE INDEX IF NOT EXISTS idx_samples_result ON sample_level_results(result_id, was_flipped);
"""

cur.execute(schema)
conn.commit()
print("✅ Schema created successfully")

cur.close()
conn.close()