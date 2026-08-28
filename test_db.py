import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()
print(repr(os.getenv("DATABASE_URL")))

try:
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    print("✅ Connected to RDS successfully")
    conn.close()
except Exception as e:
    print(f"❌ Failed: {e}")