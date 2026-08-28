# app/config.py
import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    SECRET_KEY         = os.getenv("SECRET_KEY")
    DATABASE_URL       = os.getenv("DATABASE_URL")
    REDIS_URL          = os.getenv("REDIS_URL")
    AWS_REGION         = os.getenv("AWS_REGION", "ap-south-1")
    AWS_ACCESS_KEY_ID  = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    S3_BUCKET          = os.getenv("S3_BUCKET")
    COGNITO_DOMAIN     = os.getenv("COGNITO_DOMAIN")
    COGNITO_CLIENT_ID  = os.getenv("COGNITO_CLIENT_ID")
    COGNITO_CLIENT_SECRET = os.getenv("COGNITO_CLIENT_SECRET")
    COGNITO_REDIRECT_URI  = os.getenv("COGNITO_REDIRECT_URI")
    GITHUB_CLIENT_ID      = os.getenv("GITHUB_CLIENT_ID")
    GITHUB_CLIENT_SECRET  = os.getenv("GITHUB_CLIENT_SECRET")
    GITHUB_REDIRECT_URI   = os.getenv("GITHUB_REDIRECT_URI")