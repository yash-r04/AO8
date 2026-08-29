# app/services/storage.py
import boto3
from app.config import Config

s3 = boto3.client(
    "s3",
    region_name=Config.AWS_REGION,
    aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
)

def generate_upload_url(s3_key: str, expires_in: int = 900) -> str:
    """
    15-minute presigned URL for direct browser → S3 upload.
    Model never passes through your Flask server.
    """
    return s3.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": Config.S3_BUCKET,
            "Key": s3_key,
            "ServerSideEncryption": "AES256",
        },
        ExpiresIn=expires_in,
    )

def generate_download_url(s3_key: str, expires_in: int = 300) -> str:
    """5-minute presigned download URL."""
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": Config.S3_BUCKET, "Key": s3_key},
        ExpiresIn=expires_in,
    )

def download_file(s3_key: str) -> bytes:
    """Download file bytes — used by worker to load model."""
    response = s3.get_object(Bucket=Config.S3_BUCKET, Key=s3_key)
    return response["Body"].read()

def delete_file(s3_key: str) -> None:
    """Hard delete from S3."""
    s3.delete_object(Bucket=Config.S3_BUCKET, Key=s3_key)