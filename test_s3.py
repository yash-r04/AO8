# test_s3.py
import boto3
from dotenv import load_dotenv
import os

load_dotenv()

s3 = boto3.client(
    "s3",
    region_name=os.getenv("AWS_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

bucket = os.getenv("S3_BUCKET")

# Upload a tiny test file
s3.put_object(Bucket=bucket, Key="test/hello.txt", Body=b"AO8 S3 test")
print("✅ Upload worked")

# Read it back
obj = s3.get_object(Bucket=bucket, Key="test/hello.txt")
print("✅ Download worked:", obj["Body"].read())

# Clean up
s3.delete_object(Bucket=bucket, Key="test/hello.txt")
print("✅ Delete worked")