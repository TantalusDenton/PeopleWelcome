"""S3 service for image upload, download, and presigned URL generation."""

import os
import uuid
from pathlib import Path
from typing import BinaryIO, Optional

import boto3
from botocore.exceptions import ClientError

# S3 configuration from environment
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME", "peoplewelcome-images")
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Default presigned URL expiration (1 hour)
DEFAULT_EXPIRATION = 3600


def get_s3_client():
    """Create and return an S3 client with configured credentials."""
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        return boto3.client(
            "s3",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
    # Fall back to default credential chain (IAM role, env vars, etc.)
    return boto3.client("s3", region_name=AWS_REGION)


def generate_s3_key(user_id: str, ai_id: str, filename: str) -> str:
    """
    Generate a unique S3 key for an image.

    Structure: users/{user_id}/ais/{ai_id}/{uuid}.{ext}
    """
    ext = Path(filename).suffix.lower() if filename else ".jpg"
    image_id = str(uuid.uuid4())
    return f"users/{user_id}/ais/{ai_id}/{image_id}{ext}"


def upload_image(
    file: BinaryIO,
    user_id: str,
    ai_id: str,
    filename: str,
    content_type: Optional[str] = None
) -> dict:
    """
    Upload an image to S3.

    Args:
        file: File-like object to upload
        user_id: Owner's user ID
        ai_id: Associated AI ID
        filename: Original filename (used for extension)
        content_type: MIME type of the file

    Returns:
        dict with s3_key and image_id
    """
    s3_client = get_s3_client()
    s3_key = generate_s3_key(user_id, ai_id, filename)

    # Extract image_id from the s3_key
    image_id = Path(s3_key).stem

    extra_args = {}
    if content_type:
        extra_args["ContentType"] = content_type

    try:
        s3_client.upload_fileobj(
            file,
            AWS_BUCKET_NAME,
            s3_key,
            ExtraArgs=extra_args
        )
        return {
            "s3_key": s3_key,
            "image_id": image_id,
            "bucket": AWS_BUCKET_NAME,
            "region": AWS_REGION
        }
    except ClientError as e:
        raise RuntimeError(f"Failed to upload image to S3: {e}")


def upload_image_from_path(
    file_path: str,
    user_id: str,
    ai_id: str,
    content_type: Optional[str] = None
) -> dict:
    """
    Upload an image from a local file path to S3.

    Args:
        file_path: Local path to the file
        user_id: Owner's user ID
        ai_id: Associated AI ID
        content_type: MIME type of the file

    Returns:
        dict with s3_key and image_id
    """
    path = Path(file_path)
    with open(path, "rb") as f:
        return upload_image(f, user_id, ai_id, path.name, content_type)


def get_presigned_url(s3_key: str, expiration: int = DEFAULT_EXPIRATION) -> str:
    """
    Generate a presigned URL for downloading an image.

    Args:
        s3_key: The S3 object key
        expiration: URL expiration time in seconds (default 1 hour)

    Returns:
        Presigned URL string
    """
    s3_client = get_s3_client()

    try:
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": AWS_BUCKET_NAME,
                "Key": s3_key
            },
            ExpiresIn=expiration
        )
        return url
    except ClientError as e:
        raise RuntimeError(f"Failed to generate presigned URL: {e}")


def get_presigned_upload_url(
    s3_key: str,
    content_type: str = "image/jpeg",
    expiration: int = DEFAULT_EXPIRATION
) -> dict:
    """
    Generate a presigned URL for client-side upload.

    Args:
        s3_key: The S3 object key
        content_type: MIME type of the file to upload
        expiration: URL expiration time in seconds

    Returns:
        dict with url and fields for POST upload
    """
    s3_client = get_s3_client()

    try:
        response = s3_client.generate_presigned_post(
            AWS_BUCKET_NAME,
            s3_key,
            Fields={"Content-Type": content_type},
            Conditions=[
                {"Content-Type": content_type},
                ["content-length-range", 1, 10485760]  # 1 byte to 10MB
            ],
            ExpiresIn=expiration
        )
        return response
    except ClientError as e:
        raise RuntimeError(f"Failed to generate presigned upload URL: {e}")


def delete_image(s3_key: str) -> bool:
    """
    Delete an image from S3.

    Args:
        s3_key: The S3 object key to delete

    Returns:
        True if deleted successfully
    """
    s3_client = get_s3_client()

    try:
        s3_client.delete_object(
            Bucket=AWS_BUCKET_NAME,
            Key=s3_key
        )
        return True
    except ClientError as e:
        raise RuntimeError(f"Failed to delete image from S3: {e}")


def check_image_exists(s3_key: str) -> bool:
    """
    Check if an image exists in S3.

    Args:
        s3_key: The S3 object key to check

    Returns:
        True if exists, False otherwise
    """
    s3_client = get_s3_client()

    try:
        s3_client.head_object(Bucket=AWS_BUCKET_NAME, Key=s3_key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise RuntimeError(f"Failed to check image existence: {e}")


def list_images_by_prefix(prefix: str, max_keys: int = 1000) -> list:
    """
    List all images under a given S3 prefix.

    Args:
        prefix: S3 key prefix (e.g., "users/{user_id}/ais/{ai_id}/")
        max_keys: Maximum number of keys to return

    Returns:
        List of S3 keys
    """
    s3_client = get_s3_client()

    try:
        response = s3_client.list_objects_v2(
            Bucket=AWS_BUCKET_NAME,
            Prefix=prefix,
            MaxKeys=max_keys
        )

        keys = []
        if "Contents" in response:
            keys = [obj["Key"] for obj in response["Contents"]]

        return keys
    except ClientError as e:
        raise RuntimeError(f"Failed to list images: {e}")


def copy_image(source_key: str, dest_key: str) -> bool:
    """
    Copy an image to a new location within the same bucket.

    Args:
        source_key: Source S3 key
        dest_key: Destination S3 key

    Returns:
        True if copied successfully
    """
    s3_client = get_s3_client()

    try:
        s3_client.copy_object(
            Bucket=AWS_BUCKET_NAME,
            CopySource={"Bucket": AWS_BUCKET_NAME, "Key": source_key},
            Key=dest_key
        )
        return True
    except ClientError as e:
        raise RuntimeError(f"Failed to copy image: {e}")


def get_image_metadata(s3_key: str) -> dict:
    """
    Get metadata for an image in S3.

    Args:
        s3_key: The S3 object key

    Returns:
        dict with metadata (content_type, size, last_modified)
    """
    s3_client = get_s3_client()

    try:
        response = s3_client.head_object(Bucket=AWS_BUCKET_NAME, Key=s3_key)
        return {
            "content_type": response.get("ContentType"),
            "size": response.get("ContentLength"),
            "last_modified": response.get("LastModified").isoformat() if response.get("LastModified") else None,
            "etag": response.get("ETag", "").strip('"')
        }
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return None
        raise RuntimeError(f"Failed to get image metadata: {e}")
