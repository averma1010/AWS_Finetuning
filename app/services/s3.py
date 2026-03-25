import boto3
from app.config import get_settings


def _get_client():
    settings = get_settings()
    return boto3.client(
        "s3",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key
    )


def upload_dataset(dataset_id: str, file_content: bytes, filename: str) -> str:
    settings = get_settings()
    client = _get_client()
    s3_key = f"datasets/{dataset_id}/data.jsonl"
    client.put_object(
        Bucket=settings.s3_bucket,
        Key=s3_key,
        Body=file_content,
    )
    return f"s3://{settings.s3_bucket}/{s3_key}"



def get_dataset_s3_uri(dataset_id: str) -> str:
    settings = get_settings()
    return f"s3://{settings.s3_bucket}/datasets/{dataset_id}"


def get_model_artifact_path(job_id: str) -> str:
    settings = get_settings()
    return f"s3://{settings.s3_bucket}/models/{job_id}"


def generate_presigned_url(s3_key: str, expiration: int = 3600) -> str:
    client = _get_client()
    settings = get_settings()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.s3_bucket, "Key": s3_key},
        ExpiresIn=expiration,
    )
