from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_default_region: str = "us-east-1"
    dynamodb_region: str = "us-east-1"
    s3_bucket: str = "finetuning-platform"
    dynamodb_jobs_table: str = "finetuning-jobs"
    dynamodb_models_table: str = "finetuned-models"
    sagemaker_role_arn: str = ""
    hf_token: str = ""
    max_dataset_size_mb: int = 500
    min_dataset_rows: int = 10
    default_instance_type: str = "ml.g5.xlarge"
    default_serverless_memory_mb: int = 4096
    serverless_max_concurrency: int = 10
    otel_enabled: bool = False
    otel_endpoint: str = ""
    otel_service_name: str = "finetuning-service"

    model_config = {"env_prefix": "FT_", "env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
