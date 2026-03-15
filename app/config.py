from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    aws_region: str = "us-east-1"
    s3_bucket: str = "finetuning-platform"
    dynamodb_jobs_table: str = "finetuning-jobs"
    dynamodb_models_table: str = "finetuned-models"
    sagemaker_role_arn: str = ""
    max_dataset_size_mb: int = 500
    min_dataset_rows: int = 10
    default_instance_type: str = "ml.g5.xlarge"

    model_config = {"env_prefix": "FT_"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
