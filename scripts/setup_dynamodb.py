"""Create DynamoDB tables for the finetuning platform."""
from typing import Optional
import boto3
import sys
import os
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import get_settings

REGION = "us-east-1"


def create_tables(endpoint_url: Optional[str] = None):
    settings = get_settings()
    kwargs = {
        "region_name": settings.aws_region,
        "aws_access_key_id": settings.aws_access_key_id,
        "aws_secret_access_key": settings.aws_secret_access_key
    }
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url

    dynamodb = boto3.client("dynamodb", **kwargs)

    # Jobs table
    try:
        dynamodb.create_table(
            TableName=settings.dynamodb_jobs_table,
            KeySchema=[{"AttributeName": "job_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "job_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        print(f"Created table: {settings.dynamodb_jobs_table}")
    except dynamodb.exceptions.ResourceInUseException:
        print(f"Table {settings.dynamodb_jobs_table} already exists")

    # Models table
    try:
        dynamodb.create_table(
            TableName=settings.dynamodb_models_table,
            KeySchema=[{"AttributeName": "model_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "model_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        print(f"Created table: {settings.dynamodb_models_table}")
    except dynamodb.exceptions.ResourceInUseException:
        print(f"Table {settings.dynamodb_models_table} already exists")


if __name__ == "__main__":
    endpoint = sys.argv[1] if len(sys.argv) > 1 else None
    create_tables(endpoint)
    print("Done!")
