"""Create DynamoDB tables for the finetuning platform."""
from typing import Optional
import boto3
import sys

REGION = "us-east-1"


def create_tables(endpoint_url: Optional[str] = None):
    kwargs = {"region_name": REGION}
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url

    dynamodb = boto3.client("dynamodb", **kwargs)

    # Jobs table
    try:
        dynamodb.create_table(
            TableName="finetuning-jobs",
            KeySchema=[{"AttributeName": "job_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "job_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        print("Created table: finetuning-jobs")
    except dynamodb.exceptions.ResourceInUseException:
        print("Table finetuning-jobs already exists")

    # Models table
    try:
        dynamodb.create_table(
            TableName="finetuned-models",
            KeySchema=[{"AttributeName": "model_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "model_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        print("Created table: finetuned-models")
    except dynamodb.exceptions.ResourceInUseException:
        print("Table finetuned-models already exists")


if __name__ == "__main__":
    endpoint = sys.argv[1] if len(sys.argv) > 1 else None
    create_tables(endpoint)
    print("Done!")
