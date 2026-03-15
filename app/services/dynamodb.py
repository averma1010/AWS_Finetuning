import boto3
from typing import Optional, List, Dict
from datetime import datetime, timezone
from app.config import get_settings


def _get_table(table_name: str):
    settings = get_settings()
    dynamodb = boto3.resource("dynamodb", region_name=settings.aws_region)
    return dynamodb.Table(table_name)


# --- Jobs ---

def create_job(job_data: dict) -> dict:
    settings = get_settings()
    table = _get_table(settings.dynamodb_jobs_table)
    now = datetime.now(timezone.utc).isoformat()
    job_data["created_at"] = now
    job_data["updated_at"] = now
    job_data["status"] = "pending"
    table.put_item(Item=job_data)
    return job_data


def update_job(job_id: str, updates: dict) -> dict:
    settings = get_settings()
    table = _get_table(settings.dynamodb_jobs_table)
    updates["updated_at"] = datetime.now(timezone.utc).isoformat()

    update_expr_parts = []
    expr_attr_values = {}
    expr_attr_names = {}
    for i, (key, value) in enumerate(updates.items()):
        placeholder = f":val{i}"
        name_placeholder = f"#attr{i}"
        update_expr_parts.append(f"{name_placeholder} = {placeholder}")
        expr_attr_values[placeholder] = value
        expr_attr_names[name_placeholder] = key

    response = table.update_item(
        Key={"job_id": job_id},
        UpdateExpression="SET " + ", ".join(update_expr_parts),
        ExpressionAttributeValues=expr_attr_values,
        ExpressionAttributeNames=expr_attr_names,
        ReturnValues="ALL_NEW",
    )
    return response["Attributes"]


def get_job(job_id: str) -> Optional[dict]:
    settings = get_settings()
    table = _get_table(settings.dynamodb_jobs_table)
    response = table.get_item(Key={"job_id": job_id})
    return response.get("Item")


def list_jobs(user_id: Optional[str] = None) -> List[dict]:
    settings = get_settings()
    table = _get_table(settings.dynamodb_jobs_table)
    if user_id:
        response = table.scan(
            FilterExpression=boto3.dynamodb.conditions.Attr("user_id").eq(user_id)
        )
    else:
        response = table.scan()
    return response.get("Items", [])


# --- Models ---

def create_model(model_data: dict) -> dict:
    settings = get_settings()
    table = _get_table(settings.dynamodb_models_table)
    now = datetime.now(timezone.utc).isoformat()
    model_data["created_at"] = now
    model_data["status"] = "ready"
    table.put_item(Item=model_data)
    return model_data


def update_model(model_id: str, updates: dict) -> dict:
    settings = get_settings()
    table = _get_table(settings.dynamodb_models_table)

    update_expr_parts = []
    expr_attr_values = {}
    expr_attr_names = {}
    for i, (key, value) in enumerate(updates.items()):
        placeholder = f":val{i}"
        name_placeholder = f"#attr{i}"
        update_expr_parts.append(f"{name_placeholder} = {placeholder}")
        expr_attr_values[placeholder] = value
        expr_attr_names[name_placeholder] = key

    response = table.update_item(
        Key={"model_id": model_id},
        UpdateExpression="SET " + ", ".join(update_expr_parts),
        ExpressionAttributeValues=expr_attr_values,
        ExpressionAttributeNames=expr_attr_names,
        ReturnValues="ALL_NEW",
    )
    return response["Attributes"]


def get_model(model_id: str) -> Optional[dict]:
    settings = get_settings()
    table = _get_table(settings.dynamodb_models_table)
    response = table.get_item(Key={"model_id": model_id})
    return response.get("Item")


def list_models() -> List[dict]:
    settings = get_settings()
    table = _get_table(settings.dynamodb_models_table)
    response = table.scan()
    return response.get("Items", [])
