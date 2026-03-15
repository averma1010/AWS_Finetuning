from typing import Optional
from pydantic import BaseModel


class FinetunedModel(BaseModel):
    model_id: str
    job_id: str
    base_model: str
    s3_artifact_path: str
    endpoint_name: Optional[str] = None
    status: str
    created_at: str


class ModelListResponse(BaseModel):
    base_models: list
    finetuned_models: list


class DeployRequest(BaseModel):
    instance_type: Optional[str] = None
    instance_count: int = 1


class DeployResponse(BaseModel):
    model_id: str
    endpoint_name: str
    status: str
