from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


class EndpointType(str, Enum):
    real_time = "real-time"
    serverless = "serverless"


class ServerlessConfig(BaseModel):
    memory_size_in_mb: int = Field(default=4096, ge=1024, le=6144)
    max_concurrency: int = Field(default=10, ge=1, le=200)


class FinetunedModel(BaseModel):
    model_id: str
    job_id: str
    base_model: str
    s3_artifact_path: str
    endpoint_name: Optional[str] = None
    endpoint_type: Optional[str] = None
    status: str
    created_at: str


class ModelListResponse(BaseModel):
    base_models: list
    finetuned_models: list


class DeployRequest(BaseModel):
    endpoint_type: EndpointType = Field(default=EndpointType.serverless)
    instance_type: Optional[str] = Field(default=None, description="For real-time only")
    instance_count: int = Field(default=1, ge=1, le=10, description="For real-time only")
    serverless_config: Optional[ServerlessConfig] = Field(default=None, description="For serverless only")


class DeployResponse(BaseModel):
    model_id: str
    endpoint_name: str
    status: str
