from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


class FinetuneMethod(str, Enum):
    sft = "sft"


class HyperParameters(BaseModel):
    learning_rate: float = Field(default=2e-4, ge=1e-6, le=1e-2)
    num_epochs: int = Field(default=3, ge=1, le=20)
    batch_size: int = Field(default=4, ge=1, le=64)
    max_seq_length: Optional[int] = None
    lora_r: int = Field(default=16, ge=4, le=128)
    lora_alpha: int = Field(default=32, ge=4, le=256)


class FinetuneRequest(BaseModel):
    base_model: str = Field(description="Model key from the curated registry, e.g. 'llama-3.2-1b'")
    dataset_id: str = Field(description="Dataset ID returned from /datasets upload")
    method: FinetuneMethod = Field(default=FinetuneMethod.sft)
    hyperparams: HyperParameters = Field(default_factory=HyperParameters)
    instance_type: Optional[str] = Field(default=None, description="Override default instance type")
    user_id: str = Field(default="default", description="User identifier")


class FinetuneResponse(BaseModel):
    job_id: str
    status: str
    base_model: str
    method: str
    sagemaker_job_name: str
