from typing import Optional
from pydantic import BaseModel


class JobStatus(BaseModel):
    job_id: str
    user_id: str
    base_model: str
    method: str
    status: str
    sagemaker_job_name: str
    hyperparams: dict
    created_at: str
    updated_at: str
    metrics: Optional[dict] = None
    error: Optional[str] = None


class JobListResponse(BaseModel):
    jobs: list
