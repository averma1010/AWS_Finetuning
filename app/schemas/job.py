from typing import Optional
from pydantic import BaseModel


class TrainingProgress(BaseModel):
    stage: str
    stage_message: Optional[str] = None
    current_epoch: Optional[float] = None
    loss: Optional[float] = None


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
    progress: Optional[TrainingProgress] = None


class JobListResponse(BaseModel):
    jobs: list
