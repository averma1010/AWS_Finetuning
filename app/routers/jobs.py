from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from app.schemas.job import JobStatus, JobListResponse
from app.services import dynamodb, sagemaker

router = APIRouter()


@router.get("/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    job = dynamodb.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Sync status with SageMaker if job is still running
    if job["status"] in ("pending", "in_progress"):
        try:
            sm_status = sagemaker.get_training_job_status(job["sagemaker_job_name"])
            updates = {"status": sm_status["status"]}
            if "metrics" in sm_status:
                updates["metrics"] = sm_status["metrics"]
            if "error" in sm_status:
                updates["error"] = sm_status["error"]

            # If completed, register the finetuned model
            if sm_status["status"] == "completed" and "model_artifact_path" in sm_status:
                updates["model_artifact_path"] = sm_status["model_artifact_path"]
                _register_finetuned_model(job, sm_status["model_artifact_path"])

            job = dynamodb.update_job(job_id, updates)
        except Exception:
            pass  # Return stale DDB data if SageMaker call fails

    return JobStatus(**job)


@router.get("", response_model=JobListResponse)
async def list_jobs(user_id: Optional[str] = Query(default=None)):
    jobs = dynamodb.list_jobs(user_id=user_id)
    return JobListResponse(jobs=[JobStatus(**j) for j in jobs])


def _register_finetuned_model(job: dict, artifact_path: str):
    """Create a finetuned model record when training completes."""
    import uuid
    model_id = str(uuid.uuid4())
    dynamodb.create_model({
        "model_id": model_id,
        "job_id": job["job_id"],
        "base_model": job["base_model"],
        "s3_artifact_path": artifact_path,
    })
