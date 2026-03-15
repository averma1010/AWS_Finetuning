import uuid
from fastapi import APIRouter, HTTPException

from app.schemas.finetune import FinetuneRequest, FinetuneResponse
from app.models.registry import get_model_spec
from app.services import dynamodb, s3 as s3_service, sagemaker

router = APIRouter()


@router.post("", response_model=FinetuneResponse)
async def create_finetune_job(request: FinetuneRequest):
    model_spec = get_model_spec(request.base_model)
    if model_spec is None:
        raise HTTPException(status_code=400, detail=f"Unknown model: {request.base_model}")

    if request.method.value not in model_spec.supported_methods:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.base_model}' does not support method '{request.method.value}'",
        )

    job_id = str(uuid.uuid4())
    sagemaker_job_name = f"ft-{request.method.value}-{job_id[:8]}"

    hyperparams = request.hyperparams.model_dump()
    if hyperparams.get("max_seq_length") is None:
        hyperparams["max_seq_length"] = model_spec.max_seq_length

    # Create job record in DynamoDB
    job_data = {
        "job_id": job_id,
        "user_id": request.user_id,
        "base_model": request.base_model,
        "dataset_id": request.dataset_id,
        "method": request.method.value,
        "sagemaker_job_name": sagemaker_job_name,
        "hyperparams": hyperparams,
    }
    dynamodb.create_job(job_data)

    # Launch SageMaker training job
    dataset_s3_uri = s3_service.get_dataset_s3_uri(request.dataset_id)
    try:
        sagemaker.launch_training_job(
            job_id=job_id,
            sagemaker_job_name=sagemaker_job_name,
            base_model_key=request.base_model,
            dataset_s3_uri=dataset_s3_uri,
            method=request.method.value,
            hyperparams=hyperparams,
            instance_type=request.instance_type,
        )
        dynamodb.update_job(job_id, {"status": "in_progress"})
    except Exception as e:
        dynamodb.update_job(job_id, {"status": "failed", "error": str(e)})
        raise HTTPException(status_code=500, detail=f"Failed to launch training job: {e}")

    return FinetuneResponse(
        job_id=job_id,
        status="in_progress",
        base_model=request.base_model,
        method=request.method.value,
        sagemaker_job_name=sagemaker_job_name,
    )
