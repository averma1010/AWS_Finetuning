from fastapi import APIRouter, HTTPException

from app.schemas.model import ModelListResponse, FinetunedModel, DeployRequest, DeployResponse
from app.models.registry import list_base_models
from app.services import dynamodb, sagemaker

router = APIRouter()


@router.get("", response_model=ModelListResponse)
async def list_models():
    base = list_base_models()
    finetuned_items = dynamodb.list_models()
    finetuned = [FinetunedModel(**m) for m in finetuned_items]
    return ModelListResponse(base_models=base, finetuned_models=finetuned)


@router.post("/{model_id}/deploy", response_model=DeployResponse)
async def deploy_model(model_id: str, request: DeployRequest = DeployRequest()):
    model = dynamodb.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Finetuned model not found")

    if model.get("endpoint_name"):
        raise HTTPException(status_code=409, detail="Model already has an active endpoint")

    try:
        dynamodb.update_model(model_id, {"status": "deploying"})
        endpoint_name = sagemaker.create_endpoint(
            model_id=model_id,
            model_artifact_path=model["s3_artifact_path"],
            base_model_key=model["base_model"],
            instance_type=request.instance_type,
            instance_count=request.instance_count,
        )
        dynamodb.update_model(model_id, {"endpoint_name": endpoint_name, "status": "deployed"})
    except Exception as e:
        dynamodb.update_model(model_id, {"status": "failed"})
        raise HTTPException(status_code=500, detail=f"Deployment failed: {e}")

    return DeployResponse(model_id=model_id, endpoint_name=endpoint_name, status="deployed")


@router.delete("/{model_id}/endpoint")
async def delete_model_endpoint(model_id: str):
    model = dynamodb.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Finetuned model not found")

    endpoint_name = model.get("endpoint_name")
    if not endpoint_name:
        raise HTTPException(status_code=404, detail="No active endpoint for this model")

    try:
        sagemaker.delete_endpoint(endpoint_name)
        dynamodb.update_model(model_id, {"endpoint_name": None, "status": "ready"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete endpoint: {e}")

    return {"detail": "Endpoint deleted", "model_id": model_id}
