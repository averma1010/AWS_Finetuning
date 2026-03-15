from fastapi import APIRouter, HTTPException

from app.schemas.inference import InferenceRequest, InferenceResponse
from app.services import dynamodb, sagemaker

router = APIRouter()


@router.post("", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    model = dynamodb.get_model(request.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Finetuned model not found")

    endpoint_name = model.get("endpoint_name")
    if not endpoint_name:
        raise HTTPException(status_code=400, detail="Model has no deployed endpoint. Deploy it first.")

    payload = {
        "inputs": request.prompt,
        "parameters": {
            "max_new_tokens": request.max_new_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        },
    }

    try:
        result = sagemaker.invoke_endpoint(endpoint_name, payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    generated_text = result[0]["generated_text"] if isinstance(result, list) else result.get("generated_text", "")

    return InferenceResponse(
        model_id=request.model_id,
        generated_text=generated_text,
    )
