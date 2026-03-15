import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.schemas.dataset import DatasetUploadResponse
from app.services.validation import validate_dataset, ValidationError
from app.services.s3 import upload_dataset

router = APIRouter()


@router.post("", response_model=DatasetUploadResponse)
async def upload_dataset_endpoint(file: UploadFile = File(...)):
    if not file.filename.endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="Only .jsonl files are supported")

    content = await file.read()

    try:
        result = validate_dataset(content)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.detail)

    dataset_id = str(uuid.uuid4())
    s3_path = upload_dataset(dataset_id, content, file.filename)

    return DatasetUploadResponse(
        dataset_id=dataset_id,
        filename=file.filename,
        s3_path=s3_path,
        num_rows=result["num_rows"],
        format=result["format"],
    )
