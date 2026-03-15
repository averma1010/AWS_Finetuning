from typing import Optional
from pydantic import BaseModel


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    filename: str
    s3_path: str
    num_rows: int
    format: str


class DatasetValidationError(BaseModel):
    detail: str
    row: Optional[int] = None
