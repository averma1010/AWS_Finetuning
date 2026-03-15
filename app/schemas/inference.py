from typing import Optional
from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    model_id: str = Field(description="Finetuned model ID to run inference on")
    prompt: str = Field(description="Input prompt text")
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


class InferenceResponse(BaseModel):
    model_id: str
    generated_text: str
    usage: Optional[dict] = None
