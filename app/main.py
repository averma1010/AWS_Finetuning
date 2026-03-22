from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import datasets, finetune, jobs, models, inference

app = FastAPI(
    title="Finetuning-as-a-Service",
    description="Internal API for SFT and DPO finetuning of curated SLMs via AWS SageMaker",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router, prefix="/datasets", tags=["Datasets"])
app.include_router(finetune.router, prefix="/finetune", tags=["Finetuning"])
app.include_router(jobs.router, prefix="/jobs", tags=["Jobs"])
app.include_router(models.router, prefix="/models", tags=["Models"])
app.include_router(inference.router, prefix="/inference", tags=["Inference"])

# Initialize telemetry after routers are registered so FastAPIInstrumentor sees all routes
from app.telemetry import setup_telemetry
setup_telemetry(app, get_settings())


@app.get("/health")
def health_check():
    return {"status": "healthy"}
