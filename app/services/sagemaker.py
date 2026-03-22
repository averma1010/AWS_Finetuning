import boto3
import json
import os
from typing import Optional, Dict
from fastapi import HTTPException
from app.config import get_settings
from app.models.registry import get_model_spec
from app.services import s3 as s3_service
from app import telemetry


def _get_client():
    settings = get_settings()
    return boto3.client(
        "sagemaker",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key
    )


def launch_training_job(
    job_id: str,
    sagemaker_job_name: str,
    base_model_key: str,
    dataset_s3_uri: str,
    method: str,
    hyperparams: dict,
    instance_type: Optional[str] = None,
) -> str:
    tracer = telemetry.get_tracer()
    with tracer.start_as_current_span("sagemaker.launch_training_job") as span:
        span.set_attribute("job.id", job_id)
        span.set_attribute("job.sagemaker_name", sagemaker_job_name)
        span.set_attribute("model.base", base_model_key)
        span.set_attribute("training.method", method)
        return _launch_training_job_inner(
            job_id, sagemaker_job_name, base_model_key, dataset_s3_uri, method, hyperparams, instance_type
        )


def _launch_training_job_inner(
    job_id: str,
    sagemaker_job_name: str,
    base_model_key: str,
    dataset_s3_uri: str,
    method: str,
    hyperparams: dict,
    instance_type: Optional[str] = None,
) -> str:
    settings = get_settings()
    client = _get_client()
    model_spec = get_model_spec(base_model_key)

    resolved_instance_type = instance_type or model_spec.default_instance_type

    # Upload the appropriate training script
    script_dir = os.path.join(os.path.dirname(__file__), "..", "training_scripts")
    script_name = "sft_train.py" if method == "sft" else "dpo_train.py"
    script_path = os.path.abspath(os.path.join(script_dir, script_name))
    source_s3_uri = s3_service.upload_training_script(job_id, script_path)

    training_hyperparams = {
        "sagemaker_program": script_name,
        "sagemaker_submit_directory": source_s3_uri,
        "model_name": model_spec.hf_model_id,
        "learning_rate": str(hyperparams.get("learning_rate", 2e-4)),
        "num_epochs": str(hyperparams.get("num_epochs", 3)),
        "batch_size": str(hyperparams.get("batch_size", 4)),
        "max_seq_length": str(hyperparams.get("max_seq_length", model_spec.max_seq_length)),
        "lora_r": str(hyperparams.get("lora_r", model_spec.default_lora_r)),
        "lora_alpha": str(hyperparams.get("lora_alpha", model_spec.default_lora_alpha)),
    }
    if method == "dpo":
        training_hyperparams["beta"] = str(hyperparams.get("beta", 0.1))

    output_path = s3_service.get_model_artifact_path(job_id)

    environment = {}
    if settings.hf_token:
        environment["HUGGING_FACE_HUB_TOKEN"] = settings.hf_token

    client.create_training_job(
        TrainingJobName=sagemaker_job_name,
        AlgorithmSpecification={
            "TrainingImage": _get_training_image(settings.aws_region),
            "TrainingInputMode": "File",
            "MetricDefinitions": [
                {"Name": "train:loss", "Regex": r"'loss': ([0-9\.]+)"},
                {"Name": "train:epoch", "Regex": r"'epoch': ([0-9\.]+)"},
                {"Name": "train:learning_rate", "Regex": r"'learning_rate': ([0-9\.e\-]+)"},
                {"Name": "train:step", "Regex": r"\[PROGRESS\] step=([0-9]+)"},
            ],
        },
        Environment=environment,
        RoleArn=settings.sagemaker_role_arn,
        HyperParameters=training_hyperparams,
        InputDataConfig=[
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": dataset_s3_uri,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            },
        ],
        OutputDataConfig={"S3OutputPath": output_path},
        ResourceConfig={
            "InstanceType": resolved_instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": 100,
        },
        StoppingCondition={"MaxRuntimeInSeconds": 86400},
    )

    return sagemaker_job_name


def get_training_job_status(sagemaker_job_name: str) -> dict:
    client = _get_client()
    response = client.describe_training_job(TrainingJobName=sagemaker_job_name)

    status_map = {
        "InProgress": "in_progress",
        "Completed": "completed",
        "Failed": "failed",
        "Stopping": "stopping",
        "Stopped": "stopped",
    }

    result = {
        "status": status_map.get(response["TrainingJobStatus"], response["TrainingJobStatus"]),
        "sagemaker_status": response["TrainingJobStatus"],
    }

    if "FinalMetricDataList" in response:
        result["metrics"] = {
            m["MetricName"]: m["Value"] for m in response["FinalMetricDataList"]
        }

    if "FailureReason" in response:
        result["error"] = response["FailureReason"]

    if response["TrainingJobStatus"] == "Completed":
        result["model_artifact_path"] = response["ModelArtifacts"]["S3ModelArtifacts"]

    # Build progress from SecondaryStatus
    progress: dict = {"stage": response.get("SecondaryStatus", "Unknown")}
    transitions = response.get("SecondaryStatusTransitions", [])
    if transitions:
        progress["stage_message"] = transitions[-1].get("StatusMessage", "")
    metrics = result.get("metrics", {})
    if "train:epoch" in metrics:
        progress["current_epoch"] = metrics["train:epoch"]
    if "train:loss" in metrics:
        progress["loss"] = metrics["train:loss"]
    result["progress"] = progress

    return result


def create_endpoint(
    model_id: str,
    model_artifact_path: str,
    base_model_key: str,
    instance_type: Optional[str] = None,
    instance_count: int = 1,
) -> str:
    settings = get_settings()
    client = _get_client()
    model_spec = get_model_spec(base_model_key)
    resolved_instance_type = instance_type or model_spec.default_instance_type
    endpoint_name = f"ft-{model_id}"

    # Create model
    model_name = f"ft-model-{model_id}"
    client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": _get_inference_image(settings.aws_region),
            "ModelDataUrl": model_artifact_path,
            "Environment": {
                "HF_MODEL_ID": model_spec.hf_model_id,
                "HF_TASK": "text-generation",
            },
        },
        ExecutionRoleArn=settings.sagemaker_role_arn,
    )

    # Create endpoint config
    config_name = f"ft-config-{model_id}"
    client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "primary",
                "ModelName": model_name,
                "InstanceType": resolved_instance_type,
                "InitialInstanceCount": instance_count,
            }
        ],
    )

    # Create endpoint
    client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=config_name,
    )

    return endpoint_name


def create_serverless_endpoint(
    model_id: str,
    model_artifact_path: str,
    base_model_key: str,
    memory_size_mb: int = 4096,
    max_concurrency: int = 10,
) -> str:
    """Create a serverless inference endpoint."""
    settings = get_settings()
    client = _get_client()
    model_spec = get_model_spec(base_model_key)
    endpoint_name = f"ft-serverless-{model_id}"

    # Create model (same as real-time)
    model_name = f"ft-model-{model_id}"
    client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": _get_inference_image(settings.aws_region),
            "ModelDataUrl": model_artifact_path,
            "Environment": {
                "HF_MODEL_ID": model_spec.hf_model_id,
                "HF_TASK": "text-generation",
            },
        },
        ExecutionRoleArn=settings.sagemaker_role_arn,
    )

    # Create endpoint config with serverless configuration
    config_name = f"ft-config-{model_id}"
    client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "primary",
                "ModelName": model_name,
                "ServerlessConfig": {
                    "MemorySizeInMB": memory_size_mb,
                    "MaxConcurrency": max_concurrency,
                },
            }
        ],
    )

    # Create endpoint
    client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=config_name,
    )

    return endpoint_name


def delete_endpoint(endpoint_name: str, endpoint_type: str = "real-time"):
    """Delete endpoint (works for both real-time and serverless)."""
    client = _get_client()
    try:
        response = client.describe_endpoint(EndpointName=endpoint_name)
        config_name = response["EndpointConfigName"]

        # Delete endpoint
        client.delete_endpoint(EndpointName=endpoint_name)

        # Delete config and model
        config_response = client.describe_endpoint_config(EndpointConfigName=config_name)
        model_name = config_response["ProductionVariants"][0]["ModelName"]
        client.delete_endpoint_config(EndpointConfigName=config_name)
        client.delete_model(ModelName=model_name)
    except client.exceptions.ClientError as e:
        raise


def invoke_endpoint(endpoint_name: str, payload: dict) -> dict:
    settings = get_settings()
    runtime_client = boto3.client(
        "sagemaker-runtime",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key
    )

    try:
        body = json.dumps(payload)
        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=body,
        )
        result = json.loads(response["Body"].read().decode("utf-8"))
        return result
    except runtime_client.exceptions.ModelNotReadyException:
        # Serverless endpoints have cold starts
        raise HTTPException(
            status_code=503,
            detail="Endpoint is warming up. Please retry in 10-30 seconds."
        )


def _get_training_image(region: str) -> str:
    account_map = {
        "us-east-1": "763104351884",
        "us-west-2": "763104351884",
        "eu-west-1": "763104351884",
    }
    account = account_map.get(region, "763104351884")
    return f"{account}.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-training:2.1.0-transformers4.36.0-gpu-py310-cu121-ubuntu20.04"


def _get_inference_image(region: str) -> str:
    account_map = {
        "us-east-1": "763104351884",
        "us-west-2": "763104351884",
        "eu-west-1": "763104351884",
    }
    account = account_map.get(region, "763104351884")
    return f"{account}.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-inference:2.1.0-transformers4.37.0-gpu-py310-cu121-ubuntu22.04"
