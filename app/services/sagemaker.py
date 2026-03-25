import boto3
import json
from typing import Optional
from fastapi import HTTPException
from sagemaker import Session as SageMakerSession
from sagemaker.jumpstart.estimator import JumpStartEstimator
from sagemaker.jumpstart.model import JumpStartModel
from sagemaker.serverless import ServerlessInferenceConfig

from app.config import get_settings
from app.models.registry import get_model_spec
from app.services import s3 as s3_service
from app import telemetry


def _get_sagemaker_session(settings) -> SageMakerSession:
    boto_session = boto3.Session(
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )
    return SageMakerSession(boto_session=boto_session)


def _get_boto_client(settings):
    return boto3.client(
        "sagemaker",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
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
    model_spec = get_model_spec(base_model_key)
    resolved_instance_type = instance_type or model_spec.default_instance_type
    output_path = s3_service.get_model_artifact_path(job_id)

    sm_session = _get_sagemaker_session(settings)

    estimator = JumpStartEstimator(
        model_id=model_spec.jumpstart_model_id,
        role=settings.sagemaker_role_arn,
        instance_type=resolved_instance_type,
        output_path=output_path,
        sagemaker_session=sm_session,
        hyperparameters={
            "epoch": str(hyperparams.get("num_epochs", 3)),
            "learning_rate": str(hyperparams.get("learning_rate", 2e-4)),
            "per_device_train_batch_size": str(hyperparams.get("batch_size", 4)),
            "max_input_length": str(hyperparams.get("max_seq_length", model_spec.max_seq_length)),
            "lora_r": str(hyperparams.get("lora_r", model_spec.default_lora_r)),
            "lora_alpha": str(hyperparams.get("lora_alpha", model_spec.default_lora_alpha)),
            "instruction_tuned": "True",
        },
    )

    estimator.fit(
        {"training": dataset_s3_uri},
        job_name=sagemaker_job_name,
        wait=False,
    )

    return sagemaker_job_name


def get_training_job_status(sagemaker_job_name: str) -> dict:
    settings = get_settings()
    client = _get_boto_client(settings)
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
    model_spec = get_model_spec(base_model_key)
    resolved_instance_type = instance_type or model_spec.default_instance_type
    endpoint_name = f"ft-{model_id}"

    sm_session = _get_sagemaker_session(settings)

    model = JumpStartModel(
        model_id=model_spec.jumpstart_model_id,
        model_data=model_artifact_path,
        role=settings.sagemaker_role_arn,
        sagemaker_session=sm_session,
    )

    model.deploy(
        initial_instance_count=instance_count,
        instance_type=resolved_instance_type,
        endpoint_name=endpoint_name,
        wait=False,
    )

    return endpoint_name


def create_serverless_endpoint(
    model_id: str,
    model_artifact_path: str,
    base_model_key: str,
    memory_size_mb: int = 4096,
    max_concurrency: int = 10,
) -> str:
    settings = get_settings()
    model_spec = get_model_spec(base_model_key)
    endpoint_name = f"ft-serverless-{model_id}"

    sm_session = _get_sagemaker_session(settings)

    model = JumpStartModel(
        model_id=model_spec.jumpstart_model_id,
        model_data=model_artifact_path,
        role=settings.sagemaker_role_arn,
        sagemaker_session=sm_session,
    )

    model.deploy(
        serverless_inference_config=ServerlessInferenceConfig(
            memory_size_in_mb=memory_size_mb,
            max_concurrency=max_concurrency,
        ),
        endpoint_name=endpoint_name,
        wait=False,
    )

    return endpoint_name


def delete_endpoint(endpoint_name: str, endpoint_type: str = "real-time"):
    settings = get_settings()
    client = _get_boto_client(settings)
    try:
        response = client.describe_endpoint(EndpointName=endpoint_name)
        config_name = response["EndpointConfigName"]

        client.delete_endpoint(EndpointName=endpoint_name)

        config_response = client.describe_endpoint_config(EndpointConfigName=config_name)
        model_name = config_response["ProductionVariants"][0]["ModelName"]
        client.delete_endpoint_config(EndpointConfigName=config_name)
        client.delete_model(ModelName=model_name)
    except client.exceptions.ClientError:
        raise


def invoke_endpoint(endpoint_name: str, payload: dict) -> dict:
    settings = get_settings()
    runtime_client = boto3.client(
        "sagemaker-runtime",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
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
        raise HTTPException(
            status_code=503,
            detail="Endpoint is warming up. Please retry in 10-30 seconds.",
        )
