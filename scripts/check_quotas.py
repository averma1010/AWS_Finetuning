"""Check SageMaker instance quotas for training and inference."""
import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import get_settings
import boto3

def check_quotas():
    settings = get_settings()
    service_quotas = boto3.client(
        'service-quotas',
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key
    )

    # SageMaker service code
    service_code = 'sagemaker'

    # Instance types to check
    training_instances = [
        'ml.m5.xlarge',
        'ml.m5.2xlarge',
        'ml.g4dn.xlarge',
        'ml.g5.xlarge',
        'ml.g5.2xlarge',
        'ml.p3.2xlarge'
    ]

    print("=" * 80)
    print("TRAINING JOB QUOTAS")
    print("=" * 80)

    for instance in training_instances:
        quota_name = f"{instance} for training job usage"
        try:
            # List quotas and find matching ones
            paginator = service_quotas.get_paginator('list_service_quotas')
            found = False

            for page in paginator.paginate(ServiceCode=service_code):
                for quota in page['Quotas']:
                    if quota_name.lower() in quota['QuotaName'].lower():
                        value = quota.get('Value', 0)
                        print(f"✓ {instance:20s} → {int(value):3d} instances")
                        found = True
                        break
                if found:
                    break

            if not found:
                print(f"? {instance:20s} → Unable to find quota")

        except Exception as e:
            print(f"✗ {instance:20s} → Error: {str(e)[:50]}")

    print("\n" + "=" * 80)
    print("ENDPOINT (INFERENCE) QUOTAS")
    print("=" * 80)

    for instance in training_instances:
        quota_name = f"{instance} for endpoint usage"
        try:
            paginator = service_quotas.get_paginator('list_service_quotas')
            found = False

            for page in paginator.paginate(ServiceCode=service_code):
                for quota in page['Quotas']:
                    if quota_name.lower() in quota['QuotaName'].lower():
                        value = quota.get('Value', 0)
                        print(f"✓ {instance:20s} → {int(value):3d} instances")
                        found = True
                        break
                if found:
                    break

            if not found:
                print(f"? {instance:20s} → Unable to find quota")

        except Exception as e:
            print(f"✗ {instance:20s} → Error: {str(e)[:50]}")

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print("Instances with >0 quota are available to use immediately.")
    print("For others, request quota increase via AWS Service Quotas console.")
    print("\nCPU instances (ml.m5.*): Slower but often have quota available")
    print("GPU instances (ml.g4dn.*, ml.g5.*, ml.p3.*): Faster but need quota request")


if __name__ == "__main__":
    check_quotas()
