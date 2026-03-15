from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class BaseModelSpec:
    hf_model_id: str
    display_name: str
    default_instance_type: str
    supported_methods: List[str] = field(default_factory=lambda: ["sft", "dpo"])
    max_seq_length: int = 2048
    default_lora_r: int = 16
    default_lora_alpha: int = 32


CURATED_MODELS: Dict[str, BaseModelSpec] = {
    "llama-3.2-1b": BaseModelSpec(
        hf_model_id="meta-llama/Llama-3.2-1B",
        display_name="Llama 3.2 1B",
        default_instance_type="ml.g5.xlarge",
        max_seq_length=4096,
    ),
    "llama-3.2-3b": BaseModelSpec(
        hf_model_id="meta-llama/Llama-3.2-3B",
        display_name="Llama 3.2 3B",
        default_instance_type="ml.g5.xlarge",
        max_seq_length=4096,
    ),
    "mistral-7b-v0.3": BaseModelSpec(
        hf_model_id="mistralai/Mistral-7B-v0.3",
        display_name="Mistral 7B v0.3",
        default_instance_type="ml.g5.2xlarge",
        max_seq_length=8192,
    ),
}


def get_model_spec(model_key: str) -> Optional[BaseModelSpec]:
    return CURATED_MODELS.get(model_key)


def list_base_models() -> List[dict]:
    return [
        {
            "model_key": key,
            "display_name": spec.display_name,
            "hf_model_id": spec.hf_model_id,
            "supported_methods": spec.supported_methods,
            "default_instance_type": spec.default_instance_type,
            "max_seq_length": spec.max_seq_length,
        }
        for key, spec in CURATED_MODELS.items()
    ]
