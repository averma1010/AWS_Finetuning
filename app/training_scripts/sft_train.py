"""SageMaker entry point for SFT finetuning using trl + peft."""
import argparse
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import torch


class SageMakerProgressCallback(TrainerCallback):
    """Emits progress lines that SageMaker captures via MetricDefinitions regex."""

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None or not state.is_local_process_zero:
            return
        step = state.global_step
        epoch = logs.get("epoch", state.epoch or 0)
        loss = logs.get("loss")
        lr = logs.get("learning_rate")
        # SageMaker-parseable line (captured by MetricDefinitions regex)
        parts = [f"[PROGRESS] step={step}", f"'epoch': {epoch:.4f}"]
        if loss is not None:
            parts.append(f"'loss': {loss:.6f}")
        if lr is not None:
            parts.append(f"'learning_rate': {lr:.2e}")
        print(" ".join(parts), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)

    # SageMaker environment
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--training", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    return parser.parse_args()


def main():
    args = parse_args()

    # Load dataset
    data_files = os.path.join(args.training, "data.jsonl")
    dataset = load_dataset("json", data_files=data_files, split="train")

    # QLoRA quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # LoRA config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # Determine formatting based on dataset columns
    if "messages" in dataset.column_names:
        if tokenizer.chat_template is not None:
            # Instruct model: apply_chat_template converts list-of-dicts to a plain string
            def formatting_func(examples):
                return [
                    tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                    for msgs in examples["messages"]
                ]
        else:
            # Base model without a chat template: concatenate role/content pairs manually
            def formatting_func(examples):
                texts = []
                for msgs in examples["messages"]:
                    parts = []
                    for msg in msgs:
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        parts.append(f"### {role.capitalize()}:\n{content}")
                    texts.append("\n\n".join(parts))
                return texts
    else:
        def formatting_func(examples):
            texts = []
            for prompt, completion in zip(examples["prompt"], examples["completion"]):
                texts.append(f"### Instruction:\n{prompt}\n\n### Response:\n{completion}")
            return texts

    # Training config
    max_seq_length = args.max_seq_length if args.max_seq_length > 0 else 2048
    training_args = SFTConfig(
        output_dir=args.model_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=max_seq_length,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        gradient_accumulation_steps=4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        dataset_text_field=None,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_func,
        args=training_args,
        tokenizer=tokenizer,
        callbacks=[SageMakerProgressCallback()],
    )

    trainer.train()

    # Save the LoRA adapter
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)


if __name__ == "__main__":
    main()
