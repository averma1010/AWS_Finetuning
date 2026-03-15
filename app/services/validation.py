import json
from typing import Optional, Dict
from app.config import get_settings


class ValidationError(Exception):
    def __init__(self, detail: str, row: Optional[int] = None):
        self.detail = detail
        self.row = row
        super().__init__(detail)


def validate_dataset(file_content: bytes) -> dict:
    """Validate a JSONL dataset file and return its format and row count."""
    settings = get_settings()
    max_size = settings.max_dataset_size_mb * 1024 * 1024
    if len(file_content) > max_size:
        raise ValidationError(f"File exceeds maximum size of {settings.max_dataset_size_mb}MB")

    try:
        text = file_content.decode("utf-8")
    except UnicodeDecodeError:
        raise ValidationError("File is not valid UTF-8 text")

    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if len(lines) < settings.min_dataset_rows:
        raise ValidationError(
            f"Dataset must have at least {settings.min_dataset_rows} rows, found {len(lines)}"
        )

    detected_format = None
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise ValidationError(f"Invalid JSON on line {i + 1}", row=i + 1)

        if not isinstance(row, dict):
            raise ValidationError(f"Row {i + 1} is not a JSON object", row=i + 1)

        row_format = _detect_row_format(row)
        if row_format is None:
            raise ValidationError(
                f"Row {i + 1} does not match any supported format. "
                "Expected: SFT (prompt+completion or messages) or DPO (prompt+chosen+rejected)",
                row=i + 1,
            )

        if detected_format is None:
            detected_format = row_format
        elif row_format != detected_format:
            raise ValidationError(
                f"Inconsistent format at row {i + 1}: expected '{detected_format}', got '{row_format}'",
                row=i + 1,
            )

    return {"format": detected_format, "num_rows": len(lines)}


def _detect_row_format(row: dict) -> Optional[str]:
    if "prompt" in row and "chosen" in row and "rejected" in row:
        return "dpo"
    if "messages" in row and isinstance(row["messages"], list):
        for msg in row["messages"]:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                return None
        return "sft_messages"
    if "prompt" in row and "completion" in row:
        return "sft_prompt_completion"
    return None
