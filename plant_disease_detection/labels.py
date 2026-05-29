"""Label metadata helpers."""

from __future__ import annotations

import json
from pathlib import Path

from plant_disease_detection.config import EXPECTED_CLASS_COUNT, configured_labels_path
from plant_disease_detection.errors import LabelContractError

LabelMap = dict[int, str]


def load_labels(path: str | Path | None = None) -> LabelMap:
    """Load and validate the model label contract."""

    labels_path = Path(path).expanduser() if path else configured_labels_path()
    if not labels_path.exists():
        raise LabelContractError(f"Labels file not found: {labels_path}")

    try:
        raw_labels = json.loads(labels_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LabelContractError(f"Labels file is not valid JSON: {labels_path}") from exc

    if isinstance(raw_labels, list):
        labels = {index: label for index, label in enumerate(raw_labels)}
    elif isinstance(raw_labels, dict):
        try:
            labels = {int(index): str(label) for index, label in raw_labels.items()}
        except (TypeError, ValueError) as exc:
            raise LabelContractError("Labels JSON keys must be class indexes.") from exc
    else:
        raise LabelContractError("Labels JSON must be either a list or an index-to-label object.")

    validate_labels(labels)
    return dict(sorted(labels.items()))


def validate_labels(labels: LabelMap) -> None:
    """Ensure labels are contiguous and match the trained 38-class model."""

    expected_indexes = set(range(EXPECTED_CLASS_COUNT))
    actual_indexes = set(labels)

    if len(labels) != EXPECTED_CLASS_COUNT:
        raise LabelContractError(
            f"Expected {EXPECTED_CLASS_COUNT} labels, found {len(labels)} labels."
        )
    if actual_indexes != expected_indexes:
        missing = sorted(expected_indexes - actual_indexes)
        unexpected = sorted(actual_indexes - expected_indexes)
        raise LabelContractError(
            f"Labels must be indexed 0..{EXPECTED_CLASS_COUNT - 1}; "
            f"missing={missing}, unexpected={unexpected}."
        )
    if any(not label.strip() for label in labels.values()):
        raise LabelContractError("Labels cannot be empty.")


def format_label(label: str) -> str:
    """Convert PlantVillage directory labels into portfolio-friendly display text."""

    if "___" in label:
        crop, condition = label.split("___", maxsplit=1)
    else:
        crop, condition = "", label

    crop = _clean_label_part(crop)
    condition = _clean_label_part(condition)

    if not crop:
        return condition
    if condition.lower() == "healthy":
        return f"{crop} healthy"
    return f"{crop}: {condition}"


def _clean_label_part(value: str) -> str:
    return " ".join(value.replace("_", " ").strip().split())
