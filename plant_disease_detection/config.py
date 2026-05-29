"""Project paths and runtime configuration."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "compressed_model.h5"
DEFAULT_LABELS_PATH = REPO_ROOT / "models" / "labels.json"
DEFAULT_HF_SPACE = "LeonSilva15/plant-diseases-detection"
MODEL_INPUT_SIZE = (224, 224)
EXPECTED_CLASS_COUNT = 38


def configured_model_path() -> Path:
    """Return the model path, allowing deployments to override it with an env var."""

    value = os.getenv("PLANT_DISEASE_MODEL_PATH")
    return Path(value).expanduser() if value else _default_artifact_path("compressed_model.h5")


def configured_labels_path() -> Path:
    """Return the labels path, allowing deployments to override it with an env var."""

    value = os.getenv("PLANT_DISEASE_LABELS_PATH")
    return Path(value).expanduser() if value else _default_artifact_path("labels.json")


def configured_hf_space() -> str:
    """Return the Hugging Face Space repo id used for runtime artifact fallback."""

    return os.getenv("HF_SPACE", DEFAULT_HF_SPACE)


def _default_artifact_path(filename: str) -> Path:
    repo_candidate = REPO_ROOT / "models" / filename
    cwd_candidate = Path.cwd() / "models" / filename
    return repo_candidate if repo_candidate.exists() else cwd_candidate
