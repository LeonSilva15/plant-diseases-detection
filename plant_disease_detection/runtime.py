"""Model runtime loading."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from plant_disease_detection.config import configured_model_path
from plant_disease_detection.errors import ModelLoadError


@lru_cache(maxsize=1)
def load_keras_model(model_path: str | None = None) -> Any:
    """Load the compressed Keras model with TensorFlow only when inference is requested."""

    resolved_path = Path(model_path).expanduser() if model_path else configured_model_path()
    if not resolved_path.exists():
        raise ModelLoadError(f"Model file not found: {resolved_path}")

    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ModelLoadError(
            "TensorFlow is required for inference. Install the project dependencies first."
        ) from exc

    try:
        return keras.models.load_model(resolved_path, compile=False)
    except Exception as exc:  # pragma: no cover - TensorFlow raises framework-specific errors.
        raise ModelLoadError(f"Could not load model artifact: {resolved_path}") from exc
