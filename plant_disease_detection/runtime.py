"""Model runtime loading."""

from __future__ import annotations

import json
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

    return _load_model_with_fallback(keras, resolved_path)


def _load_model_with_fallback(keras: Any, resolved_path: Path) -> Any:
    """Load a Keras model, falling back to legacy H5 config repair when needed."""

    try:
        return keras.models.load_model(resolved_path, compile=False)
    except Exception as primary_exc:
        try:
            return _load_legacy_h5_model(keras, resolved_path)
        except Exception as fallback_exc:
            raise ModelLoadError(
                "Could not load model artifact: "
                f"{resolved_path}. "
                "Standard Keras load failed with "
                f"{type(primary_exc).__name__}: {primary_exc}. "
                "Legacy H5 fallback failed with "
                f"{type(fallback_exc).__name__}: {fallback_exc}."
            ) from fallback_exc


def _load_legacy_h5_model(keras: Any, resolved_path: Path) -> Any:
    """Load a legacy full-model H5 after repairing incompatible layer config."""

    import h5py

    with h5py.File(resolved_path, "r") as h5_file:
        model_config = h5_file.attrs.get("model_config")

    if model_config is None:
        raise ValueError("H5 file does not contain a model_config attribute")
    if isinstance(model_config, bytes):
        model_config = model_config.decode("utf-8")

    config = json.loads(model_config)
    repaired_config = _strip_legacy_depthwise_groups(config)
    model = keras.models.model_from_json(json.dumps(repaired_config))
    model.load_weights(resolved_path)
    return model


def _strip_legacy_depthwise_groups(value: Any) -> Any:
    """Return a copy of a Keras config without legacy DepthwiseConv2D groups values."""

    if isinstance(value, dict):
        cleaned = {key: _strip_legacy_depthwise_groups(item) for key, item in value.items()}
        if cleaned.get("class_name") == "DepthwiseConv2D":
            layer_config = cleaned.get("config")
            if isinstance(layer_config, dict):
                layer_config.pop("groups", None)
        return cleaned

    if isinstance(value, list):
        return [_strip_legacy_depthwise_groups(item) for item in value]

    return value
