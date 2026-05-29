"""Model runtime loading."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from plant_disease_detection.config import configured_model_path
from plant_disease_detection.errors import ModelLoadError

HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


@lru_cache(maxsize=1)
def load_keras_model(model_path: str | None = None) -> Any:
    """Load the compressed Keras model with TensorFlow only when inference is requested."""

    resolved_path = Path(model_path).expanduser() if model_path else configured_model_path()
    if not resolved_path.exists():
        raise ModelLoadError(f"Model file not found: {resolved_path}")
    if resolved_path.suffix.lower() in {".h5", ".hdf5"}:
        _validate_hdf5_model_file(resolved_path)

    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ModelLoadError(
            "TensorFlow is required for inference. Install the project dependencies first."
        ) from exc

    return _load_model_with_fallback(keras, resolved_path)


def _validate_hdf5_model_file(model_path: Path) -> None:
    """Fail early when a deployed H5 model is a pointer, placeholder, or corrupt file."""

    try:
        signature = _read_file_signature(model_path)
        size = model_path.stat().st_size
    except OSError as exc:
        raise ModelLoadError(f"Could not read model artifact: {model_path}. {exc}") from exc

    if signature != HDF5_SIGNATURE:
        actual_signature = signature.hex() if signature else "<empty>"
        raise ModelLoadError(
            f"Model artifact is not a valid HDF5 file: {model_path}. "
            f"Expected signature {HDF5_SIGNATURE.hex()}, got {actual_signature}; "
            f"file size is {size} bytes."
        )


def _read_file_signature(model_path: Path) -> bytes:
    with model_path.open("rb") as model_file:
        return model_file.read(len(HDF5_SIGNATURE))


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
