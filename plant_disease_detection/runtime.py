"""Model runtime loading."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from plant_disease_detection.config import configured_hf_space, configured_model_path
from plant_disease_detection.errors import ModelLoadError

HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
GIT_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/"
GIT_LFS_POINTER_MIN_BYTES = 100
GIT_LFS_POINTER_MAX_BYTES = 200


@lru_cache(maxsize=1)
def load_keras_model(model_path: str | None = None) -> Any:
    """Load the compressed Keras model with TensorFlow only when inference is requested."""

    resolved_path = Path(model_path).expanduser() if model_path else configured_model_path()
    if not resolved_path.exists():
        raise ModelLoadError(f"Model file not found: {resolved_path}")
    if resolved_path.suffix.lower() in {".h5", ".hdf5"}:
        resolved_path = _resolve_hdf5_model_path(resolved_path)

    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ModelLoadError(
            "TensorFlow is required for inference. Install the project dependencies first."
        ) from exc

    return _load_model_with_fallback(keras, resolved_path)


def _resolve_hdf5_model_path(model_path: Path) -> Path:
    """Return a valid HDF5 model path, downloading LFS content when needed."""

    if _has_hdf5_signature(model_path):
        return model_path
    if _is_git_lfs_pointer_file(model_path):
        downloaded_path = _download_hf_space_model(model_path)
        _validate_hdf5_model_file(downloaded_path)
        return downloaded_path

    _validate_hdf5_model_file(model_path)
    return model_path


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


def _has_hdf5_signature(model_path: Path) -> bool:
    try:
        return _read_file_signature(model_path) == HDF5_SIGNATURE
    except OSError:
        return False


def _is_git_lfs_pointer_file(model_path: Path) -> bool:
    try:
        size = model_path.stat().st_size
        if not GIT_LFS_POINTER_MIN_BYTES <= size <= GIT_LFS_POINTER_MAX_BYTES:
            return False
        return _read_file_prefix(model_path, len(GIT_LFS_POINTER_PREFIX)) == GIT_LFS_POINTER_PREFIX
    except OSError:
        return False


def _download_hf_space_model(model_path: Path) -> Path:
    """Download the real model artifact when a Space checkout contains an LFS pointer."""

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ModelLoadError(
            "Model artifact is a Git LFS pointer, and huggingface_hub is required to download "
            "the real model file."
        ) from exc

    filename = _space_model_filename(model_path)
    try:
        downloaded_path = hf_hub_download(
            repo_id=configured_hf_space(),
            repo_type="space",
            filename=filename,
            force_download=True,
        )
    except Exception as exc:
        raise ModelLoadError(
            f"Model artifact is a Git LFS pointer and could not be downloaded from "
            f"Hugging Face Space {configured_hf_space()}: {filename}. {exc}"
        ) from exc

    return Path(downloaded_path)


def _space_model_filename(model_path: Path) -> str:
    return f"models/{model_path.name}"


def _read_file_signature(model_path: Path) -> bytes:
    return _read_file_prefix(model_path, len(HDF5_SIGNATURE))


def _read_file_prefix(model_path: Path, length: int) -> bytes:
    with model_path.open("rb") as model_file:
        return model_file.read(length)


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
