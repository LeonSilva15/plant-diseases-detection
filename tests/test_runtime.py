import sys
import types
from pathlib import Path

import pytest

from plant_disease_detection import runtime
from plant_disease_detection.config import DEFAULT_HF_SPACE
from plant_disease_detection.errors import ModelLoadError


def _git_lfs_pointer_bytes(size: int = 14314088) -> bytes:
    return (
        b"version https://git-lfs.github.com/spec/v1\n"
        + b"oid sha256:"
        + b"a" * 64
        + f"\nsize {size}\n".encode()
    )


def test_strip_legacy_depthwise_groups_only_updates_depthwise_layers():
    config = {
        "class_name": "Functional",
        "config": {
            "layers": [
                {
                    "class_name": "DepthwiseConv2D",
                    "config": {
                        "name": "conv_dw_1",
                        "groups": 1,
                        "depth_multiplier": 1,
                    },
                },
                {
                    "class_name": "Conv2D",
                    "config": {
                        "name": "conv_pw_1",
                        "groups": 1,
                        "filters": 64,
                    },
                },
            ]
        },
    }

    cleaned = runtime._strip_legacy_depthwise_groups(config)
    layers = cleaned["config"]["layers"]

    assert "groups" not in layers[0]["config"]
    assert layers[0]["config"]["depth_multiplier"] == 1
    assert layers[1]["config"]["groups"] == 1
    assert config["config"]["layers"][0]["config"]["groups"] == 1


def test_validate_hdf5_model_file_accepts_hdf5_signature(tmp_path):
    model_path = tmp_path / "model.h5"
    model_path.write_bytes(runtime.HDF5_SIGNATURE + b"payload")

    runtime._validate_hdf5_model_file(model_path)


def test_load_keras_model_rejects_non_hdf5_artifact_before_tensorflow_import(tmp_path):
    runtime.load_keras_model.cache_clear()
    model_path = tmp_path / "model.h5"
    model_path.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")

    with pytest.raises(ModelLoadError) as exc_info:
        runtime.load_keras_model(str(model_path))

    message = str(exc_info.value)
    assert f"Model artifact is not a valid HDF5 file: {model_path}" in message
    assert "76657273696f6e20" in message
    assert "file size is 43 bytes" in message


def test_resolve_hdf5_model_path_downloads_git_lfs_pointer(monkeypatch, tmp_path):
    pointer_path = tmp_path / "compressed_model.h5"
    pointer_bytes = _git_lfs_pointer_bytes()
    assert (
        runtime.GIT_LFS_POINTER_MIN_BYTES
        <= len(pointer_bytes)
        <= runtime.GIT_LFS_POINTER_MAX_BYTES
    )
    pointer_path.write_bytes(pointer_bytes)
    downloaded_path = tmp_path / "downloaded-compressed_model.h5"
    downloaded_path.write_bytes(runtime.HDF5_SIGNATURE + b"payload")
    calls = []

    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        return str(downloaded_path)

    fake_hub = types.SimpleNamespace(hf_hub_download=fake_hf_hub_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    resolved_path = runtime._resolve_hdf5_model_path(pointer_path)

    assert resolved_path == downloaded_path
    assert calls == [
        {
            "repo_id": DEFAULT_HF_SPACE,
            "repo_type": "space",
            "filename": "models/compressed_model.h5",
            "force_download": True,
        }
    ]


def test_resolve_hdf5_model_path_honors_hf_space_env(monkeypatch, tmp_path):
    pointer_path = tmp_path / "compressed_model.h5"
    pointer_path.write_bytes(_git_lfs_pointer_bytes())
    downloaded_path = tmp_path / "downloaded-compressed_model.h5"
    downloaded_path.write_bytes(runtime.HDF5_SIGNATURE + b"payload")
    calls = []

    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        return str(downloaded_path)

    fake_hub = types.SimpleNamespace(hf_hub_download=fake_hf_hub_download)
    monkeypatch.setenv("HF_SPACE", "Example/custom-space")
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    resolved_path = runtime._resolve_hdf5_model_path(pointer_path)

    assert resolved_path == downloaded_path
    assert calls[0]["repo_id"] == "Example/custom-space"


def test_resolve_hdf5_model_path_rejects_invalid_non_pointer_artifact(tmp_path):
    model_path = tmp_path / "compressed_model.h5"
    model_path.write_bytes(b"not-a-pointer".ljust(133, b"!"))

    with pytest.raises(ModelLoadError) as exc_info:
        runtime._resolve_hdf5_model_path(model_path)

    message = str(exc_info.value)
    assert "Model artifact is not a valid HDF5 file" in message
    assert "file size is 133 bytes" in message


def test_load_model_with_fallback_reports_primary_and_fallback_failures(monkeypatch):
    class FailingModels:
        def load_model(self, path, compile=False):
            raise ValueError("unrecognized keyword argument groups")

    class FailingKeras:
        models = FailingModels()

    def fail_legacy_load(keras, resolved_path):
        raise RuntimeError("fallback reconstruction failed")

    monkeypatch.setattr(runtime, "_load_legacy_h5_model", fail_legacy_load)

    with pytest.raises(ModelLoadError) as exc_info:
        runtime._load_model_with_fallback(FailingKeras(), Path("models/compressed_model.h5"))

    message = str(exc_info.value)
    assert "Could not load model artifact: models/compressed_model.h5" in message
    assert "ValueError: unrecognized keyword argument groups" in message
    assert "RuntimeError: fallback reconstruction failed" in message


def test_load_model_with_fallback_returns_legacy_model_after_primary_failure(monkeypatch):
    sentinel_model = object()

    class FailingModels:
        def load_model(self, path, compile=False):
            raise ValueError("legacy h5 config")

    class FailingKeras:
        models = FailingModels()

    def load_legacy_model(keras, resolved_path):
        return sentinel_model

    monkeypatch.setattr(runtime, "_load_legacy_h5_model", load_legacy_model)

    model = runtime._load_model_with_fallback(FailingKeras(), Path("models/compressed_model.h5"))

    assert model is sentinel_model
