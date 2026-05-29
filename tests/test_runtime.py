from pathlib import Path

import pytest

from plant_disease_detection import runtime
from plant_disease_detection.errors import ModelLoadError


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
