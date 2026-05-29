import importlib
import importlib.util
from pathlib import Path

import pytest


def test_app_builds_gradio_blocks():
    try:
        gr = importlib.import_module("gradio")
    except ModuleNotFoundError as exc:
        if exc.name == "gradio":
            pytest.skip("gradio is not installed in this test environment")
        raise

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location("app", app_path)
    assert spec is not None
    assert spec.loader is not None
    app_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_module)
    demo = app_module.build_demo()

    assert isinstance(demo, gr.Blocks)
    assert "Educational demo only" in app_module.DISCLAIMER
