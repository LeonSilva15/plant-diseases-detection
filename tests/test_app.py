import importlib

import pytest


def test_app_builds_gradio_blocks():
    try:
        gr = importlib.import_module("gradio")
    except ModuleNotFoundError as exc:
        if exc.name == "gradio":
            pytest.skip("gradio is not installed in this test environment")
        raise

    app_module = importlib.import_module("app")
    demo = app_module.build_demo()

    assert isinstance(demo, gr.Blocks)
    assert "Educational demo only" in app_module.DISCLAIMER
