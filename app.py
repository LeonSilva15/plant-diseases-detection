"""Gradio app for the plant disease classifier."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

_MPL_CACHE_ROOT = Path(__file__).resolve().parent / ".tmp" / "matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_ROOT))

import gradio as gr  # noqa: E402

from plant_disease_detection.errors import PlantDiseaseDetectionError  # noqa: E402
from plant_disease_detection.predictor import predict_image  # noqa: E402

APP_TITLE = "Plant Disease Detection"
MODEL_SUMMARY = (
    "MobileNet transfer-learning classifier, compressed from the notebook-trained model, "
    "serving 38 PlantVillage classes."
)
DISCLAIMER = (
    "Educational demo only. Results are not a professional agricultural diagnosis; confirm severe "
    "crop issues with a qualified agronomist or extension service."
)


def classify_image(image: Any) -> tuple[str, dict[str, float], list[list[str]], str]:
    """Run model inference and return Gradio-friendly outputs."""

    started_at = time.perf_counter()
    try:
        result = predict_image(image, top_k=5)
    except PlantDiseaseDetectionError as exc:
        raise gr.Error(str(exc)) from exc

    elapsed_seconds = time.perf_counter() - started_at
    top_prediction = result.top_prediction
    summary = (
        f"## {top_prediction.display_name}\n"
        f"Confidence: **{top_prediction.confidence:.2%}**"
    )
    rows = [
        [
            str(prediction.rank),
            prediction.display_name,
            f"{prediction.confidence:.2%}",
            prediction.label,
        ]
        for prediction in result.predictions
    ]
    metadata = (
        f"{MODEL_SUMMARY}\n\n"
        f"Classes: {result.class_count} | Inference time: {elapsed_seconds:.3f}s"
    )
    return summary, result.confidence_chart(), rows, metadata


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title=APP_TITLE,
        theme=gr.themes.Soft(primary_hue="green", neutral_hue="slate"),
        css="""
        .result-note {font-size: 0.92rem}
        """,
    ) as demo:
        gr.Markdown(f"# {APP_TITLE}")
        gr.Markdown(MODEL_SUMMARY)

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Leaf image",
                    sources=["upload", "webcam"],
                    type="pil",
                    height=360,
                )
                submit_button = gr.Button("Analyze image", variant="primary")

            with gr.Column(scale=1):
                top_prediction = gr.Markdown("## Waiting for an image")
                confidence_chart = gr.Label(label="Top 5 confidence scores", num_top_classes=5)
                details_table = gr.Dataframe(
                    headers=["Rank", "Prediction", "Confidence", "Raw label"],
                    datatype=["str", "str", "str", "str"],
                    row_count=5,
                    col_count=(4, "fixed"),
                    interactive=False,
                    label="Prediction details",
                )

        metadata = gr.Markdown(elem_classes=["result-note"])
        gr.Markdown(f"**Use note:** {DISCLAIMER}")

        submit_button.click(
            fn=classify_image,
            inputs=image_input,
            outputs=[top_prediction, confidence_chart, details_table, metadata],
        )

    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch()
