"""Command-line prediction entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from plant_disease_detection.errors import PlantDiseaseDetectionError
from plant_disease_detection.labels import load_labels
from plant_disease_detection.predictor import predict_image
from plant_disease_detection.runtime import load_keras_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict plant diseases from a leaf image.")
    parser.add_argument("image", type=Path, help="Path to an image file.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of classes to return.")
    parser.add_argument("--model-path", type=Path, default=None, help="Path to a Keras .h5 model.")
    parser.add_argument("--labels-path", type=Path, default=None, help="Path to labels JSON.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started_at = time.perf_counter()

    try:
        labels = load_labels(args.labels_path)
        model = load_keras_model(str(args.model_path) if args.model_path else None)
        result = predict_image(args.image, model=model, labels=labels, top_k=args.top_k)
    except PlantDiseaseDetectionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    elapsed_seconds = time.perf_counter() - started_at
    if args.json:
        payload = result.to_dict()
        payload["inference_seconds"] = elapsed_seconds
        print(json.dumps(payload, indent=2))
    else:
        print(f"Top prediction: {result.top_prediction.display_name}")
        print(f"Confidence: {result.top_prediction.confidence:.2%}")
        print(f"Inference time: {elapsed_seconds:.3f}s")
        print()
        for prediction in result.predictions:
            print(f"{prediction.rank}. {prediction.display_name} - {prediction.confidence:.2%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
