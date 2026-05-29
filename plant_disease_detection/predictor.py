"""Prediction service layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from plant_disease_detection.config import EXPECTED_CLASS_COUNT
from plant_disease_detection.errors import ModelOutputError
from plant_disease_detection.labels import LabelMap, format_label, load_labels
from plant_disease_detection.preprocessing import ImageInput, preprocess_image
from plant_disease_detection.runtime import load_keras_model


@dataclass(frozen=True)
class Prediction:
    rank: int
    label: str
    display_name: str
    confidence: float


@dataclass(frozen=True)
class PredictionResult:
    predictions: list[Prediction]
    class_count: int

    @property
    def top_prediction(self) -> Prediction:
        return self.predictions[0]

    def confidence_chart(self) -> dict[str, float]:
        return {prediction.display_name: prediction.confidence for prediction in self.predictions}

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_count": self.class_count,
            "predictions": [asdict(prediction) for prediction in self.predictions],
        }


def predict_image(
    image: ImageInput,
    *,
    model: Any | None = None,
    labels: LabelMap | None = None,
    top_k: int = 5,
) -> PredictionResult:
    """Run top-k model inference for one image."""

    label_map = labels if labels is not None else load_labels()
    if top_k < 1:
        raise ModelOutputError("top_k must be at least 1.")
    top_k = min(top_k, len(label_map))

    runtime_model = model if model is not None else load_keras_model()
    batch = preprocess_image(image)
    probabilities = _predict_probabilities(runtime_model, batch, expected_classes=len(label_map))
    ranked_indexes = np.argsort(probabilities)[::-1][:top_k]

    predictions = [
        Prediction(
            rank=rank,
            label=label_map[int(index)],
            display_name=format_label(label_map[int(index)]),
            confidence=float(probabilities[int(index)]),
        )
        for rank, index in enumerate(ranked_indexes, start=1)
    ]

    return PredictionResult(predictions=predictions, class_count=len(label_map))


def _predict_probabilities(model: Any, batch: np.ndarray, expected_classes: int) -> np.ndarray:
    raw_prediction = model.predict(batch, verbose=0)
    probabilities = np.asarray(raw_prediction, dtype=np.float32).reshape(-1)

    if probabilities.shape[0] != expected_classes:
        raise ModelOutputError(
            f"Model returned {probabilities.shape[0]} classes; expected {expected_classes}."
        )
    if expected_classes != EXPECTED_CLASS_COUNT:
        raise ModelOutputError(
            f"Model label contract has {expected_classes} classes; expected {EXPECTED_CLASS_COUNT}."
        )
    if not np.all(np.isfinite(probabilities)):
        raise ModelOutputError("Model returned non-finite confidence values.")

    return probabilities
