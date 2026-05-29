"""Plant disease detection inference package."""

from plant_disease_detection.labels import format_label, load_labels
from plant_disease_detection.predictor import Prediction, PredictionResult, predict_image

__all__ = [
    "Prediction",
    "PredictionResult",
    "format_label",
    "load_labels",
    "predict_image",
]

__version__ = "0.1.0"
