import numpy as np
import pytest
from PIL import Image

from plant_disease_detection.errors import InvalidImageError, ModelOutputError
from plant_disease_detection.labels import load_labels
from plant_disease_detection.predictor import predict_image


class FakeModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray([probabilities], dtype=np.float32)
        self.seen_batch_shape = None

    def predict(self, batch, verbose=0):
        self.seen_batch_shape = batch.shape
        return self.probabilities


def test_predict_image_returns_sorted_top_k_predictions():
    labels = load_labels()
    probabilities = np.zeros(38, dtype=np.float32)
    probabilities[3] = 0.1
    probabilities[7] = 0.9
    probabilities[12] = 0.4
    model = FakeModel(probabilities)
    image = Image.new("RGB", (24, 24), color="green")

    result = predict_image(image, model=model, labels=labels, top_k=3)

    assert model.seen_batch_shape == (1, 224, 224, 3)
    assert [prediction.label for prediction in result.predictions] == [
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Grape___Esca_(Black_Measles)",
        "Apple___healthy",
    ]
    assert [prediction.confidence for prediction in result.predictions] == pytest.approx(
        [0.9, 0.4, 0.1]
    )


def test_predict_image_rejects_missing_input():
    labels = load_labels()
    model = FakeModel(np.ones(38, dtype=np.float32) / 38)

    with pytest.raises(InvalidImageError):
        predict_image(None, model=model, labels=labels)


def test_predict_image_rejects_wrong_output_shape():
    labels = load_labels()
    model = FakeModel(np.ones(4, dtype=np.float32) / 4)
    image = Image.new("RGB", (24, 24), color="green")

    with pytest.raises(ModelOutputError):
        predict_image(image, model=model, labels=labels)
