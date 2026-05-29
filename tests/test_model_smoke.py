import os

import numpy as np
import pytest
from PIL import Image

from plant_disease_detection.labels import load_labels
from plant_disease_detection.predictor import predict_image
from plant_disease_detection.runtime import load_keras_model


@pytest.mark.model
def test_checked_in_model_loads_and_predicts():
    if os.getenv("RUN_MODEL_SMOKE") != "1":
        pytest.skip("Set RUN_MODEL_SMOKE=1 to load the TensorFlow model artifact.")

    pytest.importorskip("tensorflow")

    labels = load_labels()
    model = load_keras_model()
    image_array = np.full((224, 224, 3), fill_value=128, dtype=np.uint8)
    image = Image.fromarray(image_array)

    result = predict_image(image, model=model, labels=labels, top_k=5)

    assert result.class_count == 38
    assert len(result.predictions) == 5
    assert result.top_prediction.label in labels.values()
    assert 0.0 <= result.top_prediction.confidence <= 1.0
