import numpy as np
from PIL import Image

from plant_disease_detection.preprocessing import preprocess_image


def test_preprocess_image_matches_mobilenet_shape_and_scale():
    image = Image.new("RGB", (32, 48), color=(255, 127, 0))

    batch = preprocess_image(image)

    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32
    assert np.isclose(batch.max(), 1.0)
    assert batch.min() >= -1.0


def test_preprocess_accepts_numpy_arrays():
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    batch = preprocess_image(image)

    assert batch.shape == (1, 224, 224, 3)
