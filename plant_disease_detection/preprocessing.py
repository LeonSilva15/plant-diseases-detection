"""Image loading and preprocessing for MobileNet inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from plant_disease_detection.config import MODEL_INPUT_SIZE
from plant_disease_detection.errors import InvalidImageError

ImageInput = str | Path | Image.Image | np.ndarray


def load_image(image: ImageInput) -> Image.Image:
    """Open an image input and return an RGB Pillow image."""

    if image is None:
        raise InvalidImageError("No image was provided.")

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, np.ndarray):
        return _image_from_array(image).convert("RGB")

    if isinstance(image, str | Path):
        image_path = Path(image).expanduser()
        if not image_path.exists():
            raise InvalidImageError(f"Image file not found: {image_path}")

        try:
            with Image.open(image_path) as opened_image:
                return opened_image.convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise InvalidImageError(f"Could not open image: {image_path}") from exc

    raise InvalidImageError(f"Unsupported image input type: {type(image).__name__}")


def preprocess_image(
    image: ImageInput,
    target_size: tuple[int, int] = MODEL_INPUT_SIZE,
) -> np.ndarray:
    """Resize and scale an image like Keras MobileNet `preprocess_input`.

    The training notebook used `keras.applications.mobilenet.preprocess_input`, which scales RGB
    pixels from [0, 255] to [-1, 1]. Reimplementing that small transform keeps preprocessing
    testable without importing TensorFlow.
    """

    rgb_image = load_image(image)
    resized_image = rgb_image.resize(target_size)
    image_array = np.asarray(resized_image, dtype=np.float32)
    batched_image = np.expand_dims(image_array, axis=0)
    return (batched_image / 127.5) - 1.0


def _image_from_array(array: np.ndarray) -> Image.Image:
    if array.size == 0:
        raise InvalidImageError("Image arrays cannot be empty.")
    if array.ndim not in {2, 3}:
        raise InvalidImageError("Image arrays must have 2 or 3 dimensions.")
    if array.ndim == 3 and array.shape[-1] not in {1, 3, 4}:
        raise InvalidImageError("Image arrays must have 1, 3, or 4 channels.")

    normalized_array: Any = array
    if np.issubdtype(array.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(array)) <= 1.0 else 1.0
        normalized_array = np.clip(array * scale, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        normalized_array = np.clip(array, 0, 255).astype(np.uint8)

    if normalized_array.ndim == 3 and normalized_array.shape[-1] == 1:
        normalized_array = normalized_array[:, :, 0]

    return Image.fromarray(normalized_array)
