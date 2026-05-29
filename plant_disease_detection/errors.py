"""Domain-specific exceptions for controlled CLI and app errors."""


class PlantDiseaseDetectionError(Exception):
    """Base exception for expected application errors."""


class InvalidImageError(PlantDiseaseDetectionError):
    """Raised when an image cannot be opened or converted for inference."""


class LabelContractError(PlantDiseaseDetectionError):
    """Raised when the label metadata does not match the model contract."""


class ModelLoadError(PlantDiseaseDetectionError):
    """Raised when TensorFlow or the model artifact cannot be loaded."""


class ModelOutputError(PlantDiseaseDetectionError):
    """Raised when the model output does not match the expected class contract."""
