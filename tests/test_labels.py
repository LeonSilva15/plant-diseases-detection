from plant_disease_detection.config import EXPECTED_CLASS_COUNT
from plant_disease_detection.labels import format_label, load_labels


def test_labels_match_model_contract():
    labels = load_labels()

    assert len(labels) == EXPECTED_CLASS_COUNT
    assert list(labels) == list(range(EXPECTED_CLASS_COUNT))
    assert labels[0] == "Apple___Apple_scab"
    assert labels[37] == "Tomato___healthy"


def test_format_label_makes_directory_names_readable():
    assert format_label("Apple___Apple_scab") == "Apple: Apple scab"
    assert format_label("Tomato___healthy") == "Tomato healthy"
    assert format_label("Corn_(maize)___Common_rust_") == "Corn (maize): Common rust"
