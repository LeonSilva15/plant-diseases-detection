---
title: Plant Disease Detection
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
python_version: "3.10"
license: mit
tags:
  - computer-vision
  - image-classification
  - tensorflow
  - keras
  - gradio
  - agriculture
---

# Plant Disease Detection

Production-ready portfolio demo for classifying plant leaf diseases from images. The
application serves a compressed MobileNet transfer-learning model through a Gradio interface
that accepts upload or webcam input and returns top disease predictions with confidence scores.

Live demo target: https://huggingface.co/spaces/LeonSilva15/plant-diseases-detection

## What This Demonstrates

- Image classification with TensorFlow/Keras and MobileNet transfer learning.
- A deployable inference path separated from the original training notebook.
- A stable model contract: `models/compressed_model.h5` plus `models/labels.json`.
- Local CLI inference, Gradio UI inference, automated tests, linting, and CI deployment wiring.
- Clear product limitations for portfolio review.

## Model

The original experiment is preserved in `model-creation.ipynb`. It trains a MobileNet-based
classifier with ImageNet weights, adds dense classification layers for 38 PlantVillage classes,
tunes learning rate and dropout with Bayesian optimization, then prunes and strips the model for
smaller inference artifacts.

Recorded notebook metrics for the checked-in compressed model:

- Validation set size: 17,558 images.
- Class count: 38.
- Compressed-model validation accuracy: `0.9301742911338806`.
- Macro average F1-score: `0.93`.
- Weighted average F1-score: `0.93`.

Dataset: [New Plant Diseases Dataset on Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

## Architecture

```mermaid
flowchart LR
    A[Leaf image upload or webcam] --> B[Gradio app.py]
    B --> C[Preprocess RGB image to 224x224]
    C --> D[MobileNet scaling to -1..1]
    D --> E[compressed_model.h5]
    E --> F[Top-k prediction formatter]
    G[labels.json] --> F
    F --> H[Prediction, confidence chart, model note]
```

## Repository Layout

```text
.
|-- app.py                         # Hugging Face Spaces / Gradio entrypoint
|-- model-creation.ipynb           # Original training and evaluation notebook
|-- models/
|   |-- compressed_model.h5        # Runtime model for v1
|   |-- labels.json                # 38-class label contract
|   |-- model.h5                   # Original trained model artifact
|   `-- pruned_model.h5            # Pruned intermediate artifact
|-- plant_disease_detection/       # Inference package
|-- tests/                         # Unit and smoke tests
`-- .github/workflows/ci.yml       # Lint, tests, optional Space deploy
```

## Local Setup

Use Python 3.10 to match the notebook and Hugging Face Space configuration.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the app locally:

```bash
python app.py
```

Run CLI inference:

```bash
python -m plant_disease_detection.predict path/to/leaf.jpg --top-k 5 --json
```

Run checks:

```bash
ruff check .
pytest -m "not model"
```

Run the optional TensorFlow model artifact smoke test:

```bash
RUN_MODEL_SMOKE=1 pytest -m model
```

## Deployment

This repo is configured for Hugging Face Spaces with Gradio through the README front matter.
The GitHub Actions workflow runs lint and fast tests on every push and pull request. The
TensorFlow model artifact smoke test is available from the manual workflow dispatch path. On
pushes to `main`, the workflow deploys to a Space when both values are configured:

- Repository secret `HF_TOKEN`: Hugging Face token with write access.
- Repository variable `HF_SPACE`: target Space id, for example
  `LeonSilva15/plant-diseases-detection`.

The Space can also be created manually by uploading this repository to Hugging Face with
`sdk: gradio` and `app_file: app.py`.

## Limitations

- This is an educational portfolio demo, not a professional agricultural diagnosis tool.
- The model was trained on the Kaggle PlantVillage-style dataset, so real field images may
  differ in lighting, background, camera quality, leaf age, and disease stage.
- v1 packages the current compressed model only; it does not retrain, recalibrate, or improve
  accuracy.
- Dataset files are intentionally excluded from git. Use the Kaggle link to reproduce training.

## Evidence From The Original Notebook

Classification report:

![classification_report](https://github.com/LeonSilva15/plant-diseases-detection/assets/36859776/1a24753b-3850-4680-9575-236e1ef8d7ab)

Confusion matrix:

![confusion_matrix](https://github.com/LeonSilva15/plant-diseases-detection/assets/36859776/e5f9d399-2711-4be1-b6d5-7b157c27fd52)

Example predictions:

|  |  |
|--|--|
| ![Prediction example 1](https://github.com/LeonSilva15/plant-diseases-detection/assets/36859776/c1cf2672-4ad1-4a5e-904c-c25204ebb219) | ![Prediction example 2](https://github.com/LeonSilva15/plant-diseases-detection/assets/36859776/8681d2f7-fead-40aa-a7d4-35e919eba38e) |
| ![Prediction example 3](https://github.com/LeonSilva15/plant-diseases-detection/assets/36859776/d5a9e381-1d5a-4a43-90b0-44a77be6a0a2) | ![Prediction example 4](https://github.com/LeonSilva15/plant-diseases-detection/assets/36859776/da0169be-f90e-4d69-b86d-d85a433ba167) |
