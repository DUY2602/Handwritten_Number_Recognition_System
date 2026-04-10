# Handwritten Math Expression Recognition System

A comprehensive end-to-end pipeline for recognizing handwritten mathematical expressions. This system integrates advanced image processing, deep learning classification, and a context-aware expression parser into a user-friendly Flask web interface.

## 🚀 Key Features

- **Dual-Mode Interface:** Supports both image uploads (phone photos) and direct drawing on a digital canvas.
- **Advanced Segmentation:** Robust multi-stage segmentation that handles notebook ruling lines, overlapping characters, and fragmented strokes.
- **VGG-style Classifier:** A deep Convolutional Neural Network (CNN) trained on a massive combined dataset (MNIST, EMNIST, and custom operators).
- **Contextual Post-Processing:** Uses geometric and sequence logic to resolve ambiguous characters (e.g., distinguishing '1' from '-' or '7' from '/').
- **Feedback Loop:** A built-in system to collect user corrections and rejections to improve future model training.
- **Evaluation Suite:** Comprehensive metrics including Accuracy, Precision, Recall, and Confusion Matrices.

## 📂 Project Structure

```text
├── app.py                      # Flask Web Application entry point
├── requirements.txt            # Project dependencies
├── setup_env.bat               # One-click install and run script
├── static/                     # Web assets (JS, CSS, Uploads)
├── templates/                  # HTML templates
└── src/
    ├── main.py                 # CLI entry point for batch processing
    ├── model/                  # Classification & Training logic
    │   ├── artifacts/          # Trained models, datasets, and feedback data
    │   ├── evaluation/         # Model and Pipeline evaluation scripts
    │   ├── build_baseline.py   # Script to train the Keras VGG model
    │   └── operator_classifier.py # Dataset merging & Prediction API
    ├── preprocessing/          # Image normalization & ROI cleaning
    └── segmentation/           # Character extraction & Expression parsing
        ├── segmentation.py     # Main segmentation engine
        ├── rect_ops.py         # Box merging and splitting logic
        └── expression_parser.py # Math logic and validation
```

## Main modules

### Web app

`app.py`

- serves the upload/draw interface
- runs segmentation and character prediction
- returns annotated output images
- lets users correct bad predictions
- saves corrections and rejected segments for later retraining

### CLI pipeline

`src/main.py`

- runs the end-to-end expression pipeline from the terminal
- can show or skip the matplotlib visualization

### Segmentation

`src/segmentation/segmentation.py`

- prepares the full image for contour detection
- removes small connected-component noise
- merges broken fragments
- crops each detected character
- normalizes each ROI into a classifier-friendly 28x28 image

### ROI preprocessing

`src/preprocessing/preprocessing.py`

- cleans a single cropped character
- centers it on a 28x28 canvas
- prepares tensors/arrays for the classifier runtime

### Prediction refinement

`src/segmentation/prediction_refiner.py`

- applies shape-aware rules for confusing classes such as `-`, `/`, `(`, `)`
- improves the raw model output before building the expression

### Expression parsing

`src/segmentation/expression_parser.py`

- validates token order
- builds the final expression string
- computes the result when the expression is valid

### Evaluation utilities

`src/evaluation/evaluation.py`

- provides accuracy, precision, recall, F1, confusion matrix, and per-class metrics

## Setup

Install dependencies from the project root:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` already includes both TensorFlow and PyTorch dependencies used by the current repo.

## Run the web app

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

The web app supports:

- uploading an image
- drawing directly on the canvas
- reviewing segmented characters
- saving corrected labels
- rejecting bad segments caused by wrong segmentation

## Run the CLI pipeline

```powershell
python src/main.py --input path\to\image.png
```

If you do not want the matplotlib window:

```powershell
python src/main.py --input path\to\image.png --no-display
```

## Runtime model

The runtime classifier loads:

`src/model/artifacts/models/best_expression_model.keras`

Class names are read from:

`src/model/artifacts/models/training_summary.json`

## Feedback flow

When a user presses `Save Feedback` in the web app:

- corrected characters are saved to `src/model/artifacts/feedback/corrections/`
- rejected bad segments are saved to `src/model/artifacts/feedback/rejections/`
- a JSON report is written to `src/model/artifacts/feedback/reports/`
- each saved action is appended to `src/model/artifacts/feedback/feedback_log.jsonl`

Important:

- feedback does not trigger retraining automatically
- the current repo still needs a separate retraining step if you want the model to learn from saved feedback

## Generated local artifacts

These folders are generated by running the project and are not core source code:

- `src/segmentation/output_digit/`
- `src/segmentation/output_debug/`
- `src/model/artifacts/feedback/`
- `synthetic_operators/`
- `eval_artifacts/`
- `.tmp/`
- `static/uploads/`
- `data/`

## Known limitations

- segmentation can still over-merge or over-split characters in difficult images
- the runtime Keras model and the legacy PyTorch training path are not fully unified yet
- feedback collection is implemented, but feedback-based retraining is not wired into the app yet
