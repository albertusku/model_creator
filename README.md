# Model Creator

Local web app for building object-detection datasets from videos, reviewing bounding-box annotations, using YOLO assistance, tracking instances, exporting training-ready snapshots, and training CSV classification models.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776ab)
![FastAPI](https://img.shields.io/badge/FastAPI-local%20app-009688)
![Frontend](https://img.shields.io/badge/Frontend-HTML%20%2B%20CSS%20%2B%20JS-f7df1e)
![Tests](https://img.shields.io/badge/tests-pytest-0a7)

## What It Does

Model Creator is a local dashboard for object detection and CSV classification workflows:

- Create and open dataset projects stored in local folders.
- Import videos and extract frames into the annotation queue.
- Draw, move, resize, delete, and review bounding boxes on a canvas.
- Use a configured YOLO model to suggest boxes.
- Auto-review pending images when detections are good enough.
- Generate tracking candidate frames, select an instance, and render a trajectory video.
- Add videos only for tracking without adding their frames to the dataset.
- Export reviewed annotations as YOLO or COCO snapshots.
- Start YOLO training runs from reviewed images.
- Create CSV classification projects.
- Import tabular CSV datasets, choose a target column and feature columns, inspect metrics and PCA clusters, train a scikit-learn classifier, and download predictions for new CSV files.

The app is intended for local use. It does not add authentication or public deployment hardening.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn model_creator.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

For YOLO suggestions, auto-review, tracking, and training support:

```bash
pip install -e ".[dev,inference]"
```

## Typical Workflow

Object detection:

1. Create a project with a local path, name, and class list.
2. Import a video with `Video to Images` to extract annotation frames.
3. Annotate frames on the central canvas.
4. Mark reviewed images when they are ready.
5. Optionally configure a YOLO `.pt` model for suggestions or auto-review.
6. Export reviewed images as YOLO or COCO.
7. Optionally start a training run from the reviewed dataset.

CSV classification:

1. Create or open a project with type `CSV classification`.
2. Import a CSV in `CSV Dataset`.
3. Select the target column to predict.
4. Select one or more feature columns. Numeric and categorical columns are handled automatically.
5. Optionally adjust `Test size` and click `Plot clusters` to preview a PCA projection by target class.
6. Click `Train classifier`.
7. Review accuracy, macro F1, weighted F1, class distribution, and confusion matrix in the classification workspace.
8. Upload another CSV with the same feature columns in `Predict CSV`.
9. Download the generated predictions CSV.

Classification projects must be opened by backend path for import, training, and prediction actions. Browser folder mode can inspect files, but backend operations need the real local project path.

## Classification Workflow

Classification uses normal CSV files and stores all artifacts inside the project folder.

- Project type: `csv_classification`.
- Imported datasets: `classification/datasets/<dataset_id>.csv`.
- Training runs: `classification_runs/<run_id>/`.
- Model artifact: `model.joblib`.
- Run metadata: `metadata.json`.
- Run metrics: `metrics.json`.
- Test predictions: `classification_runs/<run_id>/predictions.csv`.
- Uploaded prediction outputs: `classification_predictions/<prediction_id>.csv`.

The built-in trainer is a scikit-learn pipeline. Numeric features are imputed with the median and scaled. Categorical features are imputed with the most frequent value and one-hot encoded. The classifier is a balanced `RandomForestClassifier`.

Training requires a target column with at least two classes and at least one feature column. The target column cannot also be used as a feature. Prediction CSV files must include all feature columns used by the selected training run; extra columns are kept in the output. Prediction output includes a `prediction` column and probability columns when the classifier supports probabilities.

## Tracking Workflow

Tracking is separate from the normal annotation queue.

1. Configure a YOLO model in `Assisted Annotation`.
2. In `Tracking`, choose an imported project video or add an external video with `Add tracking video`.
3. Select the class and confidence.
4. Click `Generate candidate frames`.
5. Open the `Tracking candidates` tab in the right panel.
6. Click any candidate frame to show it in the central viewer.
7. Click the desired bounding box instance.
8. Click `Track selected instance`.
9. Use `Show trajectory` when the tracking video is ready.

External tracking videos are copied into the project `videos/` folder, but their frames are not added to the annotation image list.

## Project Layout

```text
model_creator/
  main.py          Compatibility entrypoint for uvicorn
  app/
    main.py        FastAPI app and HTTP endpoints
  core/
    schemas.py     Pydantic request/data models
    storage.py     Project JSON, folders, images, videos, annotations
    file_dialog.py Local folder picker integration
  datasets/
    exporters.py   Dataset validation plus YOLO and COCO export
  media/
    video.py       Video frame extraction
  models/
    object_detection/
      inference.py   YOLO prediction mapping
      auto_review.py Background auto-review jobs
      tracking.py    Candidate frames, instance tracking, trajectory video
      training.py    YOLO training jobs
    classification/
      training.py    CSV import, tabular training, PCA clusters, predictions
    pose/
      pose.py        Human pose video rendering
  static/
    index.html     Plain HTML UI
    styles.css     Dashboard styling
    app.js         Canvas and API interaction logic
tests/
  test_*.py        Pytest coverage
```

## Project Data

A project folder contains:

```text
project.json
images/
videos/
exports/
classification/
  datasets/
classification_runs/
classification_predictions/
```

`project.json` stores project metadata, project type, classes, imported videos, image records, annotations, split config, model config, and classification metadata.

Generated exports, local datasets, classification predictions, and training runs should stay out of version control.

## Development Commands

Run tests:

```bash
python3 -m pytest
```

Check Python syntax:

```bash
python3 -m compileall model_creator tests
```

Check frontend JavaScript syntax:

```bash
node --check model_creator/static/app.js
```

Run the app:

```bash
uvicorn model_creator.main:app --reload
```

If `uvicorn` is not on PATH, use:

```bash
python3 -m uvicorn model_creator.main:app --reload
```

## Notes

- Only reviewed images are exported for object-detection projects.
- Browser folder mode can open a project, but backend-only operations require opening by project path.
- YOLO-assisted features require a valid model path and the optional `inference` dependencies.
- ByteTrack tracking requires the `lap` package, included in the `inference` extra.
- CSV classification features use the base dependencies `pandas`, `scikit-learn`, and `joblib`.
- This is a local tool; do not expose it publicly without adding authentication and path access controls.
