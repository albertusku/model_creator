# Repository Guidelines

## Project Structure & Module Organization

This repository contains a local Python web app for bounding-box annotation, YOLO assistance, tracking, object-detection training, CSV classification, and dataset export.

- `model_creator/main.py` is a compatibility entrypoint for `uvicorn model_creator.main:app`.
- `model_creator/app/main.py` defines the FastAPI app and API endpoints.
- `model_creator/core/storage.py` manages project JSON files and local project folders.
- `model_creator/core/schemas.py` contains Pydantic request and data models.
- `model_creator/core/file_dialog.py` contains local folder picker integration.
- `model_creator/media/video.py` extracts frames from imported videos.
- `model_creator/datasets/exporters.py` validates annotations and exports YOLO/COCO snapshots.
- `model_creator/models/object_detection/inference.py` maps YOLO model predictions into project classes.
- `model_creator/models/object_detection/auto_review.py` runs background auto-review jobs for pending images.
- `model_creator/models/object_detection/tracking.py` generates tracking candidates, starts ByteTrack jobs, and renders trajectory videos.
- `model_creator/models/object_detection/training.py` creates YOLO snapshots and runs training jobs.
- `model_creator/models/classification/training.py` imports CSV datasets, trains tabular classifiers, renders PCA cluster data, and generates CSV predictions.
- `model_creator/models/pose/pose.py` renders human pose videos.
- `model_creator/static/` contains the browser UI: `index.html`, `styles.css`, and `app.js`.
- `tests/` contains pytest coverage for project creation, annotation persistence, validation, export, auto-review, tracking, object-detection training, and CSV classification.

Do not commit generated folders such as `.venv/`, `__pycache__/`, `.pytest_cache/`, `model_creator.egg-info/`, exported datasets, local videos, project folders, `classification_predictions/`, `classification_runs/`, or training runs.

## Build, Test, and Development Commands

Create and install the development environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the web app locally:

```bash
uvicorn model_creator.main:app --reload
```

Open `http://127.0.0.1:8000`.

Install model-assisted annotation dependencies when working on YOLO suggestions, auto-review, tracking, or object-detection training:

```bash
pip install -e ".[dev,inference]"
```

Run tests:

```bash
python3 -m pytest
```

Check Python syntax without running the app:

```bash
python3 -m compileall model_creator tests
```

Check frontend JavaScript syntax:

```bash
node --check model_creator/static/app.js
```

## Coding Style & Naming Conventions

Use Python 3.11+ and 4-space indentation. Prefer `snake_case` for modules, functions, variables, and JSON field names. Keep backend modules focused by responsibility: API routing in `app/main.py`, shared project concerns in `core/`, dataset import/export concerns in `datasets/` and `media/`, and model-specific workflows under `models/<model_type>/`.

Frontend code is plain HTML/CSS/JavaScript. Use descriptive DOM IDs and keep canvas interaction logic in `static/app.js`.

Do not add frontend frameworks or new browser dependencies unless the user explicitly asks for them. Keep the UI as a dense local tool, not a marketing page.

## Classification Notes

- Classification projects use `project_type: "csv_classification"` and do not require object-detection classes at creation time.
- CSV classification state is stored under the `classification` key in `project.json`, with imported dataset metadata in `classification.datasets`, training run metadata in `classification.runs`, and the selected/default model in `classification.last_run_id`.
- Imported CSV files are copied into `classification/datasets/`.
- Training artifacts are written under `classification_runs/<run_id>/`, including `model.joblib`, `metrics.json`, `metadata.json`, and test-set `predictions.csv`.
- Uploaded prediction outputs are written under `classification_predictions/` and exposed through `/api/classification/predictions/{prediction_id}/download`.
- Classification endpoints require opening the project by backend path, not browser folder mode.
- The default classifier is a scikit-learn pipeline: numeric columns use median imputation plus standard scaling, categorical columns use most-frequent imputation plus one-hot encoding, and the model is `RandomForestClassifier`.
- Training validates that the target column exists, feature columns exist, target is not selected as a feature, at least one feature is selected, and the target has at least two non-null classes.
- Prediction CSV files must contain every feature column saved in the selected run metadata; extra columns are preserved in the output.
- The cluster view uses the selected target/features and returns PCA-projected points for frontend rendering.

### Classification UI Notes

- `body.classificationMode` switches the UI from the object-detection dashboard to the CSV classification workspace.
- Preserve existing DOM IDs used by `app.js`, especially `projectType`, `classificationCsvFile`, `importClassificationCsv`, `classificationTarget`, `classificationFeatures`, `classificationTestSize`, `plotClassificationClusters`, `startClassificationTraining`, `classificationRun`, `classificationPredictFile`, `predictClassificationCsv`, `classificationDownload`, `classificationMetrics`, `classificationPredictions`, and `classificationClusterCanvas`.
- Object-detection-only canvas interactions must stay hidden/disabled in classification mode; classification uses the results workspace rather than the annotation canvas.

## Frontend Behavior Notes

- The main UI is a three-column dashboard: left controls, central canvas/viewer, and right review queue.
- The right column has two modes: annotation images and tracking candidates.
- Annotation images use the main `canvas` and allow creating/editing boxes.
- Tracking candidates use `trackingCandidateCanvas`; they show generated frames with bounding boxes and allow selecting an instance, but must not allow annotation-box creation.
- `trajectoryVideo` is separate from both canvases and is shown only for rendered trajectory playback.
- Preserve existing DOM IDs used by `app.js`, especially `canvas`, `trackingCandidateCanvas`, `trajectoryVideo`, `trackingCandidates`, `trackingVideo`, `trackSelectedInstance`, `showTrajectory`, and `generateTrackingCandidates`.
- If a visual change touches hidden/shown canvas states, verify that `[hidden]` and the `candidateMode` / `trajectoryMode` classes still prevent canvas overlap.

## Tracking Notes

- Imported videos can be used for both frame extraction and tracking.
- External tracking-only videos are added through `POST /api/tracking/videos`; they are copied into the project `videos/` folder and registered in `project.json`, but their frames are not added to `images/`.
- Candidate frames come from `POST /api/tracking/candidates` and include base64 JPEG data plus detection boxes.
- Tracking starts from a selected candidate box through `POST /api/tracking/start`.
- ByteTrack requires the optional `lap` dependency from the `inference` extra.

## Testing Guidelines

Use `pytest`. Place tests in `tests/` and name files `test_*.py`. Test names should describe behavior, for example `test_export_yolo_snapshot`. Add tests for any change that affects project persistence, annotation validation, frame metadata, tracking video registration, candidate generation, export output, auto-review, object-detection training, or CSV classification import/training/cluster/prediction behavior.

Before finishing code changes, run the relevant subset when possible. For broad changes, run:

```bash
python3 -m pytest
python3 -m compileall model_creator tests
node --check model_creator/static/app.js
```

## Commit & Pull Request Guidelines

This directory is not currently a Git repository, so no historical commit convention exists. Use short imperative commit messages, for example `Add COCO export validation` or `Fix reviewed image filtering`.

Pull requests should include a concise summary, testing performed, linked issue if applicable, and screenshots or notes for UI changes.

## Security & Configuration Tips

This app is intended for local use. Do not expose it publicly without adding authentication and path access controls. Never commit real datasets, large videos, secrets, or local environment files.
