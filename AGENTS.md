# Repository Guidelines

## Project Structure & Module Organization

This repository contains a local Python web app for bounding-box annotation, YOLO assistance, tracking, training, and dataset export.

- `model_creator/main.py` defines the FastAPI app and API endpoints.
- `model_creator/storage.py` manages project JSON files and local project folders.
- `model_creator/video.py` extracts frames from imported videos.
- `model_creator/inference.py` maps YOLO model predictions into project classes.
- `model_creator/auto_review.py` runs background auto-review jobs for pending images.
- `model_creator/tracking.py` generates tracking candidates, starts ByteTrack jobs, and renders trajectory videos.
- `model_creator/training.py` creates YOLO snapshots and runs training jobs.
- `model_creator/exporters.py` validates annotations and exports YOLO/COCO snapshots.
- `model_creator/schemas.py` contains Pydantic request and data models.
- `model_creator/static/` contains the browser UI: `index.html`, `styles.css`, and `app.js`.
- `tests/` contains pytest coverage for project creation, annotation persistence, validation, export, auto-review, tracking, and training.

Do not commit generated folders such as `.venv/`, `__pycache__/`, `.pytest_cache/`, `model_creator.egg-info/`, exported datasets, local videos, project folders, or training runs.

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

Install model-assisted annotation dependencies when working on YOLO suggestions, auto-review, tracking, or training:

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

Use Python 3.11+ and 4-space indentation. Prefer `snake_case` for modules, functions, variables, and JSON field names. Keep backend modules focused by responsibility: API routing in `main.py`, persistence in `storage.py`, export logic in `exporters.py`, video processing in `video.py`, tracking in `tracking.py`, and training in `training.py`.

Frontend code is plain HTML/CSS/JavaScript. Use descriptive DOM IDs and keep canvas interaction logic in `static/app.js`.

Do not add frontend frameworks or new browser dependencies unless the user explicitly asks for them. Keep the UI as a dense local tool, not a marketing page.

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

Use `pytest`. Place tests in `tests/` and name files `test_*.py`. Test names should describe behavior, for example `test_export_yolo_snapshot`. Add tests for any change that affects project persistence, annotation validation, frame metadata, tracking video registration, candidate generation, export output, auto-review, or training.

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
