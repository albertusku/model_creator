# Model Creator

Local web app for building object-detection datasets from videos, reviewing bounding-box annotations, using YOLO assistance, tracking instances, and exporting training-ready snapshots.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776ab)
![FastAPI](https://img.shields.io/badge/FastAPI-local%20app-009688)
![Frontend](https://img.shields.io/badge/Frontend-HTML%20%2B%20CSS%20%2B%20JS-f7df1e)
![Tests](https://img.shields.io/badge/tests-pytest-0a7)

## What It Does

Model Creator is a local annotation dashboard for object detection workflows:

- Create and open dataset projects stored in local folders.
- Import videos and extract frames into the annotation queue.
- Draw, move, resize, delete, and review bounding boxes on a canvas.
- Use a configured YOLO model to suggest boxes.
- Auto-review pending images when detections are good enough.
- Generate tracking candidate frames, select an instance, and render a trajectory video.
- Add videos only for tracking without adding their frames to the dataset.
- Export reviewed annotations as YOLO or COCO snapshots.
- Start YOLO training runs from reviewed images.

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

1. Create a project with a local path, name, and class list.
2. Import a video with `Video to Images` to extract annotation frames.
3. Annotate frames on the central canvas.
4. Mark reviewed images when they are ready.
5. Optionally configure a YOLO `.pt` model for suggestions or auto-review.
6. Export reviewed images as YOLO or COCO.
7. Optionally start a training run from the reviewed dataset.

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
  main.py          FastAPI app and HTTP endpoints
  storage.py       Project JSON, folders, images, videos, annotations
  video.py         Video frame extraction
  inference.py     YOLO prediction mapping
  auto_review.py   Background auto-review jobs
  tracking.py      Candidate frames, instance tracking, trajectory video
  exporters.py     YOLO and COCO export
  training.py      YOLO training jobs
  schemas.py       Pydantic request/data models
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
```

`project.json` stores project metadata, classes, imported videos, image records, annotations, split config, and model config.

Generated exports and local datasets should stay out of version control.

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

- Only reviewed images are exported.
- Browser folder mode can open a project, but backend-only operations require opening by project path.
- YOLO-assisted features require a valid model path and the optional `inference` dependencies.
- ByteTrack tracking requires the `lap` package, included in the `inference` extra.
- This is a local tool; do not expose it publicly without adding authentication and path access controls.
