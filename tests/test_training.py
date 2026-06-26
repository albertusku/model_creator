from __future__ import annotations

import pytest
from fastapi import HTTPException

from model_creator.app.main import api_start_training, api_training_asset
from model_creator.core.schemas import Box, SplitConfig, TrainingStartRequest
from model_creator.core.storage import add_images, create_project, load_project, save_annotations
from model_creator.models.object_detection.training import get_job, start_training_job


def make_reviewed_project(tmp_path):
    root = tmp_path / "dataset"
    create_project(str(root), "Dataset", ["car"], SplitConfig())
    image_file = root / "images" / "frame.jpg"
    image_file.parent.mkdir(parents=True, exist_ok=True)
    image_file.write_bytes(b"fake image")
    add_images(
        str(root),
        [
            {
                "id": "img-1",
                "file": "images/frame.jpg",
                "width": 100,
                "height": 80,
                "video_id": "video-1",
                "source_frame": 30,
                "sequence": 0,
            }
        ],
    )
    save_annotations(
        str(root),
        "img-1",
        [Box(id="box-1", class_id=0, x=10, y=20, width=40, height=20)],
        reviewed=True,
    )
    return root


def successful_runner(request, snapshot_path, run_path):
    weights = run_path / "weights"
    weights.mkdir(parents=True)
    best = weights / "best.pt"
    last = weights / "last.pt"
    best.write_bytes(b"best")
    last.write_bytes(b"last")
    (run_path / "results.csv").write_text("epoch,metrics/mAP50(B)\n1,0.75\n", encoding="utf-8")
    (run_path / "results.png").write_bytes(b"png")
    assert (snapshot_path / "data.yaml").exists()
    return {"best_model_path": best, "last_model_path": last, "logs": ["fake training"]}


def test_start_training_creates_snapshot_and_completes(tmp_path):
    root = make_reviewed_project(tmp_path)
    request = TrainingStartRequest(project_path=str(root), epochs=1)

    job = start_training_job(request, runner=successful_runner, run_in_background=False)

    assert job["status"] == "completed"
    assert "fake training" in job["logs"]
    assert job["snapshot_path"]
    assert job["best_model_path"].endswith("best.pt")
    assert job["metrics"]["metrics/mAP50(B)"] == 0.75
    assert job["assets"] == ["results.png"]
    data = load_project(root)
    assert data["last_trained_model"]["best_model_path"].endswith("best.pt")
    assert data["model"]["path"].endswith("best.pt")


def test_training_job_records_failure(tmp_path):
    root = make_reviewed_project(tmp_path)

    def failing_runner(request, snapshot_path, run_path):
        raise RuntimeError("training failed")

    job = start_training_job(
        TrainingStartRequest(project_path=str(root), epochs=1),
        runner=failing_runner,
        run_in_background=False,
    )

    assert job["status"] == "failed"
    assert job["error"] == "training failed"
    assert get_job(job["id"])["status"] == "failed"


def test_training_start_rejects_missing_backend_path():
    with pytest.raises(HTTPException) as exc:
        api_start_training(TrainingStartRequest(project_path=""))

    assert exc.value.status_code == 400
    assert "backend path" in exc.value.detail


def test_training_asset_endpoint_serves_existing_assets(tmp_path):
    root = make_reviewed_project(tmp_path)
    job = start_training_job(
        TrainingStartRequest(project_path=str(root), epochs=1),
        runner=successful_runner,
        run_in_background=False,
    )
    found = api_training_asset(job["id"], "results.png")
    with pytest.raises(HTTPException) as exc:
        api_training_asset(job["id"], "missing.png")

    assert str(found.path).endswith("results.png")
    assert exc.value.status_code == 404
