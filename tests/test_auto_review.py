from __future__ import annotations

import time

import pytest
from fastapi import HTTPException

from model_creator.auto_review import get_auto_review_job, start_auto_review_job
from model_creator.main import api_start_auto_review
from model_creator.schemas import AutoReviewStartRequest, Box, SplitConfig
from model_creator.storage import add_images, configure_model, create_project, load_project, save_annotations


def make_project(tmp_path, image_count: int = 4):
    root = tmp_path / "dataset"
    create_project(str(root), "Dataset", ["car"], SplitConfig())
    images = []
    for index in range(image_count):
        image_file = root / "images" / f"frame-{index}.jpg"
        image_file.parent.mkdir(parents=True, exist_ok=True)
        image_file.write_bytes(b"fake image")
        images.append(
            {
                "id": f"img-{index}",
                "file": f"images/frame-{index}.jpg",
                "width": 100,
                "height": 80,
                "video_id": "video-1",
                "source_frame": index * 30,
                "sequence": index,
            }
        )
    add_images(str(root), images)
    return root


def configure_dummy_model(root, tmp_path, confidence: float = 0.5) -> None:
    model_file = tmp_path / "best.pt"
    model_file.write_bytes(b"weights")
    configure_model(str(root), str(model_file), confidence)


def wait_for_job(job_id: str) -> dict:
    deadline = time.time() + 5
    while time.time() < deadline:
        job = get_auto_review_job(job_id)
        if job and job["status"] == "completed":
            return job
        time.sleep(0.01)
    raise AssertionError("auto-review job did not complete")


def model_box(box_id: str, confidence: float) -> dict:
    return {
        "id": box_id,
        "class_id": 0,
        "x": 10,
        "y": 12,
        "width": 30,
        "height": 20,
        "confidence": confidence,
        "source": "model",
    }


def test_auto_review_rejects_missing_project_path():
    with pytest.raises(HTTPException) as exc:
        api_start_auto_review(AutoReviewStartRequest(project_path=""))

    assert exc.value.status_code == 400
    assert "project_path is required" in exc.value.detail


def test_auto_review_rejects_project_without_configured_model(tmp_path):
    root = make_project(tmp_path)

    with pytest.raises(HTTPException) as exc:
        api_start_auto_review(AutoReviewStartRequest(project_path=str(root)))

    assert exc.value.status_code == 400
    assert "no model configured" in exc.value.detail


def test_auto_review_processes_only_unreviewed_and_skips_existing_boxes(tmp_path):
    root = make_project(tmp_path, image_count=3)
    configure_dummy_model(root, tmp_path, confidence=0.5)
    save_annotations(
        str(root),
        "img-1",
        [Box(id="manual", class_id=0, x=1, y=1, width=10, height=10)],
        reviewed=False,
    )
    save_annotations(
        str(root),
        "img-2",
        [Box(id="done", class_id=0, x=2, y=2, width=10, height=10)],
        reviewed=True,
    )
    called = []

    def predictor(project_path, image_id, confidence):
        called.append(image_id)
        return [model_box("auto", 0.9)]

    job = start_auto_review_job(str(root), 0.5, predictor=predictor)
    finished = wait_for_job(job["id"])
    data = load_project(root)

    assert called == ["img-0"]
    assert data["annotations"]["img-0"]["reviewed"] is True
    assert data["annotations"]["img-1"]["reviewed"] is False
    assert data["annotations"]["img-1"]["boxes"][0]["id"] == "manual"
    assert finished["counts"] == {
        "total": 2,
        "processed": 2,
        "approved": 1,
        "insufficient": 0,
        "skipped": 1,
        "failed": 0,
    }


def test_auto_review_keeps_insufficient_images_unreviewed(tmp_path):
    root = make_project(tmp_path, image_count=3)
    configure_dummy_model(root, tmp_path, confidence=0.5)

    predictions = {
        "img-0": [model_box("high", 0.8)],
        "img-1": [model_box("low", 0.2)],
        "img-2": [],
    }

    def predictor(project_path, image_id, confidence):
        return predictions[image_id]

    job = start_auto_review_job(str(root), 0.5, predictor=predictor)
    finished = wait_for_job(job["id"])
    data = load_project(root)

    assert data["annotations"]["img-0"]["reviewed"] is True
    assert data["annotations"]["img-1"]["reviewed"] is False
    assert data["annotations"]["img-1"]["boxes"][0]["id"] == "low"
    assert data["annotations"]["img-2"]["reviewed"] is False
    assert data["annotations"]["img-2"]["boxes"] == []
    assert finished["counts"]["approved"] == 1
    assert finished["counts"]["insufficient"] == 2
    assert finished["manual_review_image_ids"] == ["img-1", "img-2"]
