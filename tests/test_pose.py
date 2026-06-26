from __future__ import annotations

from pathlib import Path

import pytest

from model_creator.pose import get_pose_video_path, start_pose_job
from model_creator.schemas import SplitConfig
from model_creator.storage import add_video, create_project


def make_project(tmp_path):
    root = tmp_path / "dataset"
    create_project(str(root), "Dataset", ["barbell"], SplitConfig())
    video_path = root / "videos" / "clip.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"video")
    video_id = add_video(str(root), "clip.mp4", "clip.mp4", 0, 0)
    return root, video_id


def fake_pose_runner(output_dir: Path):
    def runner(video_path: Path, confidence: float, model_name: str) -> dict:
        assert video_path.name == "clip.mp4"
        assert confidence == 0.4
        assert model_name == "yolo11n-pose.pt"
        output = output_dir / "pose.webm"
        output.write_bytes(b"pose-video")
        return {"video_path": output, "processed": 3, "total": 5}

    return runner


def test_start_pose_job_renders_original_video_with_runner(tmp_path):
    root, video_id = make_project(tmp_path)

    job = start_pose_job(
        str(root),
        video_id,
        "original",
        confidence=0.4,
        runner=fake_pose_runner(tmp_path),
        run_in_background=False,
    )

    assert job["status"] == "completed"
    assert job["source"] == "original"
    assert job["progress"] == {"processed": 3, "total": 5}
    assert job["video_url"] == f"/api/pose/{job['id']}/video"
    assert get_pose_video_path(job["id"]).read_bytes() == b"pose-video"


def test_pose_job_rejects_tracking_source_without_tracking_job(tmp_path):
    root, video_id = make_project(tmp_path)

    with pytest.raises(ValueError, match="tracking_job_id is required"):
        start_pose_job(str(root), video_id, "tracking", runner=fake_pose_runner(tmp_path), run_in_background=False)


def test_pose_job_uses_tracking_video_when_available(tmp_path, monkeypatch):
    root, video_id = make_project(tmp_path)
    tracking_video = root / "videos" / "clip.mp4"
    monkeypatch.setattr("model_creator.pose.get_tracking_video_path", lambda job_id: tracking_video)

    job = start_pose_job(
        str(root),
        video_id,
        "tracking",
        tracking_job_id="track-job",
        confidence=0.4,
        runner=fake_pose_runner(tmp_path),
        run_in_background=False,
    )

    assert job["status"] == "completed"
    assert job["source"] == "tracking"
    assert job["tracking_job_id"] == "track-job"


def test_pose_job_rejects_invalid_source(tmp_path):
    root, video_id = make_project(tmp_path)

    with pytest.raises(ValueError, match="source must be original or tracking"):
        start_pose_job(str(root), video_id, "other", runner=fake_pose_runner(tmp_path), run_in_background=False)
