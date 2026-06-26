from __future__ import annotations

import pytest

from model_creator.schemas import SplitConfig
from model_creator.storage import configure_model, create_project, load_project, save_project
from model_creator.tracking import (
    _best_iou_track,
    _boxes_from_result,
    generate_candidate_frames,
    get_tracking_video_path,
    start_tracking_job,
)

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover
    cv2 = None
    np = None


def make_project(tmp_path, *, with_model: bool = True):
    root = tmp_path / "dataset"
    create_project(str(root), "Dataset", ["car", "person"], SplitConfig())
    video_path = root / "videos" / "clip.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    if cv2 is not None:
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (32, 24))
        for index in range(25):
            frame = np.full((24, 32, 3), index, dtype=np.uint8)
            writer.write(frame)
        writer.release()
    else:
        video_path.write_bytes(b"fake")
    data = load_project(root)
    data["videos"].append(
        {
            "id": "video-1",
            "source_name": "clip.mp4",
            "stored_name": "clip.mp4",
            "every_n_frames": 1,
            "frame_count": 25,
            "imported_at": "2026-01-01T00:00:00+00:00",
        }
    )
    save_project(root, data)
    if with_model:
        model_file = tmp_path / "best.pt"
        model_file.write_bytes(b"weights")
        configure_model(str(root), str(model_file), 0.4)
    return root


def test_tracking_rejects_project_without_model(tmp_path):
    root = make_project(tmp_path, with_model=False)

    with pytest.raises(ValueError, match="no model configured"):
        generate_candidate_frames(str(root), "video-1", 0)


def test_tracking_rejects_invalid_video_and_class(tmp_path):
    root = make_project(tmp_path)

    with pytest.raises(ValueError, match="unknown video id"):
        generate_candidate_frames(str(root), "missing", 0)
    with pytest.raises(ValueError, match="unknown class id"):
        generate_candidate_frames(str(root), "video-1", 99)


@pytest.mark.skipif(cv2 is None, reason="opencv-python is required")
def test_candidate_generation_falls_back_to_source_video_name(tmp_path):
    root = make_project(tmp_path)
    data = load_project(root)
    data["videos"][0]["stored_name"] = "missing-prefixed_clip.mp4"
    save_project(root, data)

    def detector(model_path, frames, confidence, data, class_id):
        return [[] for _frame in frames]

    result = generate_candidate_frames(str(root), "video-1", 0, detector=detector)

    assert len(result["candidates"]) == 20
    assert all(candidate["boxes"] == [] for candidate in result["candidates"])


@pytest.mark.skipif(cv2 is None, reason="opencv-python is required")
def test_candidate_generation_uses_at_most_first_20_seconds_and_filters_class(tmp_path):
    root = make_project(tmp_path)
    before = (root / "project.json").read_text(encoding="utf-8")

    def detector(model_path, frames, confidence, data, class_id):
        return [
            [
                {"id": f"car-{index}", "class_id": 0, "x": 1, "y": 2, "width": 3, "height": 4, "confidence": 0.9},
                {"id": f"person-{index}", "class_id": 1, "x": 5, "y": 6, "width": 7, "height": 8, "confidence": 0.8},
            ]
            for index, _frame in enumerate(frames)
        ]

    result = generate_candidate_frames(str(root), "video-1", 1, 0.2, detector=detector)

    after = (root / "project.json").read_text(encoding="utf-8")
    assert len(result["candidates"]) == 20
    assert result["candidates"][0]["frame"] == 0
    assert result["candidates"][-1]["frame"] == 19
    assert all(box["class_id"] == 1 for candidate in result["candidates"] for box in candidate["boxes"])
    assert result["candidates"][0]["image"].startswith("data:image/jpeg;base64,")
    assert after == before


def test_start_tracking_job_creates_job_from_start_frame(tmp_path):
    root = make_project(tmp_path)

    def tracker(model_path, video_path, confidence, data, class_id, start_frame, start_box):
        assert start_frame == 7
        return [
            {
                "frame": 7,
                "time_sec": 7.0,
                "x": 10,
                "y": 11,
                "width": 12,
                "height": 13,
                "center_x": 16,
                "center_y": 17.5,
                "confidence": 0.9,
                "track_id": 42,
            }
        ]

    job = start_tracking_job(
        str(root),
        "video-1",
        0,
        7,
        {"id": "start", "class_id": 0, "x": 10, "y": 11, "width": 12, "height": 13},
        0.3,
        tracker=tracker,
        run_in_background=False,
    )

    assert job["status"] == "completed"
    assert job["start_frame"] == 7
    assert job["selected_track_id"] == 42
    assert job["trajectory"][0]["frame"] == 7
    assert job["video_url"] == f"/api/tracking/{job['id']}/video"
    video_path = get_tracking_video_path(job["id"])
    assert video_path.exists()
    assert video_path.suffix == ".webm"


def test_start_tracking_job_reports_failure_when_no_track_matches(tmp_path):
    root = make_project(tmp_path)

    job = start_tracking_job(
        str(root),
        "video-1",
        0,
        7,
        {"id": "start", "class_id": 0, "x": 10, "y": 11, "width": 12, "height": 13},
        tracker=lambda *args: [],
        run_in_background=False,
    )

    assert job["status"] == "failed"
    assert "could not associate" in job["error"]


def test_start_tracking_rejects_missing_lap_before_creating_job(tmp_path, monkeypatch):
    root = make_project(tmp_path)
    monkeypatch.setattr("model_creator.tracking.importlib.util.find_spec", lambda name: None if name == "lap" else object())

    with pytest.raises(RuntimeError, match="ByteTrack requires"):
        start_tracking_job(
            str(root),
            "video-1",
            0,
            7,
            {"id": "start", "class_id": 0, "x": 10, "y": 11, "width": 12, "height": 13},
        )


def test_best_iou_track_selects_largest_overlap():
    selected = _best_iou_track(
        {"x": 10, "y": 10, "width": 20, "height": 20},
        [
            {"x": 0, "y": 0, "width": 5, "height": 5, "track_id": 1},
            {"x": 12, "y": 12, "width": 20, "height": 20, "track_id": 2},
            {"x": 15, "y": 15, "width": 5, "height": 5, "track_id": 3},
        ],
    )

    assert selected["track_id"] == 2


def test_tracking_boxes_can_fallback_to_selected_class_for_unmapped_model_class():
    class Scalar:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    class Coordinates(list):
        def tolist(self):
            return list(self)

    class RawBox:
        xyxy = [Coordinates([1, 2, 11, 22])]
        cls = [Scalar(99)]
        conf = [Scalar(0.87)]

    class Result:
        boxes = [RawBox()]

    data = {"classes": [{"id": 0, "name": "car"}]}

    assert _boxes_from_result(Result(), {99: "unmapped"}, data, class_id=0) == []

    boxes = _boxes_from_result(
        Result(),
        {99: "unmapped"},
        data,
        class_id=0,
        fallback_to_requested_class=True,
    )

    assert len(boxes) == 1
    assert boxes[0]["class_id"] == 0
    assert boxes[0]["x"] == 1
    assert boxes[0]["width"] == 10
