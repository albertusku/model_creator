from __future__ import annotations

import gc
import tempfile
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .storage import load_project, resolve_project_video_path
from .tracking import get_tracking_video_path

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

POSE_MODEL = "yolo11n-pose.pt"
COCO_POSE_EDGES = (
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)
BODY_KEYPOINTS = set(range(5, 17))

PoseRunner = Callable[[Path, float, str], dict[str, Any]]

_JOBS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": job["id"],
        "status": job["status"],
        "project_path": job["project_path"],
        "video_id": job["video_id"],
        "source": job["source"],
        "tracking_job_id": job.get("tracking_job_id"),
        "progress": dict(job["progress"]),
        "video_url": f"/api/pose/{job['id']}/video" if job.get("video_path") else None,
        "error": job.get("error"),
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }


def get_pose_job(job_id: str) -> dict[str, Any] | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        return _public_job(job) if job else None


def get_pose_video_path(job_id: str) -> Path | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        video_path = Path(job["video_path"]) if job and job.get("video_path") else None
    if not video_path or not video_path.exists() or not video_path.is_file():
        return None
    return video_path


def start_pose_job(
    project_path: str,
    video_id: str,
    source: str = "original",
    tracking_job_id: str | None = None,
    confidence: float | None = None,
    *,
    runner: PoseRunner | None = None,
    run_in_background: bool = True,
) -> dict[str, Any]:
    data, video_path, threshold = _validate_pose_request(project_path, video_id, source, tracking_job_id, confidence)
    if runner is None and cv2 is None:
        raise RuntimeError("opencv-python is required to render human pose videos")

    job_id = uuid.uuid4().hex
    job = {
        "id": job_id,
        "status": "queued",
        "project_path": project_path,
        "video_id": video_id,
        "source": source,
        "tracking_job_id": tracking_job_id,
        "progress": {"processed": 0, "total": None},
        "video_path": None,
        "error": None,
        "created_at": _now(),
        "updated_at": _now(),
        "project_name": data.get("name"),
    }
    with _LOCK:
        _JOBS[job_id] = job

    args = (job_id, video_path, threshold, runner)
    if run_in_background:
        threading.Thread(target=_run_pose_job, args=args, daemon=True).start()
    else:
        _run_pose_job(*args)
    return get_pose_job(job_id) or _public_job(job)


def _validate_pose_request(
    project_path: str,
    video_id: str,
    source: str,
    tracking_job_id: str | None,
    confidence: float | None,
) -> tuple[dict[str, Any], Path, float]:
    if not project_path.strip():
        raise ValueError("project_path is required")
    if source not in {"original", "tracking"}:
        raise ValueError("source must be original or tracking")
    data = load_project(project_path)
    video = next((item for item in data.get("videos", []) if item["id"] == video_id), None)
    if not video:
        raise ValueError("unknown video id")
    threshold = 0.25 if confidence is None else confidence
    if threshold < 0 or threshold > 1:
        raise ValueError("confidence must be between 0 and 1")
    if source == "tracking":
        if not tracking_job_id:
            raise ValueError("tracking_job_id is required for tracking pose")
        video_path = get_tracking_video_path(tracking_job_id)
        if not video_path:
            raise ValueError("tracking video is not available")
        return data, video_path, float(threshold)
    video_path = resolve_project_video_path(project_path, video)
    if not video_path.exists():
        raise ValueError(f"video file not found: {video_path}")
    return data, video_path, float(threshold)


def _run_pose_job(job_id: str, video_path: Path, confidence: float, runner: PoseRunner | None) -> None:
    _mutate_job(job_id, lambda job: job.update({"status": "running"}))
    try:
        result = (runner or _render_pose_video)(video_path, confidence, POSE_MODEL)

        def finish(job: dict[str, Any]) -> None:
            job["status"] = "completed"
            job["video_path"] = str(result["video_path"])
            job["progress"]["processed"] = result.get("processed", 0)
            job["progress"]["total"] = result.get("total")

        _mutate_job(job_id, finish)
    except Exception as exc:
        message = str(exc)

        def fail(job: dict[str, Any]) -> None:
            job["status"] = "failed"
            job["error"] = message
            job.setdefault("traceback", traceback.format_exc())

        _mutate_job(job_id, fail)


def _mutate_job(job_id: str, mutator: Callable[[dict[str, Any]], None]) -> None:
    with _LOCK:
        job = _JOBS[job_id]
        mutator(job)
        job["updated_at"] = _now()


def _render_pose_video(video_path: Path, confidence: float, model_name: str) -> dict[str, Any]:
    if cv2 is None:
        raise RuntimeError("opencv-python is required to render human pose videos")
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("ultralytics is not installed. Install it with: pip install -e '.[dev,inference]'") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    fps = fps if fps > 0 else 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError(f"could not read video dimensions: {video_path}")

    output = tempfile.NamedTemporaryFile(prefix="model_creator_pose_", suffix=".webm", delete=False)
    output_path = Path(output.name)
    output.close()
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"VP80"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        output_path.unlink(missing_ok=True)
        raise ValueError("could not create human pose video")

    model = None
    processed = 0
    try:
        model = YOLO(model_name)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            results = model.predict(frame, conf=confidence, verbose=False)
            result = results[0] if results else None
            _draw_pose_result(frame, result)
            writer.write(frame)
            processed += 1
        return {"video_path": output_path, "processed": processed, "total": total}
    finally:
        capture.release()
        writer.release()
        del model
        gc.collect()
        _release_cuda_cache()


def _draw_pose_result(frame: Any, result: Any) -> None:
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None:
        return
    raw_points = getattr(keypoints, "data", None)
    if raw_points is None:
        return
    for person in raw_points.cpu().numpy().tolist():
        points = [_normalize_keypoint(point) for point in person[:17]]
        for left, right in COCO_POSE_EDGES:
            if left >= len(points) or right >= len(points):
                continue
            left_point = points[left]
            right_point = points[right]
            if not left_point or not right_point:
                continue
            cv2.line(frame, left_point, right_point, (37, 99, 235), 3, cv2.LINE_AA)
        for index, point in enumerate(points):
            if index not in BODY_KEYPOINTS:
                continue
            if point:
                cv2.circle(frame, point, 4, (216, 137, 6), -1, cv2.LINE_AA)


def _normalize_keypoint(point: list[float]) -> tuple[int, int] | None:
    if len(point) < 2:
        return None
    confidence = float(point[2]) if len(point) >= 3 else 1.0
    if confidence <= 0:
        return None
    return int(round(float(point[0]))), int(round(float(point[1])))


def _release_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
