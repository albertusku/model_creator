from __future__ import annotations

import base64
import gc
import importlib.util
import tempfile
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .inference import map_model_class
from .storage import load_project, project_root, resolve_project_video_path

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

DetectionRunner = Callable[[str, list[Any], float, dict[str, Any], int], list[list[dict[str, Any]]]]
TrackingRunner = Callable[[str, Path, float, dict[str, Any], int, int, dict[str, Any]], list[dict[str, Any]]]

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
        "class_id": job["class_id"],
        "start_frame": job["start_frame"],
        "selected_track_id": job.get("selected_track_id"),
        "progress": dict(job["progress"]),
        "trajectory": list(job["trajectory"]),
        "video_url": f"/api/tracking/{job['id']}/video" if job.get("video_path") else None,
        "error": job.get("error"),
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }


def get_tracking_job(job_id: str) -> dict[str, Any] | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        return _public_job(job) if job else None


def get_tracking_video_path(job_id: str) -> Path | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        video_path = Path(job["video_path"]) if job and job.get("video_path") else None
    if not video_path or not video_path.exists() or not video_path.is_file():
        return None
    return video_path


def generate_candidate_frames(
    project_path: str,
    video_id: str,
    class_id: int,
    confidence: float | None = None,
    *,
    detector: DetectionRunner | None = None,
) -> dict[str, Any]:
    data, video, video_path, threshold = _validate_tracking_request(project_path, video_id, class_id, confidence)
    if cv2 is None:
        raise RuntimeError("opencv-python is required to extract tracking candidates")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 0
        fps = fps if fps > 0 else 30.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        available_seconds = 20 if frame_count <= 0 else min(20, max(1, int((frame_count - 1) // fps) + 1))
        frames: list[Any] = []
        metadata: list[dict[str, Any]] = []
        for second in range(available_seconds):
            frame_number = int(round(second * fps))
            if frame_count > 0 and frame_number >= frame_count:
                break
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                raise ValueError(f"could not encode candidate frame {frame_number}")
            frames.append(frame)
            metadata.append(
                {
                    "frame": frame_number,
                    "time_sec": float(second),
                    "width": int(width),
                    "height": int(height),
                    "image": "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii"),
                }
            )
    finally:
        capture.release()

    detections = (detector or _detect_candidate_boxes)(str(Path(data["model"]["path"]).expanduser().resolve()), frames, threshold, data, class_id)
    candidates = []
    for index, item in enumerate(metadata):
        boxes = detections[index] if index < len(detections) else []
        candidates.append({**item, "boxes": [box for box in boxes if box["class_id"] == class_id]})
    return {"video_id": video["id"], "class_id": class_id, "confidence": threshold, "candidates": candidates}


def start_tracking_job(
    project_path: str,
    video_id: str,
    class_id: int,
    start_frame: int,
    start_box: dict[str, Any],
    confidence: float | None = None,
    *,
    tracker: TrackingRunner | None = None,
    run_in_background: bool = True,
) -> dict[str, Any]:
    data, video, video_path, threshold = _validate_tracking_request(project_path, video_id, class_id, confidence)
    if start_frame < 0:
        raise ValueError("start_frame must be >= 0")
    _validate_box(start_box)
    if tracker is None:
        _require_tracking_dependencies()

    job_id = uuid.uuid4().hex
    job = {
        "id": job_id,
        "status": "queued",
        "project_path": project_path,
        "video_id": video_id,
        "class_id": class_id,
        "start_frame": start_frame,
        "selected_track_id": None,
        "progress": {"processed": 0, "total": None},
        "trajectory": [],
        "video_path": None,
        "error": None,
        "created_at": _now(),
        "updated_at": _now(),
    }
    with _LOCK:
        _JOBS[job_id] = job

    args = (job_id, str(Path(data["model"]["path"]).expanduser().resolve()), video_path, threshold, data, class_id, start_frame, start_box, tracker)
    if run_in_background:
        threading.Thread(target=_run_tracking_job, args=args, daemon=True).start()
    else:
        _run_tracking_job(*args)
    return get_tracking_job(job_id) or _public_job(job)


def _validate_tracking_request(
    project_path: str,
    video_id: str,
    class_id: int,
    confidence: float | None,
) -> tuple[dict[str, Any], dict[str, Any], Path, float]:
    if not project_path.strip():
        raise ValueError("project_path is required")
    data = load_project(project_path)
    model_config = data.get("model")
    if not model_config:
        raise ValueError("no model configured for this project")
    if class_id not in {item["id"] for item in data.get("classes", [])}:
        raise ValueError(f"unknown class id: {class_id}")
    video = next((item for item in data.get("videos", []) if item["id"] == video_id), None)
    if not video:
        raise ValueError("unknown video id")
    video_path = resolve_project_video_path(project_path, video)
    if not video_path.exists():
        raise ValueError(f"video file not found: {video_path}")
    threshold = model_config.get("confidence", 0.25) if confidence is None else confidence
    if threshold < 0 or threshold > 1:
        raise ValueError("confidence must be between 0 and 1")
    return data, video, video_path, float(threshold)


def _validate_box(box: dict[str, Any]) -> None:
    for key in ("x", "y", "width", "height"):
        if key not in box:
            raise ValueError(f"start_box missing {key}")
    if float(box["width"]) <= 0 or float(box["height"]) <= 0:
        raise ValueError("start_box width and height must be positive")


def _require_tracking_dependencies() -> None:
    if importlib.util.find_spec("lap") is None:
        raise RuntimeError(
            "ByteTrack requires the 'lap' package. Install it with: pip install -e '.[dev,inference]' "
            "or pip install 'lap>=0.5.12', then restart the app."
        )


def _run_tracking_job(
    job_id: str,
    model_path: str,
    video_path: Path,
    confidence: float,
    data: dict[str, Any],
    class_id: int,
    start_frame: int,
    start_box: dict[str, Any],
    tracker: TrackingRunner | None,
) -> None:
    _mutate_job(job_id, lambda job: job.update({"status": "running"}))
    try:
        trajectory = (tracker or _track_video)(model_path, video_path, confidence, data, class_id, start_frame, start_box)
        if not trajectory:
            raise ValueError("could not associate selected box with a track_id at the start frame")
        selected_track_id = trajectory[0].get("track_id")
        rendered_video_path = _render_trajectory_video(video_path, start_frame, trajectory)

        def finish(job: dict[str, Any]) -> None:
            job["status"] = "completed"
            job["trajectory"] = trajectory
            job["selected_track_id"] = selected_track_id
            job["progress"]["processed"] = len(trajectory)
            job["video_path"] = str(rendered_video_path)

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


def _detect_candidate_boxes(
    model_path: str,
    frames: list[Any],
    confidence: float,
    data: dict[str, Any],
    class_id: int,
) -> list[list[dict[str, Any]]]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("ultralytics is not installed. Install it with: pip install -e '.[dev,inference]'") from exc

    model = None
    results = None
    try:
        model = YOLO(model_path)
        results = model.predict(frames, conf=confidence, verbose=False) if frames else []
        model_names = getattr(results[0], "names", None) if results else getattr(model, "names", None)
        return [
            _boxes_from_result(result, model_names, data, class_id=class_id, fallback_to_requested_class=True)
            for result in results
        ]
    finally:
        del results
        del model
        gc.collect()
        _release_cuda_cache()


def _track_video(
    model_path: str,
    video_path: Path,
    confidence: float,
    data: dict[str, Any],
    class_id: int,
    start_frame: int,
    start_box: dict[str, Any],
) -> list[dict[str, Any]]:
    if cv2 is None:
        raise RuntimeError("opencv-python is required to track video")
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("ultralytics is not installed. Install it with: pip install -e '.[dev,inference]'") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    fps = fps if fps > 0 else 30.0
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    model = None
    selected_track_id = None
    trajectory: list[dict[str, Any]] = []
    frame_number = start_frame
    try:
        model = YOLO(model_path)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            results = model.track(frame, conf=confidence, persist=True, tracker="bytetrack.yaml", verbose=False)
            result = results[0] if results else None
            model_names = getattr(result, "names", None) if result is not None else getattr(model, "names", None)
            boxes = (
                _boxes_from_result(
                    result,
                    model_names,
                    data,
                    class_id=class_id,
                    include_track_id=True,
                    fallback_to_requested_class=True,
                )
                if result
                else []
            )
            if frame_number == start_frame:
                selected = _best_iou_track(start_box, boxes)
                if selected is None:
                    break
                selected_track_id = selected["track_id"]
            if selected_track_id is not None:
                for box in boxes:
                    if box.get("track_id") == selected_track_id:
                        trajectory.append(_trajectory_point(box, frame_number, fps))
                        break
            frame_number += 1
        return trajectory
    finally:
        capture.release()
        del model
        gc.collect()
        _release_cuda_cache()


def _render_trajectory_video(video_path: Path, start_frame: int, trajectory: list[dict[str, Any]]) -> Path:
    if cv2 is None:
        raise RuntimeError("opencv-python is required to render trajectory video")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    fps = fps if fps > 0 else 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError(f"could not read video dimensions: {video_path}")

    output = tempfile.NamedTemporaryFile(prefix="model_creator_track_", suffix=".webm", delete=False)
    output_path = Path(output.name)
    output.close()
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"VP80"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        output_path.unlink(missing_ok=True)
        raise ValueError("could not create trajectory video")

    points_by_frame = {int(point["frame"]): point for point in trajectory}
    line_points: list[tuple[int, int]] = []
    frame_number = start_frame
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            point = points_by_frame.get(frame_number)
            if point is not None:
                line_points.append((int(round(point["center_x"])), int(round(point["center_y"]))))
            if len(line_points) >= 2:
                cv2.polylines(frame, [line_points_array(line_points)], False, (109, 47, 255), 3, cv2.LINE_AA)
            writer.write(frame)
            frame_number += 1
    finally:
        capture.release()
        writer.release()
    return output_path


def line_points_array(points: list[tuple[int, int]]) -> Any:
    import numpy as np

    return np.array(points, dtype=np.int32).reshape((-1, 1, 2))


def _boxes_from_result(
    result: Any,
    model_names: dict | list | None,
    data: dict[str, Any],
    *,
    class_id: int | None = None,
    include_track_id: bool = False,
    fallback_to_requested_class: bool = False,
) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    raw_boxes = getattr(result, "boxes", None)
    if raw_boxes is None:
        return boxes
    for raw_box in raw_boxes:
        xyxy = raw_box.xyxy[0].tolist()
        model_class_id = int(raw_box.cls[0].item())
        project_class_id = map_model_class(model_class_id, model_names, data)
        if project_class_id is None and fallback_to_requested_class and class_id is not None:
            project_class_id = class_id
        if project_class_id is None or (class_id is not None and project_class_id != class_id):
            continue
        x1, y1, x2, y2 = [float(value) for value in xyxy]
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        if width <= 0 or height <= 0:
            continue
        box = {
            "id": str(uuid.uuid4()),
            "class_id": project_class_id,
            "x": max(0.0, x1),
            "y": max(0.0, y1),
            "width": width,
            "height": height,
            "confidence": float(raw_box.conf[0].item()),
            "source": "model",
        }
        if include_track_id:
            raw_id = getattr(raw_box, "id", None)
            if raw_id is None:
                continue
            box["track_id"] = int(raw_id[0].item())
        boxes.append(box)
    return boxes


def _best_iou_track(start_box: dict[str, Any], boxes: list[dict[str, Any]]) -> dict[str, Any] | None:
    best_box = None
    best_iou = 0.0
    for box in boxes:
        if box.get("track_id") is None:
            continue
        score = box_iou(start_box, box)
        if score > best_iou:
            best_iou = score
            best_box = box
    return best_box if best_iou > 0 else None


def box_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_x2 = float(left["x"]) + float(left["width"])
    left_y2 = float(left["y"]) + float(left["height"])
    right_x2 = float(right["x"]) + float(right["width"])
    right_y2 = float(right["y"]) + float(right["height"])
    inter_x1 = max(float(left["x"]), float(right["x"]))
    inter_y1 = max(float(left["y"]), float(right["y"]))
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_left = float(left["width"]) * float(left["height"])
    area_right = float(right["width"]) * float(right["height"])
    union = area_left + area_right - intersection
    return intersection / union if union > 0 else 0.0


def _trajectory_point(box: dict[str, Any], frame: int, fps: float) -> dict[str, Any]:
    return {
        "frame": frame,
        "time_sec": frame / fps,
        "x": box["x"],
        "y": box["y"],
        "width": box["width"],
        "height": box["height"],
        "center_x": box["x"] + box["width"] / 2,
        "center_y": box["y"] + box["height"] / 2,
        "confidence": box.get("confidence"),
        "track_id": box.get("track_id"),
    }


def _release_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
