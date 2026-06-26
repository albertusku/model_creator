from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from .inference import predict_boxes
from .schemas import Box
from .storage import load_project, save_annotations

Predictor = Callable[[str, str, float | None], list[dict[str, Any]]]

_JOBS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_counts() -> dict[str, int]:
    return {
        "total": 0,
        "processed": 0,
        "approved": 0,
        "insufficient": 0,
        "skipped": 0,
        "failed": 0,
    }


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": job["id"],
        "status": job["status"],
        "project_path": job["project_path"],
        "confidence": job["confidence"],
        "counts": dict(job["counts"]),
        "updated_annotations": dict(job["updated_annotations"]),
        "manual_review_image_ids": list(job["manual_review_image_ids"]),
        "errors": list(job["errors"]),
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }


def get_auto_review_job(job_id: str) -> dict[str, Any] | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        return _public_job(job) if job else None


def start_auto_review_job(
    project_path: str,
    confidence: float | None = None,
    predictor: Predictor | None = None,
) -> dict[str, Any]:
    if not project_path.strip():
        raise ValueError("project_path is required")

    data = load_project(project_path)
    model_config = data.get("model")
    if not model_config:
        raise ValueError("no model configured for this project")

    threshold = model_config.get("confidence", 0.25) if confidence is None else confidence
    if threshold < 0 or threshold > 1:
        raise ValueError("confidence must be between 0 and 1")

    pending_images = [
        image
        for image in data.get("images", [])
        if not data.get("annotations", {}).get(image["id"], {"reviewed": False}).get("reviewed", False)
    ]
    job_id = str(uuid.uuid4())
    job = {
        "id": job_id,
        "status": "running",
        "project_path": project_path,
        "confidence": threshold,
        "counts": {**_empty_counts(), "total": len(pending_images)},
        "updated_annotations": {},
        "manual_review_image_ids": [],
        "errors": [],
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    with _LOCK:
        _JOBS[job_id] = job

    thread = threading.Thread(
        target=_run_auto_review_job,
        args=(job_id, project_path, pending_images, threshold, predictor or predict_boxes),
        daemon=True,
    )
    thread.start()
    return get_auto_review_job(job_id) or _public_job(job)


def _update_job(job_id: str, mutator: Callable[[dict[str, Any]], None]) -> None:
    with _LOCK:
        job = _JOBS[job_id]
        mutator(job)
        job["updated_at"] = _now_iso()


def _run_auto_review_job(
    job_id: str,
    project_path: str,
    images: list[dict[str, Any]],
    confidence: float,
    predictor: Predictor,
) -> None:
    try:
        for image in images:
            image_id = image["id"]
            try:
                data = load_project(project_path)
                annotation = data.get("annotations", {}).get(image_id, {"reviewed": False, "boxes": []})
                if annotation.get("reviewed", False):
                    _update_job(job_id, lambda job: _increment(job, "processed"))
                    continue
                if annotation.get("boxes"):
                    _update_job(job_id, lambda job: _increment_many(job, ("processed", "skipped")))
                    continue

                suggestions = predictor(project_path, image_id, confidence)
                reviewed = bool(suggestions) and all(
                    float(box.get("confidence", 0.0)) >= confidence for box in suggestions
                )
                boxes = [Box(**box) for box in suggestions]
                saved = save_annotations(project_path, image_id, boxes, reviewed)

                def record(job: dict[str, Any]) -> None:
                    _increment(job, "processed")
                    _increment(job, "approved" if reviewed else "insufficient")
                    job["updated_annotations"][image_id] = saved
                    if not reviewed and image_id not in job["manual_review_image_ids"]:
                        job["manual_review_image_ids"].append(image_id)

                _update_job(job_id, record)
            except Exception as exc:
                error_message = str(exc)

                def record_failure(job: dict[str, Any]) -> None:
                    _increment_many(job, ("processed", "failed"))
                    job["errors"].append({"image_id": image_id, "error": error_message})

                _update_job(job_id, record_failure)
    finally:
        def finish(job: dict[str, Any]) -> None:
            if job["status"] == "running":
                job["status"] = "completed"

        _update_job(job_id, finish)


def _increment(job: dict[str, Any], key: str) -> None:
    job["counts"][key] += 1


def _increment_many(job: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        _increment(job, key)
