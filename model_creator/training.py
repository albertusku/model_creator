from __future__ import annotations

import csv
import gc
import io
import threading
import traceback
import uuid
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .exporters import export_dataset
from .schemas import SplitConfig, TrainingStartRequest
from .storage import load_project, project_root, save_project

TRAINING_ASSETS = {
    "results.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "F1_curve.png",
    "P_curve.png",
    "PR_curve.png",
    "R_curve.png",
    "labels.jpg",
    "labels_correlogram.jpg",
}

TrainingRunner = Callable[[TrainingStartRequest, Path, Path], dict[str, Any]]

_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": job["id"],
        "status": job["status"],
        "logs": list(job.get("logs", [])),
        "snapshot_path": str(job["snapshot_path"]) if job.get("snapshot_path") else None,
        "run_path": str(job["run_path"]) if job.get("run_path") else None,
        "best_model_path": str(job["best_model_path"]) if job.get("best_model_path") else None,
        "last_model_path": str(job["last_model_path"]) if job.get("last_model_path") else None,
        "metrics": job.get("metrics") or {},
        "assets": list(job.get("assets", [])),
        "error": job.get("error"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
    }


def _append_log(job_id: str, message: str) -> None:
    with _lock:
        job = _jobs[job_id]
        job.setdefault("logs", []).append(message)


def get_job(job_id: str) -> dict[str, Any] | None:
    with _lock:
        job = _jobs.get(job_id)
        return _public_job(job) if job else None


def get_asset_path(job_id: str, name: str) -> Path | None:
    if name != Path(name).name or name not in TRAINING_ASSETS:
        return None
    with _lock:
        job = _jobs.get(job_id)
        run_path = job.get("run_path") if job else None
    if not run_path:
        return None
    path = Path(run_path) / name
    return path if path.exists() and path.is_file() else None


def start_training_job(
    request: TrainingStartRequest,
    *,
    runner: TrainingRunner | None = None,
    run_in_background: bool = True,
) -> dict[str, Any]:
    if not request.project_path.strip():
        raise ValueError("training requires opening the project by backend path")
    root = project_root(request.project_path)
    if not (root / "project.json").exists():
        raise ValueError("training requires opening the project by backend path")

    data = load_project(root)
    split = SplitConfig(train=request.train, val=request.val, test=request.test)
    snapshot_path = export_dataset(str(root), "yolo", split)
    job_id = uuid.uuid4().hex
    run_path = root / "training_runs" / job_id

    job = {
        "id": job_id,
        "status": "queued",
        "logs": [f"Created YOLO snapshot: {snapshot_path}"],
        "snapshot_path": snapshot_path,
        "run_path": run_path,
        "best_model_path": None,
        "last_model_path": None,
        "metrics": {},
        "assets": [],
        "error": None,
        "created_at": _now(),
        "started_at": None,
        "finished_at": None,
        "project_path": str(root),
        "request": request.dict(),
        "project_name": data.get("name"),
    }
    with _lock:
        _jobs[job_id] = job

    target = _run_training_job
    args = (job_id, request, snapshot_path, run_path, runner)
    if run_in_background:
        thread = threading.Thread(target=target, args=args, daemon=True)
        thread.start()
    else:
        target(*args)
    return get_job(job_id) or _public_job(job)


def _run_training_job(
    job_id: str,
    request: TrainingStartRequest,
    snapshot_path: Path,
    run_path: Path,
    runner: TrainingRunner | None,
) -> None:
    with _lock:
        job = _jobs[job_id]
        job["status"] = "running"
        job["started_at"] = _now()
    try:
        _append_log(job_id, "Training started")
        result = (runner or _ultralytics_runner)(request, snapshot_path, run_path)
        for line in result.get("logs", []):
            _append_log(job_id, str(line))
        best_path = Path(result.get("best_model_path") or run_path / "weights" / "best.pt")
        last_path = Path(result.get("last_model_path") or run_path / "weights" / "last.pt")
        metrics = result.get("metrics") or _read_metrics(run_path)
        assets = sorted(path.name for path in run_path.iterdir() if path.name in TRAINING_ASSETS) if run_path.exists() else []
        _record_trained_model(
            request.project_path,
            best_path,
            last_path,
            snapshot_path,
            run_path,
            metrics,
        )
        with _lock:
            job = _jobs[job_id]
            job["status"] = "completed"
            job["finished_at"] = _now()
            job["best_model_path"] = best_path if best_path.exists() else None
            job["last_model_path"] = last_path if last_path.exists() else None
            job["metrics"] = metrics
            job["assets"] = assets
            job["logs"].append("Training completed")
    except Exception as exc:
        with _lock:
            job = _jobs[job_id]
            job["status"] = "failed"
            job["finished_at"] = _now()
            job["error"] = str(exc)
            job["logs"].append(str(exc))
            job["logs"].append(traceback.format_exc())


def _ultralytics_runner(request: TrainingStartRequest, snapshot_path: Path, run_path: Path) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install training dependencies with: pip install -e '.[dev,inference]'") from exc

    batch = _parse_batch(request.batch)
    device = _resolve_device(request.device)
    stream = io.StringIO()
    model = None
    try:
        model = YOLO(request.model)
        with redirect_stdout(stream), redirect_stderr(stream):
            model.train(
                data=str(snapshot_path / "data.yaml"),
                epochs=request.epochs,
                imgsz=request.image_size,
                batch=batch,
                device=device,
                project=str(run_path.parent),
                name=run_path.name,
                exist_ok=True,
                workers=0,
            )
        return {
            "best_model_path": run_path / "weights" / "best.pt",
            "last_model_path": run_path / "weights" / "last.pt",
            "metrics": _read_metrics(run_path),
            "logs": stream.getvalue().splitlines(),
        }
    finally:
        del model
        gc.collect()
        _release_cuda_cache()


def _release_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _parse_batch(value: str) -> int:
    text = str(value).strip().lower()
    if text in {"", "auto"}:
        return -1
    batch = int(text)
    if batch < 1:
        raise ValueError("batch must be auto or a positive integer")
    return batch


def _resolve_device(value: str) -> str:
    text = str(value).strip().lower()
    if text and text != "auto":
        return text
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _read_metrics(run_path: Path) -> dict[str, float]:
    results_csv = run_path / "results.csv"
    if not results_csv.exists():
        return {}
    with results_csv.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    metrics: dict[str, float] = {}
    for key, value in rows[-1].items():
        cleaned = key.strip()
        try:
            metrics[cleaned] = float(str(value).strip())
        except (TypeError, ValueError):
            continue
    return metrics


def _record_trained_model(
    project_path: str,
    best_path: Path,
    last_path: Path,
    snapshot_path: Path,
    run_path: Path,
    metrics: dict[str, float],
) -> None:
    data = load_project(project_path)
    data["last_trained_model"] = {
        "best_model_path": str(best_path) if best_path.exists() else None,
        "last_model_path": str(last_path) if last_path.exists() else None,
        "trained_at": _now(),
        "snapshot_path": str(snapshot_path),
        "run_path": str(run_path),
        "metrics": metrics,
    }
    if best_path.exists():
        model = data.get("model") or {}
        model["path"] = str(best_path)
        model.setdefault("confidence", 0.25)
        data["model"] = model
    save_project(project_path, data)
