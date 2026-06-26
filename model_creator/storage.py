from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schemas import Box, ClassDef, SplitConfig

PROJECT_FILE = "project.json"
DEFAULT_PROJECTS_DIR = Path(__file__).resolve().parent.parent / "projects"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def project_root(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def project_file(path: str | Path) -> Path:
    return project_root(path) / PROJECT_FILE


def projects_base(path: str | Path | None = None) -> Path:
    if path and str(path).strip():
        return Path(path).expanduser().resolve()
    return DEFAULT_PROJECTS_DIR.resolve()


def ensure_dirs(root: Path) -> None:
    for name in ("images", "videos", "exports"):
        (root / name).mkdir(parents=True, exist_ok=True)


def default_project(name: str, classes: list[str], split: SplitConfig) -> dict[str, Any]:
    cleaned = []
    seen = set()
    for class_name in classes:
        value = class_name.strip()
        if value and value not in seen:
            seen.add(value)
            cleaned.append(value)
    if not cleaned:
        raise ValueError("at least one class is required")

    return {
        "version": 1,
        "name": name.strip() or "Untitled Dataset",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "classes": [ClassDef(id=i, name=value).dict() for i, value in enumerate(cleaned)],
        "split": split.normalized().dict(),
        "videos": [],
        "images": [],
        "annotations": {},
        "model": None,
    }


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = now_iso()
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temp_name = handle.name
    Path(temp_name).replace(path)


def create_project(path: str, name: str, classes: list[str], split: SplitConfig) -> dict[str, Any]:
    root = project_root(path)
    root.mkdir(parents=True, exist_ok=True)
    ensure_dirs(root)
    data = default_project(name, classes, split)
    atomic_write_json(project_file(root), data)
    return data


def load_project(path: str | Path) -> dict[str, Any]:
    file_path = project_file(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Project file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_projects(base_path: str | Path | None = None) -> list[dict[str, str]]:
    base = projects_base(base_path)
    base.mkdir(parents=True, exist_ok=True)
    projects = []
    for child in sorted(base.iterdir(), key=lambda item: item.name.lower()):
        if not child.is_dir():
            continue
        file_path = child / PROJECT_FILE
        if not file_path.exists():
            continue
        project_name = child.name
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                project_name = json.load(handle).get("name") or project_name
        except Exception:
            pass
        projects.append({"name": child.name, "path": str(child.resolve()), "project_name": project_name})
    return projects


def discover_project_models(project_path: str | Path) -> list[dict[str, str]]:
    root = project_root(project_path)
    if not project_file(root).exists():
        raise FileNotFoundError(f"Project file not found: {project_file(root)}")
    models = []
    for path in sorted(root.rglob("*.pt"), key=lambda item: str(item.relative_to(root)).lower()):
        resolved = path.resolve()
        if not path.is_file() or not resolved.is_relative_to(root):
            continue
        models.append({"name": path.relative_to(root).as_posix(), "path": str(resolved)})
    return models


def resolve_project_video_path(project_path: str | Path, video: dict[str, Any]) -> Path:
    root = project_root(project_path)
    candidates = []
    stored_name = str(video.get("stored_name") or "").strip()
    source_name = str(video.get("source_name") or "").strip()
    if stored_name:
        stored_path = Path(stored_name)
        candidates.append(root / stored_path)
        candidates.append(root / "videos" / stored_path.name)
    if source_name:
        candidates.append(root / "videos" / Path(source_name).name)
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return candidates[0] if candidates else root / "videos" / ""


def save_project(path: str | Path, data: dict[str, Any]) -> None:
    ensure_dirs(project_root(path))
    atomic_write_json(project_file(path), data)


def add_video(project_path: str, source_name: str, stored_name: str, every_n_frames: int, frame_count: int) -> str:
    data = load_project(project_path)
    video_id = str(uuid.uuid4())
    data["videos"].append(
        {
            "id": video_id,
            "source_name": source_name,
            "stored_name": stored_name,
            "every_n_frames": every_n_frames,
            "frame_count": frame_count,
            "imported_at": now_iso(),
        }
    )
    save_project(project_path, data)
    return video_id


def add_images(project_path: str, images: list[dict[str, Any]]) -> None:
    data = load_project(project_path)
    data["images"].extend(images)
    for image in images:
        data["annotations"].setdefault(image["id"], {"reviewed": False, "boxes": []})
    save_project(project_path, data)


def save_annotations(project_path: str, image_id: str, boxes: list[Box], reviewed: bool) -> dict[str, Any]:
    data = load_project(project_path)
    image_ids = {image["id"] for image in data["images"]}
    class_ids = {class_def["id"] for class_def in data["classes"]}
    if image_id not in image_ids:
        raise ValueError("unknown image id")
    for box in boxes:
        if box.class_id not in class_ids:
            raise ValueError(f"unknown class id: {box.class_id}")
    data["annotations"][image_id] = {"reviewed": reviewed, "boxes": [box.dict() for box in boxes]}
    save_project(project_path, data)
    return data["annotations"][image_id]


def configure_model(project_path: str, model_path: str, confidence: float) -> dict[str, Any]:
    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"model file not found: {path}")
    if confidence < 0 or confidence > 1:
        raise ValueError("confidence must be between 0 and 1")
    data = load_project(project_path)
    data["model"] = {"path": str(path), "confidence": confidence}
    save_project(project_path, data)
    return data["model"]


def copy_upload_to_project(project_path: str, filename: str, source_path: Path) -> Path:
    root = project_root(project_path)
    target = root / "videos" / f"{uuid.uuid4()}_{Path(filename).name}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target)
    return target
