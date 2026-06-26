from __future__ import annotations

import uuid
import gc
from pathlib import Path
from typing import Any

from .storage import load_project, project_root


def class_name_map(data: dict[str, Any]) -> dict[str, int]:
    return {item["name"].strip().lower(): item["id"] for item in data.get("classes", [])}


def map_model_class(model_class_id: int, model_names: dict | list | None, data: dict[str, Any]) -> int | None:
    project_class_ids = {item["id"] for item in data.get("classes", [])}
    if model_names is not None:
        if isinstance(model_names, dict):
            model_name = model_names.get(model_class_id) or model_names.get(str(model_class_id))
        else:
            model_name = model_names[model_class_id] if model_class_id < len(model_names) else None
        if model_name is not None:
            mapped = class_name_map(data).get(str(model_name).strip().lower())
            if mapped is not None:
                return mapped
    if model_class_id in project_class_ids:
        return model_class_id
    return None


def predict_boxes(project_path: str, image_id: str, confidence: float | None = None) -> list[dict[str, Any]]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("ultralytics is not installed. Install it with: pip install -e '.[inference]'") from exc

    data = load_project(project_path)
    model_config = data.get("model")
    if not model_config:
        raise ValueError("no model configured for this project")
    image = next((item for item in data.get("images", []) if item["id"] == image_id), None)
    if not image:
        raise ValueError("unknown image id")

    model_path = Path(model_config["path"]).expanduser().resolve()
    image_path = project_root(project_path) / image["file"]
    conf = model_config.get("confidence", 0.25) if confidence is None else confidence
    if conf < 0 or conf > 1:
        raise ValueError("confidence must be between 0 and 1")

    model = None
    results = None
    try:
        model = YOLO(str(model_path))
        results = model.predict(str(image_path), conf=conf, verbose=False)
        if not results:
            return []

        model_names = getattr(results[0], "names", None) or getattr(model, "names", None)
        suggestions: list[dict[str, Any]] = []
        for raw_box in results[0].boxes:
            xyxy = raw_box.xyxy[0].tolist()
            model_class_id = int(raw_box.cls[0].item())
            project_class_id = map_model_class(model_class_id, model_names, data)
            if project_class_id is None:
                continue
            x1, y1, x2, y2 = [float(value) for value in xyxy]
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            if width <= 0 or height <= 0:
                continue
            suggestions.append(
                {
                    "id": str(uuid.uuid4()),
                    "class_id": project_class_id,
                    "x": max(0.0, x1),
                    "y": max(0.0, y1),
                    "width": min(width, image["width"] - max(0.0, x1)),
                    "height": min(height, image["height"] - max(0.0, y1)),
                    "confidence": float(raw_box.conf[0].item()),
                    "source": "model",
                }
            )
        return suggestions
    finally:
        del results
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
