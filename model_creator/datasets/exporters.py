from __future__ import annotations

import json
import math
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.schemas import SplitConfig
from ..core.storage import load_project, project_root


def validate_project(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    classes = {item["id"] for item in data.get("classes", [])}
    images = {item["id"]: item for item in data.get("images", [])}
    for image_id, state in data.get("annotations", {}).items():
        image = images.get(image_id)
        if not image:
            errors.append(f"annotation references unknown image {image_id}")
            continue
        for box in state.get("boxes", []):
            if box.get("class_id") not in classes:
                errors.append(f"{image_id}: unknown class id {box.get('class_id')}")
            if box.get("width", 0) <= 0 or box.get("height", 0) <= 0:
                errors.append(f"{image_id}: box has non-positive size")
            if box.get("x", 0) < 0 or box.get("y", 0) < 0:
                errors.append(f"{image_id}: box starts outside image")
            if box.get("x", 0) + box.get("width", 0) > image["width"] + 0.01:
                errors.append(f"{image_id}: box exceeds image width")
            if box.get("y", 0) + box.get("height", 0) > image["height"] + 0.01:
                errors.append(f"{image_id}: box exceeds image height")
    return errors


def reviewed_images(data: dict[str, Any]) -> list[dict[str, Any]]:
    annotations = data.get("annotations", {})
    return [image for image in data.get("images", []) if annotations.get(image["id"], {}).get("reviewed")]


def split_images(images: list[dict[str, Any]], split: SplitConfig) -> dict[str, list[dict[str, Any]]]:
    normalized = split.normalized()
    ordered = sorted(images, key=lambda item: item["id"])
    total = len(ordered)
    train_end = math.floor(total * normalized.train)
    val_end = train_end + math.floor(total * normalized.val)
    return {
        "train": ordered[:train_end],
        "val": ordered[train_end:val_end],
        "test": ordered[val_end:],
    }


def export_dataset(project_path: str, fmt: str, split: SplitConfig) -> Path:
    root = project_root(project_path)
    data = load_project(root)
    errors = validate_project(data)
    if errors:
        raise ValueError("; ".join(errors[:10]))
    images = reviewed_images(data)
    if not images:
        raise ValueError("no reviewed images to export")

    export_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = root / "exports" / f"{fmt}_{export_id}_{uuid.uuid4().hex[:8]}"
    target.mkdir(parents=True, exist_ok=False)

    splits = split_images(images, split)
    if fmt == "yolo":
        export_yolo(root, target, data, splits)
    elif fmt == "coco":
        export_coco(root, target, data, splits)
    else:
        raise ValueError(f"unsupported export format: {fmt}")

    manifest = {
        "project": data["name"],
        "format": fmt,
        "created_at": datetime.now().isoformat(),
        "classes": data["classes"],
        "split": split.normalized().dict(),
        "counts": {name: len(items) for name, items in splits.items()},
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return target


def export_yolo(root: Path, target: Path, data: dict[str, Any], splits: dict[str, list[dict[str, Any]]]) -> None:
    annotations = data["annotations"]
    class_names = [item["name"] for item in sorted(data["classes"], key=lambda item: item["id"])]
    (target / "data.yaml").write_text(
        f"path: {target.as_posix()!r}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names: {class_names!r}\n",
        encoding="utf-8",
    )
    for split_name, images in splits.items():
        image_dir = target / "images" / split_name
        label_dir = target / "labels" / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image in images:
            source = root / image["file"]
            shutil.copyfile(source, image_dir / source.name)
            lines = []
            for box in annotations.get(image["id"], {}).get("boxes", []):
                cx = (box["x"] + box["width"] / 2) / image["width"]
                cy = (box["y"] + box["height"] / 2) / image["height"]
                width = box["width"] / image["width"]
                height = box["height"] / image["height"]
                lines.append(f"{box['class_id']} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}")
            (label_dir / f"{source.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def export_coco(root: Path, target: Path, data: dict[str, Any], splits: dict[str, list[dict[str, Any]]]) -> None:
    annotations = data["annotations"]
    categories = [{"id": item["id"] + 1, "name": item["name"]} for item in data["classes"]]
    for split_name, images in splits.items():
        image_dir = target / "images" / split_name
        ann_dir = target / "annotations"
        image_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)
        coco_images = []
        coco_annotations = []
        ann_id = 1
        for idx, image in enumerate(images, start=1):
            source = root / image["file"]
            shutil.copyfile(source, image_dir / source.name)
            coco_images.append(
                {
                    "id": idx,
                    "file_name": f"images/{split_name}/{source.name}",
                    "width": image["width"],
                    "height": image["height"],
                }
            )
            for box in annotations.get(image["id"], {}).get("boxes", []):
                coco_annotations.append(
                    {
                        "id": ann_id,
                        "image_id": idx,
                        "category_id": box["class_id"] + 1,
                        "bbox": [box["x"], box["y"], box["width"], box["height"]],
                        "area": box["width"] * box["height"],
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
        payload = {"images": coco_images, "annotations": coco_annotations, "categories": categories}
        (ann_dir / f"instances_{split_name}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
