from __future__ import annotations

import json

import pytest

from model_creator.exporters import export_dataset, validate_project
from model_creator.inference import map_model_class
from model_creator.schemas import Box, SplitConfig
from model_creator.storage import add_images, configure_model, create_project, load_project, save_annotations


def make_project(tmp_path):
    root = tmp_path / "dataset"
    create_project(str(root), "Dataset", ["car", "person"], SplitConfig())
    image_file = root / "images" / "frame.jpg"
    image_file.parent.mkdir(parents=True, exist_ok=True)
    image_file.write_bytes(b"fake image")
    add_images(
        str(root),
        [
            {
                "id": "img-1",
                "file": "images/frame.jpg",
                "width": 100,
                "height": 80,
                "video_id": "video-1",
                "source_frame": 30,
                "sequence": 0,
            }
        ],
    )
    return root


def test_create_project_and_save_annotations(tmp_path):
    root = make_project(tmp_path)

    saved = save_annotations(
        str(root),
        "img-1",
        [Box(id="box-1", class_id=0, x=10, y=12, width=30, height=20)],
        reviewed=True,
    )

    assert saved["reviewed"] is True
    data = load_project(root)
    assert data["classes"][0]["name"] == "car"
    assert data["annotations"]["img-1"]["boxes"][0]["width"] == 30


def test_validation_rejects_box_outside_image(tmp_path):
    root = make_project(tmp_path)
    data = load_project(root)
    data["annotations"]["img-1"] = {
        "reviewed": True,
        "boxes": [{"id": "box-1", "class_id": 0, "x": 90, "y": 10, "width": 20, "height": 10}],
    }

    errors = validate_project(data)

    assert any("exceeds image width" in error for error in errors)


def test_export_yolo_snapshot(tmp_path):
    root = make_project(tmp_path)
    save_annotations(
        str(root),
        "img-1",
        [Box(id="box-1", class_id=0, x=10, y=20, width=40, height=20)],
        reviewed=True,
    )

    target = export_dataset(str(root), "yolo", SplitConfig(train=1, val=0, test=0))

    label = target / "labels" / "train" / "frame.txt"
    assert label.exists()
    assert label.read_text(encoding="utf-8").strip() == "0 0.300000 0.375000 0.400000 0.250000"
    data_yaml = target / "data.yaml"
    assert data_yaml.exists()
    assert f"path: {target.as_posix()!r}" in data_yaml.read_text(encoding="utf-8")


def test_export_coco_snapshot_uses_positive_category_ids(tmp_path):
    root = make_project(tmp_path)
    save_annotations(
        str(root),
        "img-1",
        [Box(id="box-1", class_id=0, x=10, y=20, width=40, height=20)],
        reviewed=True,
    )

    target = export_dataset(str(root), "coco", SplitConfig(train=1, val=0, test=0))

    payload = json.loads((target / "annotations" / "instances_train.json").read_text(encoding="utf-8"))
    assert payload["categories"][0]["id"] == 1
    assert payload["annotations"][0]["category_id"] == 1


def test_export_requires_reviewed_images(tmp_path):
    root = make_project(tmp_path)

    with pytest.raises(ValueError, match="no reviewed images"):
        export_dataset(str(root), "yolo", SplitConfig())


def test_export_uses_only_reviewed_images(tmp_path):
    root = make_project(tmp_path)
    second_image = root / "images" / "unreviewed.jpg"
    second_image.write_bytes(b"fake image")
    add_images(
        str(root),
        [
            {
                "id": "img-2",
                "file": "images/unreviewed.jpg",
                "width": 100,
                "height": 80,
                "video_id": "video-1",
                "source_frame": 60,
                "sequence": 1,
            }
        ],
    )
    save_annotations(
        str(root),
        "img-1",
        [Box(id="box-1", class_id=0, x=10, y=20, width=40, height=20)],
        reviewed=True,
    )

    target = export_dataset(str(root), "yolo", SplitConfig(train=1, val=0, test=0))

    assert (target / "images" / "train" / "frame.jpg").exists()
    assert not (target / "images" / "train" / "unreviewed.jpg").exists()


def test_configure_model_saves_path_and_confidence(tmp_path):
    root = make_project(tmp_path)
    model_file = tmp_path / "best.pt"
    model_file.write_bytes(b"weights")

    config = configure_model(str(root), str(model_file), 0.4)

    data = load_project(root)
    assert config["confidence"] == 0.4
    assert data["model"]["path"] == str(model_file.resolve())


def test_model_class_mapping_prefers_matching_names(tmp_path):
    root = make_project(tmp_path)
    data = load_project(root)

    mapped = map_model_class(7, {7: "person"}, data)

    assert mapped == 1


def test_model_class_mapping_falls_back_to_same_class_id(tmp_path):
    root = make_project(tmp_path)
    data = load_project(root)

    mapped = map_model_class(0, {0: "unknown"}, data)

    assert mapped == 0
