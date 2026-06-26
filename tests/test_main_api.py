from __future__ import annotations

from io import BytesIO

import pytest
from fastapi import UploadFile

from model_creator.core.file_dialog import choose_directory
from model_creator.app.main import api_add_tracking_video, api_choose_project_directory, api_discover_projects, api_model_files
from model_creator.core.schemas import SplitConfig
from model_creator.core.storage import create_project, load_project


def test_project_directory_dialog_endpoint(monkeypatch):
    monkeypatch.setattr("model_creator.app.main.choose_directory", lambda: "/tmp/project")

    response = api_choose_project_directory()

    assert response == {"path": "/tmp/project"}


def test_choose_directory_uses_zenity(monkeypatch):
    class Completed:
        returncode = 0
        stdout = "/tmp/project\n"
        stderr = ""

    monkeypatch.setattr("model_creator.core.file_dialog.shutil.which", lambda name: "/usr/bin/zenity")
    monkeypatch.setattr("model_creator.core.file_dialog.subprocess.run", lambda *args, **kwargs: Completed())

    assert choose_directory() == "/tmp/project"


def test_discover_projects_lists_direct_children_with_project_json(tmp_path):
    base = tmp_path / "projects"
    first = base / "alpha"
    second = base / "beta"
    ignored = base / "not-a-project"
    create_project(str(first), "Alpha Dataset", ["car"], SplitConfig())
    create_project(str(second), "Beta Dataset", ["person"], SplitConfig())
    ignored.mkdir(parents=True)

    response = api_discover_projects(str(base))

    assert response["base_path"] == str(base.resolve())
    assert response["projects"] == [
        {"name": "alpha", "path": str(first.resolve()), "project_name": "Alpha Dataset"},
        {"name": "beta", "path": str(second.resolve()), "project_name": "Beta Dataset"},
    ]


def test_model_files_lists_pt_files_inside_project_only(tmp_path):
    root = tmp_path / "dataset"
    create_project(str(root), "Dataset", ["car"], SplitConfig())
    weights = root / "training_runs" / "job-1" / "weights"
    weights.mkdir(parents=True)
    best = weights / "best.pt"
    last = weights / "last.pt"
    best.write_bytes(b"best")
    last.write_bytes(b"last")
    (tmp_path / "outside.pt").write_bytes(b"outside")

    response = api_model_files(str(root))

    assert response["models"] == [
        {"name": "training_runs/job-1/weights/best.pt", "path": str(best.resolve())},
        {"name": "training_runs/job-1/weights/last.pt", "path": str(last.resolve())},
    ]


@pytest.mark.anyio
async def test_add_tracking_video_registers_video_without_extracting_images(tmp_path):
    root = tmp_path / "dataset"
    create_project(str(root), "Dataset", ["car"], SplitConfig())

    response = await api_add_tracking_video(
        project_path=str(root),
        file=UploadFile(BytesIO(b"video-bytes"), filename="tracking.mp4"),
    )

    data = load_project(root)
    assert response["video_id"] == data["videos"][0]["id"]
    assert data["videos"][0]["source_name"] == "tracking.mp4"
    assert data["videos"][0]["every_n_frames"] == 0
    assert data["images"] == []
    assert (root / "videos" / data["videos"][0]["stored_name"]).read_bytes() == b"video-bytes"
