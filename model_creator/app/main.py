from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..models.object_detection.auto_review import get_auto_review_job, start_auto_review_job
from ..datasets.exporters import export_dataset
from ..core.file_dialog import choose_directory
from ..models.object_detection.inference import predict_boxes
from ..core.schemas import (
    AutoReviewStartRequest,
    ConfigureModelRequest,
    CreateProjectRequest,
    ExportRequest,
    PoseStartRequest,
    ProjectPathRequest,
    SaveAnnotationsRequest,
    SplitConfig,
    SuggestBoxesRequest,
    TrackingCandidatesRequest,
    TrackingStartRequest,
    TrainingStartRequest,
)
from ..models.pose.pose import get_pose_job, get_pose_video_path, start_pose_job
from ..core.storage import (
    add_images,
    add_video,
    configure_model,
    copy_upload_to_project,
    create_project,
    discover_project_models,
    discover_projects,
    load_project,
    projects_base,
    project_root,
    save_annotations,
)
from ..media.video import extract_frames
from ..models.object_detection.tracking import generate_candidate_frames, get_tracking_job, get_tracking_video_path, start_tracking_job
from ..models.object_detection.training import get_asset_path, get_job, start_training_job

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(title="Model Creator")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/projects")
def api_create_project(payload: CreateProjectRequest) -> dict:
    try:
        return create_project(payload.path, payload.name, payload.classes, payload.split)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/projects/open")
def api_open_project(payload: ProjectPathRequest) -> dict:
    try:
        return load_project(payload.path)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/projects/discover")
def api_discover_projects(base_path: str | None = None) -> dict:
    try:
        base = projects_base(base_path)
        return {"base_path": str(base), "projects": discover_projects(base)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/dialogs/project-directory")
def api_choose_project_directory() -> dict[str, str | None]:
    try:
        return {"path": choose_directory()}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/images/{image_id}")
def api_image(image_id: str, project_path: str) -> FileResponse:
    data = load_project(project_path)
    image = next((item for item in data["images"] if item["id"] == image_id), None)
    if not image:
        raise HTTPException(status_code=404, detail="image not found")
    path = project_root(project_path) / image["file"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="image file not found")
    return FileResponse(path)


@app.post("/api/videos/import")
async def api_import_video(
    project_path: str = Form(...),
    every_n_frames: int = Form(...),
    file: UploadFile = File(...),
) -> dict:
    if every_n_frames < 1:
        raise HTTPException(status_code=400, detail="every_n_frames must be >= 1")
    try:
        root = project_root(project_path)
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            temp_path = Path(handle.name)
            while chunk := await file.read(1024 * 1024):
                handle.write(chunk)
        stored_video = copy_upload_to_project(project_path, file.filename or "video", temp_path)
        video_id = add_video(project_path, file.filename or stored_video.name, stored_video.name, every_n_frames, 0)
        images = extract_frames(root, stored_video, video_id, every_n_frames)
        data = load_project(project_path)
        for video in data["videos"]:
            if video["id"] == video_id:
                video["frame_count"] = len(images)
        from ..core.storage import save_project

        save_project(project_path, data)
        add_images(project_path, images)
        temp_path.unlink(missing_ok=True)
        return {"video_id": video_id, "frames": len(images), "project": load_project(project_path)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/tracking/videos")
async def api_add_tracking_video(
    project_path: str = Form(...),
    file: UploadFile = File(...),
) -> dict:
    temp_path: Path | None = None
    try:
        root = project_root(project_path)
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            temp_path = Path(handle.name)
            while chunk := await file.read(1024 * 1024):
                handle.write(chunk)
        stored_video = copy_upload_to_project(project_path, file.filename or "tracking_video", temp_path)
        video_id = add_video(project_path, file.filename or stored_video.name, stored_video.name, 0, 0)
        return {"video_id": video_id, "project": load_project(project_path)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)


@app.post("/api/annotations")
def api_save_annotations(payload: SaveAnnotationsRequest) -> dict:
    try:
        return save_annotations(payload.project_path, payload.image_id, payload.boxes, payload.reviewed)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/model/config")
def api_configure_model(payload: ConfigureModelRequest) -> dict:
    try:
        return configure_model(payload.project_path, payload.model_path, payload.confidence)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/model/files")
def api_model_files(project_path: str) -> dict:
    try:
        return {"models": discover_project_models(project_path)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/model/suggest")
def api_suggest_boxes(payload: SuggestBoxesRequest) -> dict:
    try:
        return {"boxes": predict_boxes(payload.project_path, payload.image_id, payload.confidence)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/model/auto-review/start")
def api_start_auto_review(payload: AutoReviewStartRequest) -> dict:
    try:
        return start_auto_review_job(payload.project_path, payload.confidence)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/model/auto-review/{job_id}")
def api_auto_review_status(job_id: str) -> dict:
    job = get_auto_review_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="auto-review job not found")
    return job


@app.post("/api/tracking/candidates")
def api_tracking_candidates(payload: TrackingCandidatesRequest) -> dict:
    try:
        return generate_candidate_frames(payload.project_path, payload.video_id, payload.class_id, payload.confidence)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/tracking/start")
def api_start_tracking(payload: TrackingStartRequest) -> dict:
    try:
        return start_tracking_job(
            payload.project_path,
            payload.video_id,
            payload.class_id,
            payload.start_frame,
            payload.start_box.dict(),
            payload.confidence,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/tracking/{job_id}")
def api_tracking_status(job_id: str) -> dict:
    job = get_tracking_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="tracking job not found")
    return job


@app.get("/api/tracking/{job_id}/video")
def api_tracking_video(job_id: str) -> FileResponse:
    path = get_tracking_video_path(job_id)
    if not path:
        raise HTTPException(status_code=404, detail="tracking video not found")
    return FileResponse(path, media_type="video/webm", filename=f"trajectory_{job_id}.webm")


@app.post("/api/pose/start")
def api_start_pose(payload: PoseStartRequest) -> dict:
    try:
        return start_pose_job(
            payload.project_path,
            payload.video_id,
            payload.source,
            payload.tracking_job_id,
            payload.confidence,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/pose/{job_id}")
def api_pose_status(job_id: str) -> dict:
    job = get_pose_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="pose job not found")
    return job


@app.get("/api/pose/{job_id}/video")
def api_pose_video(job_id: str) -> FileResponse:
    path = get_pose_video_path(job_id)
    if not path:
        raise HTTPException(status_code=404, detail="pose video not found")
    return FileResponse(path, media_type="video/webm", filename=f"pose_{job_id}.webm")


@app.post("/api/export")
def api_export(payload: ExportRequest) -> dict[str, str]:
    try:
        split = SplitConfig(train=payload.train, val=payload.val, test=payload.test)
        target = export_dataset(payload.project_path, payload.format, split)
        return {"path": str(target)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/training/start")
def api_start_training(payload: TrainingStartRequest) -> dict:
    try:
        return start_training_job(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/training/{job_id}")
def api_training_status(job_id: str) -> dict:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="training job not found")
    return job


@app.get("/api/training/{job_id}/assets/{name}")
def api_training_asset(job_id: str, name: str) -> FileResponse:
    path = get_asset_path(job_id, name)
    if not path:
        raise HTTPException(status_code=404, detail="training asset not found")
    return FileResponse(path)


app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")
