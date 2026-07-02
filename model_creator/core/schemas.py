from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, validator


class ClassDef(BaseModel):
    id: int
    name: str


class SplitConfig(BaseModel):
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    @validator("train", "val", "test")
    @classmethod
    def ratio_range(cls, value: float) -> float:
        if value < 0:
            raise ValueError("split ratios must be non-negative")
        return value

    def normalized(self) -> "SplitConfig":
        total = self.train + self.val + self.test
        if total <= 0:
            return SplitConfig()
        return SplitConfig(train=self.train / total, val=self.val / total, test=self.test / total)


class CreateProjectRequest(BaseModel):
    path: str
    name: str
    classes: list[str] = Field(default_factory=list)
    project_type: Literal["object_detection", "csv_classification"] = "object_detection"
    split: SplitConfig = Field(default_factory=SplitConfig)


class SetProjectTypeRequest(BaseModel):
    path: str
    project_type: Literal["object_detection", "csv_classification"]


class ProjectPathRequest(BaseModel):
    path: str


class Box(BaseModel):
    id: str
    class_id: int
    x: float
    y: float
    width: float
    height: float


class SaveAnnotationsRequest(BaseModel):
    project_path: str
    image_id: str
    boxes: list[Box]
    reviewed: bool = False


class ConfigureModelRequest(BaseModel):
    project_path: str
    model_path: str
    confidence: float = 0.25


class SuggestBoxesRequest(BaseModel):
    project_path: str
    image_id: str
    confidence: float | None = None


class TrackingCandidatesRequest(BaseModel):
    project_path: str
    video_id: str
    class_id: int
    confidence: float | None = None


class TrackingStartRequest(BaseModel):
    project_path: str
    video_id: str
    class_id: int
    start_frame: int
    start_box: Box
    confidence: float | None = None


class PoseStartRequest(BaseModel):
    project_path: str
    video_id: str
    source: Literal["original", "tracking"] = "original"
    tracking_job_id: str | None = None
    confidence: float | None = None


class AutoReviewStartRequest(BaseModel):
    project_path: str
    confidence: float | None = None


class ExportRequest(BaseModel):
    project_path: str
    format: Literal["yolo", "coco"]
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1


class TrainingStartRequest(BaseModel):
    project_path: str
    model: str = "yolo11n.pt"
    epochs: int = 50
    image_size: int = 640
    batch: str = "auto"
    device: str = "auto"
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    @validator("epochs", "image_size")
    @classmethod
    def positive_int(cls, value: int) -> int:
        if value < 1:
            raise ValueError("training numeric values must be positive")
        return value


class ClassificationDatasetImportResponse(BaseModel):
    dataset_id: str
    columns: list[str]
    rows: int
    summary: dict


class ClassificationTrainRequest(BaseModel):
    project_path: str
    dataset_id: str
    target_column: str
    feature_columns: list[str]
    test_size: float = 0.2

    @validator("test_size")
    @classmethod
    def valid_test_size(cls, value: float) -> float:
        if value <= 0 or value >= 1:
            raise ValueError("test_size must be between 0 and 1")
        return value


class ClassificationClusterRequest(BaseModel):
    project_path: str
    dataset_id: str
    target_column: str
    feature_columns: list[str]


class ClassificationPredictionRequest(BaseModel):
    project_path: str
    run_id: str | None = None
