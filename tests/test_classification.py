from __future__ import annotations

from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile

from model_creator.app.main import (
    api_classification_clusters,
    api_import_classification_dataset,
    api_predict_classification,
    api_set_project_type,
    api_start_classification_training,
)
from model_creator.core.schemas import ClassificationClusterRequest, ClassificationTrainRequest, SetProjectTypeRequest, SplitConfig
from model_creator.core.storage import create_project, load_project
from model_creator.models.classification.training import start_classification_training_job


def test_create_csv_classification_project_without_classes(tmp_path):
    root = tmp_path / "classification"

    data = create_project(str(root), "CSV Model", [], SplitConfig(), "csv_classification")

    assert data["project_type"] == "csv_classification"
    assert data["classes"] == []
    assert data["classification"]["datasets"] == []


def test_set_project_type_persists_legacy_choice(tmp_path):
    root = tmp_path / "legacy"
    create_project(str(root), "Legacy", ["car"], SplitConfig())
    data = load_project(root)
    del data["project_type"]
    (root / "project.json").write_text(__import__("json").dumps(data), encoding="utf-8")

    response = api_set_project_type(SetProjectTypeRequest(path=str(root), project_type="csv_classification"))

    assert response["project_type"] == "csv_classification"
    assert load_project(root)["project_type"] == "csv_classification"


@pytest.mark.anyio
async def test_import_classification_csv_returns_summary(tmp_path):
    root = tmp_path / "classification"
    create_project(str(root), "CSV Model", [], SplitConfig(), "csv_classification")

    response = await api_import_classification_dataset(
        project_path=str(root),
        file=UploadFile(BytesIO(b"age,color,target\n10,red,A\n20,blue,B\n"), filename="data.csv"),
    )

    assert response["rows"] == 2
    assert response["columns"] == ["age", "color", "target"]
    assert response["summary"]["columns"]["age"]["dtype"].startswith("int")
    assert (root / "classification" / "datasets" / f"{response['dataset_id']}.csv").exists()


@pytest.mark.anyio
async def test_import_classification_csv_detects_semicolon_delimiter(tmp_path):
    root = tmp_path / "classification"
    create_project(str(root), "CSV Model", [], SplitConfig(), "csv_classification")

    response = await api_import_classification_dataset(
        project_path=str(root),
        file=UploadFile(BytesIO("edad;color;objetivo\n10;rojo;A\n20;azul;B\n".encode()), filename="data.csv"),
    )

    assert response["columns"] == ["edad", "color", "objetivo"]
    assert response["rows"] == 2


@pytest.mark.anyio
async def test_import_classification_csv_rejects_empty(tmp_path):
    root = tmp_path / "classification"
    create_project(str(root), "CSV Model", [], SplitConfig(), "csv_classification")

    with pytest.raises(HTTPException) as exc:
        await api_import_classification_dataset(
            project_path=str(root),
            file=UploadFile(BytesIO(b""), filename="empty.csv"),
        )

    assert exc.value.status_code == 400
    assert "empty" in exc.value.detail.lower()


def test_classification_training_validates_target_and_features(tmp_path):
    root = tmp_path / "classification"
    create_project(str(root), "CSV Model", [], SplitConfig(), "csv_classification")
    dataset = root / "classification" / "datasets" / "dataset-1.csv"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_text("age,target\n10,A\n20,B\n", encoding="utf-8")
    data = load_project(root)
    data["classification"]["datasets"].append({"id": "dataset-1", "file": "classification/datasets/dataset-1.csv"})
    (root / "project.json").write_text(__import__("json").dumps(data), encoding="utf-8")

    with pytest.raises(HTTPException) as exc:
        api_start_classification_training(
            ClassificationTrainRequest(
                project_path=str(root),
                dataset_id="dataset-1",
                target_column="missing",
                feature_columns=["age"],
            )
        )

    assert exc.value.status_code == 400
    assert "target_column" in exc.value.detail


def test_classification_clusters_returns_pca_points(tmp_path):
    root = tmp_path / "classification"
    create_project(str(root), "CSV Model", [], SplitConfig(), "csv_classification")
    dataset = root / "classification" / "datasets" / "dataset-1.csv"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_text(
        "age,color,target\n10,red,A\n12,red,A\n40,blue,B\n42,blue,B\n",
        encoding="utf-8",
    )
    data = load_project(root)
    data["classification"]["datasets"].append(
        {"id": "dataset-1", "file": "classification/datasets/dataset-1.csv", "columns": ["age", "color", "target"]}
    )
    (root / "project.json").write_text(__import__("json").dumps(data), encoding="utf-8")

    response = api_classification_clusters(
        ClassificationClusterRequest(
            project_path=str(root),
            dataset_id="dataset-1",
            target_column="target",
            feature_columns=["age", "color"],
        )
    )

    assert response["method"] == "pca"
    assert response["labels"] == ["A", "B"]
    assert response["rows"] == 4
    assert len(response["points"]) == 4
    assert set(response["points"][0]) == {"x", "y", "label"}


def test_classification_training_completes_and_predicts(tmp_path):
    root = tmp_path / "classification"
    create_project(str(root), "CSV Model", [], SplitConfig(), "csv_classification")
    dataset = root / "classification" / "datasets" / "dataset-1.csv"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_text(
        "age,color,target\n"
        "10,red,A\n"
        "12,red,A\n"
        "40,blue,B\n"
        "42,blue,B\n"
        "11,red,A\n"
        "41,blue,B\n",
        encoding="utf-8",
    )
    data = load_project(root)
    data["classification"]["datasets"].append(
        {"id": "dataset-1", "file": "classification/datasets/dataset-1.csv", "columns": ["age", "color", "target"]}
    )
    (root / "project.json").write_text(__import__("json").dumps(data), encoding="utf-8")
    request = ClassificationTrainRequest(
        project_path=str(root),
        dataset_id="dataset-1",
        target_column="target",
        feature_columns=["age", "color"],
        test_size=0.34,
    )

    job = start_classification_training_job(request, run_in_background=False)

    assert job["status"] == "completed"
    assert job["metrics"]["accuracy"] >= 0
    assert (root / "classification_runs" / job["id"] / "model.joblib").exists()
    assert (root / "classification_runs" / job["id"] / "metrics.json").exists()
    predictions = (root / "classification_runs" / job["id"] / "predictions.csv").read_text(encoding="utf-8").splitlines()[0]
    assert predictions == "actual,prediction,prob_A,prob_B"
    assert load_project(root)["classification"]["last_run_id"] == job["id"]


@pytest.mark.anyio
async def test_classification_prediction_rejects_missing_columns(tmp_path):
    root = tmp_path / "classification"
    create_project(str(root), "CSV Model", [], SplitConfig(), "csv_classification")
    dataset = root / "classification" / "datasets" / "dataset-1.csv"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_text(
        "age,color,target\n10,red,A\n12,red,A\n40,blue,B\n42,blue,B\n11,red,A\n41,blue,B\n",
        encoding="utf-8",
    )
    data = load_project(root)
    data["classification"]["datasets"].append(
        {"id": "dataset-1", "file": "classification/datasets/dataset-1.csv", "columns": ["age", "color", "target"]}
    )
    (root / "project.json").write_text(__import__("json").dumps(data), encoding="utf-8")
    job = start_classification_training_job(
        ClassificationTrainRequest(
            project_path=str(root),
            dataset_id="dataset-1",
            target_column="target",
            feature_columns=["age", "color"],
            test_size=0.34,
        ),
        run_in_background=False,
    )

    with pytest.raises(HTTPException) as exc:
        await api_predict_classification(
            project_path=str(root),
            run_id=job["id"],
            file=UploadFile(BytesIO(b"age\n10\n"), filename="predict.csv"),
        )

    assert exc.value.status_code == 400
    assert "missing required columns" in exc.value.detail


@pytest.mark.anyio
async def test_classification_prediction_returns_rows_and_download(tmp_path):
    root = tmp_path / "classification"
    create_project(str(root), "CSV Model", [], SplitConfig(), "csv_classification")
    dataset = root / "classification" / "datasets" / "dataset-1.csv"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_text(
        "age,color,target\n10,red,A\n12,red,A\n40,blue,B\n42,blue,B\n11,red,A\n41,blue,B\n",
        encoding="utf-8",
    )
    data = load_project(root)
    data["classification"]["datasets"].append(
        {"id": "dataset-1", "file": "classification/datasets/dataset-1.csv", "columns": ["age", "color", "target"]}
    )
    (root / "project.json").write_text(__import__("json").dumps(data), encoding="utf-8")
    job = start_classification_training_job(
        ClassificationTrainRequest(
            project_path=str(root),
            dataset_id="dataset-1",
            target_column="target",
            feature_columns=["age", "color"],
            test_size=0.34,
        ),
        run_in_background=False,
    )

    response = await api_predict_classification(
        project_path=str(root),
        run_id=job["id"],
        file=UploadFile(BytesIO(b"age,color\n10,red\n40,blue\n"), filename="predict.csv"),
    )

    assert response["run_id"] == job["id"]
    assert response["columns"] == ["age", "color", "prediction", "prob_A", "prob_B"]
    assert len(response["rows"]) == 2
    assert 0 <= response["rows"][0]["prob_A"] <= 1
    assert 0 <= response["rows"][0]["prob_B"] <= 1
    assert response["download_url"].startswith("/api/classification/predictions/")
