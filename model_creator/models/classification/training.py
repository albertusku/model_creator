from __future__ import annotations

import json
import shutil
import threading
import traceback
import uuid
from csv import Error as CsvError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ...core.schemas import ClassificationClusterRequest, ClassificationTrainRequest
from ...core.storage import load_project, project_root, save_project

ClassificationRunner = Callable[[ClassificationTrainRequest, Path, Path], dict[str, Any]]

_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": job["id"],
        "status": job["status"],
        "logs": list(job.get("logs", [])),
        "run_path": str(job["run_path"]) if job.get("run_path") else None,
        "model_path": str(job["model_path"]) if job.get("model_path") else None,
        "metrics_path": str(job["metrics_path"]) if job.get("metrics_path") else None,
        "metadata_path": str(job["metadata_path"]) if job.get("metadata_path") else None,
        "metrics": job.get("metrics") or {},
        "error": job.get("error"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
    }


def get_classification_job(job_id: str) -> dict[str, Any] | None:
    with _lock:
        job = _jobs.get(job_id)
        return _public_job(job) if job else None


def import_classification_dataset(project_path: str, filename: str, source_path: Path) -> dict[str, Any]:
    root = _require_classification_project(project_path)
    frame = _read_csv(source_path)
    if frame.empty or not list(frame.columns):
        raise ValueError("CSV must contain at least one row and one column")

    dataset_id = uuid.uuid4().hex
    stored_name = f"{dataset_id}.csv"
    target = root / "classification" / "datasets" / stored_name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target)

    summary = summarize_frame(frame)
    data = load_project(root)
    classification = data.setdefault("classification", {"datasets": [], "runs": [], "last_run_id": None})
    classification.setdefault("datasets", []).append(
        {
            "id": dataset_id,
            "source_name": filename,
            "file": f"classification/datasets/{stored_name}",
            "rows": int(len(frame)),
            "columns": list(map(str, frame.columns)),
            "imported_at": _now(),
            "summary": summary,
        }
    )
    save_project(root, data)
    return {"dataset_id": dataset_id, "columns": list(map(str, frame.columns)), "rows": int(len(frame)), "summary": summary}


def summarize_frame(frame: pd.DataFrame) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for name in frame.columns:
        series = frame[name]
        non_null = int(series.notna().sum())
        item: dict[str, Any] = {
            "dtype": str(series.dtype),
            "non_null": non_null,
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            described = series.describe()
            item.update(
                {
                    "min": _json_value(described.get("min")),
                    "max": _json_value(described.get("max")),
                    "mean": _json_value(described.get("mean")),
                }
            )
        else:
            values = [str(value) for value in series.dropna().astype(str).unique()[:10]]
            item["sample_values"] = values
        columns[str(name)] = item
    return {"columns": columns}


def start_classification_training_job(
    request: ClassificationTrainRequest,
    *,
    runner: ClassificationRunner | None = None,
    run_in_background: bool = True,
) -> dict[str, Any]:
    root = _require_classification_project(request.project_path)
    dataset_path = _dataset_path(root, request.dataset_id)
    _validate_training_request(dataset_path, request)
    job_id = uuid.uuid4().hex
    run_path = root / "classification_runs" / job_id

    job = {
        "id": job_id,
        "status": "queued",
        "logs": ["Classification training queued"],
        "run_path": run_path,
        "model_path": None,
        "metrics_path": None,
        "metadata_path": None,
        "metrics": {},
        "error": None,
        "created_at": _now(),
        "started_at": None,
        "finished_at": None,
        "project_path": str(root),
        "request": request.dict(),
    }
    with _lock:
        _jobs[job_id] = job

    target = _run_classification_job
    args = (job_id, request, dataset_path, run_path, runner)
    if run_in_background:
        thread = threading.Thread(target=target, args=args, daemon=True)
        thread.start()
    else:
        target(*args)
    return get_classification_job(job_id) or _public_job(job)


def predict_classification_csv(project_path: str, source_path: Path, run_id: str | None = None) -> dict[str, Any]:
    root = _require_classification_project(project_path)
    run_path = _resolve_run_path(root, run_id)
    model_path = run_path / "model.joblib"
    metadata_path = run_path / "metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise ValueError("classification model artifacts were not found")

    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_columns = list(metadata.get("feature_columns") or [])
    frame = _read_csv(source_path)
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"prediction CSV is missing required columns: {', '.join(missing)}")

    predictions = model.predict(frame[feature_columns])
    output = frame.copy()
    output["prediction"] = predictions
    _append_prediction_probabilities(output, model, frame[feature_columns])
    prediction_id = uuid.uuid4().hex
    output_path = root / "classification_predictions" / f"{prediction_id}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    rows = output.head(25).to_dict(orient="records")
    return {
        "prediction_id": prediction_id,
        "run_id": run_path.name,
        "rows": rows,
        "columns": list(map(str, output.columns)),
        "download_url": f"/api/classification/predictions/{prediction_id}/download",
    }


def classification_cluster_plot(request: ClassificationClusterRequest) -> dict[str, Any]:
    root = _require_classification_project(request.project_path)
    dataset_path = _dataset_path(root, request.dataset_id)
    frame = _read_csv(dataset_path)
    _validate_cluster_frame(frame, request)
    x = frame[request.feature_columns]
    y = frame[request.target_column].astype(str).fillna("(missing)")
    preprocessor = _build_preprocessor(x, request.feature_columns)
    transformed = preprocessor.fit_transform(x)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    if transformed.shape[1] == 1:
        xs = transformed[:, 0]
        ys = [0.0] * len(xs)
        explained_variance = [1.0, 0.0]
    else:
        pca = PCA(n_components=2, random_state=42)
        projected = pca.fit_transform(transformed)
        xs = projected[:, 0]
        ys = projected[:, 1]
        explained_variance = [float(value) for value in pca.explained_variance_ratio_]
    labels = sorted(y.unique().tolist())
    points = [
        {"x": float(x_value), "y": float(y_value), "label": label}
        for x_value, y_value, label in zip(xs, ys, y.tolist(), strict=False)
    ]
    return {
        "method": "pca",
        "points": points,
        "labels": labels,
        "explained_variance": explained_variance,
        "rows": len(points),
        "feature_columns": request.feature_columns,
        "target_column": request.target_column,
    }


def get_prediction_path(project_path: str, prediction_id: str) -> Path | None:
    if prediction_id != Path(prediction_id).name:
        return None
    root = project_root(project_path)
    path = root / "classification_predictions" / f"{prediction_id}.csv"
    return path if path.exists() and path.is_file() else None


def _run_classification_job(
    job_id: str,
    request: ClassificationTrainRequest,
    dataset_path: Path,
    run_path: Path,
    runner: ClassificationRunner | None,
) -> None:
    with _lock:
        job = _jobs[job_id]
        job["status"] = "running"
        job["started_at"] = _now()
    try:
        result = (runner or _sklearn_runner)(request, dataset_path, run_path)
        metrics = result.get("metrics") or {}
        model_path = Path(result.get("model_path") or run_path / "model.joblib")
        metrics_path = Path(result.get("metrics_path") or run_path / "metrics.json")
        metadata_path = Path(result.get("metadata_path") or run_path / "metadata.json")
        _record_classification_run(request.project_path, job_id, run_path, model_path, metrics_path, metadata_path, metrics, request)
        with _lock:
            job = _jobs[job_id]
            job["status"] = "completed"
            job["finished_at"] = _now()
            job["model_path"] = model_path
            job["metrics_path"] = metrics_path
            job["metadata_path"] = metadata_path
            job["metrics"] = metrics
            job["logs"].extend(str(line) for line in result.get("logs", []))
            job["logs"].append("Classification training completed")
    except Exception as exc:
        with _lock:
            job = _jobs[job_id]
            job["status"] = "failed"
            job["finished_at"] = _now()
            job["error"] = str(exc)
            job["logs"].append(str(exc))
            job["logs"].append(traceback.format_exc())


def _sklearn_runner(request: ClassificationTrainRequest, dataset_path: Path, run_path: Path) -> dict[str, Any]:
    frame = _read_csv(dataset_path)
    _validate_training_frame(frame, request)
    x = frame[request.feature_columns]
    y = frame[request.target_column]
    stratify = y if y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=request.test_size,
        random_state=42,
        stratify=stratify,
    )
    transformers = _feature_transformers(x, request.feature_columns)
    model = Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers=transformers)),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")),
        ]
    )
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    labels = sorted(pd.Series(y).dropna().unique().tolist(), key=lambda value: str(value))
    matrix = confusion_matrix(y_test, predicted, labels=labels)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "macro_f1": float(f1_score(y_test, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, predicted, average="weighted", zero_division=0)),
        "class_distribution": {str(key): int(value) for key, value in y.value_counts(dropna=False).items()},
        "labels": [str(label) for label in labels],
        "confusion_matrix": matrix.astype(int).tolist(),
        "test_rows": int(len(y_test)),
        "train_rows": int(len(y_train)),
    }
    run_path.mkdir(parents=True, exist_ok=True)
    model_path = run_path / "model.joblib"
    metrics_path = run_path / "metrics.json"
    metadata_path = run_path / "metadata.json"
    predictions_path = run_path / "predictions.csv"
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    metadata = {
        "dataset_id": request.dataset_id,
        "target_column": request.target_column,
        "feature_columns": request.feature_columns,
        "test_size": request.test_size,
        "created_at": _now(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    test_predictions = pd.DataFrame({"actual": y_test.reset_index(drop=True), "prediction": predicted})
    _append_prediction_probabilities(test_predictions, model, x_test)
    test_predictions.to_csv(predictions_path, index=False)
    return {
        "model_path": model_path,
        "metrics_path": metrics_path,
        "metadata_path": metadata_path,
        "metrics": metrics,
        "logs": [f"Trained on {len(y_train)} rows and evaluated on {len(y_test)} rows"],
    }


def _append_prediction_probabilities(output: pd.DataFrame, model: Any, features: pd.DataFrame) -> None:
    if not hasattr(model, "predict_proba"):
        return
    probabilities = model.predict_proba(features)
    classes = getattr(model, "classes_", [])
    for index, class_name in enumerate(classes):
        output[f"prob_{class_name}"] = probabilities[:, index]


def _record_classification_run(
    project_path: str,
    run_id: str,
    run_path: Path,
    model_path: Path,
    metrics_path: Path,
    metadata_path: Path,
    metrics: dict[str, Any],
    request: ClassificationTrainRequest,
) -> None:
    data = load_project(project_path)
    classification = data.setdefault("classification", {"datasets": [], "runs": [], "last_run_id": None})
    runs = [run for run in classification.get("runs", []) if run.get("id") != run_id]
    runs.append(
        {
            "id": run_id,
            "run_path": str(run_path),
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
            "metadata_path": str(metadata_path),
            "dataset_id": request.dataset_id,
            "target_column": request.target_column,
            "feature_columns": request.feature_columns,
            "test_size": request.test_size,
            "trained_at": _now(),
            "metrics": metrics,
        }
    )
    classification["runs"] = runs
    classification["last_run_id"] = run_id
    save_project(project_path, data)


def _require_classification_project(project_path: str) -> Path:
    if not str(project_path).strip():
        raise ValueError("classification requires opening the project by backend path")
    root = project_root(project_path)
    data = load_project(root)
    if data.get("project_type") != "csv_classification":
        raise ValueError("classification endpoints require a csv_classification project")
    return root


def _dataset_path(root: Path, dataset_id: str) -> Path:
    data = load_project(root)
    datasets = data.get("classification", {}).get("datasets", [])
    item = next((dataset for dataset in datasets if dataset.get("id") == dataset_id), None)
    if not item:
        raise ValueError("unknown classification dataset")
    path = root / item["file"]
    if not path.exists():
        raise ValueError("classification dataset file not found")
    return path


def _resolve_run_path(root: Path, run_id: str | None) -> Path:
    data = load_project(root)
    classification = data.get("classification", {})
    selected_id = run_id or classification.get("last_run_id")
    if not selected_id:
        raise ValueError("train a classification model before predicting")
    run = next((item for item in classification.get("runs", []) if item.get("id") == selected_id), None)
    if not run:
        raise ValueError("unknown classification run")
    return Path(run["run_path"])


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except (pd.errors.ParserError, CsvError) as exc:
        if "Could not determine delimiter" in str(exc):
            raise ValueError("CSV is empty") from exc
        raise ValueError(str(exc)) from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError("CSV is empty") from exc


def _build_preprocessor(frame: pd.DataFrame, feature_columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer(transformers=_feature_transformers(frame, feature_columns))


def _feature_transformers(frame: pd.DataFrame, feature_columns: list[str]) -> list[tuple[str, Pipeline, list[str]]]:
    numeric_features = [column for column in feature_columns if pd.api.types.is_numeric_dtype(frame[column])]
    categorical_features = [column for column in feature_columns if column not in numeric_features]
    transformers = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
                categorical_features,
            )
        )
    return transformers


def _validate_training_request(dataset_path: Path, request: ClassificationTrainRequest) -> None:
    frame = _read_csv(dataset_path)
    _validate_training_frame(frame, request)


def _validate_training_frame(frame: pd.DataFrame, request: ClassificationTrainRequest) -> None:
    if frame.empty:
        raise ValueError("CSV must contain at least one row")
    if request.target_column not in frame.columns:
        raise ValueError("target_column does not exist in the dataset")
    if not request.feature_columns:
        raise ValueError("feature_columns must contain at least one column")
    missing = [column for column in request.feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"feature columns do not exist in the dataset: {', '.join(missing)}")
    if request.target_column in request.feature_columns:
        raise ValueError("target_column cannot also be a feature column")
    if frame[request.target_column].nunique(dropna=True) < 2:
        raise ValueError("target_column must contain at least two classes")


def _validate_cluster_frame(frame: pd.DataFrame, request: ClassificationClusterRequest) -> None:
    if frame.empty:
        raise ValueError("CSV must contain at least one row")
    if request.target_column not in frame.columns:
        raise ValueError("target_column does not exist in the dataset")
    if not request.feature_columns:
        raise ValueError("feature_columns must contain at least one column")
    missing = [column for column in request.feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"feature columns do not exist in the dataset: {', '.join(missing)}")
    if request.target_column in request.feature_columns:
        raise ValueError("target_column cannot also be a feature column")


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value
