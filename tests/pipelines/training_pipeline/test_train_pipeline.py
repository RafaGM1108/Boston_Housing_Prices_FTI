"""Tests para el training pipeline."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipelines.training_pipeline.train_pipeline import (
    build_models,
    build_preprocessor,
    evaluate_model,
    load_features,
    save_metrics,
    save_model,
    split_data,
    train_and_evaluate,
    validate_train_test_split,
)

EXPECTED_FEATURES = 13
TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_TRAIN_RATIO = 0.75
MAX_TRAIN_RATIO = 0.85


def _make_features_df(n: int = 50) -> pd.DataFrame:
    """Crea un DataFrame de features sintéticos."""
    rng = np.random.default_rng(RANDOM_STATE)
    return pd.DataFrame(
        {
            "crim": rng.uniform(0, 10, n),
            "zn": rng.uniform(0, 100, n),
            "indus": rng.uniform(0, 28, n),
            "chas": rng.choice([0.0, 1.0], n),
            "nox": rng.uniform(0.3, 0.9, n),
            "rm": rng.uniform(3, 9, n),
            "age": rng.uniform(0, 100, n),
            "dis": rng.uniform(1, 12, n),
            "rad": rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 24], n).astype(float),
            "tax": rng.uniform(180, 720, n),
            "ptratio": rng.uniform(12, 22, n),
            "black": rng.uniform(0, 400, n),
            "lstat": rng.uniform(1, 38, n),
            "medv": rng.uniform(5, 50, n),
        }
    )


def test_split_data_sizes() -> None:
    df = _make_features_df()
    x_train, x_test, y_train, y_test = split_data(df, "medv", TEST_SIZE, RANDOM_STATE)
    assert len(x_train) + len(x_test) == len(df)
    assert len(y_train) == len(x_train)
    assert len(y_test) == len(x_test)


def test_split_data_no_target_in_features() -> None:
    df = _make_features_df()
    x_train, x_test, _, _ = split_data(df, "medv", TEST_SIZE, RANDOM_STATE)
    assert "medv" not in x_train.columns
    assert "medv" not in x_test.columns


def test_build_preprocessor_transforms() -> None:
    df = _make_features_df()
    x_train, _, _, _ = split_data(df, "medv", TEST_SIZE, RANDOM_STATE)
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(x_train)
    assert transformed.shape[0] == len(x_train)
    assert transformed.shape[1] == EXPECTED_FEATURES


def test_build_models_returns_pipelines() -> None:
    models = build_models()
    assert len(models) > 0
    for _name, pipeline in models.items():
        assert hasattr(pipeline, "fit")
        assert hasattr(pipeline, "predict")


def test_evaluate_model_returns_metrics() -> None:
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.8, 4.1, 5.0])
    metrics = evaluate_model(y_true, y_pred)
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert metrics["r2"] > 0


def test_train_and_evaluate() -> None:
    df = _make_features_df(100)
    x_train, x_test, y_train, y_test = split_data(df, "medv", TEST_SIZE, RANDOM_STATE)
    models = build_models()
    pipeline = next(iter(models.values()))
    trained, train_m, test_m = train_and_evaluate(pipeline, x_train, x_test, y_train, y_test)
    assert hasattr(trained, "predict")
    assert "rmse" in train_m
    assert "rmse" in test_m


def test_save_model_creates_file(tmp_path: object) -> None:
    df = _make_features_df(100)
    x_train, _, y_train, _ = split_data(df, "medv", TEST_SIZE, RANDOM_STATE)
    models = build_models()
    pipeline = next(iter(models.values()))
    pipeline.fit(x_train, y_train)
    out_path = str(tmp_path) + "/model.joblib"  # type: ignore[operator]
    save_model(pipeline, out_path)
    assert Path(out_path).exists()


EXPECTED_RMSE = 3.5


def test_save_metrics_creates_file(tmp_path: object) -> None:
    metrics = {"rmse": EXPECTED_RMSE, "mae": 2.1, "r2": 0.85}
    out_path = str(tmp_path) + "/metrics.json"  # type: ignore[operator]
    save_metrics(metrics, out_path)
    assert Path(out_path).exists()
    with open(out_path) as f:
        loaded = json.load(f)
    assert loaded["rmse"] == EXPECTED_RMSE


def test_load_features(tmp_path: object) -> None:
    df = _make_features_df()
    csv_path = str(tmp_path) + "/features.csv"  # type: ignore[operator]
    df.to_csv(csv_path, index=False)
    loaded = load_features(csv_path)
    assert len(loaded) == len(df)


# --- Tests de validación train/test split (issue 4) ---


def test_validate_split_passes_valid_split() -> None:
    df = _make_features_df(200)
    x_train, x_test, y_train, y_test = split_data(df, "medv", TEST_SIZE, RANDOM_STATE)
    result = validate_train_test_split(x_train, x_test, y_train, y_test)
    assert result["passed"] is True
    assert result["checks"]["no_index_overlap"]["passed"] is True


def test_validate_split_detects_index_overlap() -> None:
    df = _make_features_df(100)
    x_train, x_test, y_train, y_test = split_data(df, "medv", TEST_SIZE, RANDOM_STATE)
    # Forzar overlap copiando filas de train a test
    x_test_leaked = pd.concat([x_test, x_train.iloc[:5]])
    y_test_leaked = pd.concat([y_test, y_train.iloc[:5]])
    with pytest.raises(ValueError, match="FAILED"):
        validate_train_test_split(x_train, x_test_leaked, y_train, y_test_leaked)


def test_validate_split_reports_size_ratio() -> None:
    df = _make_features_df(200)
    x_train, x_test, y_train, y_test = split_data(df, "medv", TEST_SIZE, RANDOM_STATE)
    result = validate_train_test_split(x_train, x_test, y_train, y_test)
    ratio = result["checks"]["size_ratio"]["train_ratio"]
    assert MIN_TRAIN_RATIO < ratio < MAX_TRAIN_RATIO


def test_validate_split_checks_target_distribution() -> None:
    df = _make_features_df(200)
    x_train, x_test, y_train, y_test = split_data(df, "medv", TEST_SIZE, RANDOM_STATE)
    result = validate_train_test_split(x_train, x_test, y_train, y_test)
    assert "target_distribution" in result["checks"]
    assert "p_value" in result["checks"]["target_distribution"]


def test_validate_split_checks_feature_distributions() -> None:
    df = _make_features_df(200)
    x_train, x_test, y_train, y_test = split_data(df, "medv", TEST_SIZE, RANDOM_STATE)
    result = validate_train_test_split(x_train, x_test, y_train, y_test)
    assert "feature_distributions" in result["checks"]
    assert len(result["checks"]["feature_distributions"]) > 0
