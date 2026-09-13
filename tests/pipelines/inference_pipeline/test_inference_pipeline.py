"""Tests para el inference pipeline."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pipelines.inference_pipeline.inference_pipeline import (
    load_model,
    load_new_data,
    predict,
    save_predictions,
)
from pipelines.training_pipeline.train_pipeline import (
    build_models,
    split_data,
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
EXPECTED_FEATURE_COLS = 13


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


def _train_dummy_model(tmp_path: Path) -> Path:
    """Entrena y guarda un modelo dummy, retorna el path."""
    df = _make_features_df(100)
    x_train, _, y_train, _ = split_data(df, "medv", TEST_SIZE, RANDOM_STATE)
    models = build_models()
    pipeline = next(iter(models.values()))
    pipeline.fit(x_train, y_train)
    model_path = tmp_path / "model.joblib"
    joblib.dump(pipeline, model_path)
    return model_path


def test_load_model(tmp_path: Path) -> None:
    model_path = _train_dummy_model(tmp_path)
    model = load_model(str(model_path))
    assert hasattr(model, "predict")


def test_load_new_data(tmp_path: Path) -> None:
    df = _make_features_df()
    df.insert(0, "ID", range(len(df)))
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    loaded = load_new_data(str(csv_path), "ID", "medv")
    assert "ID" not in loaded.columns
    assert "medv" not in loaded.columns
    assert len(loaded.columns) == EXPECTED_FEATURE_COLS


def test_predict_generates_predictions(tmp_path: Path) -> None:
    model_path = _train_dummy_model(tmp_path)
    model = load_model(str(model_path))
    df = _make_features_df()
    x_train, _, _, _ = split_data(df, "medv", TEST_SIZE, RANDOM_STATE)
    result = predict(model, x_train)
    assert "prediction" in result.columns
    assert len(result) == len(x_train)


def test_save_predictions_creates_file(tmp_path: Path) -> None:
    df = pd.DataFrame({"col1": [1.0, 2.0], "prediction": [10.0, 20.0]})
    out_path = str(tmp_path / "predictions.csv")
    save_predictions(df, out_path)
    assert Path(out_path).exists()
    loaded = pd.read_csv(out_path)
    assert len(loaded) == len(df)


def test_end_to_end_inference(tmp_path: Path) -> None:
    model_path = _train_dummy_model(tmp_path)
    model = load_model(str(model_path))
    df = _make_features_df()
    df.insert(0, "ID", range(len(df)))
    csv_path = tmp_path / "new_data.csv"
    df.to_csv(csv_path, index=False)
    data = load_new_data(str(csv_path), "ID", "medv")
    result = predict(model, data)
    out_path = str(tmp_path / "preds.csv")
    save_predictions(result, out_path)
    saved = pd.read_csv(out_path)
    assert "prediction" in saved.columns
    assert len(saved) == len(data)
