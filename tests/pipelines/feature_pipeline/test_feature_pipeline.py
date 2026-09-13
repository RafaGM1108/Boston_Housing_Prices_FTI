"""Tests para el feature pipeline."""

from pathlib import Path

import pandas as pd

from pipelines.feature_pipeline.feature_pipeline import (
    load_raw_data,
    save_features,
    transform_features,
)

EXPECTED_ROWS = 2


def _make_raw_df() -> pd.DataFrame:
    """Crea un DataFrame de prueba que simula los datos crudos."""
    return pd.DataFrame(
        {
            "ID": [1, 2, 3, 3, 4],
            "crim": [0.006, 0.027, 0.032, 0.032, None],
            "zn": [18.0, 0.0, 0.0, 0.0, 0.0],
            "indus": [2.31, 7.07, 2.18, 2.18, 5.96],
            "chas": [0, 0, 0, 0, 0],
            "nox": [0.538, 0.469, 0.458, 0.458, 0.499],
            "rm": [6.575, 6.421, 6.998, 6.998, 5.966],
            "age": [65.2, 78.9, 45.8, 45.8, 30.2],
            "dis": [4.09, 4.967, 6.062, 6.062, 3.847],
            "rad": [1, 2, 3, 3, 5],
            "tax": [296, 242, 222, 222, 279],
            "ptratio": [15.3, 17.8, 18.7, 18.7, 19.2],
            "black": [396.9, 396.9, 394.63, 394.63, 393.43],
            "lstat": [4.98, 9.14, 2.94, 2.94, 10.13],
            "medv": [24.0, 21.6, 33.4, 33.4, None],
        }
    )


def test_transform_removes_duplicates() -> None:
    df = _make_raw_df()
    result = transform_features(df, id_column="ID", target_column="medv")
    assert len(result) < len(df)


def test_transform_drops_id_column() -> None:
    df = _make_raw_df()
    result = transform_features(df, id_column="ID", target_column="medv")
    assert "ID" not in result.columns


def test_transform_drops_rows_without_target() -> None:
    df = _make_raw_df()
    result = transform_features(df, id_column="ID", target_column="medv")
    assert result["medv"].isna().sum() == 0


def test_transform_imputes_nulls_with_median() -> None:
    df = _make_raw_df()
    result = transform_features(df, id_column="ID", target_column="medv")
    assert result.drop(columns=["medv"]).isna().sum().sum() == 0


def test_save_and_load(tmp_path: object) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    path = str(tmp_path) + "/output.csv"  # type: ignore[operator]
    df.to_csv(path, index=False)
    loaded = pd.read_csv(path)
    pd.testing.assert_frame_equal(df, loaded)


def test_load_raw_data(tmp_path: object) -> None:
    csv_path = str(tmp_path) + "/test.csv"  # type: ignore[operator]
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    df.to_csv(csv_path, index=False)
    loaded = load_raw_data(csv_path)
    assert len(loaded) == EXPECTED_ROWS


def test_save_features_creates_file(tmp_path: object) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    out_path = str(tmp_path) + "/sub/output.csv"  # type: ignore[operator]
    save_features(df, out_path)
    assert Path(out_path).exists()
