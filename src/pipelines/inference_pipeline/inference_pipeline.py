"""Inference pipeline para el proyecto Boston Home Prices."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from loguru import logger
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_config() -> dict[str, Any]:
    """Carga la configuración desde conf/config.yml."""
    config_path = PROJECT_ROOT / "conf" / "config.yml"
    with open(config_path) as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config


def load_model(path: str) -> Pipeline:
    """Carga el modelo entrenado desde disco."""
    full_path = Path(path) if Path(path).is_absolute() else PROJECT_ROOT / path
    logger.info(f"Cargando modelo desde {full_path}")
    model: Pipeline = joblib.load(full_path)
    logger.info("Modelo cargado exitosamente")
    return model


def load_new_data(path: str, id_column: str, target_column: str) -> pd.DataFrame:
    """Lee datos nuevos y los prepara para predicción."""
    full_path = Path(path) if Path(path).is_absolute() else PROJECT_ROOT / path
    logger.info(f"Leyendo datos desde {full_path}")
    df: pd.DataFrame = pd.read_csv(full_path)
    logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")

    if id_column in df.columns:
        df = df.drop(columns=[id_column])
    if target_column in df.columns:
        df = df.drop(columns=[target_column])

    df = df.drop_duplicates()
    df = df.dropna(how="all")

    df["chas"] = df["chas"].astype(str)
    df["rad"] = df["rad"].astype(float)

    df = df.reset_index(drop=True)
    logger.info(f"Datos preparados: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def predict(model: Pipeline, data: pd.DataFrame) -> pd.DataFrame:
    """Genera predicciones con el modelo cargado."""
    logger.info(f"Generando predicciones para {len(data)} registros")
    predictions = model.predict(data)
    result = data.copy()
    result["prediction"] = predictions
    logger.info(
        f"Predicciones generadas — "
        f"min: {predictions.min():.2f}, max: {predictions.max():.2f}, "
        f"mean: {predictions.mean():.2f}"
    )
    return result


def save_predictions(df: pd.DataFrame, path: str) -> None:
    """Guarda las predicciones en un archivo CSV."""
    full_path = Path(path) if Path(path).is_absolute() else PROJECT_ROOT / path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(full_path, index=False)
    logger.info(f"Predicciones guardadas en {full_path}")


def run() -> None:
    """Ejecuta el inference pipeline completo."""
    config = load_config()

    model = load_model(config["model_path"])
    data = load_new_data(
        config["inference_data_path"],
        config["id_column"],
        config["target_column"],
    )
    results = predict(model, data)
    save_predictions(results, config["predictions_path"])

    logger.info("Inference pipeline completado exitosamente")


if __name__ == "__main__":
    run()
