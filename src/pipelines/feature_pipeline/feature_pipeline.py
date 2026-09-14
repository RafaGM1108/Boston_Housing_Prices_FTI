"""Feature pipeline para el proyecto Boston Home Prices."""

from pathlib import Path

import pandas as pd
import pandera.pandas as pa
import yaml
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MAX_NULL_FRACTION = 0.05


def load_config() -> dict[str, str]:
    """Carga la configuración desde conf/config.yml."""
    config_path = PROJECT_ROOT / "conf" / "config.yml"
    with open(config_path) as f:
        config: dict[str, str] = yaml.safe_load(f)
    return config


def load_raw_data(path: str) -> pd.DataFrame:
    """Lee los datos crudos desde CSV."""
    full_path = Path(path) if Path(path).is_absolute() else PROJECT_ROOT / path
    logger.info(f"Leyendo datos crudos desde {full_path}")
    df: pd.DataFrame = pd.read_csv(full_path)
    logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def transform_features(df: pd.DataFrame, id_column: str, target_column: str) -> pd.DataFrame:
    """Transforma los datos crudos en features para el modelo."""
    logger.info("Iniciando transformación de features")

    df = df.drop(columns=[id_column])
    logger.info(f"Columna '{id_column}' eliminada")

    scale_corrections = {"crim": 100, "nox": 1, "rm": 100, "dis": 100}
    for col, threshold in scale_corrections.items():
        mask = df[col] > threshold
        affected = mask.sum()
        if affected > 0:
            df.loc[mask, col] = df.loc[mask, col] / 1000
            logger.info(f"Escala corregida en '{col}': {affected} valores divididos por 1000")

    rows_before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Duplicados eliminados: {rows_before - len(df)}")

    rows_before = len(df)
    df = df.dropna(subset=[target_column])
    logger.info(f"Filas sin target eliminadas: {rows_before - len(df)}")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        if col != target_column:
            median_val = df[col].median()
            nulls = df[col].isna().sum()
            if nulls > 0:
                df[col] = df[col].fillna(median_val)
                logger.info(f"Columna '{col}': {nulls} nulos imputados con mediana ({median_val})")

    df = df.reset_index(drop=True)
    logger.info(f"Features finales: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def build_feature_schema() -> pa.DataFrameSchema:
    """Define el esquema de validación para los features procesados."""
    return pa.DataFrameSchema(
        columns={
            "crim": pa.Column(float, pa.Check.in_range(0, 90), nullable=False),
            "zn": pa.Column(float, pa.Check.in_range(0, 100), nullable=False),
            "indus": pa.Column(float, pa.Check.in_range(0, 30), nullable=False),
            "chas": pa.Column(float, pa.Check.isin([0.0, 1.0]), nullable=False),
            "nox": pa.Column(float, pa.Check.in_range(0.3, 0.9), nullable=False),
            "rm": pa.Column(float, pa.Check.in_range(3, 9), nullable=False),
            "age": pa.Column(float, pa.Check.in_range(0, 100), nullable=False),
            "dis": pa.Column(float, pa.Check.in_range(1, 13), nullable=False),
            "rad": pa.Column(float, pa.Check.ge(1), nullable=False),
            "tax": pa.Column(float, pa.Check.gt(0), nullable=False),
            "ptratio": pa.Column(float, pa.Check.in_range(10, 25), nullable=False),
            "black": pa.Column(float, pa.Check.ge(0), nullable=False),
            "lstat": pa.Column(float, pa.Check.in_range(0, 40), nullable=False),
            "medv": pa.Column(float, pa.Check.gt(0), nullable=False),
        },
        checks=[
            pa.Check(
                lambda df: len(df) == len(df.drop_duplicates()),
                error="El dataset contiene filas duplicadas",
            ),
        ],
    )


def validate_raw_data(df: pd.DataFrame, id_column: str, target_column: str) -> None:
    """Valida los datos crudos antes de la transformación."""
    logger.info("Validando datos crudos")

    expected_columns = [
        id_column,
        "crim",
        "zn",
        "indus",
        "chas",
        "nox",
        "rm",
        "age",
        "dis",
        "rad",
        "tax",
        "ptratio",
        "black",
        "lstat",
        target_column,
    ]
    missing = set(expected_columns) - set(df.columns)
    if missing:
        msg = f"Columnas faltantes en datos crudos: {missing}"
        raise ValueError(msg)

    if df.empty:
        msg = "El dataset crudo está vacío"
        raise ValueError(msg)

    n_rows = len(df)
    for col in df.columns:
        null_fraction = df[col].isna().sum() / n_rows
        if null_fraction > MAX_NULL_FRACTION:
            logger.warning(
                f"Columna '{col}' tiene {null_fraction:.1%} nulos (máximo: {MAX_NULL_FRACTION:.0%})"
            )

    logger.info("Validación de datos crudos completada")


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Valida los features procesados con pandera."""
    logger.info("Validando features procesados")
    schema = build_feature_schema()
    validated: pd.DataFrame = schema.validate(df)
    logger.info("Validación de features exitosa")
    return validated


def save_features(df: pd.DataFrame, path: str) -> None:
    """Almacena los features procesados en un archivo CSV."""
    full_path = Path(path) if Path(path).is_absolute() else PROJECT_ROOT / path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(full_path, index=False)
    logger.info(f"Features guardados en {full_path}")


def run() -> None:
    """Ejecuta el feature pipeline completo."""
    config = load_config()

    df = load_raw_data(config["raw_data_path"])
    validate_raw_data(df, config["id_column"], config["target_column"])
    df = transform_features(df, config["id_column"], config["target_column"])
    validate_features(df)
    save_features(df, config["feature_data_path"])

    logger.info("Feature pipeline completado exitosamente")


if __name__ == "__main__":
    run()
