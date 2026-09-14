"""Training pipeline para el proyecto Boston Home Prices."""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]

LOG_COLS = ["crim", "zn", "dis", "lstat"]
NUM_COLS = ["indus", "nox", "rm", "age", "tax", "ptratio", "black"]
CAT_COLS = ["chas"]
ORD_COLS = ["rad"]

N_FOLDS = 10


def load_config() -> dict[str, Any]:
    """Carga la configuración desde conf/config.yml."""
    config_path = PROJECT_ROOT / "conf" / "config.yml"
    with open(config_path) as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config


def load_features(path: str) -> pd.DataFrame:
    """Lee los features procesados."""
    full_path = Path(path) if Path(path).is_absolute() else PROJECT_ROOT / path
    logger.info(f"Leyendo features desde {full_path}")
    df: pd.DataFrame = pd.read_csv(full_path)
    logger.info(f"Features cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def split_data(
    df: pd.DataFrame, target_column: str, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:  # type: ignore[type-arg]
    """Separa los datos en train y test."""
    x = df.drop(columns=[target_column])
    y = df[target_column]

    x["chas"] = x["chas"].astype(str)
    x["rad"] = x["rad"].astype(float)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"Train: {len(x_train)} filas, Test: {len(x_test)} filas")
    return x_train, x_test, y_train, y_test


KS_ALPHA = 0.05


def validate_train_test_split(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,  # type: ignore[type-arg]
    y_test: pd.Series,  # type: ignore[type-arg]
) -> dict[str, Any]:
    """Valida la separación train/test: no-overlap, distribución target y features."""
    results: dict[str, Any] = {"passed": True, "checks": {}}

    # 1. No hay filas compartidas entre train y test (data leakage)
    train_idx = set(x_train.index.tolist())
    test_idx = set(x_test.index.tolist())
    overlap = train_idx & test_idx
    no_leakage = len(overlap) == 0
    results["checks"]["no_index_overlap"] = {
        "passed": no_leakage,
        "overlap_count": len(overlap),
    }
    if not no_leakage:
        results["passed"] = False
        logger.error(f"Data leakage detectado: {len(overlap)} índices compartidos")

    # 2. Distribución del target (KS test)
    ks_stat, ks_pvalue = stats.ks_2samp(y_train.values, y_test.values)
    target_ok = ks_pvalue > KS_ALPHA
    results["checks"]["target_distribution"] = {
        "passed": target_ok,
        "ks_statistic": float(ks_stat),
        "p_value": float(ks_pvalue),
    }
    if not target_ok:
        results["passed"] = False
        logger.warning(f"Distribución del target difiere significativamente (KS p={ks_pvalue:.4f})")
    else:
        logger.info(f"Target: distribución similar (KS p={ks_pvalue:.4f})")

    # 3. Distribución de features numéricos (KS test por columna)
    numeric_cols = x_train.select_dtypes(include=["number"]).columns
    feature_checks: dict[str, dict[str, Any]] = {}
    for col in numeric_cols:
        f_stat, f_pvalue = stats.ks_2samp(
            x_train[col].dropna().values,
            x_test[col].dropna().values,
        )
        col_ok = f_pvalue > KS_ALPHA
        feature_checks[col] = {
            "passed": col_ok,
            "ks_statistic": float(f_stat),
            "p_value": float(f_pvalue),
        }
        if not col_ok:
            logger.warning(f"Feature '{col}': distribución difiere (KS p={f_pvalue:.4f})")
    results["checks"]["feature_distributions"] = feature_checks

    # 4. Proporciones de tamaño
    total = len(x_train) + len(x_test)
    train_ratio = len(x_train) / total
    results["checks"]["size_ratio"] = {
        "train_ratio": float(train_ratio),
        "train_size": len(x_train),
        "test_size": len(x_test),
    }
    logger.info(f"Split ratio: train={train_ratio:.2%}, test={1 - train_ratio:.2%}")

    if results["passed"]:
        logger.info("Validación train/test split: PASSED")
    else:
        msg = "Validación train/test split: FAILED"
        raise ValueError(msg)

    return results


def build_preprocessor() -> ColumnTransformer:
    """Construye el preprocesador sklearn."""
    log_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("log_transform", FunctionTransformer(np.log1p, validate=True)),
            ("scaler", StandardScaler()),
        ]
    )
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(drop="if_binary", sparse_output=False, handle_unknown="ignore"),
            ),
        ]
    )
    ord_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("log", log_pipeline, LOG_COLS),
            ("num", num_pipeline, NUM_COLS),
            ("cat", cat_pipeline, CAT_COLS),
            ("ord", ord_pipeline, ORD_COLS),
        ],
        remainder="drop",
    )


def build_models() -> dict[str, Pipeline]:
    """Construye los pipelines de modelos candidatos."""
    preprocessor = build_preprocessor()
    return {
        "Ridge": Pipeline([("preprocessor", preprocessor), ("model", Ridge(random_state=42))]),
        "RandomForest": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
            ]
        ),
        "GradientBoosting": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", GradientBoostingRegressor(random_state=42)),
            ]
        ),
    }


def evaluate_model(
    y_true: pd.Series,
    y_pred: np.ndarray,  # type: ignore[type-arg]
) -> dict[str, float]:
    """Calcula métricas de evaluación."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def select_best_model(
    models: dict[str, Pipeline],
    x_train: pd.DataFrame,
    y_train: pd.Series,  # type: ignore[type-arg]
) -> tuple[str, Pipeline, dict[str, Any]]:
    """Selecciona el mejor modelo por CV RMSE."""
    scoring = {
        "RMSE": "neg_root_mean_squared_error",
        "MAE": "neg_mean_absolute_error",
        "R2": "r2",
    }

    best_name = ""
    best_rmse = float("inf")
    best_pipeline = None
    cv_results: dict[str, Any] = {}

    for name, pipeline in models.items():
        logger.info(f"Evaluando {name} con {N_FOLDS}-fold CV")
        cv = cross_validate(
            pipeline, x_train, y_train, cv=N_FOLDS, scoring=scoring, return_train_score=True
        )
        mean_rmse = float(-cv["test_RMSE"].mean())
        mean_mae = float(-cv["test_MAE"].mean())
        mean_r2 = float(cv["test_R2"].mean())

        cv_results[name] = {
            "cv_rmse": mean_rmse,
            "cv_mae": mean_mae,
            "cv_r2": mean_r2,
            "cv_train_rmse": float(-cv["train_RMSE"].mean()),
            "cv_train_r2": float(cv["train_R2"].mean()),
        }
        logger.info(f"{name} — CV RMSE: {mean_rmse:.4f}, MAE: {mean_mae:.4f}, R2: {mean_r2:.4f}")

        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_name = name
            best_pipeline = pipeline

    logger.info(f"Mejor modelo: {best_name} (CV RMSE: {best_rmse:.4f})")
    assert best_pipeline is not None
    return best_name, best_pipeline, cv_results


def train_and_evaluate(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,  # type: ignore[type-arg]
    y_test: pd.Series,  # type: ignore[type-arg]
) -> tuple[Pipeline, dict[str, float], dict[str, float]]:
    """Entrena el modelo y evalúa en train y test."""
    pipeline.fit(x_train, y_train)

    train_metrics = evaluate_model(y_train, pipeline.predict(x_train))
    test_metrics = evaluate_model(y_test, pipeline.predict(x_test))

    logger.info(f"Train — RMSE: {train_metrics['rmse']:.4f}, R2: {train_metrics['r2']:.4f}")
    logger.info(f"Test  — RMSE: {test_metrics['rmse']:.4f}, R2: {test_metrics['r2']:.4f}")

    return pipeline, train_metrics, test_metrics


OVERFIT_THRESHOLD = 0.3
UNDERFIT_R2_THRESHOLD = 0.5


def validate_model(
    best_name: str,
    cv_results: dict[str, Any],
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> dict[str, Any]:
    """Compara train, CV y test; detecta overfitting/underfitting."""
    cv_best = cv_results[best_name]

    comparison = {
        "train_rmse": train_metrics["rmse"],
        "cv_train_rmse": cv_best["cv_train_rmse"],
        "cv_test_rmse": cv_best["cv_rmse"],
        "test_rmse": test_metrics["rmse"],
        "train_r2": train_metrics["r2"],
        "cv_train_r2": cv_best["cv_train_r2"],
        "cv_test_r2": cv_best["cv_r2"],
        "test_r2": test_metrics["r2"],
    }

    logger.info("=== Comparación Train / CV / Test ===")
    logger.info(
        f"RMSE — Train: {comparison['train_rmse']:.4f}, "
        f"CV-train: {comparison['cv_train_rmse']:.4f}, "
        f"CV-test: {comparison['cv_test_rmse']:.4f}, "
        f"Test: {comparison['test_rmse']:.4f}"
    )
    logger.info(
        f"R2   — Train: {comparison['train_r2']:.4f}, "
        f"CV-train: {comparison['cv_train_r2']:.4f}, "
        f"CV-test: {comparison['cv_test_r2']:.4f}, "
        f"Test: {comparison['test_r2']:.4f}"
    )

    diagnosis: list[str] = []

    rmse_gap = comparison["train_rmse"] - comparison["cv_test_rmse"]
    if abs(rmse_gap) > OVERFIT_THRESHOLD * comparison["cv_test_rmse"]:
        if rmse_gap < 0:
            diagnosis.append("overfitting")
            logger.warning(
                "Overfitting detectado: train RMSE muy inferior a CV RMSE "
                f"(gap={abs(rmse_gap):.4f})"
            )
        else:
            diagnosis.append("underfitting")

    if comparison["cv_test_r2"] < UNDERFIT_R2_THRESHOLD:
        diagnosis.append("underfitting")
        logger.warning(
            f"Underfitting: CV R2={comparison['cv_test_r2']:.4f} < {UNDERFIT_R2_THRESHOLD}"
        )

    if not diagnosis:
        diagnosis.append("good_fit")
        logger.info("Modelo con buen ajuste: sin señales de overfitting ni underfitting")

    unique_diagnosis = list(dict.fromkeys(diagnosis))

    validation: dict[str, Any] = {
        "comparison": comparison,
        "diagnosis": unique_diagnosis,
    }

    logger.info(f"Diagnóstico: {unique_diagnosis}")
    return validation


def save_model(model: Pipeline, path: str) -> None:
    """Guarda el modelo entrenado."""
    full_path = Path(path) if Path(path).is_absolute() else PROJECT_ROOT / path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, full_path)
    logger.info(f"Modelo guardado en {full_path}")


def save_metrics(metrics: dict[str, Any], path: str) -> None:
    """Guarda las métricas de evaluación."""
    full_path = Path(path) if Path(path).is_absolute() else PROJECT_ROOT / path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Métricas guardadas en {full_path}")


def run() -> None:
    """Ejecuta el training pipeline completo."""
    config = load_config()

    df = load_features(config["feature_data_path"])
    x_train, x_test, y_train, y_test = split_data(
        df, config["target_column"], config["test_size"], config["random_state"]
    )

    split_validation = validate_train_test_split(x_train, x_test, y_train, y_test)

    models = build_models()
    best_name, best_pipeline, cv_results = select_best_model(models, x_train, y_train)

    trained_model, train_metrics, test_metrics = train_and_evaluate(
        best_pipeline, x_train, x_test, y_train, y_test
    )

    model_validation = validate_model(best_name, cv_results, train_metrics, test_metrics)

    save_model(trained_model, config["model_path"])
    save_metrics(
        {
            "best_model": best_name,
            "split_validation": split_validation,
            "cv_results": cv_results,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "model_validation": model_validation,
        },
        config["metrics_path"],
    )

    logger.info("Training pipeline completado exitosamente")


if __name__ == "__main__":
    run()
