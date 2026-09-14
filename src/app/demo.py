"""Demo funcional - Predicción de Precios de Viviendas en Boston.

Ejecutar con: streamlit run src/app/demo.py
"""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "data" / "06_models" / "model.joblib"
EXAMPLE_PATH = PROJECT_ROOT / "data" / "ejemplo_batch.csv"
PRICE_SCALE = 1000

st.set_page_config(
    page_title="Predicción Precios Boston",
    page_icon="🏠",
    layout="wide",
)

FEATURE_CONFIG: dict[str, dict[str, Any]] = {
    "crim": {
        "label": "Tasa de criminalidad per cápita",
        "min": 0.01,
        "max": 89.0,
        "default": 0.26,
        "step": 0.01,
        "help": "Tasa de criminalidad per cápita de la zona. Valores bajos indican barrios seguros.",
    },
    "zn": {
        "label": "Suelo residencial amplio (%)",
        "min": 0.0,
        "max": 100.0,
        "default": 0.0,
        "step": 1.0,
        "help": "Porcentaje de terreno residencial destinado a lotes grandes (más de 25,000 pies²).",
    },
    "indus": {
        "label": "Suelo industrial (%)",
        "min": 0.7,
        "max": 28.0,
        "default": 9.8,
        "step": 0.1,
        "help": "Porcentaje de la zona dedicado a negocios no minoristas (industria).",
    },
    "chas": {
        "label": "Cercanía al río Charles",
        "options": {0: "No colinda con el río", 1: "Colinda con el río Charles"},
        "help": "Indica si la zona colinda con el río Charles.",
    },
    "nox": {
        "label": "Concentración óxidos de nitrógeno",
        "min": 0.38,
        "max": 0.87,
        "default": 0.54,
        "step": 0.01,
        "help": "Concentración de óxidos de nitrógeno (partes por 10M). Mide la contaminación.",
    },
    "rm": {
        "label": "Habitaciones promedio",
        "min": 3.5,
        "max": 8.8,
        "default": 6.2,
        "step": 0.1,
        "help": "Número promedio de habitaciones por vivienda en la zona.",
    },
    "age": {
        "label": "Antigüedad viviendas (%)",
        "min": 0.0,
        "max": 100.0,
        "default": 77.0,
        "step": 1.0,
        "help": "Porcentaje de viviendas de la zona construidas antes de 1940.",
    },
    "dis": {
        "label": "Distancia a centros de empleo",
        "min": 1.1,
        "max": 12.0,
        "default": 3.37,
        "step": 0.1,
        "help": "Distancia ponderada a los cinco principales centros de empleo de Boston.",
    },
    "rad": {
        "label": "Accesibilidad a autopistas",
        "options": {
            1: "1",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            24: "24",
        },
        "default_index": 4,
        "help": "Índice de acceso a las autopistas radiales de Boston.",
    },
    "tax": {
        "label": "Impuesto a la propiedad (por $10k)",
        "min": 188.0,
        "max": 711.0,
        "default": 330.0,
        "step": 1.0,
        "help": "Tasa de impuesto a la propiedad por cada $10,000 de valor.",
    },
    "ptratio": {
        "label": "Alumnos por profesor",
        "min": 12.6,
        "max": 21.2,
        "default": 19.0,
        "step": 0.1,
        "help": "Número de alumnos por cada profesor en las escuelas de la zona.",
    },
    "black": {
        "label": "Índice demográfico (variable histórica)",
        "min": 3.5,
        "max": 396.9,
        "default": 392.0,
        "step": 1.0,
        "help": "Variable histórica del dataset original de 1978. Su uso es éticamente cuestionable.",
    },
    "lstat": {
        "label": "Población estatus socioeconómico bajo (%)",
        "min": 1.7,
        "max": 38.0,
        "default": 11.2,
        "step": 0.1,
        "help": "Porcentaje de la población clasificada como de estatus socioeconómico bajo.",
    },
}

EXPECTED_COLS = [
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
]

PRICE_SEGMENTS = [
    (15, "Económico", "#3b82f6", "Menos de $15,000"),
    (25, "Medio", "#22c55e", "$15,000 - $25,000"),
    (35, "Alto", "#f59e0b", "$25,000 - $35,000"),
    (float("inf"), "Premium", "#ef4444", "Más de $35,000"),
]


@st.cache_resource
def load_model() -> Any:
    return joblib.load(MODEL_PATH)


def get_price_segment(prediction: float) -> tuple[str, str, str]:
    for threshold, name, color, label in PRICE_SEGMENTS:
        if prediction < threshold:
            return name, color, label
    return "Premium", "#ef4444", "Más de $35,000"


def prepare_input(data: pd.DataFrame) -> pd.DataFrame:
    data["chas"] = data["chas"].astype(str)
    data["rad"] = data["rad"].astype(float)
    return data


# --- Header ---
st.title("🏠 Predicción de Precios de Viviendas en Boston")
st.markdown("Modelo entrenado con el pipeline FTI del curso de MLOps.")

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "No se encontró el modelo entrenado en `data/06_models/model.joblib`. "
        "Ejecuta primero el training pipeline."
    )
    st.stop()

# --- Tabs ---
tab_online, tab_batch = st.tabs(["🔮 Predicción Online", "📦 Predicción Batch"])

# =============================================================================
# TAB 1: Predicción Online
# =============================================================================
with tab_online:
    st.markdown("Ajusta las características de la zona para obtener una estimación del precio.")

    features = {}

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🏡 Vivienda")
        cfg = FEATURE_CONFIG["rm"]
        features["rm"] = st.slider(
            cfg["label"],
            cfg["min"],
            cfg["max"],
            cfg["default"],
            cfg["step"],
            help=cfg["help"],
        )
        cfg = FEATURE_CONFIG["age"]
        features["age"] = st.slider(
            cfg["label"],
            cfg["min"],
            cfg["max"],
            cfg["default"],
            cfg["step"],
            help=cfg["help"],
        )

    with col2:
        st.subheader("🌳 Entorno")
        cfg = FEATURE_CONFIG["nox"]
        features["nox"] = st.slider(
            cfg["label"],
            cfg["min"],
            cfg["max"],
            cfg["default"],
            cfg["step"],
            help=cfg["help"],
        )
        cfg = FEATURE_CONFIG["dis"]
        features["dis"] = st.slider(
            cfg["label"],
            cfg["min"],
            cfg["max"],
            cfg["default"],
            cfg["step"],
            help=cfg["help"],
        )
        cfg = FEATURE_CONFIG["indus"]
        features["indus"] = st.slider(
            cfg["label"],
            cfg["min"],
            cfg["max"],
            cfg["default"],
            cfg["step"],
            help=cfg["help"],
        )
        cfg = FEATURE_CONFIG["zn"]
        features["zn"] = st.slider(
            cfg["label"],
            cfg["min"],
            cfg["max"],
            cfg["default"],
            cfg["step"],
            help=cfg["help"],
        )
        cfg = FEATURE_CONFIG["chas"]
        features["chas"] = st.selectbox(
            cfg["label"],
            options=list(cfg["options"].keys()),
            format_func=lambda x: cfg["options"][x],
            help=cfg["help"],
        )
        cfg = FEATURE_CONFIG["rad"]
        features["rad"] = st.selectbox(
            cfg["label"],
            options=list(cfg["options"].keys()),
            index=cfg["default_index"],
            help=cfg["help"],
        )

    with col3:
        st.subheader("🏘️ Vecindario")
        cfg = FEATURE_CONFIG["lstat"]
        features["lstat"] = st.slider(
            cfg["label"],
            cfg["min"],
            cfg["max"],
            cfg["default"],
            cfg["step"],
            help=cfg["help"],
        )
        cfg = FEATURE_CONFIG["crim"]
        features["crim"] = st.slider(
            cfg["label"],
            cfg["min"],
            cfg["max"],
            cfg["default"],
            cfg["step"],
            help=cfg["help"],
        )
        cfg = FEATURE_CONFIG["tax"]
        features["tax"] = st.slider(
            cfg["label"],
            cfg["min"],
            cfg["max"],
            cfg["default"],
            cfg["step"],
            help=cfg["help"],
        )
        cfg = FEATURE_CONFIG["ptratio"]
        features["ptratio"] = st.slider(
            cfg["label"],
            cfg["min"],
            cfg["max"],
            cfg["default"],
            cfg["step"],
            help=cfg["help"],
        )
        cfg = FEATURE_CONFIG["black"]
        features["black"] = st.slider(
            cfg["label"],
            cfg["min"],
            cfg["max"],
            cfg["default"],
            cfg["step"],
            help=cfg["help"],
        )

    st.divider()

    _, col_btn, _ = st.columns([1, 2, 1])
    with col_btn:
        predict_online = st.button("🔮 Estimar precio", type="primary", use_container_width=True)

    if predict_online:
        input_df = pd.DataFrame([features])
        input_df = prepare_input(input_df)
        prediction = float(model.predict(input_df)[0])
        price_usd = prediction * PRICE_SCALE
        seg_name, seg_color, seg_label = get_price_segment(prediction)

        st.markdown("### Resultado")
        col_price, col_seg = st.columns(2)
        with col_price:
            st.metric("Precio estimado", f"${price_usd:,.0f}")
        with col_seg:
            st.markdown(
                f'<div style="background-color:{seg_color}22; border-left:6px solid {seg_color}; '
                f'padding:16px 20px; border-radius:8px;">'
                f'<div style="font-size:0.85rem; color:#666;">Segmento</div>'
                f'<div style="font-size:1.4rem; font-weight:700; color:{seg_color};">{seg_name}</div>'
                f'<div style="font-size:0.9rem; color:#444;">{seg_label}</div></div>',
                unsafe_allow_html=True,
            )

        st.info(
            "Esta estimación se basa en el dataset histórico de Boston Housing "
            "usado en el curso. No representa precios de mercado actuales.",
            icon="💡",
        )

# =============================================================================
# TAB 2: Predicción Batch
# =============================================================================
with tab_batch:
    st.markdown(
        "Sube un archivo CSV con múltiples registros para obtener predicciones en lote. "
        "El archivo debe tener las siguientes columnas (sin la columna target `medv`):"
    )
    st.code(", ".join(EXPECTED_COLS))

    if EXAMPLE_PATH.exists():
        with open(EXAMPLE_PATH, "rb") as f:
            st.download_button(
                "📥 Descargar CSV de ejemplo",
                f,
                file_name="ejemplo_batch.csv",
                mime="text/csv",
            )

    uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except (ValueError, pd.errors.ParserError) as e:
            st.error(f"Error al leer el archivo: {e}")
            st.stop()

        if "ID" in df.columns:
            df = df.drop(columns=["ID"])
        if "medv" in df.columns:
            df = df.drop(columns=["medv"])

        missing = set(EXPECTED_COLS) - set(df.columns)
        if missing:
            st.error(f"Columnas faltantes en el archivo: {missing}")
        else:
            st.markdown(f"**Registros cargados:** {len(df)}")
            st.dataframe(df.head(10), use_container_width=True)

            if st.button("🚀 Generar predicciones", type="primary"):
                df_input = prepare_input(df[EXPECTED_COLS].copy())
                predictions = model.predict(df_input)
                df_result = df.copy()
                df_result["precio_estimado_usd"] = (predictions * PRICE_SCALE).round(0).astype(int)
                df_result["segmento"] = [get_price_segment(p)[0] for p in predictions]

                st.markdown("### Resultados")

                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Registros", len(df_result))
                col_m2.metric("Precio promedio", f"${predictions.mean() * PRICE_SCALE:,.0f}")
                col_m3.metric(
                    "Rango",
                    f"${predictions.min() * PRICE_SCALE:,.0f} - ${predictions.max() * PRICE_SCALE:,.0f}",
                )

                st.dataframe(df_result, use_container_width=True)

                csv_output = df_result.to_csv(index=False)
                st.download_button(
                    "📥 Descargar predicciones CSV",
                    csv_output,
                    file_name="predicciones_boston.csv",
                    mime="text/csv",
                )
