import os
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from branca.element import Template, MacroElement

# --- Hacer visible el paquete del proyecto (src) ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Importa la clase para que pickle pueda reconstruir el pipeline (opción B)
from src.transformers import IntensityImputer  # noqa: F401

st.set_page_config(page_title="Tsunami ML • Streamlit", layout="wide")

st.title("Predicción y visualización de tsunamis a partir de terremotos")
st.markdown("Carga tu dataset para visualizar terremotos y tsunamis. Opcionalmente, carga tu modelo para predicciones puntuales.")

# ==========================
# 1) CARGA DE DATOS
# ==========================
st.sidebar.header("1) Datos")
uploaded = st.sidebar.file_uploader("Sube tu CSV procesado", type=["csv"])
ruta_local = st.sidebar.text_input("...o indica una ruta local (opcional)", value="")
df = None

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.sidebar.success("CSV cargado correctamente (subida).")
    except Exception as e:
        st.sidebar.error(f"No pude leer el CSV subido: {e}")
elif ruta_local.strip():
    try:
        df = pd.read_csv(ruta_local.strip())
        st.sidebar.success("CSV cargado correctamente (ruta).")
    except Exception as e:
        st.sidebar.error(f"No pude leer el CSV de la ruta indicada: {e}")

# columnas requeridas y opcionales (para mapa/visualización)
cols_requeridas = ["latitude_eq", "longitude_eq", "oceanicTsunami", "magnitude_Mw"]
cols_opcionales = {"year": np.nan, "locationName": "Sin nombre"}

if df is not None:
    faltan_req = [c for c in cols_requeridas if c not in df.columns]
    if faltan_req:
        st.error(f"Faltan columnas **requeridas** en tu dataset: {faltan_req}")
        df = None
    else:
        # crear opcionales si faltan
        faltan_opt = [c for c in cols_opcionales if c not in df.columns]
        for c in faltan_opt:
            df[c] = cols_opcionales[c]
        if faltan_opt:
            st.warning(f"Faltaban columnas **opcionales** y se han creado por defecto: {faltan_opt}")

# ==========================
# 2) FILTROS Y MAPA
# ==========================
if df is not None:
    st.sidebar.header("2) Filtros")

    df = df.copy()
    # casting seguro
    for c in ["latitude_eq", "longitude_eq", "magnitude_Mw", "year", "oceanicTsunami"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "oceanicTsunami" in df.columns:
        df["oceanicTsunami"] = df["oceanicTsunami"].fillna(0).astype(int)

    vista = st.sidebar.selectbox("Vista", ["Todos", "Solo tsunami", "Solo no tsunami"])
    if vista == "Solo tsunami":
        df_f = df[df["oceanicTsunami"] == 1]
    elif vista == "Solo no tsunami":
        df_f = df[df["oceanicTsunami"] == 0]
    else:
        df_f = df

    # Filtro por año si existe
    if "year" in df_f.columns and df_f["year"].notna().any():
        anios_validos = df_f["year"].dropna().astype(int)
        a_min, a_max = int(anios_validos.min()), int(anios_validos.max())
        if a_min > a_max:  # sanity
            a_min, a_max = a_max, a_min
        rango = st.sidebar.slider("Rango de año", min_value=a_min, max_value=a_max, value=(a_min, a_max))
        df_f = df_f[df_f["year"].between(rango[0], rango[1], inclusive="both")]

    # Quitar filas sin coordenadas del terremoto
    df_f = df_f.dropna(subset=["latitude_eq", "longitude_eq"])

    st.subheader("🗺️ Mapa histórico")
    if df_f.empty:
        st.info("No hay registros para los filtros seleccionados.")
    else:
        lat_centro = float(df_f["latitude_eq"].mean())
        lon_centro = float(df_f["longitude_eq"].mean())
        m = folium.Map(location=[lat_centro, lon_centro], zoom_start=2, tiles="CartoDB positron")

        # --- Capa de terremotos (verde=no tsunami, rojo=tsunami)
        for _, row in df_f.iterrows():
            color = "green" if row.get("oceanicTsunami", 0) == 0 else "red"
            popup = folium.Popup(
                f"""<b>{row.get('locationName','Sin nombre')}</b><br>
                Año: {row.get('year', '—')}<br>
                Magnitud: {row.get('magnitude_Mw', '—')}<br>
                Tsunami: {"Sí" if row.get("oceanicTsunami", 0)==1 else "No"}""",
                max_width=300
            )
            folium.CircleMarker(
                location=[row["latitude_eq"], row["longitude_eq"]],
                radius=3,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=popup
            ).add_to(m)

        # --- Leyenda fija
        legend_template = """
        {% macro html(this, kwargs) %}
        <div style="
            position: fixed; 
            bottom: 30px; left: 30px; z-index:9999; 
            background-color: white; padding: 10px 12px; 
            border: 2px solid #bbb; border-radius: 8px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.2); font-size: 14px;">
          <div style="font-weight:600; margin-bottom:6px;">Leyenda</div>
          <div style="display:flex; align-items:center; gap:8px;">
            <span style="display:inline-block;width:12px;height:12px;background:green;border-radius:50%;"></span>
            <span>Terremoto sin tsunami</span>
          </div>
          <div style="height:6px;"></div>
          <div style="display:flex; align-items:center; gap:8px;">
            <span style="display:inline-block;width:12px;height:12px;background:red;border-radius:50%;"></span>
            <span>Terremoto con tsunami</span>
          </div>
        </div>
        {% endmacro %}
        """
        macro = MacroElement()
        macro._template = Template(legend_template)
        m.get_root().add_child(macro)

        st_folium(m, width=None, height=600)

# ==========================
# 3) PREDICCIÓN OPCIONAL (ampliada)
# ==========================
st.sidebar.header("3) (Opcional) Modelo")
modelo_file = st.sidebar.file_uploader("Carga tu modelo .pkl (pipeline)", type=["pkl"])
modelo_path = st.sidebar.text_input("...o ruta local al .pkl", value="")
model = None
error_modelo = None

# Pre-imports habituales si el pipeline los usó
for lib in ("lightgbm", "xgboost", "catboost", "sklearn", "imblearn"):
    try:
        __import__(lib)
    except Exception:
        pass

try:
    if modelo_file is not None:
        model = pickle.load(modelo_file)  # buffer subido
        st.sidebar.success("Modelo cargado correctamente (subida).")
    elif modelo_path.strip():
        if not os.path.exists(modelo_path):
            raise FileNotFoundError(f"No existe el archivo: {modelo_path}")
        with open(modelo_path, "rb") as f:
            model = pickle.load(f)
        st.sidebar.success("Modelo cargado correctamente (ruta).")
    else:
        st.sidebar.info("Carga un .pkl para habilitar la predicción.")
except Exception as e:
    error_modelo = f"Error al cargar el modelo: {e}"

if error_modelo:
    st.sidebar.error(error_modelo)

# Inputs para predicción puntual (ampliados)
st.subheader("🔮 Predicción puntual (opcional)")
col1, col2, col3, col4 = st.columns(4)
with col1:
    magnitude = st.number_input("Magnitud (Mw)", min_value=0.0, max_value=10.0, step=0.1, value=6.5)
with col2:
    depth = st.number_input("Profundidad (km)", min_value=0.0, max_value=700.0, step=1.0, value=20.0)
with col3:
    latitude = st.number_input("Latitud", min_value=-90.0, max_value=90.0, step=0.01, value=0.0)
with col4:
    longitude = st.number_input("Longitud", min_value=-180.0, max_value=180.0, step=0.01, value=0.0)

col5, col6, col7, col8 = st.columns(4)
with col5:
    intensity_str = st.text_input("Intensidad (opcional)", value="")
    try:
        intensity = float(intensity_str) if intensity_str.strip() != "" else np.nan
    except Exception:
        st.warning("La intensidad debe ser numérica. Se usará NaN para imputación.")
        intensity = np.nan
with col6:
    # país desde el df si existe; si no, campo libre
    if df is not None and "country" in df.columns:
        countries = sorted([c for c in df["country"].dropna().unique().tolist() if str(c).strip() != ""])
        country = st.selectbox("País", options=countries if countries else ["UNKNOWN"], index=0 if countries else 0)
    else:
        country = st.text_input("País", value="UNKNOWN")
with col7:
    regionCode_eq = st.number_input("regionCode_eq", min_value=-9999, max_value=9999, step=1, value=0)
with col8:
    year_opt = st.text_input("Año (opcional)", value="")
    try:
        year_val = int(year_opt) if year_opt.strip() != "" else np.nan
    except Exception:
        st.warning("El año debe ser entero. Se usará NaN.")
        year_val = np.nan

if st.button("Predecir tsunami"):
    if model is None:
        st.warning("Carga un modelo .pkl en la barra lateral para habilitar la predicción.")
    else:
        # Construimos X con TODAS las columnas usadas al entrenar (según tu notebook)
        X_infer = pd.DataFrame([{
            "magnitude_Mw": magnitude,
            "eqDepth": depth,
            "latitude_eq": latitude,
            "longitude_eq": longitude,
            "intensity": intensity,        # se imputará si es NaN por IntensityImputer
            "year": year_val,              # opcional
            "country": country,
            "regionCode_eq": regionCode_eq,
        }])

        try:
            pred = model.predict(X_infer)[0]
            prob = model.predict_proba(X_infer)[0][1] if hasattr(model, "predict_proba") else None
            etiqueta = "Tsunami probable 🌊" if int(pred) == 1 else "Tsunami poco probable ✅"
            if prob is not None and np.isfinite(prob):
                st.success(f"Predicción: **{etiqueta}** — Prob: **{prob:.2%}**")
            else:
                st.success(f"Predicción: **{etiqueta}**")
        except Exception as e:
            st.error(
                "No se pudo realizar la predicción. Tu pipeline probablemente espera más columnas o un orden distinto. "
                "Asegúrate de exportar un pipeline de **inferencia** (sin SMOTE) con el mismo preprocesado."
            )
            st.exception(e)

        # Mini-mapa del punto consultado
        m_pred = folium.Map(location=[latitude, longitude], zoom_start=5, tiles="CartoDB positron")
        color_pred = "red" if 'pred' in locals() and int(pred) == 1 else "green"
        folium.CircleMarker(
            location=[latitude, longitude],
            radius=6,
            color=color_pred,
            fill=True,
            fill_color=color_pred,
            fill_opacity=0.85,
            popup=f"Input → Mw:{magnitude} | Prof:{depth}km"
        ).add_to(m_pred)
        st_folium(m_pred, width=None, height=400)

st.caption(
    "Nota: si tu pipeline de *producción* solo debe usar entradas del usuario, exporta un pipeline de **inferencia** "
    "(sin SMOTE) que espere exactamente estas columnas. Si entrenaste con más columnas (p.ej., `country`, `regionCode_eq`), "
    "debes proporcionarlas también aquí o ajustar tu pipeline."
)
