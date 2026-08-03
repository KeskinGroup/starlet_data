"""app.py – COFSpace Diffusion Predictor
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import base64
import io
import matplotlib.pyplot as plt
from typing import List

# -----------------------------------------------------------------------------
# Helper functions -------------------------------------------------------------
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data(show_spinner=False)
def load_feature_ranges(csv_path: Path, target: str):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[target])
    return X.min(), X.max(), X.columns.tolist()

# -----------------------------------------------------------------------------
# Constants --------------------------------------------------------------------
# -----------------------------------------------------------------------------
MODEL_DIR = Path("Models")
DATA_DIR = Path("data")

GASES = {
    "H2":  {"pkl": "model_H2.pkl",  "csv": "HypoCoRE - H2 - 1 BAR - LOG - 7800.csv",  "target": "LOG(DH2 - 1 bar)"},
    "N2":  {"pkl": "model_N2.pkl",  "csv": "HypoCoRE - N2 - 1 BAR - LOG - 7800.csv",  "target": "LOG(DN2 - 1 bar)"},
    "O2":  {"pkl": "model_O2.pkl",  "csv": "HypoCoRE - O2 - 1 BAR - LOG - 7800.csv",  "target": "LOG(DO2 - 1 bar)"},
    "CO2": {"pkl": "model_CO2.pkl", "csv": "HypoCoRE - CO2 - 1 BAR - LOG - 7800.csv", "target": "LOG(DCO2 - 1 bar)"},
    "CH4": {"pkl": "model_CH4.pkl", "csv": "HypoCoRE - CH4 - 1 BAR - LOG - 7800.csv", "target": "LOG(DCH4 - 1 bar)"},
}

# -----------------------------------------------------------------------------
# Page & Theme -----------------------------------------------------------------
# -----------------------------------------------------------------------------
st.set_page_config(page_title="COFSpace Diffusion Predictor", page_icon="🧪", layout="wide")

# Sidebar ---------------------------------------------------------------------
with st.sidebar:
    light_mode = st.toggle("🔆 Light mode", value=False, help="Toggle between light and dark themes.")
    st.markdown("## Gas Selection")
    gas_key = st.selectbox("Choose a gas", list(GASES.keys()))
    st.info(
        "💡 **Tip:** Keep your inputs within the training domain to maximise predictive reliability.",
        icon="ℹ️",
    )
    st.markdown("---")
    st.caption("Built with ❤️ & Streamlit — July 2025")

# Inject CSS depending on theme -----------------------------------------------
if light_mode:
    bg_css = "background: radial-gradient(ellipse at top left, #fafafa 0%, #e8e8e8 60%); color:#222;"
else:
    bg_css = "background: radial-gradient(ellipse at top left, #1f2024 0%, #0f1013 60%); color:#f5f5f5;"

st.markdown(
    f"""
    <style>
        #MainMenu, footer, header {{visibility: hidden;}}
        .stApp {{{bg_css}}}
        div[data-baseweb="input"] input {{border-radius: 0.6rem !important;}}
        .hero-title {{text-align:center; font-size:3rem; font-weight:800; margin-top:-0.5rem;}}
        .hero-subtitle {{text-align:center; font-size:1.1rem; margin-bottom:1.6rem; opacity:0.85;}}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Hero section -----------------------------------------------------------------
# -----------------------------------------------------------------------------
col_h1, col_h2, col_h3 = st.columns([1, 2, 1])
with col_h2:
    st.markdown("<div class='hero-title'>COFSpace Diffusion Predictor</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='hero-subtitle'>
            Machine‑learning‑driven estimation of single‑component self‑diffusion coefficients
            in covalent organic frameworks at 298 K and 1 bar.
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Load model & feature ranges --------------------------------------------------
# -----------------------------------------------------------------------------
info = GASES[gas_key]
model = load_model(MODEL_DIR / info["pkl"])
fmin, fmax, features = load_feature_ranges(DATA_DIR / info["csv"], info["target"])

# -----------------------------------------------------------------------------
# Single‑prediction input form -------------------------------------------------
# -----------------------------------------------------------------------------
with st.form("input_form", border=False):
    st.markdown("#### Input Features")
    cols = st.columns(3)
    user_vals = {}
    for i, feat in enumerate(features):
        col = cols[i % 3]
        user_vals[feat] = col.number_input(
            f"{feat} ({fmin[feat]:.2f} – {fmax[feat]:.2f})",
            value=float(np.round((fmin[feat] + fmax[feat]) / 2, 6)),
            min_value=float(fmin[feat]),
            max_value=float(fmax[feat]),
            format="%.6f",
            step=float(np.round((fmax[feat] - fmin[feat]) / 200, 6)),
            key=feat,
        )
    submitted = st.form_submit_button("✨ Predict Diffusion", type="primary")

if submitted:
    X_input = pd.DataFrame([user_vals])

    # OOB warning -------------------------------------------------------------
    oob = [f"• **{k}** ({v})" for k, v in user_vals.items() if v < fmin[k] or v > fmax[k]]
    if oob:
        st.warning(
            "The following inputs fall outside the model's training domain and may reduce reliability:\n" + "\n".join(oob)
        )

    # Prediction --------------------------------------------------------------
    pred_logD = float(model.predict(X_input)[0])
    pred_D = 10 ** pred_logD

    # Display ----------------------------------------------------------------
    col1, col2 = st.columns([1, 3], gap="small")
    col1.metric("Log(D) [cm²/s]", f"{pred_logD:.4f}")
    col2.metric("D [cm²/s]", f"{pred_D:.3e}")

    st.success("Prediction complete ✅")

# -----------------------------------------------------------------------------
# CSV bulk prediction ----------------------------------------------------------
# -----------------------------------------------------------------------------
st.markdown("---")
# --------------------------------------------------------------------------
# Bulk CSV prediction section
# --------------------------------------------------------------------------
st.markdown("### 📄 Batch prediction from CSV")
csv_file = st.file_uploader(
    "Upload a CSV that contains *exactly* the same feature columns "
    f"({len(features)} columns) in any order. The model will ignore extra columns.",
    type=["csv"],
)

if csv_file is not None:
    try:
        df_in = pd.read_csv(csv_file)
        # Keep only known features and in the original training order
        df_in = df_in[[c for c in features if c in df_in.columns]]
        missing = [c for c in features if c not in df_in.columns]
        if missing:
            st.error(f"Missing required feature columns: {', '.join(missing)}")
        else:
            with st.spinner("Running predictions …"):
                # Vectorised prediction
                predictions = model.predict(df_in)

                df_out = df_in.copy()
                df_out["logD_pred"] = predictions
                df_out["D_pred"] = 10 ** df_out["logD_pred"]

                # Provide a download button
                towrite = io.BytesIO()
                df_out.to_csv(towrite, index=False)
                towrite.seek(0)
                b64 = base64.b64encode(towrite.read()).decode()
                dl_link = f'<a href="data:file/csv;base64,{b64}" download="HypoCOF_{gas_key}_predictions.csv">📥 Download results</a>'
                st.markdown(dl_link, unsafe_allow_html=True)

                st.success(f"Finished! Predicted {len(df_out)} rows.")
    except Exception as e:
        st.exception(e)
