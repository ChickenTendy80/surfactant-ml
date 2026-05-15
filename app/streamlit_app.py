import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import streamlit as st

from streamlit_ketcher import st_ketcher

from src.predict import predict_pcmc

# ============================================================
# PAGE TITLE
# ============================================================

st.title("Surfactant CMC Predictor")

st.write(
    "Draw a surfactant molecule or paste a SMILES string." \
    "Press Apply once complete"
)

# ============================================================
# MOLECULE DRAWER
# ============================================================

smiles = st_ketcher()

# ============================================================
# IONIC TYPE
# ============================================================

ionic_type = st.selectbox(
    "Select ionic type",
    [
        "anionic",
        "cationic",
        "zwitterionic",
        "nonionic"
    ]
)

# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button("Predict CMC"):

    if smiles:
        try:
            pred_pcmc, pred_cmc = predict_pcmc(
                smiles,
                ionic_type
            )

            st.success("Prediction Complete")
            st.write(f"SMILES: {smiles}")
            st.metric(
                "Predicted pCMC",
                f"{pred_pcmc:.3f}"
            )

            st.metric(
                "Predicted CMC (mM)",
                f"{pred_cmc:.4f}"
            )

        except Exception as e:
            st.error(str(e))

    else:
        st.warning("Please draw a molecule.")

#Run streamlit run app/streamlit_app.py