import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import streamlit as st

from streamlit_ketcher import st_ketcher

from src.predict import predict_pcmc

# ============================================================
# COMMON SURFACTANTS
# ============================================================

common_surfactants = {
    "SDS": {
        "smiles": "CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]",
        "ionic": "anionic"
    },

    "CTABr": {
        "smiles": "CCCCCCCCCCCCCCCC[N+](C)(C)C.[Br-]",
        "ionic": "cationic"
    },

    "Tween 80": {
        "smiles": "CCCCCCCC=CCCCCCCCCCCC(=O)OCC(CO)OCC(CO)OCC(CO)O",
        "ionic": "nonionic"
    },

    "Brij 35": {
        "smiles": "CCCCCCCCCCCCOCCOCCOCCO",
        "ionic": "nonionic"
    },

    "SLES": {
        "smiles": "CCCCCCCCCCCCOCCOCCOS(=O)(=O)[O-].[Na+]",
        "ionic": "anionic"
    }
}

# ============================================================
# PAGE TITLE
# ============================================================

st.title("Surfactant CMC Predictor")

st.write(
    "Draw a surfactant molecule or paste a SMILES string." \
    "Press Apply once complete"
)

#Dropdown for common surfactant
selected = st.selectbox(
    "Load common surfactant",
    ["Custom"] + list(common_surfactants.keys())
)

#Set default
default_smiles = ""

default_ionic = "nonionic"

if selected != "Custom":

    default_smiles = common_surfactants[selected]["smiles"]

    default_ionic = common_surfactants[selected]["ionic"]

# ============================================================
# MOLECULE DRAWER
# ============================================================

smiles = st_ketcher(default_smiles)

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
    ],
    index=[
        "anionic",
        "cationic",
        "zwitterionic",
        "nonionic"
    ].index(default_ionic)
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
#End by pressing control^ + C (NOT command + C) on mac

#Next goal: Compare with literaure values
# 1. Exact Matching: search raw_surfactant.csv for the exact same smile
# 2. Molecular similarity matching: look for SIMILAR molecules, 
#       display similar molecule and compare their similarity (using tanimoto similarity) 
#       then show experimental pCMC
# Can try to plot predicted vs literature??