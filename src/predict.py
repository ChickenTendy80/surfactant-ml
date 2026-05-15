# ============================================================
# src/predict.py
# ============================================================

import joblib
import pandas as pd
import numpy as np

from src.descriptors import (
    calc_descriptors,
    #calc_fp
)

from src.feature_engineering import (
    generate_engineered_features
)

# ============================================================
# LOAD MODEL + SCALER
# ============================================================

model_xgb = joblib.load(
    "models/trained_model_xgb.pkl"
)

model_rf = joblib.load(
    "models/trained_model_rf.pkl"
)

feature_columns = joblib.load(
    "models/feature_columns.pkl"
)

# ============================================================
# GENERATE COMBINED FEATURE VECTOR
# ============================================================

def generate_combined_features(
    smiles,
    ionic_type="nonionic"
):
    """
    Generate complete feature dataframe for prediction.

    Includes:
    - RDKit descriptors
    - engineered surfactant features
    - Morgan fingerprints

    Parameters
    ----------
    smiles : str
        Molecule SMILES string

    ionic_type : str
        Surfactant ionic class
        (anionic/cationic/zwitterionic/nonionic)

    Returns
    -------
    pandas.DataFrame
        Single-row feature dataframe
    """

    # --------------------------------------------------------
    # CREATE TEMP DATAFRAME
    # --------------------------------------------------------

    temp_df = pd.DataFrame({
        "smiles": [smiles],
        "ionic_type": [ionic_type]
    })

    # --------------------------------------------------------
    # RDKit DESCRIPTORS
    # --------------------------------------------------------

    desc_df = pd.DataFrame([
        calc_descriptors(smiles)
    ])

    # --------------------------------------------------------
    # ENGINEERED FEATURES
    # --------------------------------------------------------

    engineered_df = generate_engineered_features(
        temp_df.copy()
    )

    engineered_df = engineered_df[
        [
            "aromatic_flag",
            "tail_length",
            "ionic_encoded",
            "ethoxylate_count"
        ]
    ]

    # --------------------------------------------------------
    # MORGAN FINGERPRINTS
    # --------------------------------------------------------
    """
    fp_array = calc_fp(smiles)

    fp_df = pd.DataFrame(
        [fp_array],
        columns=[
            f"FP_{i}"
            for i in range(len(fp_array))
        ]
    )
    """
    # --------------------------------------------------------
    # COMBINE EVERYTHING
    # --------------------------------------------------------

    combined_df = pd.concat(
        [
            desc_df.reset_index(drop=True),
            engineered_df.reset_index(drop=True)
        ],
        axis=1
    )

    return combined_df

# ============================================================
# PREDICT pCMC
# ============================================================

def predict_pcmc(
    smiles,
    ionic_type="nonionic"
):

    # --------------------------------------------------------
    # GENERATE FEATURES
    # --------------------------------------------------------

    X_new = generate_combined_features(
        smiles=smiles,
        ionic_type=ionic_type
    )

    # --------------------------------------------------------
    # MATCH TRAINING COLUMN ORDER
    # --------------------------------------------------------

    X_new = X_new[feature_columns]
    #print(X_new.columns.tolist()[:20])
    #print(feature_columns[:20])
    # --------------------------------------------------------
    # SCALE FEATURES
    # --------------------------------------------------------
    #print("X_new shape:")
    #print(X_new.shape)

    #print("\nFirst 20 features:")
    #print(X_new.iloc[0, :20])

    #print("\nNaN count:")
    #print(X_new.isna().sum().sum())

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    pred_pcmc = model_xgb.predict(X_new)[0]

    cmc_molar = 10 ** (-pred_pcmc)

    pred_cmc_mM = cmc_molar * 1000

    return pred_pcmc, pred_cmc_mM


#Test Molecules

#SDS
#Smile: CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]
#Anionic
#pCMC ~2.1

#CTABr
#Smile: CCCCCCCCCCCCCCCC[N+](C)(C)C.[Br-]
#Cationic
#pCMC ~3.0

#Tween 80
#Smile: CCCCCCCC=CCCCCCCCCCCC(=O)OCC(CO)OCC(CO)OCC(CO)O
#Nonionic
#pCMC ~5