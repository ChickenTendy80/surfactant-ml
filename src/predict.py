# ============================================================
# src/predict.py
# ============================================================

import joblib
import pandas as pd
import numpy as np

from src.descriptors import (
    calc_descriptors,
    calc_fp
)

from src.feature_engineering import (
    generate_engineered_features
)

# ============================================================
# LOAD MODEL + SCALER
# ============================================================

model = joblib.load(
    "models/trained_model.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
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

    fp_array = calc_fp(smiles)

    fp_df = pd.DataFrame(
        [fp_array],
        columns=[
            f"FP_{i}"
            for i in range(len(fp_array))
        ]
    )

    # --------------------------------------------------------
    # COMBINE EVERYTHING
    # --------------------------------------------------------

    combined_df = pd.concat(
        [
            desc_df.reset_index(drop=True),
            engineered_df.reset_index(drop=True),
            fp_df.reset_index(drop=True)
        ],
        axis=1
    )

    return combined_df

