# ============================================================
# src/descriptors.py
# ============================================================

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

# ============================================================
# BASIC RDKit DESCRIPTORS
# ============================================================

def calc_descriptors(smiles):
    """
    Generate core RDKit molecular descriptors.

    Parameters
    ----------
    smiles : str
        SMILES string of molecule

    Returns
    -------
    pandas.Series
        Molecular descriptors
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return pd.Series({
            "MolWt": np.nan,
            "MolLogP": np.nan,
            "TPSA": np.nan,
            "NumHDonors": np.nan,
            "NumHAcceptors": np.nan,
            "RingCount": np.nan,
            "HeavyAtomCount": np.nan,
            "RotatableBonds": np.nan
        })

    return pd.Series({
        "MolWt": Descriptors.MolWt(mol),
        "MolLogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "RingCount": Descriptors.RingCount(mol),
        "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
        "RotatableBonds": Descriptors.NumRotatableBonds(mol)
    })

# ============================================================
# MORGAN FINGERPRINTS
# ============================================================
"""
def calc_fp(smiles, radius=2, n_bits=128):

    mol = Chem.MolFromSmiles(smiles)

    arr = np.zeros((n_bits,), dtype=int)

    if mol is None:
        return arr

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits
    )

    fp = morgan_gen.GetFingerprint(mol)

    DataStructs.ConvertToNumpyArray(fp, arr)

    return arr
"""
# ============================================================
# GENERATE FULL FINGERPRINT DATAFRAME
# ============================================================
"""
def fingerprints_to_df(smiles_series, radius=2, n_bits=128):
    fps = np.array([
        calc_fp(s, radius=radius, n_bits=n_bits)
        for s in smiles_series
    ])

    fp_df = pd.DataFrame(
        fps,
        columns=[f"FP_{i}" for i in range(n_bits)]
    )

    return fp_df
"""