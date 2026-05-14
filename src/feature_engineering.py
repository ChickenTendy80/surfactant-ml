# ============================================================
# src/feature_engineering.py
# ============================================================

import numpy as np

from rdkit import Chem

# ============================================================
# AROMATICITY FLAG
# ============================================================

def aromatic_flag(smiles):
    """
    Detect whether molecule contains aromatic atoms.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return 0

    return int(
        any(atom.GetIsAromatic() for atom in mol.GetAtoms())
    )

# ============================================================
# LONGEST ALKYL TAIL LENGTH
# ============================================================

def alkyl_tail_length(smiles):
    """
    Estimate longest non-aromatic carbon chain.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return np.nan

    max_length = 0

    def dfs(atom_idx, visited):

        atom = mol.GetAtomWithIdx(atom_idx)

        # Must be carbon
        if atom.GetAtomicNum() != 6:
            return 0

        # Ignore aromatic carbons
        if atom.GetIsAromatic():
            return 0

        visited.add(atom_idx)

        max_path = 1

        for neighbor in atom.GetNeighbors():

            n_idx = neighbor.GetIdx()

            if n_idx not in visited:

                n_atom = mol.GetAtomWithIdx(n_idx)

                if (
                    n_atom.GetAtomicNum() == 6
                    and not n_atom.GetIsAromatic()
                ):

                    path_len = 1 + dfs(
                        n_idx,
                        visited.copy()
                    )

                    max_path = max(max_path, path_len)

        return max_path

    for atom in mol.GetAtoms():

        if (
            atom.GetAtomicNum() == 6
            and not atom.GetIsAromatic()
        ):

            length = dfs(atom.GetIdx(), set())

            max_length = max(max_length, length)

    return max_length

# ============================================================
# IONIC TYPE ENCODING
# ============================================================

def ionic_encode(x):
    """
    Encode ionic surfactant classes numerically.

    Encoding:
    0 = anionic
    1 = cationic
    2 = zwitterionic
    3 = nonionic/other
    """

    x = str(x).lower()

    if "anionic" in x:
        return 0

    elif "cationic" in x:
        return 1

    elif "zwitter" in x:
        return 2

    else:
        return 3

# ============================================================
# ETHOXYLATE COUNT
# ============================================================

def ethoxylate_count(smiles):
    """
    Count ethylene oxide repeat units.

    Detects:
    -CH2-CH2-O-
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return np.nan

    eo_pattern = Chem.MolFromSmarts("[CH2][CH2]O")

    matches = mol.GetSubstructMatches(eo_pattern)

    return len(matches)

# ============================================================
# COMBINED FEATURE GENERATOR
# ============================================================

def generate_engineered_features(df):
    """
    Add engineered surfactant features to dataframe.
    """

    df["aromatic_flag"] = df["smiles"].apply(
        aromatic_flag
    )

    df["tail_length"] = df["smiles"].apply(
        alkyl_tail_length
    )

    df["ionic_encoded"] = df["ionic_type"].apply(
        ionic_encode
    )

    df["ethoxylate_count"] = df["smiles"].apply(
        ethoxylate_count
    )

    return df