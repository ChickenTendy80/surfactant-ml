#============================================
#Import needed libraries
#============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

#Load dataset
df = pd.read_csv("data/processed_surfactant.csv")

#Target: pCMC
y = df["pCMC"]

#3 features will be comapred: descriptors only, fingerprints only, combined
#Determine which feature is the best

#==============================================
# 1. Descriptors Only
#==============================================
descriptors_col = [
    "MolWt", 
    "MolLogP", 
    "TPSA", 
    "NumHDonors", 
    "NumHAcceptors", 
    "RingCount", 
    "HeavyAtomCount",
    "RotatableBonds",
    "tail_length",
    "aromatic_flag",
    "ionic_encoded",
    "ethoxylate_count"
    ]

X_desc = df[descriptors_col]

#==============================================
# 2. Fingerprints Only
#==============================================
fp_cols = [
    col for col in df.columns
    if col.startswith("FP_")
]

X_fp = df[fp_cols]

#==============================================
# 3. Descriptor AND Fingerprints 
#==============================================
X_combined = pd.concat(
    [X_desc, X_fp],
    axis=1
)

#==============================================
# 4. Test Train Split 
#==============================================
X_train, X_test, y_train, y_test = train_test_split(
    X_combined,
    y,
    test_size = 0.2,
    random_state = 42
)
