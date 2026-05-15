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

from descriptors import (
    calc_descriptors,
    fingerprints_to_df
)

from feature_engineering import (
    generate_engineered_features
)

# ============================================================
# LOAD RAW DATA
# ============================================================

df = pd.read_csv(
    "data/raw_surfactant.csv"
)

# ============================================================
# GENERATE RDKit DESCRIPTORS
# ============================================================

desc_df = df["smiles"].apply(
    calc_descriptors
)

# ============================================================
# GENERATE FINGERPRINTS
# ============================================================

fp_df = fingerprints_to_df(
    df["smiles"]
)

# ============================================================
# GENERATE ENGINEERED FEATURES
# ============================================================

engineered_df = generate_engineered_features(
    df.copy()
)

engineered_cols = [
    "aromatic_flag",
    "tail_length",
    "ionic_encoded",
    "ethoxylate_count"
]

engineered_df = engineered_df[
    engineered_cols
]

# ============================================================
# COMBINE ALL FEATURES
# ============================================================

X_combined = pd.concat(
    [
        desc_df.reset_index(drop=True),
        engineered_df.reset_index(drop=True),
        fp_df.reset_index(drop=True)
    ],
    axis=1
)

# ============================================================
# TARGET VARIABLE
# ============================================================

y = df["pCMC"]

# ============================================================
# OPTIONAL:
# SAVE GENERATED FEATURE MATRIX
# ============================================================

processed_df = pd.concat(
    [
        X_combined,
        y
    ],
    axis=1
)

processed_df.to_csv(
    "data/processed_surfactants.csv",
    index=False
)

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

X_desc = X_combined

#==============================================
# 2. Fingerprints Only
#==============================================
fp_cols = [
    col for col in df.columns
    if col.startswith("FP_")
]

X_fp = df[fp_cols]
#print(f'Length of fingerprint: {len(X_fp)}')

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
randomState = 20
X_train, X_test, y_train, y_test = train_test_split(
    X_combined,
    y,
    test_size = 0.2,
    random_state = randomState
)

# Save feature column order
joblib.dump(
    X_combined.columns.tolist(),
    "models/feature_columns.pkl"
)

#==============================================
# Scale data so that there are equal contributions from each factor
#==============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#==============================================
# Model 1: Linear Regression
#==============================================
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

y_pred_lr = lr_model.predict(X_test_scaled)

#==============================================
# Model 2: Random Forest
#==============================================
rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=randomState
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)


#print("\nTraining predictions:")
#print(y_pred_rf[:10])

#print("\nActual values:")
#print(y_train.iloc[:10].values)

#==============================================
# Model 3: XGBoost
#==============================================
xgb_model = XGBRegressor(
    #Each tree = attempt to improve previous
    n_estimators=300, #300 trees
    learning_rate=0.05, #How aggressively each new tree updates prediction
    max_depth=6, #Complexity of each tree
    random_state=randomState
)

xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

#==============================================
# Method Accuracy Comparison
#==============================================
def evaluate_method(y_true, y_pred, model_name):
    #Root Mean Square Error
    # Lower RMSE = better
    rmse = np.sqrt(mean_squared_error(y_true,y_pred))

    #Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)

    #Variance
    r2 = r2_score(y_true, y_pred)

    return {
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

results_lr = evaluate_method(y_test, y_pred_lr, 'Linear Regression')
results_rf = evaluate_method(y_test, y_pred_rf, 'Random Forest')
results_xgb = evaluate_method(y_test, y_pred_xgb, 'XGBoost')

#Plot comparison
#Data
model_methods = ['Linear Regression', 'Random Forest', 'XGBoost']
RMSE_comparison = [results_lr["RMSE"], results_rf["RMSE"], results_xgb["RMSE"]]
MAE_comparison = [results_lr["MAE"], results_rf["MAE"], results_xgb["MAE"]]
R2_comparison = [results_lr["R2"], results_rf["R2"], results_xgb["R2"]]

width = 0.2
bar1 = np.arange(len(model_methods))
bar2 = [i+width for i in bar1]
bar3 = [i+width for i in bar2]

#MEthod comparison
plt.bar(bar1, RMSE_comparison, width, label = 'RMSE')
plt.bar(bar2, MAE_comparison, width, label = 'MAE')
plt.bar(bar3, R2_comparison, width, label = 'R2')

plt.xlabel('Methods')
plt.ylabel('Values')
plt.legend()
plt.xticks(bar1+width, model_methods)
plt.title('RMSE, MAE, and R2 for each Method')
plt.tight_layout()

plt.savefig(
    "figures/method_comparison.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

#====================================
# Parity Plot
#====================================
#Linear Regression
plt.figure(figsize=(6,6))

plt.scatter(
    y_test,
    y_pred_lr
)

plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    linestyle="--"
)

plt.xlabel("Actual pCMC")
plt.ylabel("Predicted pCMC")

plt.title("Linear Regression: Predicted vs Actual")

plt.tight_layout()

plt.savefig(
    "figures/parity_plot_lr.png",
    dpi=300
)

plt.show()

#Random Forest
plt.figure(figsize=(6,6))

plt.scatter(
    y_test,
    y_pred_rf
)

plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    linestyle="--"
)

plt.xlabel("Actual pCMC")
plt.ylabel("Predicted pCMC")

plt.title("Random Forest: Predicted vs Actual")

plt.tight_layout()

plt.savefig(
    "figures/parity_plot_rf.png",
    dpi=300
)

plt.show()

#XGB
plt.figure(figsize=(6,6))

plt.scatter(
    y_test,
    y_pred_xgb
)

plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    linestyle="--"
)

plt.xlabel("Actual pCMC")
plt.ylabel("Predicted pCMC")

plt.title("XGBoost: Predicted vs Actual")

plt.tight_layout()

plt.savefig(
    "figures/parity_plot_XGB.png",
    dpi=300
)

plt.show()

#====================================
# Feature Importance
#====================================
#Random Forest
importance_rf = rf_model.feature_importances_

feat_imp_rf = pd.Series(
    importance_rf,
    index=X_train.columns
).sort_values(ascending=False)

top_features_rf = feat_imp_rf.head(15)

plt.figure(figsize=(8,6))

top_features_rf.sort_values().plot(kind="barh")

plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")

plt.tight_layout()

plt.savefig(
    "figures/feature_importance_rf.png",
    dpi=300
)

plt.show()

#XGBoost
importance_xgb = xgb_model.feature_importances_

feat_imp_xgb = pd.Series(
    importance_xgb,
    index=X_train.columns
).sort_values(ascending=False)

top_features_xgb = feat_imp_xgb.head(15)

plt.figure(figsize=(8,6))

top_features_xgb.sort_values().plot(kind="barh")

plt.xlabel("Importance")
plt.title("XGBoost Feature Importance")

plt.tight_layout()

plt.savefig(
    "figures/feature_importance_xgb.png",
    dpi=300
)

plt.show()


#============================================
# Save Model and Scaler
#============================================
joblib.dump(
    xgb_model,
    "models/trained_model_xgb.pkl"
)

joblib.dump(
    rf_model,
    "models/trained_model_rf.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

metrics = [results_lr, results_rf, results_xgb]
metrics_df = pd.DataFrame(metrics)

metrics_df.to_csv(
    "data/model_metrics.csv",
    index=False
)


