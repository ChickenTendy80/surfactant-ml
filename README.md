# Surfactant-ML

Machine learning framework for predicting surfactant critical micelle concentration (CMC) from molecular structure using RDKit descriptors, engineered surfactant features, and XGBoost.

The project includes:

* molecular descriptor generation
* surfactant-specific feature engineering
* ML model training
* molecular similarity search
* Streamlit web app for interactive prediction

---

# Features

* Predict pCMC and CMC directly from SMILES
* Draw molecules interactively in browser
* Compare predictions against literature surfactants
* Find most chemically similar surfactant
* Visualize molecular structures
* Train custom ML models on surfactant datasets

---

# Project Structure

```text
surfactant-ml/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   │   └── raw_surfactant.csv
│   │
│   └── processed/
│
├── figures/
│
├── models/
│   ├── trained_model.pkl
│   ├── feature_columns.pkl
│
├── notebooks/
│   ├── 01_rdkit_descriptors.ipynb
│   ├── 02_pure_cmc_model.ipynb
│   ├── 03_binary_mixture_model.ipynb
│
├── src/
│   ├── descriptors.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── predict.py
│
├── README.md
│
└── requirements.txt
```

---

# Installation

Create a conda environment:

```bash
conda create -n surfactantml python=3.11
conda activate surfactantml
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Required Packages

Core packages:

```text
streamlit
streamlit-ketcher
pandas
numpy
scikit-learn
xgboost
joblib
matplotlib
rdkit
Pillow
```

---

# Dataset Format

Expected CSV columns:

| Column          | Description                            |
| --------------- | -------------------------------------- |
| surfactant_name | Name of surfactant                     |
| smiles          | Molecular SMILES                       |
| pCMC            | Experimental pCMC                      |
| ionic_type      | anionic/cationic/nonionic/zwitterionic |
| temperature     | Experimental temperature               |
| HLB             | Hydrophilic-lipophilic balance         |

Example:

```csv
surfactant_name,smiles,pCMC,ionic_type
SDS,CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+],2.09,anionic
CTAB,CCCCCCCCCCCCCCCC[N+](C)(C)C.[Br-],3.05,cationic
```

---

# Feature Engineering

The project generates:

## RDKit descriptors

* MolWt
* MolLogP
* TPSA
* NumHDonors
* NumHAcceptors
* RingCount
* HeavyAtomCount
* RotatableBonds

## Engineered surfactant features

* aromatic_flag
* tail_length
* ionic_encoded
* ethoxylate_count

## Morgan fingerprints

Circular molecular fingerprints for structural similarity and ML input.

---

# Model Training

Train models using:

```bash
python src/train_model.py
```

Models implemented:

* Linear Regression
* Random Forest
* XGBoost

Evaluation metrics:

* RMSE
* MAE
* R²

Generated outputs:

* trained_model.pkl
* feature_columns.pkl
* model_metrics.csv

---

# Prediction

Run prediction directly:

```bash
python src/predict.py
```

Example:

```text
Enter SMILES:
CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]

Enter ionic type:
anionic
```

Output:

```text
Predicted pCMC: 2.135
Predicted CMC: 7.3244 mM
```

---

# Streamlit Web App

Launch the interactive web interface:

```bash
streamlit run app/streamlit_app.py
```

Features:

* draw molecules interactively
* automatic SMILES generation
* pCMC prediction
* CMC conversion
* literature comparison
* similarity search
* molecular visualization

---

# Molecular Similarity Search

The app performs:

* Morgan fingerprint generation
* Tanimoto similarity comparison
* nearest-neighbor literature lookup

This allows:

* prediction validation
* confidence estimation
* literature benchmarking

---

# Example Results

| Molecule | Experimental pCMC | Predicted pCMC |
| -------- | ----------------: | -------------: |
| SDS      |              2.09 |           2.14 |
| CTAB     |              3.05 |           3.05 |
| Tween 80 |              4.92 |           4.92 |

---

# Future Development

Planned extensions:

## Phase 4

Binary surfactant mixture prediction

## Phase 5

Inverse surfactant design

## Phase 6

Graph neural network models

## Phase 7

Surfactant formulation optimization

---

# Notes

* XGBoost and Random Forest should NOT use feature scaling during prediction.
* Ensure prediction feature order matches training feature order.
* Always save feature columns during training.

---

# Author

Ryan Kim

---

# License

MIT License
