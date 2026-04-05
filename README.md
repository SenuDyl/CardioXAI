# CardioXAI

CardioXAI is an ensemble + explainable AI framework for binary heart disease detection using the UCI Cleveland heart disease dataset.

The project compares multiple modeling scenarios, builds a targeted weighted ensemble, and explains predictions with SHAP-based visualizations.

## Highlights

- End-to-end classical ML pipeline for heart disease prediction.
- Multiple scenarios for comparison:
	- baseline
	- feature engineering
	- PCA components
	- feature engineering + PCA
- Weighted soft-voting ensemble combining:
	- Random Forest (baseline setup)
	- SVM (feature-engineered setup)
	- KNN (PCA-augmented setup)
- Explainability with SHAP:
	- Tree-based SHAP for Random Forest
	- Kernel SHAP for the weighted ensemble

## Repository Structure

```text
CardioXAI/
|-- data_preprocessing.py       # Data loading and preprocessing pipelines
|-- EDA.py                      # Exploratory data analysis and visualization
|-- ensemble_model.py           # Targeted weighted soft-voting ensemble
|-- explainability.py           # SHAP analysis for baseline and ensemble models
|-- feature_engineering.py      # Scenario runner + synthetic feature generation
|-- tuning.py                   # Hyperparameter tuning workflow
|-- utils.py                    # Shared utility functions (CSV result saving)
|-- requirements.txt            # Python dependencies
|-- data/
|   |-- heart_disease_uci.csv
|   |-- Heart Disease UCI/
|       |-- processed.cleveland.data
|       |-- preprocessed_cleveland.csv
|       |-- ...
|-- eda_results/                # EDA outputs
|-- output/                     # Evaluation CSVs for scenarios/ensemble/tuning
|-- shap_plots/                 # SHAP figures
```

## Dataset

The code primarily uses UCI Cleveland heart disease data from:

- `data/Heart Disease UCI/processed.cleveland.data`
- Official source: https://archive.ics.uci.edu/dataset/45/heart+disease

Target variable construction in the project:

- Original severity: `num` in [0, 4]
- Binary target: `target = 1 if num > 0 else 0`

## Modeling Pipeline

### 1) Preprocessing

Implemented in `data_preprocessing.py`.

- Numeric preprocessing:
	- KNN imputation (`KNNImputer`, `n_neighbors=5`)
	- Optional scaling (`StandardScaler`) depending on model type
- Categorical preprocessing:
	- Most-frequent imputation
	- One-hot encoding (`OneHotEncoder`)
- Optional PCA branch:
	- Median imputation
	- Scaling before PCA
	- PCA on selected numeric features

### 2) Feature Engineering

Implemented in `feature_engineering.py`.

Engineered features include:

- `age_group`
- `bp_category`
- `chol_group`
- `hr_competence_ratio = thalch / (220 - age)`
- `severe_ischemia` (derived from `exang` and `oldpeak` threshold)

### 3) Models

Main classical models used across scenarios:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

### 4) Ensemble Strategy

Implemented in `ensemble_model.py`.

Weighted soft-voting probabilities are combined as:

- RF weight = 1.0
- SVM weight = 0.5
- KNN weight = 2.0

## Explainability (XAI)

Implemented in `explainability.py`.

- Baseline Random Forest explanation:
	- `shap.TreeExplainer`
- Weighted ensemble explanation:
	- `shap.KernelExplainer`
	- Produces top directional feature impact plots

Generated SHAP plots are stored in `shap_plots/`.

## Installation

### 1) Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

Dependencies listed in `requirements.txt`:

- pandas
- numpy
- matplotlib
- shap
- scikit-learn
- xgboost
- seaborn

## How To Run

Run each module from the project root.

### Baseline tuning run

```powershell
python baseline.py
```

### Scenario comparison (baseline / FE / PCA / FE+PCA)

```powershell
python feature_engineering.py
```

### Hyperparameter tuning workflow

```powershell
python tuning.py
```

### Targeted weighted ensemble

```powershell
python ensemble_model.py
```

### Explainability plots (RF + ensemble SHAP)

```powershell
python explainability.py
```

### Exploratory data analysis

```powershell
python EDA.py
```

## Outputs

Typical metrics include:

- Accuracy
- Precision
- Recall
- F1 score

### `eda_results/`

Stores EDA figures (histograms, scatter/box plots, heatmaps).

### `shap_plots/`

Stores SHAP-based explanation figures for healthy vs disease influence patterns.

## Reproducibility Notes

- Most scripts use `RANDOM_STATE = 42` for deterministic train/test splits.
- Ensure you run from repository root so relative data paths resolve correctly.
- Some scripts assume Cleveland subset fields and specific UCI file naming.

## Known Notes

- `test.py` and `test2.py` appear to be prototype/experimental scripts and are not the main pipeline.
- The active production-style workflows are in:
	- `data_preprocessing.py`
	- `feature_engineering.py`
	- `ensemble_model.py`
	- `explainability.py`
	- `tuning.py`

## Future Improvements

- Add unit tests for preprocessing and feature generation logic.
- Add a single experiment runner CLI (for scenario selection).
- Add model artifact persistence (joblib/pickle) and inference script.
- Add confidence interval reporting and calibration metrics.

## License

No license file is currently included in this repository. Add a `LICENSE` file if you plan to distribute this project.
