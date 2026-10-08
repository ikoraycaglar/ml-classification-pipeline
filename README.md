# Machine Learning Classification Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-darkblue?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> **Author:** İsmet Koray Çağlar

---

## 📌 Project Overview

This repository contains an end-to-end, production-ready **Machine Learning Classification Pipeline** developed for the **SENG 264** course assignment. 

The project addresses a multi-class tabular classification problem with:
- **Leakage-free Data Preprocessing:** Automated column standardization, training-median imputation, and robust Interquartile Range (IQR) boundary clipping.
- **Ensemble Learning Model:** A fine-tuned `RandomForestClassifier` with balanced subsample weighting to handle class distribution dynamics.
- **Stratified 5-Fold Cross-Validation:** Thorough evaluation measuring Accuracy, Macro F1-Score, and a composite score.
- **Ready-to-use Model Serialization & CLI:** Saved model binary (`model.joblib`) with a flexible inference script (`predict.py`) and demo data (`sample_test.csv`).

---

## 📂 Repository Structure

```text
machine-learning/
├── .gitignore               # Ignores bytecode, virtualenvs, IDE configs, temp outputs
├── README.md                # Detailed project documentation and usage guide
├── requirements.txt         # Core dependencies (numpy, pandas, scikit-learn, joblib)
├── model.joblib             # Serialized trained RandomForest model (~14.6 MB)
├── model_metadata.json      # Framework & model file specifications
├── preprocess.py            # Data cleaning, imputation, and outlier clipping pipeline
├── train.py                 # 5-Fold Stratified CV training and model export script
├── predict.py               # Command-line inference script for test predictions
├── sample_test.csv          # Sample input features for instant testing
└── students.txt             # Student verification and identification details
```

---

## ⚙️ Preprocessing Pipeline (`preprocess.py`)

The preprocessing module ensures input data is strictly compliant with model expectations while safeguarding against dirty, missing, or corrupted values:

| Step | Operation | Description |
| :--- | :--- | :--- |
| **1. Column Standardization** | `_standardize_columns(df)` | Validates and reorders input features to `['x1', 'x2', 'x3', 'x4']`. Features fallback recovery for unlabelled or partially formatted CSVs. |
| **2. Type Coercion** | `pd.to_numeric(..., errors='coerce')` | Coerces values to numerical types; invalid strings/characters become `NaN`. |
| **3. Infinity Handling** | `.replace([np.inf, -np.inf], np.nan)` | Neutralizes infinite values before statistical imputation. |
| **4. Median Imputation** | `.fillna(TRAIN_MEDIANS[col])` | Fills missing cells using **pre-computed medians from the training set** to prevent data leakage. |
| **5. IQR Outlier Clipping** | `.clip(lower=low, upper=high)` | Truncates extreme corrupt samples using broad safety limits derived from training distribution bounds. |

### Fitted Constants (Derived from Training Set)

```python
FEATURE_COLUMNS = ["x1", "x2", "x3", "x4"]

# Training medians for imputation
TRAIN_MEDIANS = {
    "x1": -0.0063448448419451495,
    "x2": 29.723466658654885,
    "x3": 2472.0620640620873,
    "x4": 0.5114326911209858,
}

# IQR-based safe clipping bounds
CLIP_LIMITS = {
    "x1": (-2.7143240324879514, 2.6879353690981267),
    "x2": (-2598.480987782774, 2625.881370124487),
    "x3": (-2600.989154365313, 7602.777583673671),
    "x4": (0.0, 1.0),
}
```

---

## 🧠 Machine Learning Model (`train.py`)

The classification engine utilizes an ensemble **Random Forest Classifier**:

```python
RandomForestClassifier(
    n_estimators=500,
    max_features=None,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)
```

### Architectural Decisions:
- **`n_estimators = 500`:** High ensemble size to stabilize variance and minimize prediction jitter.
- **`max_features = None`:** Evaluates all 4 features per split, optimal for low-dimensional tabular data.
- **`min_samples_leaf = 2`:** Adds mild regularization, preventing leaves from isolating single noisy observations.
- **`class_weight = "balanced_subsample"`:** Dynamically adjusts class weights based on the bootstrap sample for each individual tree, addressing class imbalance.
- **`random_state = 42`:** Guarantees deterministic, reproducible results across training runs.

### Validation Scheme
The training procedure evaluates generalization capability via **5-Fold Stratified Cross-Validation (`StratifiedKFold`)**:
- Preserves the target class ratio in every validation fold.
- Computes **Accuracy**, **Macro F1-Score**, and **Composite Metric** (`0.5 * Accuracy + 0.5 * Macro_F1`).
- Trains final model on the full preprocessed dataset and exports `model.joblib`.

---

## 🚀 Installation & Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/ikoraycaglar/machine-learning.git
cd machine-learning
```

### 2. Set Up Virtual Environment (Recommended)
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### A. Run Inference / Predict (`predict.py`)

You can generate predictions immediately using the pre-trained model and included sample data:

```bash
# Run prediction using sample_test.csv
python predict.py --input sample_test.csv --output predictions.csv
```

**Output:**
- Creates `predictions.csv` containing predicted class labels and class probability distributions (`proba_class_0`, `proba_class_1`, `proba_class_2`).
- Prints summary statistics and class distribution directly to the console.

**Options:**
- `--input` / `-i`: Path to the input CSV file containing features `x1, x2, x3, x4`.
- `--output` / `-o`: Path to save prediction results (default: `predictions.csv`).
- `--model` / `-m`: Path to custom model file (default: `model.joblib`).

### B. Preprocessing Only (`preprocess.py`)

To transform raw feature data into preprocessed features:

```bash
# Process default X_test.csv or sample_test.csv
python preprocess.py

# Or specify custom input and output paths
python preprocess.py custom_input.csv custom_preprocessed.csv
```

### C. Retraining the Model (`train.py`)

If you have the original training datasets (`X_train.csv` and `y_train.csv`), you can retrain the model from scratch:

```bash
python train.py
```

*Note: `X_train.csv` and `y_train.csv` are private course dataset files and are not included in public distribution. The pre-trained model (`model.joblib`) is already packaged and ready for inference.*

---

## 📊 Model Metadata & Git LFS Note

The model specification is logged in `model_metadata.json`:
```json
{
  "framework": "sklearn",
  "model_file": "model.joblib"
}
```

> **GitHub Storage Note:** `model.joblib` is ~14.6 MB in size, which is well within GitHub's standard 100 MB file limit. It can be tracked and pushed directly without requiring external object storage or Git LFS.


## 📜 License

This project is released under the [MIT License](LICENSE).
