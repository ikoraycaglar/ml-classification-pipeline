"""
Training script used to produce model.joblib.

Expected files in the same directory while training:
    X_train.csv
    y_train.csv

The submitted ZIP does not include the training CSV files; this script is
included to show the exact training procedure.
"""

import os
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from preprocess import preprocess


RANDOM_STATE = 42
MODEL_FILE = "model.joblib"


def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        max_features=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def main() -> None:
    if not os.path.exists("X_train.csv") or not os.path.exists("y_train.csv"):
        print("Notice: 'X_train.csv' or 'y_train.csv' was not found.")
        print("Training requires both 'X_train.csv' and 'y_train.csv'.")
        print("The pre-trained model is already serialized and ready as 'model.joblib'.")
        print("To run inference, run: python predict.py")
        return

    X_raw = pd.read_csv("X_train.csv")
    y = pd.read_csv("y_train.csv")["label"]
    X = preprocess(X_raw)

    # Validation report for transparency.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    acc_scores = []
    f1_scores = []
    for train_idx, valid_idx in cv.split(X, y):
        fold_model = build_model()
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = fold_model.predict(X.iloc[valid_idx])
        acc_scores.append(accuracy_score(y.iloc[valid_idx], pred))
        f1_scores.append(f1_score(y.iloc[valid_idx], pred, average="macro"))

    mean_acc = sum(acc_scores) / len(acc_scores)
    mean_f1 = sum(f1_scores) / len(f1_scores)
    print(f"5-fold Accuracy : {mean_acc:.5f}")
    print(f"5-fold Macro F1 : {mean_f1:.5f}")
    print(f"5-fold Composite: {(0.5 * mean_acc + 0.5 * mean_f1):.5f}")

    model = build_model()
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)

    metadata = {"framework": "sklearn", "model_file": MODEL_FILE}
    with open("model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    print(f"Saved {MODEL_FILE} and model_metadata.json")


if __name__ == "__main__":
    main()
