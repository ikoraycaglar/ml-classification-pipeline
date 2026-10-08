"""
Inference script for SENG 264 Machine Learning Model.

Loads the pre-trained Random Forest model (model.joblib), applies the standard
preprocessing pipeline, and generates predictions for test data.

Usage:
    python predict.py
    python predict.py --input sample_test.csv --output predictions.csv
"""

import argparse
import os
import sys
import joblib
import pandas as pd
from preprocess import preprocess

MODEL_FILE = "model.joblib"


def predict(
    input_path: str,
    output_path: str = "predictions.csv",
    model_path: str = MODEL_FILE,
) -> pd.DataFrame:
    """
    Load model, preprocess input CSV, generate predictions, and save results.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        sys.exit(1)

    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    print(f"Reading input data from: {input_path}")
    raw_df = pd.read_csv(input_path)
    print(f"  Raw data shape: {raw_df.shape}")

    print("Preprocessing input features ...")
    clean_df = preprocess(raw_df)
    print(f"  Preprocessed features shape: {clean_df.shape}")

    print("Generating predictions ...")
    predictions = model.predict(clean_df)

    results_df = pd.DataFrame({"prediction": predictions})

    # Include prediction probabilities if supported by the model
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(clean_df)
            for i, class_label in enumerate(model.classes_):
                results_df[f"proba_class_{class_label}"] = proba[:, i]
        except Exception as e:
            print(f"Warning: Could not compute prediction probabilities: {e}")

    results_df.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] Predictions saved to: {output_path}")
    print(f"Total samples predicted: {len(predictions)}")
    print("\nClass distribution in predictions:")
    counts = pd.Series(predictions).value_counts().sort_index()
    for cls, cnt in counts.items():
        pct = (cnt / len(predictions)) * 100
        print(f"  Class {cls}: {cnt} samples ({pct:.1f}%)")

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Run inference using trained model (model.joblib)"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to input CSV file (default: X_test.csv, falls back to sample_test.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="predictions.csv",
        help="Path to save prediction results CSV (default: predictions.csv)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=MODEL_FILE,
        help="Path to serialized model file (default: model.joblib)",
    )

    args = parser.parse_args()

    input_file = args.input
    if not input_file:
        if os.path.exists("X_test.csv"):
            input_file = "X_test.csv"
        elif os.path.exists("sample_test.csv"):
            print("Notice: 'X_test.csv' not found. Using 'sample_test.csv' for demonstration.")
            input_file = "sample_test.csv"
        else:
            print("Error: No input CSV file found.")
            print("Please provide a CSV file using --input, e.g.:")
            print("    python predict.py --input sample_test.csv")
            sys.exit(1)

    predict(input_file, args.output, args.model)


if __name__ == "__main__":
    main()
