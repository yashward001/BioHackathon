"""
predict_batch.py — Batch PCOS prediction from a CSV file.

The input CSV may use either:
  - Original raw column names  (e.g. " Age (yrs)", "FSH(mIU/mL)")
  - CLI short names            (e.g. "age", "fsh")

Missing columns and NaN cells are imputed from training medians.
Results are written to batch_predictions.csv.

Usage:
  python predict_batch.py <input.csv>
  python predict_batch.py <input.csv> --output my_results.csv
"""

import argparse
import sys
import pandas as pd
import joblib

from features import (
    RAW_INPUT_COLS, CLI_NAMES, CLI_TO_COL,
    apply_caps, apply_imputation, engineer_features,
)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_csv", help="Path to input CSV file")
    p.add_argument("--output", default="batch_predictions.csv",
                   help="Output CSV path (default: batch_predictions.csv)")
    args = p.parse_args()

    try:
        model           = joblib.load("pcos_model.pkl")
        feature_columns = joblib.load("feature_columns.pkl")
        stats           = joblib.load("training_stats.pkl")
    except FileNotFoundError as e:
        sys.exit(f"Model artifacts missing — run train_and_save.py first.\n  {e}")

    try:
        df = pd.read_csv(args.input_csv)
    except Exception as e:
        sys.exit(f"Could not read {args.input_csv}: {e}")

    print(f"Loaded {len(df)} rows from {args.input_csv}")

    # Normalise column names: accept CLI short names OR raw names
    cli_to_col = {cli: raw for cli, raw in CLI_TO_COL.items()}
    df = df.rename(columns={cli: raw for cli, raw in cli_to_col.items()
                             if cli in df.columns and raw not in df.columns})

    # Keep only recognised raw input columns; add missing ones as NaN
    for col in RAW_INPUT_COLS:
        if col not in df.columns:
            df[col] = float("nan")
    df = df[RAW_INPUT_COLS].copy()

    # Coerce to numeric (handles stray strings)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Apply training-fitted caps and imputation
    df = apply_caps(df, stats["caps"])
    df = apply_imputation(df, stats["medians"], stats["modes"])

    # Feature engineering
    X_full  = engineer_features(df)
    X_model = X_full.reindex(columns=feature_columns, fill_value=0)

    # Predict
    labels_num  = model.predict(X_model)
    proba       = model.predict_proba(X_model)[:, 1]

    # Build output: input features + predictions
    out = df.copy()
    out["predicted_label"]  = ["PCOS" if v == 1 else "Non-PCOS" for v in labels_num]
    out["pcos_probability"] = proba.round(4)

    out.to_csv(args.output, index=False)

    n_pcos     = (labels_num == 1).sum()
    n_nonpcos  = (labels_num == 0).sum()
    print(f"Results: {n_pcos} PCOS  |  {n_nonpcos} Non-PCOS  |  "
          f"mean probability {proba.mean():.3f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
