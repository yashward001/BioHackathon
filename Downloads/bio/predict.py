"""
predict.py — Single-patient PCOS prediction CLI.

All 36 raw feature flags are optional; omitted values are imputed from
training-set medians (or mode for binary flags).

Example — strong PCOS profile:
  python predict.py --age 28 --amh 7.2 --follicle_l 13 --follicle_r 12 \\
      --lh 10.5 --fsh 5.2 --cycle_ri 4 --prg 1.8 \\
      --hair_growth 1 --pimples 1 --weight_gain 1 \\
      --bmi 26.3 --waist 34.0 --waist_hip_ratio 0.87 \\
      --rbs 115 --vit_d3 15.0 --tsh 2.5
"""

import argparse
import sys
import pandas as pd
import joblib

from features import (
    RAW_INPUT_COLS, CLI_NAMES, CLI_HELP, CLI_TO_COL,
    apply_caps, apply_imputation, engineer_features, rotterdam_criteria_report,
)


def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Predict PCOS for a single patient.\n"
            "All flags are optional — missing values are imputed from training medians."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    for name in CLI_NAMES:
        p.add_argument(f"--{name}", type=float, default=None,
                       metavar="VAL", help=CLI_HELP.get(name, ""))
    return p


def main():
    args = build_parser().parse_args()

    try:
        model           = joblib.load("pcos_model.pkl")
        feature_columns = joblib.load("feature_columns.pkl")
        stats           = joblib.load("training_stats.pkl")
    except FileNotFoundError as e:
        sys.exit(f"Model artifacts missing — run train_and_save.py first.\n  {e}")

    # Build single-row DataFrame with raw column names
    row = {col: getattr(args, name) for name, col in CLI_TO_COL.items()}
    df = pd.DataFrame([row])

    # Coerce all to numeric (mirrors training pipeline)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Apply training-fitted caps and imputation
    df = apply_caps(df, stats["caps"])
    df = apply_imputation(df, stats["medians"], stats["modes"])

    # Feature engineering -> 52-col X
    X_full = engineer_features(df)
    X_model = X_full.reindex(columns=feature_columns, fill_value=0)

    # Predict
    label_num = int(model.predict(X_model)[0])
    prob      = float(model.predict_proba(X_model)[0, 1])
    label     = "PCOS" if label_num == 1 else "Non-PCOS"

    # Rotterdam criteria from the engineered row
    fired = rotterdam_criteria_report(X_full.iloc[0])

    # Report
    print()
    print("=" * 54)
    print(f"  Prediction      : {label}")
    print(f"  PCOS Probability: {prob:.1%}")
    print(f"  Rotterdam score : {len(fired)}/3 criteria met")
    if fired:
        for c in fired:
            print(f"    + {c}")
    else:
        print("    — No criteria fired")

    # Provide an indication of provided vs imputed features
    provided = [name for name in CLI_NAMES if getattr(args, name) is not None]
    missing  = [name for name in CLI_NAMES if getattr(args, name) is None]
    print(f"\n  Features provided : {len(provided)}/{len(CLI_NAMES)}")
    if missing:
        print(f"  Imputed (median)  : {', '.join(missing)}")
    print("=" * 54)
    print()


if __name__ == "__main__":
    main()
