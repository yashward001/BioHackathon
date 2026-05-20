"""
train_and_save.py
=================
Same pipeline as model.py, but also pickles three artifacts:
  pcos_model.pkl       — trained XGBoost classifier
  feature_columns.pkl  — ordered list of 52 feature names
  training_stats.pkl   — outlier caps + imputation medians/modes for predict-time use
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score,
    RocCurveDisplay, ConfusionMatrixDisplay,
)

try:
    from xgboost import XGBClassifier
    USE_XGB = True
    print("XGBoost found — using XGBClassifier")
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    USE_XGB = False
    print("XGBoost not installed — falling back to GradientBoostingClassifier")

from features import (
    TARGET, SKEWED_LABS, BINARY_COLS,
    fit_caps, apply_caps, fit_imputation, apply_imputation, engineer_features,
)

FILE_PATH = "(Main_Dataset)_PCOS_data_without_infertility.xlsx"

# 1. Load
df = pd.read_excel(FILE_PATH, sheet_name="Full_new")
print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")

df = df.drop(columns=["Sl. No", "Patient File No."])
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 2. Fit outlier caps and imputation stats, then apply
caps = fit_caps(df)
df = apply_caps(df, caps)

medians, modes = fit_imputation(df)
df = apply_imputation(df, medians, modes)

print(f"Missing values after cleaning: {df.isnull().sum().sum()}")

# 3. Feature engineering
X = engineer_features(df)
y = df[TARGET]

print(f"Features: {X.shape[1]}  |  Samples: {X.shape[0]}")
print(f"PCOS: {y.sum()} ({y.mean():.1%})  |  Non-PCOS: {(~y.astype(bool)).sum()}")

# 4. Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y,
)
print(f"Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

# 5. Build model
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

if USE_XGB:
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="auc",
        random_state=42,
        verbosity=0,
    )
else:
    model = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05,
        max_depth=4, subsample=0.8, random_state=42,
    )

# 6. Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("\n5-Fold Stratified CV (train set):")
for metric in ["roc_auc", "f1", "accuracy"]:
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric)
    print(f"  {metric:12s}: {scores.mean():.4f} +/- {scores.std():.4f}")

# 7. Fit on full train set
model.fit(X_train, y_train)
y_pred       = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
test_auc     = roc_auc_score(y_test, y_pred_proba)

print(f"\nTest ROC-AUC: {test_auc:.4f}")
print(classification_report(y_test, y_pred, target_names=["Non-PCOS", "PCOS"]))

# 8. Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("PolyClear — PCOS Classification Model", fontsize=14, fontweight="bold")

importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
top = importances.head(15)
colors = [
    "#16685a" if any(k in f for k in ("rotterdam", "follicle", "morphology")) else
    "#2f69a8" if any(k in f for k in ("hyperandro", "lh_fsh", "c1", "c2")) else
    "#b5392f" if any(k in f for k in ("metabolic", "bmi", "skin")) else
    "#8e6bbf"
    for f in top.index
]
axes[0].barh(top.index[::-1], top.values[::-1], color=colors[::-1])
axes[0].set_title("Top 15 Feature Importances")
axes[0].set_xlabel("Importance score")
axes[0].tick_params(axis="y", labelsize=9)

RocCurveDisplay.from_predictions(
    y_test, y_pred_proba,
    name=f"XGBoost (AUC={test_auc:.3f})", ax=axes[1], color="#16685a",
)
axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
axes[1].set_title("ROC Curve — Test Set")

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=["Non-PCOS", "PCOS"],
    colorbar=False, cmap="Greens", ax=axes[2],
)
axes[2].set_title("Confusion Matrix — Test Set")

plt.tight_layout()
plt.savefig("pcos_model_results.png", dpi=150, bbox_inches="tight")
print("Saved: pcos_model_results.png")

# 9. Save test predictions CSV (same as model.py)
test_results = X_test.copy()
test_results["true_label"]       = y_test.values
test_results["predicted_label"]  = y_pred
test_results["pcos_probability"] = y_pred_proba.round(4)
test_results.to_csv("pcos_test_predictions.csv", index=False)
print("Saved: pcos_test_predictions.csv")

# 10. Pickle model + feature list + training stats
feature_columns = list(X.columns)
joblib.dump(model,          "pcos_model.pkl")
joblib.dump(feature_columns,"feature_columns.pkl")
joblib.dump({"caps": caps, "medians": medians, "modes": modes}, "training_stats.pkl")

print("Saved: pcos_model.pkl")
print("Saved: feature_columns.pkl")
print("Saved: training_stats.pkl")
print("\nDone.")
