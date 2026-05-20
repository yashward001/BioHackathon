"""
PolyClear — PCOS Classification Model
======================================
Dataset : PCOS data without infertility (541 patients, 44 raw features)
Model   : XGBoost (falls back to sklearn GradientBoosting if xgboost not installed)
Target  : PCOS (Y/N) — binary classification
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    RocCurveDisplay, ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBClassifier
    USE_XGB = True
    print("✓ XGBoost found — using XGBClassifier")
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    USE_XGB = False
    print("⚠ XGBoost not installed — falling back to sklearn GradientBoostingClassifier")
    print("  To install: pip install xgboost")


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

FILE_PATH = "/Users/yash/Downloads/bio/(Main_Dataset)_PCOS_data_without_infertility.xlsx"

df = pd.read_excel(FILE_PATH, sheet_name="Full_new")
print(f"\nRaw data loaded: {df.shape[0]} rows × {df.shape[1]} columns")


# ══════════════════════════════════════════════════════════════════════════════
# 2. CLEAN
# ══════════════════════════════════════════════════════════════════════════════

# Drop admin/ID columns — not predictive
df = df.drop(columns=["Sl. No", "Patient File No."])

# Coerce everything to numeric (handles dirty cells like AMH='a', '1.99.')
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Cap extreme outliers at 99th percentile for skewed lab values
SKEWED_LABS = [
    "FSH(mIU/mL)", "LH(mIU/mL)",
    "  I   beta-HCG(mIU/mL)", "II    beta-HCG(mIU/mL)",
    "PRL(ng/mL)", "Vit D3 (ng/mL)", "PRG(ng/mL)"
]
for col in SKEWED_LABS:
    if col in df.columns:
        cap = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=cap)

# Impute binary flags with mode, numeric with median
BINARY_COLS = [
    "Weight gain(Y/N)", "hair growth(Y/N)", "Skin darkening (Y/N)",
    "Hair loss(Y/N)", "Pimples(Y/N)", "Fast food (Y/N)",
    "Reg.Exercise(Y/N)", "Pregnant(Y/N)"
]
for col in BINARY_COLS:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

for col in df.select_dtypes(include="number").columns:
    df[col] = df[col].fillna(df[col].median())

print(f"After cleaning — missing values: {df.isnull().sum().sum()}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

# ── Rotterdam Criterion 1: Hyperandrogenism ───────────────────────────────────
df["lh_fsh_ratio"]           = df["LH(mIU/mL)"] / (df["FSH(mIU/mL)"] + 1e-6)
df["c1_lh_fsh_flag"]         = (df["lh_fsh_ratio"] > 2.0).astype(int)
df["hyperandrogenism_score"] = (
    df["hair growth(Y/N)"] + df["Pimples(Y/N)"] + df["c1_lh_fsh_flag"]
)  # 0-3 composite androgenic signal count

# ── Rotterdam Criterion 2: Ovulatory Dysfunction ─────────────────────────────
df["cycle_irregular"]        = (df["Cycle(R/I)"] >= 4).astype(int)
df["anovulation_flag"]       = (df["PRG(ng/mL)"] < 3.0).astype(int)
df["c2_positive"]            = (
    (df["cycle_irregular"] == 1) | (df["anovulation_flag"] == 1)
).astype(int)

# ── Rotterdam Criterion 3: Polycystic Morphology ──────────────────────────────
df["total_follicles"]        = df["Follicle No. (L)"] + df["Follicle No. (R)"]
df["avg_follicle_size"]      = (
    df["Avg. F size (L) (mm)"] + df["Avg. F size (R) (mm)"]
) / 2
df["follicle_size_in_range"] = (
    (df["avg_follicle_size"] >= 2) & (df["avg_follicle_size"] <= 9)
).astype(int)
df["amh_elevated"]           = (df["AMH(ng/mL)"] > 4.59).astype(int)
df["c3_morphology_flag"]     = (
    (df["total_follicles"] >= 20) | (df["amh_elevated"] == 1)
).astype(int)

# ── Rotterdam Composite Score (0–3) ───────────────────────────────────────────
# This is the single most important feature — mirrors the clinical diagnostic gate
df["rotterdam_score"]        = (
    (df["hyperandrogenism_score"] > 0).astype(int)
    + df["c2_positive"]
    + df["c3_morphology_flag"]
)

# ── Metabolic Burden ─────────────────────────────────────────────────────────
df["bmi_category"]           = pd.cut(
    df["BMI"], bins=[0, 18.5, 24.9, 29.9, 200], labels=[0, 1, 2, 3]
).astype(float)
df["central_obesity_flag"]   = (df["Waist:Hip Ratio"] > 0.85).astype(int)
df["metabolic_score"]        = (
    (df["BMI"] >= 25).astype(int)
    + (df["RBS(mg/dl)"] >= 110).astype(int)
    + df["Skin darkening (Y/N)"]
    + df["Weight gain(Y/N)"]
    + df["Fast food (Y/N)"]
)  # 0–5 burden score

# ── Exclusion / Mimic Flags ───────────────────────────────────────────────────
df["thyroid_flag"]           = (
    (df["TSH (mIU/L)"] < 0.4) | (df["TSH (mIU/L)"] > 4.0)
).astype(int)
df["hyperprolactin_flag"]    = (df["PRL(ng/mL)"] > 25).astype(int)
df["vitd_deficient"]         = (df["Vit D3 (ng/mL)"] < 20).astype(int)

# ── Interaction Features ──────────────────────────────────────────────────────
df["follicles_x_amh"]        = df["total_follicles"] * df["AMH(ng/mL)"]
df["bmi_x_waist"]            = df["BMI"] * df["Waist(inch)"]
df["androgens_x_cycle"]      = df["hyperandrogenism_score"] * df["cycle_irregular"]


# ══════════════════════════════════════════════════════════════════════════════
# 4. SELECT FEATURES & TARGET
# ══════════════════════════════════════════════════════════════════════════════

TARGET = "PCOS (Y/N)"

# Drop raw columns that have been replaced by engineered versions,
# plus confounders that aren't clinically diagnostic
DROP = [
    TARGET,
    "Cycle(R/I)",                               # → cycle_irregular
    "Follicle No. (L)", "Follicle No. (R)",     # → total_follicles
    "Avg. F size (L) (mm)", "Avg. F size (R) (mm)",  # → avg_follicle_size
    "  I   beta-HCG(mIU/mL)", "II    beta-HCG(mIU/mL)",  # pregnancy exclusion
    "Pregnant(Y/N)", "No. of abortions", "Marraige Status (Yrs)",
]

X = df.drop(columns=[c for c in DROP if c in df.columns])
y = df[TARGET]

print(f"\nFeature set: {X.shape[1]} features | {X.shape[0]} samples")
print(f"Class balance — PCOS: {y.sum()} ({y.mean():.1%}) | Non-PCOS: {(1-y).sum()} ({(1-y.mean()):.1%})")


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,        # 80/20 split
    random_state=42,
    stratify=y             # preserve class ratio in both splits
)

print(f"\nTrain: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
print(f"Train PCOS rate: {y_train.mean():.1%} | Test PCOS rate: {y_test.mean():.1%}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. BUILD MODEL
# ══════════════════════════════════════════════════════════════════════════════

# Class imbalance weight: ~2.06x more weight on PCOS-positive samples
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

if USE_XGB:
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,   # handles class imbalance
        eval_metric="auc",
        random_state=42,
        verbosity=0,
        use_label_encoder=False
    )
else:
    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42
    )


# ══════════════════════════════════════════════════════════════════════════════
# 7. CROSS-VALIDATION ON TRAIN SET
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 5-Fold Stratified Cross-Validation (train set) ──")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for metric in ["roc_auc", "f1", "accuracy"]:
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric)
    print(f"  {metric:12s}: {scores.mean():.4f} ± {scores.std():.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. FIT ON FULL TRAIN SET & EVALUATE ON HELD-OUT TEST SET
# ══════════════════════════════════════════════════════════════════════════════

model.fit(X_train, y_train)

y_pred       = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
test_auc     = roc_auc_score(y_test, y_pred_proba)

print("\n── Test Set Results ──")
print(f"  ROC-AUC : {test_auc:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Non-PCOS', 'PCOS'])}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════

if USE_XGB:
    importances = pd.Series(model.feature_importances_, index=X.columns)
else:
    importances = pd.Series(model.feature_importances_, index=X.columns)

importances = importances.sort_values(ascending=False)

print("\n── Top 15 Feature Importances ──")
print(importances.head(15).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# 10. PLOTS
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("PolyClear — PCOS Classification Model", fontsize=14, fontweight="bold")

# ── Plot 1: Feature Importances ───────────────────────────────────────────────
top_n = 15
top_features = importances.head(top_n)
colors = ["#16685a" if "rotterdam" in f or "follicle" in f or "morphology" in f
          else "#2f69a8" if "hyperandro" in f or "lh_fsh" in f or "c1" in f or "c2" in f
          else "#b5392f" if "metabolic" in f or "bmi" in f or "skin" in f
          else "#8e6bbf"
          for f in top_features.index]

axes[0].barh(top_features.index[::-1], top_features.values[::-1], color=colors[::-1])
axes[0].set_title(f"Top {top_n} Feature Importances")
axes[0].set_xlabel("Importance score")
axes[0].tick_params(axis="y", labelsize=9)

# ── Plot 2: ROC Curve ─────────────────────────────────────────────────────────
RocCurveDisplay.from_predictions(
    y_test, y_pred_proba,
    name=f"XGBoost (AUC = {test_auc:.3f})",
    ax=axes[1],
    color="#16685a"
)
axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
axes[1].set_title("ROC Curve — Test Set")
axes[1].legend(fontsize=9)

# ── Plot 3: Confusion Matrix ──────────────────────────────────────────────────
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Non-PCOS", "PCOS"],
    colorbar=False,
    cmap="Greens",
    ax=axes[2]
)
axes[2].set_title("Confusion Matrix — Test Set")

plt.tight_layout()
plt.savefig("pcos_model_results.png", dpi=150, bbox_inches="tight")
print("\n✓ Saved: pcos_model_results.png")


# ══════════════════════════════════════════════════════════════════════════════
# 11. SAVE PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════

test_results = X_test.copy()
test_results["true_label"]       = y_test.values
test_results["predicted_label"]  = y_pred
test_results["pcos_probability"] = y_pred_proba.round(4)
test_results.to_csv("pcos_test_predictions.csv", index=False)
print("✓ Saved: pcos_test_predictions.csv")

print("\nDone.")