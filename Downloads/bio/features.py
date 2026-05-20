"""Shared feature engineering for the PCOS XGBoost model."""

import numpy as np
import pandas as pd

TARGET = "PCOS (Y/N)"

# Raw columns the user supplies (pre-engineering, original dataset names).
RAW_INPUT_COLS = [
    " Age (yrs)", "Weight (Kg)", "Height(Cm) ", "BMI", "Blood Group",
    "Pulse rate(bpm) ", "RR (breaths/min)", "Hb(g/dl)", "Cycle(R/I)",
    "Cycle length(days)", "FSH(mIU/mL)", "LH(mIU/mL)", "FSH/LH",
    "Hip(inch)", "Waist(inch)", "Waist:Hip Ratio", "TSH (mIU/L)",
    "AMH(ng/mL)", "PRL(ng/mL)", "Vit D3 (ng/mL)", "PRG(ng/mL)",
    "RBS(mg/dl)", "Weight gain(Y/N)", "hair growth(Y/N)",
    "Skin darkening (Y/N)", "Hair loss(Y/N)", "Pimples(Y/N)",
    "Fast food (Y/N)", "Reg.Exercise(Y/N)", "BP _Systolic (mmHg)",
    "BP _Diastolic (mmHg)", "Follicle No. (L)", "Follicle No. (R)",
    "Avg. F size (L) (mm)", "Avg. F size (R) (mm)", "Endometrium (mm)",
]

# CLI-safe names (1:1 with RAW_INPUT_COLS, same order)
CLI_NAMES = [
    "age", "weight_kg", "height_cm", "bmi", "blood_group",
    "pulse_rate", "rr", "hb", "cycle_ri",
    "cycle_length", "fsh", "lh", "fsh_lh_ratio",
    "hip", "waist", "waist_hip_ratio", "tsh",
    "amh", "prl", "vit_d3", "prg",
    "rbs", "weight_gain", "hair_growth",
    "skin_darkening", "hair_loss", "pimples",
    "fast_food", "reg_exercise", "bp_systolic",
    "bp_diastolic", "follicle_l", "follicle_r",
    "avg_f_size_l", "avg_f_size_r", "endometrium",
]

CLI_HELP = {
    "age":            "Age (years)",
    "weight_kg":      "Weight (kg)",
    "height_cm":      "Height (cm)",
    "bmi":            "Body Mass Index",
    "blood_group":    "Blood group numeric code — 11=A+, 12=A-, 13=B+, 14=B-, 15=O+, 16=O-, 17=AB+, 18=AB-",
    "pulse_rate":     "Pulse rate (bpm)",
    "rr":             "Respiratory rate (breaths/min)",
    "hb":             "Haemoglobin (g/dl)",
    "cycle_ri":       "Cycle regularity — 2=Regular, 4=Irregular",
    "cycle_length":   "Cycle length (days)",
    "fsh":            "FSH (mIU/mL)",
    "lh":             "LH (mIU/mL)",
    "fsh_lh_ratio":   "FSH/LH ratio (pre-computed; if omitted, derived from --fsh and --lh)",
    "hip":            "Hip circumference (inch)",
    "waist":          "Waist circumference (inch)",
    "waist_hip_ratio":"Waist:Hip ratio",
    "tsh":            "TSH (mIU/L)",
    "amh":            "AMH — Anti-Müllerian hormone (ng/mL)",
    "prl":            "Prolactin (ng/mL)",
    "vit_d3":         "Vitamin D3 (ng/mL)",
    "prg":            "Progesterone (ng/mL)",
    "rbs":            "Random blood sugar (mg/dl)",
    "weight_gain":    "Recent weight gain — 1=Yes, 0=No",
    "hair_growth":    "Excess hair growth (hirsutism) — 1=Yes, 0=No",
    "skin_darkening": "Skin darkening / acanthosis nigricans — 1=Yes, 0=No",
    "hair_loss":      "Hair loss / androgenic alopecia — 1=Yes, 0=No",
    "pimples":        "Pimples / acne — 1=Yes, 0=No",
    "fast_food":      "Regular fast food consumption — 1=Yes, 0=No",
    "reg_exercise":   "Regular exercise — 1=Yes, 0=No",
    "bp_systolic":    "Systolic blood pressure (mmHg)",
    "bp_diastolic":   "Diastolic blood pressure (mmHg)",
    "follicle_l":     "Antral follicle count — left ovary",
    "follicle_r":     "Antral follicle count — right ovary",
    "avg_f_size_l":   "Average follicle size left ovary (mm)",
    "avg_f_size_r":   "Average follicle size right ovary (mm)",
    "endometrium":    "Endometrium thickness (mm)",
}

# Mapping: CLI name -> raw column name
CLI_TO_COL = dict(zip(CLI_NAMES, RAW_INPUT_COLS))

SKEWED_LABS = [
    "FSH(mIU/mL)", "LH(mIU/mL)",
    "  I   beta-HCG(mIU/mL)", "II    beta-HCG(mIU/mL)",
    "PRL(ng/mL)", "Vit D3 (ng/mL)", "PRG(ng/mL)",
]

BINARY_COLS = [
    "Weight gain(Y/N)", "hair growth(Y/N)", "Skin darkening (Y/N)",
    "Hair loss(Y/N)", "Pimples(Y/N)", "Fast food (Y/N)",
    "Reg.Exercise(Y/N)", "Pregnant(Y/N)",
]

_DROP_AFTER_ENGINEERING = [
    TARGET,
    "Cycle(R/I)",
    "Follicle No. (L)", "Follicle No. (R)",
    "Avg. F size (L) (mm)", "Avg. F size (R) (mm)",
    "  I   beta-HCG(mIU/mL)", "II    beta-HCG(mIU/mL)",
    "Pregnant(Y/N)", "No. of abortions", "Marraige Status (Yrs)",
]


# ── Preprocessing helpers ────────────────────────────────────────────────────

def fit_caps(df):
    """Return dict of 99th-percentile caps fitted on df."""
    caps = {}
    for col in SKEWED_LABS:
        if col in df.columns:
            caps[col] = df[col].quantile(0.99)
    return caps


def apply_caps(df, caps):
    df = df.copy()
    for col, cap in caps.items():
        if col in df.columns:
            df[col] = df[col].clip(upper=cap)
    return df


def fit_imputation(df):
    """Return (medians, modes) dicts fitted on df."""
    modes = {}
    for col in BINARY_COLS:
        if col in df.columns and df[col].notna().any():
            modes[col] = float(df[col].mode()[0])
    medians = {
        col: float(df[col].median())
        for col in df.select_dtypes(include="number").columns
        if df[col].notna().any()
    }
    return medians, modes


def apply_imputation(df, medians, modes):
    df = df.copy()
    # Binary columns first (mode), then all numeric (median)
    for col, val in modes.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    for col in df.select_dtypes(include="number").columns:
        if col in medians:
            df[col] = df[col].fillna(medians[col])
    return df


# ── Feature engineering ──────────────────────────────────────────────────────

def engineer_features(df):
    """Apply all feature engineering and return the 52-column model-ready X."""
    df = df.copy()

    # Rotterdam Criterion 1 — Hyperandrogenism
    df["lh_fsh_ratio"]           = df["LH(mIU/mL)"] / (df["FSH(mIU/mL)"] + 1e-6)
    df["c1_lh_fsh_flag"]         = (df["lh_fsh_ratio"] > 2.0).astype(int)
    df["hyperandrogenism_score"] = (
        df["hair growth(Y/N)"] + df["Pimples(Y/N)"] + df["c1_lh_fsh_flag"]
    )

    # Rotterdam Criterion 2 — Ovulatory Dysfunction
    df["cycle_irregular"]  = (df["Cycle(R/I)"] >= 4).astype(int)
    df["anovulation_flag"] = (df["PRG(ng/mL)"] < 3.0).astype(int)
    df["c2_positive"]      = (
        (df["cycle_irregular"] == 1) | (df["anovulation_flag"] == 1)
    ).astype(int)

    # Rotterdam Criterion 3 — Polycystic Morphology
    df["total_follicles"]        = df["Follicle No. (L)"] + df["Follicle No. (R)"]
    df["avg_follicle_size"]      = (
        df["Avg. F size (L) (mm)"] + df["Avg. F size (R) (mm)"]
    ) / 2
    df["follicle_size_in_range"] = (
        (df["avg_follicle_size"] >= 2) & (df["avg_follicle_size"] <= 9)
    ).astype(int)
    df["amh_elevated"]       = (df["AMH(ng/mL)"] > 4.59).astype(int)
    df["c3_morphology_flag"] = (
        (df["total_follicles"] >= 20) | (df["amh_elevated"] == 1)
    ).astype(int)

    # Rotterdam Composite Score (0–3)
    df["rotterdam_score"] = (
        (df["hyperandrogenism_score"] > 0).astype(int)
        + df["c2_positive"]
        + df["c3_morphology_flag"]
    )

    # Metabolic Burden
    df["bmi_category"] = pd.cut(
        df["BMI"], bins=[0, 18.5, 24.9, 29.9, 200], labels=[0, 1, 2, 3]
    ).astype(float)
    df["central_obesity_flag"] = (df["Waist:Hip Ratio"] > 0.85).astype(int)
    df["metabolic_score"] = (
        (df["BMI"] >= 25).astype(int)
        + (df["RBS(mg/dl)"] >= 110).astype(int)
        + df["Skin darkening (Y/N)"]
        + df["Weight gain(Y/N)"]
        + df["Fast food (Y/N)"]
    )

    # Exclusion / Mimic Flags
    df["thyroid_flag"]        = (
        (df["TSH (mIU/L)"] < 0.4) | (df["TSH (mIU/L)"] > 4.0)
    ).astype(int)
    df["hyperprolactin_flag"] = (df["PRL(ng/mL)"] > 25).astype(int)
    df["vitd_deficient"]      = (df["Vit D3 (ng/mL)"] < 20).astype(int)

    # Interaction Features
    df["follicles_x_amh"]   = df["total_follicles"] * df["AMH(ng/mL)"]
    df["bmi_x_waist"]       = df["BMI"] * df["Waist(inch)"]
    df["androgens_x_cycle"] = df["hyperandrogenism_score"] * df["cycle_irregular"]

    return df.drop(columns=[c for c in _DROP_AFTER_ENGINEERING if c in df.columns])


# ── Clinical reporting ───────────────────────────────────────────────────────

def rotterdam_criteria_report(row):
    """Return list of fired Rotterdam criteria strings for a single engineered row (Series)."""
    fired = []

    if row.get("hyperandrogenism_score", 0) > 0:
        details = []
        if row.get("hair growth(Y/N)", 0):
            details.append("hirsutism")
        if row.get("Pimples(Y/N)", 0):
            details.append("acne")
        if row.get("c1_lh_fsh_flag", 0):
            details.append(f"LH/FSH={row.get('lh_fsh_ratio', 0):.2f} > 2.0")
        fired.append(f"C1 Hyperandrogenism  ({', '.join(details)})")

    if row.get("c2_positive", 0):
        details = []
        if row.get("cycle_irregular", 0):
            details.append("irregular cycle")
        if row.get("anovulation_flag", 0):
            prg = row.get("PRG(ng/mL)", "?")
            details.append(f"progesterone {prg:.2f} ng/mL < 3.0" if isinstance(prg, float) else "low progesterone")
        fired.append(f"C2 Ovulatory Dysfunction  ({', '.join(details)})")

    if row.get("c3_morphology_flag", 0):
        details = []
        tf = row.get("total_follicles", 0)
        if tf >= 20:
            details.append(f"{int(tf)} total follicles >= 20")
        if row.get("amh_elevated", 0):
            amh = row.get("AMH(ng/mL)", "?")
            details.append(f"AMH {amh:.2f} ng/mL > 4.59" if isinstance(amh, float) else "elevated AMH")
        fired.append(f"C3 Polycystic Morphology  ({', '.join(details)})")

    return fired


# ── Age-stratified guardrails (2023 International PCOS Guidelines) ───────────

def apply_age_guardrail(rotterdam: dict, patient_age) -> "str | None":
    """
    Apply 2023 International PCOS Guideline age-stratified diagnostic rules.

    rotterdam : dict
        Keys: 'c1', 'c2', 'c3', 'pcos' (all bool).
    patient_age : float | None
        Patient age in years.

    Returns a warning/note string when the Rotterdam result needs
    qualification, or None when standard adult criteria apply unchanged.
    """
    if patient_age is None:
        return None

    age = float(patient_age)

    # ≤13 yrs: within ~2 years of typical menarche — diagnosis not possible
    if age <= 13:
        return (
            f"Diagnosis deferred (age {int(age)}): patient is likely within 2 years of "
            "menarche onset. Per the 2023 International PCOS Guidelines, PCOS cannot be "
            "reliably diagnosed in this window. Reassess after 2 full years post-menarche."
        )

    # Adolescent 14–19: requires BOTH C1 and C2; C3 alone is insufficient
    if age <= 19:
        pcos = rotterdam.get("pcos", False)
        c1   = rotterdam.get("c1", False)
        c2   = rotterdam.get("c2", False)
        c3   = rotterdam.get("c3", False)
        if pcos and not (c1 and c2):
            fired = [k for k, v in [("C1", c1), ("C2", c2), ("C3", c3)] if v]
            return (
                f"Adolescent guardrail (age {int(age)}): the 2023 International PCOS Guidelines "
                "require BOTH hyperandrogenism (C1) AND ovulatory dysfunction (C2) in patients "
                "aged 14–19. Polycystic morphology / AMH (C3) cannot substitute for either. "
                f"Criteria fired: {', '.join(fired)}. Diagnosis is NOT confirmed — reassess at "
                "age 20 or when both C1 and C2 are present."
            )
        if pcos and c1 and c2:
            return (
                f"Adolescent note (age {int(age)}): diagnosis is supported (C1 + C2 present). "
                "Per 2023 guidelines, PCOM/AMH (C3) alone would not have been sufficient. "
                "Schedule follow-up; irregular cycles are common within 2 years of menarche."
            )
        return None

    # ≥35 yrs: AMH thresholds less reliable due to natural decline
    if age >= 35:
        return (
            f"AMH reliability note (age {int(age)}): AMH declines naturally from the mid-30s. "
            "The C3 threshold (>4.59 ng/mL) may underestimate polycystic morphology in "
            "patients aged ≥35. Ultrasound follicle count is preferred for C3 confirmation."
        )

    # Adults 20–34: standard Rotterdam unchanged
    return None


def classify_phenotype(c1: bool, c2: bool, c3: bool) -> "str | None":
    """
    Classify PCOS phenotype per Rotterdam consensus (phenotypes A–D).
    Returns None when fewer than 2 criteria are met (PCOS not positive).
    """
    if c1 and c2 and c3:
        return "Phenotype A — Classic/Full PCOS"
    if c1 and c2 and not c3:
        return "Phenotype B — Classic/NIH PCOS"
    if c1 and c3 and not c2:
        return "Phenotype C — Ovulatory PCOS"
    if c2 and c3 and not c1:
        return "Phenotype D — Non-hyperandrogenic PCOS"
    return None
