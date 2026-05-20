import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
import xgboost as xgb
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

ROOT = os.path.join(os.path.dirname(__file__), "..")

model           = joblib.load(os.path.join(ROOT, "pcos_model.pkl"))
feature_columns = joblib.load(os.path.join(ROOT, "feature_columns.pkl"))
stats           = joblib.load(os.path.join(ROOT, "training_stats.pkl"))

_booster = model.get_booster()


# ── helpers ──────────────────────────────────────────────────────────────────

def _f(val):
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _g(row, key, default=0.0):
    v = row.get(key)
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _ui_to_raw(data, medians):
    lh_fsh   = _f(data.get("lh_fsh"))
    fsh_med  = medians.get("FSH(mIU/mL)", 4.85)
    lh_val   = (lh_fsh * fsh_med) if lh_fsh is not None else None
    fsh_val  = fsh_med             if lh_fsh is not None else None
    fsh_lh   = (1.0 / lh_fsh)     if lh_fsh             else None

    waist_cm = _f(data.get("waist"))
    waist_in = (waist_cm / 2.54) if waist_cm is not None else None

    cycle_ri     = 4 if data.get("irregular_cycle") else 2
    reg_exercise = 0 if data.get("no_exercise")     else 1

    fn = _f(data.get("follicle_num"))
    fs = _f(data.get("follicle_size"))

    return {
        " Age (yrs)":           _f(data.get("age")),
        "BMI":                  _f(data.get("bmi")),
        "Cycle(R/I)":           float(cycle_ri),
        "FSH(mIU/mL)":          fsh_val,
        "LH(mIU/mL)":           lh_val,
        "FSH/LH":               fsh_lh,
        "Waist(inch)":          waist_in,
        "Waist:Hip Ratio":      _f(data.get("whr")),
        "TSH (mIU/L)":          _f(data.get("tsh")),
        "AMH(ng/mL)":           _f(data.get("amh")),
        "PRL(ng/mL)":           _f(data.get("prl")),
        "Vit D3 (ng/mL)":       _f(data.get("vit_d")),
        "PRG(ng/mL)":           _f(data.get("prg")),
        "RBS(mg/dl)":           _f(data.get("rbs")),
        "Weight gain(Y/N)":     float(int(bool(data.get("weight_gain")))),
        "hair growth(Y/N)":     float(int(bool(data.get("hair_growth")))),
        "Skin darkening (Y/N)": float(int(bool(data.get("skin_darkening")))),
        "Hair loss(Y/N)":       float(int(bool(data.get("hair_loss")))),
        "Pimples(Y/N)":         float(int(bool(data.get("pimples")))),
        "Fast food (Y/N)":      float(int(bool(data.get("fast_food")))),
        "Reg.Exercise(Y/N)":    float(reg_exercise),
        "Follicle No. (L)":     fn,
        "Follicle No. (R)":     fn,
        "Avg. F size (L) (mm)": fs,
        "Avg. F size (R) (mm)": fs,
        "Endometrium (mm)":     _f(data.get("endo_thickness")),
    }


def _apply_caps(row, caps):
    row = row.copy()
    for col, cap in caps.items():
        if row.get(col) is not None:
            row[col] = min(row[col], cap)
    return row


def _apply_imputation(row, medians, modes):
    row = row.copy()
    for col, val in modes.items():
        if row.get(col) is None:
            row[col] = val
    for col, val in medians.items():
        if row.get(col) is None:
            row[col] = val
    return row


def _engineer(row):
    row = dict(row)  # shallow copy

    lh  = _g(row, "LH(mIU/mL)")
    fsh = _g(row, "FSH(mIU/mL)")
    lh_fsh_ratio = lh / (fsh + 1e-6)
    row["lh_fsh_ratio"]           = lh_fsh_ratio
    row["c1_lh_fsh_flag"]         = int(lh_fsh_ratio > 2.0)
    row["hyperandrogenism_score"] = (
        _g(row, "hair growth(Y/N)") + _g(row, "Pimples(Y/N)") + row["c1_lh_fsh_flag"]
    )

    row["cycle_irregular"]  = int(_g(row, "Cycle(R/I)") >= 4)
    row["anovulation_flag"] = int(_g(row, "PRG(ng/mL)") < 3.0)
    row["c2_positive"]      = int(row["cycle_irregular"] == 1 or row["anovulation_flag"] == 1)

    total_follicles      = _g(row, "Follicle No. (L)") + _g(row, "Follicle No. (R)")
    avg_follicle_size    = (_g(row, "Avg. F size (L) (mm)") + _g(row, "Avg. F size (R) (mm)")) / 2
    row["total_follicles"]        = total_follicles
    row["avg_follicle_size"]      = avg_follicle_size
    row["follicle_size_in_range"] = int(2 <= avg_follicle_size <= 9)
    row["amh_elevated"]       = int(_g(row, "AMH(ng/mL)") > 4.59)
    row["c3_morphology_flag"] = int(total_follicles >= 20 or row["amh_elevated"] == 1)

    row["rotterdam_score"] = (
        int(row["hyperandrogenism_score"] > 0)
        + row["c2_positive"]
        + row["c3_morphology_flag"]
    )

    bmi = _g(row, "BMI")
    if   bmi <= 18.5: bmi_cat = 0.0
    elif bmi <= 24.9: bmi_cat = 1.0
    elif bmi <= 29.9: bmi_cat = 2.0
    else:             bmi_cat = 3.0
    row["bmi_category"]         = bmi_cat
    row["central_obesity_flag"] = int(_g(row, "Waist:Hip Ratio") > 0.85)
    row["metabolic_score"] = (
        int(bmi >= 25)
        + int(_g(row, "RBS(mg/dl)") >= 110)
        + _g(row, "Skin darkening (Y/N)")
        + _g(row, "Weight gain(Y/N)")
        + _g(row, "Fast food (Y/N)")
    )

    tsh = _g(row, "TSH (mIU/L)")
    row["thyroid_flag"]        = int(tsh < 0.4 or tsh > 4.0)
    row["hyperprolactin_flag"] = int(_g(row, "PRL(ng/mL)") > 25)
    row["vitd_deficient"]      = int(_g(row, "Vit D3 (ng/mL)") < 20)

    row["follicles_x_amh"]   = total_follicles * _g(row, "AMH(ng/mL)")
    row["bmi_x_waist"]       = bmi * _g(row, "Waist(inch)")
    row["androgens_x_cycle"] = row["hyperandrogenism_score"] * row["cycle_irregular"]

    return row


def _rotterdam_report(row):
    fired = []
    if row.get("hyperandrogenism_score", 0) > 0:
        details = []
        if row.get("hair growth(Y/N)", 0):
            details.append("hirsutism")
        if row.get("Pimples(Y/N)", 0):
            details.append("acne")
        if row.get("c1_lh_fsh_flag", 0):
            r = row.get("lh_fsh_ratio", 0)
            details.append(f"LH/FSH={r:.2f} > 2.0")
        fired.append(f"C1 Hyperandrogenism  ({', '.join(details)})")

    if row.get("c2_positive", 0):
        details = []
        if row.get("cycle_irregular", 0):
            details.append("irregular cycle")
        if row.get("anovulation_flag", 0):
            prg = row.get("PRG(ng/mL)", "?")
            details.append(
                f"progesterone {prg:.2f} ng/mL < 3.0" if isinstance(prg, float) else "low progesterone"
            )
        fired.append(f"C2 Ovulatory Dysfunction  ({', '.join(details)})")

    if row.get("c3_morphology_flag", 0):
        details = []
        tf = row.get("total_follicles", 0)
        if tf >= 20:
            details.append(f"{int(tf)} total follicles >= 20")
        if row.get("amh_elevated", 0):
            amh = row.get("AMH(ng/mL)", "?")
            details.append(
                f"AMH {amh:.2f} ng/mL > 4.59" if isinstance(amh, float) else "elevated AMH"
            )
        fired.append(f"C3 Polycystic Morphology  ({', '.join(details)})")

    return fired


def _age_guardrail(rotterdam, patient_age):
    if patient_age is None:
        return None
    age = float(patient_age)
    if age <= 13:
        return (
            f"Diagnosis deferred (age {int(age)}): patient is likely within 2 years of menarche. "
            "Per 2023 International PCOS Guidelines, PCOS cannot be reliably diagnosed in this window."
        )
    if age <= 19:
        pcos = rotterdam.get("pcos", False)
        c1 = rotterdam.get("c1", False)
        c2 = rotterdam.get("c2", False)
        c3 = rotterdam.get("c3", False)
        if pcos and not (c1 and c2):
            fired = [k for k, v in [("C1", c1), ("C2", c2), ("C3", c3)] if v]
            return (
                f"Adolescent guardrail (age {int(age)}): 2023 Guidelines require BOTH C1 and C2. "
                f"Criteria fired: {', '.join(fired)}. Diagnosis NOT confirmed."
            )
        if pcos and c1 and c2:
            return (
                f"Adolescent note (age {int(age)}): diagnosis supported (C1 + C2). "
                "Schedule follow-up; irregular cycles are common near menarche."
            )
    if age >= 35:
        return (
            f"AMH reliability note (age {int(age)}): AMH declines from the mid-30s. "
            "Ultrasound follicle count preferred for C3 confirmation."
        )
    return None


def _classify_phenotype(c1, c2, c3):
    if c1 and c2 and c3:  return "Phenotype A — Classic/Full PCOS"
    if c1 and c2:         return "Phenotype B — Classic/NIH PCOS"
    if c1 and c3:         return "Phenotype C — Ovulatory PCOS"
    if c2 and c3:         return "Phenotype D — Non-hyperandrogenic PCOS"
    return None


# Feature groups used to aggregate per-criterion SHAP contributions.
_C1_FEATS = frozenset({
    "lh_fsh_ratio", "c1_lh_fsh_flag", "hyperandrogenism_score",
    "hair growth(Y/N)", "Pimples(Y/N)", "androgens_x_cycle",
})
_C2_FEATS = frozenset({
    "cycle_irregular", "anovulation_flag", "c2_positive",
    "PRG(ng/mL)", "Cycle(R/I)",
})
_C3_FEATS = frozenset({
    "total_follicles", "avg_follicle_size", "follicle_size_in_range",
    "amh_elevated", "c3_morphology_flag", "AMH(ng/mL)", "follicles_x_amh",
})


def _ml_criterion_flags(dmat):
    """
    Use XGBoost's built-in SHAP values to decide which Rotterdam criteria
    the model considers active for this sample — no external shap library.
    Returns (c1, c2, c3) booleans: True when that criterion's features
    collectively push the prediction toward PCOS.
    """
    contribs  = _booster.predict(dmat, pred_contribs=True)[0]
    feat_names = list(feature_columns)
    shap = {feat_names[i]: float(contribs[i]) for i in range(len(feat_names))}

    c1_score = sum(shap.get(f, 0.0) for f in _C1_FEATS)
    c2_score = sum(shap.get(f, 0.0) for f in _C2_FEATS)
    c3_score = sum(shap.get(f, 0.0) for f in _C3_FEATS)

    return c1_score > 0, c2_score > 0, c3_score > 0


# ── endpoint ─────────────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True) or {}

    raw  = _ui_to_raw(data, stats["medians"])
    raw  = _apply_caps(raw, stats["caps"])
    raw  = _apply_imputation(raw, stats["medians"], stats["modes"])
    eng  = _engineer(raw)

    X_arr = np.array(
        [[float(eng.get(col, 0) or 0) for col in feature_columns]],
        dtype=np.float32,
    )
    dmat      = xgb.DMatrix(data=X_arr, feature_names=list(feature_columns))
    prob      = float(_booster.predict(dmat)[0])
    label_num = int(prob >= 0.5)
    label     = "PCOS" if label_num == 1 else "Non-PCOS"

    fired = _rotterdam_report(eng)
    _map  = {
        "C1": "C1 Hyperandrogenism",
        "C2": "C2 Ovulatory dysfunction",
        "C3": "C3 Polycystic morphology",
    }
    criteria = [_map[c[:2]] for c in fired if c[:2] in _map]

    # Rule-based flags — kept for Rotterdam display and age guardrail only
    c1_rule = bool(eng.get("hyperandrogenism_score", 0) > 0)
    c2_rule = bool(eng.get("c2_positive", 0))
    c3_rule = bool(eng.get("c3_morphology_flag", 0))
    rott_pcos = (int(c1_rule) + int(c2_rule) + int(c3_rule)) >= 2
    rott_dict = {"c1": c1_rule, "c2": c2_rule, "c3": c3_rule, "pcos": rott_pcos}

    # ML-derived criterion flags via SHAP — used exclusively for phenotype
    c1_ml, c2_ml, c3_ml = _ml_criterion_flags(dmat)

    return jsonify({
        "pcos_probability":   round(prob * 100, 1),
        "predicted_label":    label,
        "rotterdam_criteria": criteria,
        "phenotype":          _classify_phenotype(c1_ml, c2_ml, c3_ml) if label_num == 1 else None,
        "age_guardrail":      _age_guardrail(rott_dict, _f(data.get("age"))),
    })
