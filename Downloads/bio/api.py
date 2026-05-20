"""
api.py — PolyClear ML API
Flask server on port 5050 connecting the browser UI to the XGBoost model.

Endpoints:
  GET  /health   → {"status": "ok"}
  POST /predict  → {"pcos_probability": 0-100, "predicted_label": str,
                     "rotterdam_criteria": [str, ...]}
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

from features import (
    apply_caps, apply_imputation, engineer_features, rotterdam_criteria_report,
    apply_age_guardrail, classify_phenotype,
)

app = Flask(__name__)
CORS(app)

try:
    model           = joblib.load("pcos_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    stats           = joblib.load("training_stats.pkl")
    print("Model artifacts loaded — ready on http://localhost:5050")
except FileNotFoundError as exc:
    raise RuntimeError(f"Run train_and_save.py first: {exc}") from exc


def _f(val):
    """Coerce a value to float, or return None."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def ui_to_model_row(data):
    """
    Map UI form field IDs (HTML element IDs) to the 36 raw model input columns.

    Conversions applied:
      lh_fsh       LH:FSH ratio → derive LH = ratio × FSH_training_median;
                   FSH/LH feature set to 1/ratio
      waist        UI sends cm; model expects inches  (÷ 2.54)
      irregular_cycle  bool → Cycle(R/I): 4=irregular, 2=regular
      no_exercise  inverted → Reg.Exercise(Y/N): 0 when checked
      follicle_num per-ovary count → set both Follicle No. (L) and (R)
    """
    medians = stats["medians"]

    # LH/FSH: reconstruct individual values from the ratio
    lh_fsh   = _f(data.get("lh_fsh"))
    fsh_med  = medians.get("FSH(mIU/mL)", 4.85)
    lh_val   = (lh_fsh * fsh_med) if lh_fsh is not None else None
    fsh_val  = fsh_med             if lh_fsh is not None else None
    fsh_lh   = (1.0 / lh_fsh)     if lh_fsh             else None

    # Waist cm → inches
    waist_cm = _f(data.get("waist"))
    waist_in = (waist_cm / 2.54) if waist_cm is not None else None

    # Cycle regularity
    cycle_ri = 4 if data.get("irregular_cycle") else 2

    # Exercise flag is inverted in the UI ("No regular exercise")
    reg_exercise = 0 if data.get("no_exercise") else 1

    fn = _f(data.get("follicle_num"))
    fs = _f(data.get("follicle_size"))

    return {
        " Age (yrs)":           _f(data.get("age")),
        "Weight (Kg)":          None,
        "Height(Cm) ":          None,
        "BMI":                  _f(data.get("bmi")),
        "Blood Group":          None,
        "Pulse rate(bpm) ":     None,
        "RR (breaths/min)":     None,
        "Hb(g/dl)":             None,
        "Cycle(R/I)":           cycle_ri,
        "Cycle length(days)":   None,
        "FSH(mIU/mL)":          fsh_val,
        "LH(mIU/mL)":           lh_val,
        "FSH/LH":               fsh_lh,
        "Hip(inch)":            None,
        "Waist(inch)":          waist_in,
        "Waist:Hip Ratio":      _f(data.get("whr")),
        "TSH (mIU/L)":          _f(data.get("tsh")),
        "AMH(ng/mL)":           _f(data.get("amh")),
        "PRL(ng/mL)":           _f(data.get("prl")),
        "Vit D3 (ng/mL)":       _f(data.get("vit_d")),
        "PRG(ng/mL)":           _f(data.get("prg")),
        "RBS(mg/dl)":           _f(data.get("rbs")),
        "Weight gain(Y/N)":     int(bool(data.get("weight_gain"))),
        "hair growth(Y/N)":     int(bool(data.get("hair_growth"))),
        "Skin darkening (Y/N)": int(bool(data.get("skin_darkening"))),
        "Hair loss(Y/N)":       int(bool(data.get("hair_loss"))),
        "Pimples(Y/N)":         int(bool(data.get("pimples"))),
        "Fast food (Y/N)":      int(bool(data.get("fast_food"))),
        "Reg.Exercise(Y/N)":    reg_exercise,
        "BP _Systolic (mmHg)":  None,
        "BP _Diastolic (mmHg)": None,
        "Follicle No. (L)":     fn,
        "Follicle No. (R)":     fn,
        "Avg. F size (L) (mm)": fs,
        "Avg. F size (R) (mm)": fs,
        "Endometrium (mm)":     _f(data.get("endo_thickness")),
    }


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    data = request.get_json(force=True, silent=True) or {}

    row = ui_to_model_row(data)
    df  = pd.DataFrame([row])

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df     = apply_caps(df, stats["caps"])
    df     = apply_imputation(df, stats["medians"], stats["modes"])
    X_full = engineer_features(df)
    X_mod  = X_full.reindex(columns=feature_columns, fill_value=0)

    label_num = int(model.predict(X_mod)[0])
    prob      = float(model.predict_proba(X_mod)[0, 1])
    label     = "PCOS" if label_num == 1 else "Non-PCOS"

    # Shorten criterion strings to display names
    fired = rotterdam_criteria_report(X_full.iloc[0])
    _map  = {"C1": "C1 Hyperandrogenism",
             "C2": "C2 Ovulatory dysfunction",
             "C3": "C3 Polycystic morphology"}
    criteria = [_map[c[:2]] for c in fired if c[:2] in _map]

    # Age-stratified guardrail and phenotype classification
    eng_row     = X_full.iloc[0]
    c1_bool     = bool(eng_row.get("hyperandrogenism_score", 0) > 0)
    c2_bool     = bool(eng_row.get("c2_positive", 0))
    c3_bool     = bool(eng_row.get("c3_morphology_flag", 0))
    rott_pcos   = (int(c1_bool) + int(c2_bool) + int(c3_bool)) >= 2
    rott_dict   = {"c1": c1_bool, "c2": c2_bool, "c3": c3_bool, "pcos": rott_pcos}
    patient_age = _f(data.get("age"))
    age_guardrail = apply_age_guardrail(rott_dict, patient_age)
    phenotype     = classify_phenotype(c1_bool, c2_bool, c3_bool) if label_num == 1 else None

    return jsonify({
        "pcos_probability":   round(prob * 100, 1),
        "predicted_label":    label,
        "rotterdam_criteria": criteria,
        "phenotype":          phenotype,
        "age_guardrail":      age_guardrail,
    })


if __name__ == "__main__":
    app.run(port=5050, debug=False)
