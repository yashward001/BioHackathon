# PolyClear — PCOS Diagnostic Support Tool

A clinician-facing PCOS diagnostic support tool built for BioHackathon. Combines Rotterdam criteria rule-based assessment with an XGBoost ML model for PCOS prediction and phenotype classification.

---

## Features

- **Exclusion screen** — flags pregnancy, thyroid dysfunction, hyperprolactinaemia, and endometrial risk before any PCOS interpretation
- **Rotterdam criteria** — evaluates all three criteria (C1 Hyperandrogenism, C2 Ovulatory dysfunction, C3 Polycystic morphology) with supporting evidence
- **XGBoost ML assessment** — trained on a 541-patient cohort, ROC-AUC 0.959, 52 engineered features
- **SHAP-driven phenotype classification** — phenotype (A/B/C/D) derived from per-feature SHAP contributions, not hard clinical thresholds
- **Age-stratified guardrails** — applies 2023 International PCOS Guidelines for adolescent and ≥35 age groups
- **4 demo cases** — Classic PCOS, Mild PCOS (Phenotype D), Thyroid mimic, Low risk / negative

---

## Stack

| Layer | Technology |
|---|---|
| Frontend | Vanilla HTML/CSS/JS, Chart.js |
| ML Model | XGBoost (scikit-learn API), trained on PCOS dataset |
| API | Flask (`api.py` for local, `api/predict.py` for Vercel serverless) |
| Deployment | Vercel (static frontend + Python serverless function) |

---

## Running Locally

**1. Install dependencies**
```bash
pip install flask flask-cors pandas numpy joblib xgboost
```

**2. Train and save the model** (first time only)
```bash
python train_and_save.py
```

**3. Start the API server**
```bash
python api.py
# Runs on http://localhost:5050
```

**4. Open the frontend**

Open `index.html` directly in your browser — the frontend calls the local API at `localhost:5050`.

---

## Project Structure

```
├── index.html          # Frontend (single-page, multi-step form)
├── style.css           # Styles
├── script.js           # Frontend logic, demo cases, ML result rendering
├── api.py              # Flask API for local development
├── api/
│   └── predict.py      # Vercel Python serverless function
├── features.py         # Feature engineering shared between training and inference
├── train_and_save.py   # Model training pipeline — saves .pkl artifacts
├── model.py            # Model definition
├── pcos_model.pkl      # Trained XGBoost classifier
├── feature_columns.pkl # Ordered list of 52 feature names
├── training_stats.pkl  # Outlier caps + imputation medians/modes
└── requirements.txt    # Python dependencies
```

---

## ML Model

- **Algorithm**: XGBoost binary classifier (`binary:logistic`)
- **Features**: 52 engineered features including Rotterdam criterion flags, metabolic scores, interaction terms, and exclusion indicators
- **Performance**: ROC-AUC 0.959 on held-out test set
- **Phenotype classification**: Uses `booster.predict(pred_contribs=True)` (built-in SHAP) to group feature contributions by criterion (C1/C2/C3) — phenotype is assigned based on which criteria the model's own learned weights activate for each patient

### PCOS Phenotypes

| Phenotype | Criteria | Description |
|---|---|---|
| A | C1 + C2 + C3 | Classic/Full PCOS |
| B | C1 + C2 | Classic/NIH PCOS |
| C | C1 + C3 | Ovulatory PCOS |
| D | C2 + C3 | Non-hyperandrogenic PCOS |

---

## References

1. Rotterdam ESHRE/ASRM PCOS Consensus Workshop Group. *Hum Reprod* 2004;19(1):41–47.
2. Teede HJ et al. 2023 International Evidence-based Guideline for PCOS. *J Clin Endocrinol Metab* 2023;108(10):2447–2469.
3. Phenotype distribution in European populations. *Eur J Obstet Gynecol Reprod Biol* 2022.
