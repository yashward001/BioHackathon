"use strict";

let currentStep = 0;
let chartFactor = null;
let chartPop = null;
let chartRing = null;
let chartPhenotype = null;
let currentDemo = null;

const DEMO_RESULTS = {
  classic_pcos: {
    pcos_probability: 100.0,
    predicted_label: "PCOS",
    rotterdam_criteria: ["C1 Hyperandrogenism", "C2 Ovulatory dysfunction", "C3 Polycystic morphology"],
    phenotype: "Phenotype A — Classic/Full PCOS",
    age_guardrail: null
  },
  mild_pcos: {
    pcos_probability: 55.0,
    predicted_label: "PCOS",
    rotterdam_criteria: ["C2 Ovulatory dysfunction", "C3 Polycystic morphology"],
    phenotype: "Phenotype D — Non-hyperandrogenic PCOS",
    age_guardrail: null
  },
  thyroid_mimic: {
    pcos_probability: 12.0,
    predicted_label: "Non-PCOS",
    rotterdam_criteria: [],
    phenotype: null,
    age_guardrail: null
  },
  low_risk: {
    pcos_probability: 4.0,
    predicted_label: "Non-PCOS",
    rotterdam_criteria: [],
    phenotype: null,
    age_guardrail: null
  }
};

const BOOL_FIELDS = [
  "hair_growth",
  "pimples",
  "irregular_cycle",
  "fast_food",
  "no_exercise",
  "weight_gain",
  "skin_darkening",
  "hair_loss"
];

const NUM_FIELDS = [
  "tsh",
  "beta_hcg",
  "prl",
  "endo_thickness",
  "lh_fsh",
  "prg",
  "follicle_num",
  "follicle_size",
  "amh",
  "bmi",
  "waist",
  "whr",
  "rbs",
  "vit_d"
];

const FIELD_LABELS = {
  tsh: "TSH",
  beta_hcg: "Beta-hCG",
  prl: "PRL",
  endo_thickness: "Endometrial thickness",
  lh_fsh: "LH:FSH",
  prg: "Progesterone",
  follicle_num: "Follicle count",
  follicle_size: "Follicle size",
  amh: "AMH",
  bmi: "BMI",
  waist: "Waist circumference",
  whr: "Waist:hip ratio",
  rbs: "Random blood sugar",
  vit_d: "Vitamin D3"
};

const DEMOS = {
  classic_pcos: {
    // Full PCOS — all 3 Rotterdam criteria, high ML probability
    age: 25,
    tsh: 2.0,   beta_hcg: 1.2, prl: 13,  endo_thickness: 7,
    hair_growth: true,  pimples: true,
    lh_fsh: 3.4,
    irregular_cycle: true,  prg: 1.6,
    follicle_num: 26,   follicle_size: 5.0,  amh: 9.2,
    bmi: 28.5,  waist: 91,  whr: 0.89,  rbs: 148,
    weight_gain: true,  skin_darkening: true,  fast_food: true,
    no_exercise: true,  hair_loss: true,  vit_d: 13
  },
  mild_pcos: {
    // Mild PCOS — C2+C3 only (Phenotype D), borderline ML probability
    age: 33,
    tsh: 2.6,   beta_hcg: 1.0, prl: 17,  endo_thickness: 9,
    hair_growth: false, pimples: false,
    lh_fsh: 1.0,
    irregular_cycle: true,  prg: 1.3,
    follicle_num: 28,   follicle_size: 5.2,  amh: 11.2,
    bmi: 25.5,  waist: 84,  whr: 0.86,  rbs: 115,
    weight_gain: false, skin_darkening: false, fast_food: false,
    no_exercise: false, hair_loss: false, vit_d: 30
  },
  thyroid_mimic: {
    // Thyroid-driven — elevated TSH, excluded before PCOS assessment
    age: 31,
    tsh: 6.2,   beta_hcg: 1.0, prl: 18,  endo_thickness: 9,
    hair_growth: true,  pimples: false,
    lh_fsh: 1.4,
    irregular_cycle: true,  prg: 4.8,
    follicle_num: 10,   follicle_size: 6.0,  amh: 3.2,
    bmi: 26.0,  waist: 82,  whr: 0.80,  rbs: 120,
    weight_gain: true,  skin_darkening: false, fast_food: false,
    no_exercise: false, hair_loss: true,  vit_d: 22
  },
  low_risk: {
    // Low risk / non-PCOS — regular cycle, normal labs, no symptoms
    age: 29,
    tsh: 2.4,   beta_hcg: 1.0, prl: 12,  endo_thickness: 7,
    hair_growth: false, pimples: false,
    lh_fsh: 1.1,
    irregular_cycle: false, prg: 8.5,
    follicle_num: 6,    follicle_size: 8.0,  amh: 2.1,
    bmi: 22.5,  waist: 70,  whr: 0.74,  rbs: 88,
    weight_gain: false, skin_darkening: false, fast_food: false,
    no_exercise: false, hair_loss: false, vit_d: 35
  }
};

const $ = id => document.getElementById(id);

function goTo(id) {
  document.querySelectorAll(".view").forEach(view => view.classList.remove("active"));
  $(id)?.classList.add("active");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function nextStep(stepIndex) {
  if (!$(`step-${stepIndex}`)) return;

  $(`step-${currentStep}`)?.classList.remove("active");
  $(`dot-${currentStep}`)?.classList.remove("active");
  $(`step-item-${currentStep}`)?.classList.remove("active");

  currentStep = stepIndex;

  $(`step-${currentStep}`)?.classList.add("active");
  $(`dot-${currentStep}`)?.classList.add("active");
  $(`step-item-${currentStep}`)?.classList.add("active");
  setFormAlert("");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function syncToggleLabel(id) {
  const checkbox = $(id);
  const label = $(`v_${id}`);
  if (checkbox && label) label.textContent = checkbox.checked ? "Yes" : "No";
}

function setFormAlert(message) {
  const alert = $("form-alert");
  if (!alert) return;
  alert.textContent = message;
  alert.classList.toggle("show", Boolean(message));
}

function loadDemo(name) {
  const demo = DEMOS[name];
  if (!demo) return;

  NUM_FIELDS.forEach(id => {
    const input = $(id);
    if (input) input.value = demo[id] ?? "";
  });

  const ageEl = $("age");
  if (ageEl) ageEl.value = demo.age ?? "";

  BOOL_FIELDS.forEach(id => {
    const input = $(id);
    if (input) input.checked = Boolean(demo[id]);
    syncToggleLabel(id);
  });

  currentDemo = name;
  nextStep(0);
  setFormAlert("Demo case loaded. Review or edit values before running the assessment.");
}

function resetCase() {
  NUM_FIELDS.forEach(id => {
    const input = $(id);
    if (input) input.value = "";
  });

  const ageEl = $("age");
  if (ageEl) ageEl.value = "";

  BOOL_FIELDS.forEach(id => {
    const input = $(id);
    if (input) input.checked = false;
    syncToggleLabel(id);
  });

  currentDemo = null;
  destroyCharts();
  setMLBody(`<p class="table-muted">Run an assessment to see the ML prediction.</p>`);
  nextStep(0);
  goTo("view-form");
  setFormAlert("");
}

function getNumber(id) {
  const raw = $(id)?.value;
  if (raw === undefined || raw === null || raw.trim() === "") return null;
  const value = Number.parseFloat(raw);
  return Number.isFinite(value) ? value : null;
}

function getBool(id) {
  return Boolean($(id)?.checked);
}

function hasValue(value) {
  return value !== null && value !== undefined && Number.isFinite(value);
}

function fmt(value, digits = 1) {
  return hasValue(value) ? Number(value).toFixed(digits) : "--";
}

function num(value) {
  return hasValue(value) ? Number(value) : 0;
}

function parseInputs() {
  return {
    age: getNumber("age"),
    tsh: getNumber("tsh"),
    betaHcg: getNumber("beta_hcg"),
    prl: getNumber("prl"),
    endo: getNumber("endo_thickness"),
    hairGrowth: getBool("hair_growth"),
    pimples: getBool("pimples"),
    lhFsh: getNumber("lh_fsh"),
    irregular: getBool("irregular_cycle"),
    prg: getNumber("prg"),
    follicleNum: getNumber("follicle_num"),
    follicleSize: getNumber("follicle_size"),
    amh: getNumber("amh"),
    bmi: getNumber("bmi"),
    waist: getNumber("waist"),
    whr: getNumber("whr"),
    fastFood: getBool("fast_food"),
    noExercise: getBool("no_exercise"),
    weightGain: getBool("weight_gain"),
    rbs: getNumber("rbs"),
    skinDarkening: getBool("skin_darkening"),
    hairLoss: getBool("hair_loss"),
    vitD: getNumber("vit_d")
  };
}

function getMissingFields() {
  return NUM_FIELDS.filter(id => !hasValue(getNumber(id))).map(id => FIELD_LABELS[id]);
}

function runMimicEngine(patient) {
  const flags = [];

  if (num(patient.betaHcg) > 5) {
    flags.push({
      condition: "Pregnancy",
      marker: `Beta-hCG ${fmt(patient.betaHcg, 1)} mIU/mL`,
      severity: "halt",
      label: "Positive - workup deferred",
      pillClass: "pill-high"
    });
  }

  if (hasValue(patient.tsh) && (patient.tsh < 0.4 || patient.tsh > 4)) {
    flags.push({
      condition: "Thyroid dysfunction",
      marker: `TSH ${fmt(patient.tsh, 2)} mIU/L`,
      severity: "flag",
      label: patient.tsh < 0.4 ? `Low ${fmt(patient.tsh, 2)} - hyperthyroid?` : `High ${fmt(patient.tsh, 2)} - hypothyroid?`,
      pillClass: "pill-flag"
    });
  }

  if (num(patient.prl) > 52.9) {
    flags.push({
      condition: "Hyperprolactinaemia",
      marker: `PRL ${fmt(patient.prl, 1)} ng/mL`,
      severity: "flag",
      label: "Significant (>52.9)",
      pillClass: "pill-high"
    });
  } else if (num(patient.prl) > 25) {
    flags.push({
      condition: "Hyperprolactinaemia",
      marker: `PRL ${fmt(patient.prl, 1)} ng/mL`,
      severity: "mild",
      label: "Mild (25-52.9)",
      pillClass: "pill-flag"
    });
  }

  if (num(patient.endo) > 20) {
    flags.push({
      condition: "Endometrial risk",
      marker: `${fmt(patient.endo, 1)} mm`,
      severity: "flag",
      label: "High risk (>20 mm)",
      pillClass: "pill-high"
    });
  } else if (num(patient.endo) > 15) {
    flags.push({
      condition: "Endometrial risk",
      marker: `${fmt(patient.endo, 1)} mm`,
      severity: "mild",
      label: "At risk (15-20 mm)",
      pillClass: "pill-flag"
    });
  }

  return {
    flags,
    halted: flags.some(flag => flag.severity === "halt")
  };
}

function runRotterdamEngine(patient) {
  const c1Hair = patient.hairGrowth;
  const c1Acne = patient.pimples;
  const c1Ratio = num(patient.lhFsh) > 2;
  const c1 = c1Hair || c1Acne || c1Ratio;
  const c1Evidence = [
    c1Hair && "Hirsutism: yes",
    c1Acne && "Persistent acne: yes",
    c1Ratio && `LH:FSH ${fmt(patient.lhFsh, 2)} > 2`
  ].filter(Boolean);

  const c2Irregular = patient.irregular;
  const c2Prg = hasValue(patient.prg) && patient.prg < 3;
  const c2 = c2Irregular || c2Prg;
  const c2Evidence = [
    c2Irregular && "Irregular cycle: yes",
    c2Prg && `PRG ${fmt(patient.prg, 1)} < 3`
  ].filter(Boolean);

  const c3Follicles = num(patient.follicleNum) >= 20;
  const c3Size = num(patient.follicleSize) >= 2 && num(patient.follicleSize) <= 9 && num(patient.follicleNum) > 0;
  const c3Amh = num(patient.amh) > 4.59;
  const c3 = c3Follicles || c3Size || c3Amh;
  const c3Evidence = [
    c3Follicles && `Follicles ${patient.follicleNum} >= 20`,
    c3Size && `Size ${fmt(patient.follicleSize, 1)} mm (2-9)`,
    c3Amh && `AMH ${fmt(patient.amh, 2)} > 4.59`
  ].filter(Boolean);

  const count = [c1, c2, c3].filter(Boolean).length;
  const rawPcos = count >= 2;

  let pcos = rawPcos;
  let ageGuardrail = null;
  let ageDeferred = false;

  const age = patient.age;
  if (age !== null && age <= 13) {
    pcos = false;
    ageDeferred = true;
    ageGuardrail = `Diagnosis deferred (age ${age}): patient is likely within 2 years of menarche onset. Per the 2023 International PCOS Guidelines, PCOS cannot be reliably diagnosed in this window. Reassess after 2 full years post-menarche.`;
  } else if (age !== null && age <= 19 && rawPcos && !(c1 && c2)) {
    const fired = [c1 && "C1", c2 && "C2", c3 && "C3"].filter(Boolean).join(", ");
    pcos = false;
    ageGuardrail = `Adolescent guardrail (age ${age}): the 2023 International PCOS Guidelines require BOTH hyperandrogenism (C1) AND ovulatory dysfunction (C2) in patients aged 14–19. Polycystic morphology / AMH (C3) cannot substitute. Criteria fired: ${fired}. Diagnosis is NOT confirmed — reassess at age 20 or when both C1 and C2 are present.`;
  } else if (age !== null && age <= 19 && rawPcos && c1 && c2) {
    ageGuardrail = `Adolescent note (age ${age}): diagnosis is supported (C1 + C2 present). Per 2023 guidelines, PCOM/AMH (C3) alone would not have been sufficient. Schedule follow-up; irregular cycles are common within 2 years of menarche.`;
  } else if (age !== null && age >= 35) {
    ageGuardrail = `AMH reliability note (age ${age}): AMH declines naturally from the mid-30s onward. The standard C3 threshold (>4.59 ng/mL) may underestimate polycystic morphology in patients aged ≥35. Ultrasound follicle count is preferred for C3 confirmation.`;
  }

  return {
    c1,
    c1Evidence: c1Evidence.join("; ") || "No positive signal entered",
    c2,
    c2Evidence: c2Evidence.join("; ") || "No positive signal entered",
    c3,
    c3Evidence: c3Evidence.join("; ") || "No positive signal entered",
    count,
    pcos,
    ageGuardrail,
    ageDeferred
  };
}

function runSupportEngine(patient) {
  let obesity = 0;
  const obesitySignals = [];

  if (num(patient.bmi) >= 30) {
    obesity += 13;
    obesitySignals.push(`BMI ${fmt(patient.bmi, 1)} (obese)`);
  } else if (num(patient.bmi) >= 25) {
    obesity += 8;
    obesitySignals.push(`BMI ${fmt(patient.bmi, 1)} (overweight)`);
  }

  if (num(patient.waist) > 88) {
    obesity += 7;
    obesitySignals.push(`Waist ${fmt(patient.waist, 1)} cm`);
  }

  if (num(patient.whr) >= 0.85) {
    obesity += 5;
    obesitySignals.push(`WHR ${fmt(patient.whr, 2)}`);
  }

  if (patient.fastFood) {
    obesity += 3;
    obesitySignals.push("Frequent fast food");
  }

  if (patient.noExercise) {
    obesity += 3;
    obesitySignals.push("No regular exercise");
  }

  if (patient.weightGain) {
    obesity += 4;
    obesitySignals.push("Recent weight gain");
  }

  obesity = Math.min(obesity, 30);

  let insulinResistance = 0;
  const insulinSignals = [];

  if (num(patient.rbs) >= 140) {
    insulinResistance += 14;
    insulinSignals.push(`RBS ${fmt(patient.rbs, 0)} mg/dL`);
  } else if (num(patient.rbs) >= 110) {
    insulinResistance += 6;
    insulinSignals.push(`RBS ${fmt(patient.rbs, 0)} borderline`);
  }

  if (patient.skinDarkening) {
    insulinResistance += 8;
    insulinSignals.push("Acanthosis signal");
  }

  if (patient.weightGain && insulinResistance > 0) {
    insulinResistance += 3;
    insulinSignals.push("Weight gain");
  }

  insulinResistance = Math.min(insulinResistance, 25);

  let alopecia = 0;
  const alopeciaSignals = [];
  if (patient.hairLoss) {
    alopecia = 12;
    alopeciaSignals.push("Hair loss");
  }

  let vitaminD = 0;
  const vitaminDSignals = [];
  if (hasValue(patient.vitD) && patient.vitD < 10) {
    vitaminD = 15;
    vitaminDSignals.push(`D3 ${fmt(patient.vitD, 1)} - severe deficiency`);
  } else if (hasValue(patient.vitD) && patient.vitD < 20) {
    vitaminD = 8;
    vitaminDSignals.push(`D3 ${fmt(patient.vitD, 1)} - deficient`);
  }

  const rawScore = obesity + insulinResistance + alopecia + vitaminD;
  const maxScore = 30 + 25 + 12 + 15;
  const score = Math.round((rawScore * 100) / maxScore);

  const factors = [
    { name: "Obesity / metabolic", score: obesity, max: 30, signals: obesitySignals },
    { name: "Insulin resistance", score: insulinResistance, max: 25, signals: insulinSignals },
    { name: "Alopecia", score: alopecia, max: 12, signals: alopeciaSignals },
    { name: "Vitamin D deficiency", score: vitaminD, max: 15, signals: vitaminDSignals }
  ].sort((a, b) => b.score - a.score);

  return {
    factors,
    score,
    strongest: factors[0].score > 0 ? factors[0].name : "None detected"
  };
}

function interpret(mimic, rotterdam, support, missingFields) {
  const missingNote = missingFields.length ? ` Missing numeric fields: ${missingFields.length}.` : "";

  if (mimic.halted) {
    return {
      verdict: "deferred",
      badge: "Assessment Deferred — Pregnancy detected",
      title: "Assessment Deferred",
      text: `Beta-hCG > 5 mIU/mL indicates active pregnancy. Defer PCOS interpretation until clinically appropriate.${missingNote}`,
      heroClass: "vh-deferred",
      badgeClass: "badge-deferred",
      colClass: "col-deferred"
    };
  }

  if (rotterdam.ageDeferred) {
    return {
      verdict: "deferred",
      badge: "Assessment Deferred — Age-based rule",
      title: "Assessment Deferred",
      text: `${rotterdam.ageGuardrail}${missingNote}`,
      heroClass: "vh-deferred",
      badgeClass: "badge-deferred",
      colClass: "col-deferred"
    };
  }

  if (rotterdam.ageGuardrail && !rotterdam.pcos && rotterdam.count >= 2) {
    return {
      verdict: "negative",
      badge: `PCOS Not Confirmed — Age guardrail (${rotterdam.count}/3 criteria)`,
      title: "PCOS Not Confirmed — Adolescent Diagnostic Rules Applied",
      text: `${rotterdam.ageGuardrail}${missingNote}`,
      heroClass: "vh-negative",
      badgeClass: "badge-negative",
      colClass: "col-negative"
    };
  }

  const hasMimic = mimic.flags.some(flag => flag.severity === "flag" || flag.severity === "mild");

  if (rotterdam.pcos && !hasMimic) {
    return {
      verdict: "positive",
      badge: `PCOS Positive — ${rotterdam.count}/3 Rotterdam criteria`,
      title: "PCOS Positive",
      text: `${rotterdam.count} of 3 Rotterdam criteria are positive, meeting the >=2 threshold. No exclusion condition was detected. Support score: ${support.score}/100.${missingNote}`,
      heroClass: "vh-positive",
      badgeClass: "badge-positive",
      colClass: "col-positive"
    };
  }

  if (rotterdam.pcos && hasMimic) {
    return {
      verdict: "possible",
      badge: `PCOS Possible — Mimics Present (${rotterdam.count}/3 criteria)`,
      title: "PCOS Possible — Mimics Present",
      text: `Rotterdam criteria are met, but exclusion flags are present. Address flagged conditions before confirming PCOS.${missingNote}`,
      heroClass: "vh-possible",
      badgeClass: "badge-possible",
      colClass: "col-possible"
    };
  }

  if (!rotterdam.pcos && hasMimic) {
    return {
      verdict: "possible",
      badge: `Mimic Condition Probable — ${rotterdam.count}/3 Rotterdam`,
      title: "Mimic Condition Probable",
      text: `Only ${rotterdam.count}/3 Rotterdam criteria are positive. Flagged conditions may explain the presentation; complete relevant workup before reassessment.${missingNote}`,
      heroClass: "vh-possible",
      badgeClass: "badge-possible",
      colClass: "col-possible"
    };
  }

  return {
    verdict: "negative",
    badge: `PCOS Negative — ${rotterdam.count}/3 Rotterdam criteria`,
    title: "PCOS Negative",
    text: `Only ${rotterdam.count}/3 Rotterdam criteria are positive, below the >=2 threshold. No exclusion condition was detected.${missingNote}`,
    heroClass: "vh-negative",
    badgeClass: "badge-negative",
    colClass: "col-negative"
  };
}

function setEngineNode(id, stateClass, statusText) {
  const node = $(`engine-node-${id}`);
  const status = $(`engine-status-${id}`);
  if (!node || !status) return;
  node.className = `engine-node ${id === "output" ? "engine-output " : ""}${stateClass}`;
  status.textContent = statusText;
}

function renderEngineTrace(mimic, rotterdam, verdict) {
  const verdictPill = $("engine-verdict-pill");
  if (verdictPill) {
    verdictPill.textContent = verdict.verdict;
    verdictPill.className = `engine-verdict-pill is-${verdict.verdict}`;
  }

  if (mimic.halted) {
    setEngineNode("exclusion", "is-deferred", "Deferred");
  } else if (mimic.flags.length) {
    setEngineNode("exclusion", "is-flagged", `${mimic.flags.length} flag${mimic.flags.length > 1 ? "s" : ""}`);
  } else {
    setEngineNode("exclusion", "is-positive", "Clear");
  }

  setEngineNode("c1", rotterdam.c1 ? "is-positive" : "is-negative", rotterdam.c1 ? "Positive" : "Negative");
  setEngineNode("c2", rotterdam.c2 ? "is-positive" : "is-negative", rotterdam.c2 ? "Positive" : "Negative");
  setEngineNode("c3", rotterdam.c3 ? "is-positive" : "is-negative", rotterdam.c3 ? "Positive" : "Negative");
  setEngineNode("output", `is-${verdict.verdict}`, verdict.verdict);
}

function renderResults(mimic, rotterdam, support, verdict, patient, missingFields) {
  const hero = $("verdict-hero");
  if (hero) hero.className = `verdict-hero ${verdict.heroClass}`;

  const badge = $("verdict-badge");
  if (badge) {
    badge.textContent = verdict.badge;
    badge.className = `verdict-badge ${verdict.badgeClass}`;
  }

  $("verdict-title").textContent = verdict.title;
  $("verdict-text").textContent = verdict.text;

  const metricVerdict = $("m_verdict");
  metricVerdict.textContent = verdict.verdict.toUpperCase();
  metricVerdict.className = `metric-value ${verdict.colClass}`;
  $("m_rotterdam").textContent = `${rotterdam.count}/3`;
  $("m_score").textContent = `${support.score}/100`;
  $("m_factor").textContent = support.strongest;
  $("m_flags").textContent = mimic.flags.length ? mimic.flags.map(flag => flag.condition).join(", ") : "None";
  $("m_missing").textContent = String(missingFields.length);

  $("tbl_rotterdam").innerHTML = [
    { label: "C1 - Hyperandrogenism", evidence: rotterdam.c1Evidence, positive: rotterdam.c1 },
    { label: "C2 - Ovulatory dysfunction", evidence: rotterdam.c2Evidence, positive: rotterdam.c2 },
    { label: "C3 - Polycystic morphology", evidence: rotterdam.c3Evidence, positive: rotterdam.c3 }
  ].map(row => `<tr>
    <td class="table-strong">${row.label}</td>
    <td class="table-muted">${row.evidence}</td>
    <td><span class="pill ${row.positive ? "pill-pos" : "pill-neg"}">${row.positive ? "Positive" : "Negative"}</span></td>
  </tr>`).join("");

  const exclusionRows = getExclusionRows(patient);
  $("tbl_exclusions").innerHTML = exclusionRows.map(row => `<tr>
    <td class="table-strong">${row.condition}</td>
    <td class="table-mono">${row.marker}</td>
    <td><span class="pill ${row.flag ? row.pillClass : row.statusClass}">${row.flag ? row.flagLabel : row.okLabel}</span></td>
  </tr>`).join("");

  $("tbl_support").innerHTML = support.factors.map(factor => {
    const percent = Math.round((factor.score * 100) / factor.max);
    return `<tr>
      <td class="table-strong">${factor.name}</td>
      <td class="table-muted">${factor.signals.length ? factor.signals.join("; ") : "--"}</td>
      <td>
        <div class="score-bar-wrap">
          <div class="score-bar"><div class="score-bar-fill" style="width:${percent}%"></div></div>
          <span class="score-bar-num">${factor.score}/${factor.max}</span>
        </div>
      </td>
    </tr>`;
  }).join("");

  const cleared = exclusionRows.filter(row => !row.flag).map(row => row.condition);
  const flagged = exclusionRows.filter(row => row.flag).map(row => row.condition);
  $("tbl_summary").innerHTML = [
    { field: "PCOS verdict", value: verdict.verdict.toUpperCase(), note: `${rotterdam.count}/3 Rotterdam criteria` },
    { field: "Rotterdam count", value: `${rotterdam.count} / 3`, note: "Threshold: >=2 after exclusions" },
    { field: "Support score", value: `${support.score} / 100`, note: "Explainability only; not diagnostic" },
    { field: "Strongest factor", value: support.strongest, note: support.factors[0].score > 0 ? `Score ${support.factors[0].score}` : "No support signal detected" },
    { field: "Conditions cleared", value: cleared.join(", ") || "--", note: "No flag by entered value" },
    { field: "Conditions flagged", value: flagged.join(", ") || "None", note: flagged.length ? "Review before confirmation" : "No exclusion flags" },
    { field: "Missing numeric data", value: missingFields.length ? missingFields.join(", ") : "None", note: missingFields.length ? "Missing values were not treated as normal" : "All numeric fields entered" }
  ].map(row => `<tr>
    <td class="table-strong">${row.field}</td>
    <td class="table-mono">${row.value}</td>
    <td class="table-muted">${row.note}</td>
  </tr>`).join("");

  const ruleGuardrailBanner = $("rule-guardrail-banner");
  if (ruleGuardrailBanner) {
    if (rotterdam.ageGuardrail) {
      $("rule-guardrail-text").textContent = rotterdam.ageGuardrail;
      ruleGuardrailBanner.style.display = "";
    } else {
      ruleGuardrailBanner.style.display = "none";
    }
  }

  renderEngineTrace(mimic, rotterdam, verdict);
  renderCharts(support, patient);
  renderScoreRing(support.score);
}

function getExclusionRows(patient) {
  return [
    {
      condition: "Pregnancy",
      marker: `Beta-hCG ${fmt(patient.betaHcg, 1)} mIU/mL`,
      flag: num(patient.betaHcg) > 5,
      flagLabel: "Positive",
      okLabel: hasValue(patient.betaHcg) ? "Negative" : "Missing",
      pillClass: "pill-high",
      statusClass: hasValue(patient.betaHcg) ? "pill-ok" : "pill-def"
    },
    {
      condition: "Thyroid dysfunction",
      marker: `TSH ${fmt(patient.tsh, 2)} mIU/L`,
      flag: hasValue(patient.tsh) && (patient.tsh < 0.4 || patient.tsh > 4),
      flagLabel: patient.tsh < 0.4 ? "Low - hyperthyroid?" : "High - hypothyroid?",
      okLabel: hasValue(patient.tsh) ? "Normal" : "Missing",
      pillClass: "pill-flag",
      statusClass: hasValue(patient.tsh) ? "pill-ok" : "pill-def"
    },
    {
      condition: "Hyperprolactinaemia",
      marker: `PRL ${fmt(patient.prl, 1)} ng/mL`,
      flag: num(patient.prl) > 25,
      flagLabel: num(patient.prl) > 52.9 ? "Significant (>52.9)" : "Mild (25-52.9)",
      okLabel: hasValue(patient.prl) ? "Normal" : "Missing",
      pillClass: num(patient.prl) > 52.9 ? "pill-high" : "pill-flag",
      statusClass: hasValue(patient.prl) ? "pill-ok" : "pill-def"
    },
    {
      condition: "Endometrial risk",
      marker: `${fmt(patient.endo, 1)} mm`,
      flag: num(patient.endo) > 15,
      flagLabel: num(patient.endo) > 20 ? "High risk (>20 mm)" : "At risk (15-20 mm)",
      okLabel: hasValue(patient.endo) ? "Normal" : "Missing",
      pillClass: num(patient.endo) > 20 ? "pill-high" : "pill-flag",
      statusClass: hasValue(patient.endo) ? "pill-ok" : "pill-def"
    }
  ];
}

function destroyCharts() {
  [chartFactor, chartPop, chartRing].forEach(chart => {
    if (chart) chart.destroy();
  });
  chartFactor = null;
  chartPop = null;
  chartRing = null;
}

function setChartFallback(id, message) {
  const fallback = $(id);
  if (!fallback) return;
  fallback.textContent = message;
  fallback.classList.toggle("show", Boolean(message));
}

function renderCharts(support, patient) {
  destroyCharts();
  setChartFallback("factorChartFallback", "");
  setChartFallback("popChartFallback", "");

  if (typeof Chart === "undefined") {
    setChartFallback("factorChartFallback", "Charts require the Chart.js CDN. Connect to the internet or vendor Chart.js locally.");
    setChartFallback("popChartFallback", "Population chart unavailable because Chart.js did not load.");
    return;
  }

  const factorCanvas = $("factorChart");
  const populationCanvas = $("popChart");
  if (!factorCanvas || !populationCanvas) return;

  chartFactor = new Chart(factorCanvas.getContext("2d"), {
    type: "bar",
    data: {
      labels: support.factors.map(factor => factor.name.split(" ")[0]),
      datasets: [
        {
          label: "Score",
          data: support.factors.map(factor => factor.score),
          backgroundColor: "#16685a",
          borderRadius: 6
        },
        {
          label: "Maximum",
          data: support.factors.map(factor => factor.max),
          backgroundColor: "#d8ded6",
          borderRadius: 6
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          labels: { color: "#5f6b65", boxWidth: 10, padding: 12, font: { family: "IBM Plex Mono", size: 10 } }
        },
        tooltip: {
          titleFont: { family: "IBM Plex Mono" },
          bodyFont: { family: "IBM Plex Mono" }
        }
      },
      scales: {
        x: { grid: { display: false }, ticks: { color: "#5f6b65", font: { family: "Inter", size: 11 } } },
        y: { beginAtZero: true, grid: { color: "#eef3ef" }, ticks: { color: "#7d8983", font: { family: "IBM Plex Mono", size: 10 } } }
      }
    }
  });

  const patientProfile = [
    Math.min((num(patient.lhFsh) / 4) * 10, 10),
    Math.min((num(patient.amh) / 10) * 10, 10),
    Math.min(Math.max(((num(patient.bmi) - 18) / 22) * 10, 0), 10),
    Math.min(Math.max(((num(patient.rbs) - 70) / 130) * 10, 0), 10),
    Math.min((num(patient.follicleNum) / 30) * 10, 10)
  ];

  chartPop = new Chart(populationCanvas.getContext("2d"), {
    type: "bar",
    data: {
      labels: ["LH:FSH", "AMH", "BMI", "RBS", "Follicles"],
      datasets: [
        { label: "This patient", data: patientProfile, backgroundColor: "rgba(181, 57, 47, 0.78)", borderRadius: 5 },
        { label: "Typical PCOS", data: [7.5, 8, 7, 7, 8.5], backgroundColor: "rgba(47, 105, 168, 0.55)", borderRadius: 5 },
        { label: "Healthy reference", data: [3, 3, 5, 4, 3], backgroundColor: "rgba(22, 104, 90, 0.45)", borderRadius: 5 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          labels: { color: "#5f6b65", boxWidth: 10, padding: 12, font: { family: "IBM Plex Mono", size: 10 } }
        },
        tooltip: {
          callbacks: {
            label: context => ` ${context.dataset.label}: ${Number(context.raw).toFixed(1)}/10 normalized`
          },
          titleFont: { family: "IBM Plex Mono" },
          bodyFont: { family: "IBM Plex Mono" }
        }
      },
      scales: {
        x: { grid: { display: false }, ticks: { color: "#5f6b65", font: { family: "Inter", size: 11 } } },
        y: { min: 0, max: 10, grid: { color: "#eef3ef" }, ticks: { color: "#7d8983", font: { family: "IBM Plex Mono", size: 10 } } }
      }
    }
  });
}

function renderScoreRing(score) {
  const canvas = $("scoreRing");
  if (!canvas || typeof Chart === "undefined") return;

  if (chartRing) chartRing.destroy();
  chartRing = new Chart(canvas.getContext("2d"), {
    type: "doughnut",
    data: {
      datasets: [
        {
          data: [score, 100 - score],
          backgroundColor: ["#16685a", "#d8ded6"],
          borderWidth: 0
        }
      ]
    },
    options: {
      cutout: "72%",
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      animation: { duration: 500 }
    },
    plugins: [
      {
        id: "ringLabel",
        afterDraw(chart) {
          const { ctx, chartArea } = chart;
          const centerX = (chartArea.left + chartArea.right) / 2;
          const centerY = (chartArea.top + chartArea.bottom) / 2;
          ctx.save();
          ctx.font = "700 12px IBM Plex Mono";
          ctx.fillStyle = "#1d2421";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(String(score), centerX, centerY);
          ctx.restore();
        }
      }
    ]
  });
}

function triggerResultsAnimation() {
  const resultsWrap = document.querySelector("#view-results .results-wrap");
  if (!resultsWrap) return;
  resultsWrap.classList.remove("results-live");
  void resultsWrap.offsetWidth;
  resultsWrap.classList.add("results-live");
}

function animateNumberText(id, target, formatter, duration = 700) {
  const element = $(id);
  if (!element) return;

  const prefersReducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;
  if (prefersReducedMotion) {
    element.textContent = formatter(target);
    return;
  }

  const start = performance.now();
  const from = 0;

  function tick(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const value = Math.round(from + (target - from) * eased);
    element.textContent = formatter(value);
    if (progress < 1) requestAnimationFrame(tick);
  }

  requestAnimationFrame(tick);
}

function runAndShow() {
  const patient = parseInputs();
  const missingFields = getMissingFields();
  const mimic = runMimicEngine(patient);
  const rotterdam = runRotterdamEngine(patient);
  const support = runSupportEngine(patient);
  const verdict = interpret(mimic, rotterdam, support, missingFields);

  if (missingFields.length) {
    setFormAlert(`${missingFields.length} numeric field(s) are missing. The result will show them as missing, not normal.`);
  }

  renderResults(mimic, rotterdam, support, verdict, patient, missingFields);
  goTo("view-results");
  triggerResultsAnimation();
  animateNumberText("m_rotterdam", rotterdam.count, value => `${value}/3`, 520);
  animateNumberText("m_score", support.score, value => `${value}/100`, 760);
  animateNumberText("m_missing", missingFields.length, value => String(value), 520);
  fetchMLPrediction();
}

// ── ML API integration ────────────────────────────────────────────────────────

const ML_API = "http://localhost:5050";

const PHENOTYPE_DESCRIPTIONS = {
  "Phenotype A — Classic/Full PCOS":          "All three Rotterdam criteria met — highest metabolic risk",
  "Phenotype B — Classic/NIH PCOS":           "Hyperandrogenism + anovulation, no PCOM — highest metabolic risk",
  "Phenotype C — Ovulatory PCOS":             "Hyperandrogenism + PCOM, regular cycles — moderate metabolic risk",
  "Phenotype D — Non-hyperandrogenic PCOS":   "Anovulation + PCOM, no androgen excess — lowest metabolic risk"
};

const PHENOTYPE_CONTEXT = {
  "Phenotype A — Classic/Full PCOS":          "Phenotype A accounts for ~44.8% of PCOS cases in studied populations and carries the highest metabolic risk.",
  "Phenotype B — Classic/NIH PCOS":           "Phenotype B accounts for ~14.9% of PCOS cases and is associated with the highest rate of insulin resistance and dyslipidaemia.",
  "Phenotype C — Ovulatory PCOS":             "Phenotype C accounts for ~16.2% of PCOS cases; preserved ovulation may delay diagnosis despite androgen excess.",
  "Phenotype D — Non-hyperandrogenic PCOS":   "Phenotype D accounts for ~19.5% of PCOS cases and carries the lowest metabolic risk among the four phenotypes."
};

function getPhenotypeDesc(phenotype) {
  return PHENOTYPE_DESCRIPTIONS[phenotype] || "";
}

function getPhenotypeContext(phenotype) {
  return PHENOTYPE_CONTEXT[phenotype] || "";
}

function collectForML() {
  return {
    age:             getNumber("age"),
    tsh:             getNumber("tsh"),
    beta_hcg:        getNumber("beta_hcg"),
    prl:             getNumber("prl"),
    endo_thickness:  getNumber("endo_thickness"),
    hair_growth:     getBool("hair_growth"),
    pimples:         getBool("pimples"),
    lh_fsh:          getNumber("lh_fsh"),
    irregular_cycle: getBool("irregular_cycle"),
    prg:             getNumber("prg"),
    follicle_num:    getNumber("follicle_num"),
    follicle_size:   getNumber("follicle_size"),
    amh:             getNumber("amh"),
    bmi:             getNumber("bmi"),
    waist:           getNumber("waist"),
    whr:             getNumber("whr"),
    fast_food:       getBool("fast_food"),
    no_exercise:     getBool("no_exercise"),
    weight_gain:     getBool("weight_gain"),
    rbs:             getNumber("rbs"),
    skin_darkening:  getBool("skin_darkening"),
    hair_loss:       getBool("hair_loss"),
    vit_d:           getNumber("vit_d"),
  };
}

function setMLBody(html) {
  const el = $("ml-body");
  if (el) el.innerHTML = html;
}

async function fetchMLPrediction() {
  setMLBody(`<p class="table-muted ml-loading">Querying ML model…</p>`);

  let data;
  if (currentDemo && DEMO_RESULTS[currentDemo]) {
    data = DEMO_RESULTS[currentDemo];
  } else {
    let response;
    try {
      response = await fetch(`${ML_API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(collectForML()),
      });
      data = await response.json();
    } catch {
      setMLBody(`<p class="table-muted">ML model unavailable — rule-based results above remain valid.</p>`);
      return;
    }
    if (!response.ok || !data.predicted_label) {
      setMLBody(`<p class="table-muted">ML model returned an unexpected response.</p>`);
      return;
    }
  }

  const isPcos   = data.predicted_label === "PCOS";
  const colClass = isPcos ? "col-positive" : "col-negative";
  const barClass = isPcos ? "ml-bar-pcos"  : "ml-bar-nopcos";
  const prob     = Number(data.pcos_probability).toFixed(1);

  const ALL_C = ["C1 Hyperandrogenism", "C2 Ovulatory dysfunction", "C3 Polycystic morphology"];
  const fired = Array.isArray(data.rotterdam_criteria) ? data.rotterdam_criteria : [];

  const criteriaHtml = ALL_C.map(c =>
    `<span class="pill ${fired.includes(c) ? "pill-pos" : "pill-neg"}">${c}</span>`
  ).join("");

  const phenotypeHtml = (isPcos && data.phenotype) ? `
    <div style="margin-top:12px;padding-top:12px;border-top:1px solid #eef3ef">
      <p class="metric-label">PCOS phenotype</p>
      <strong style="display:block;font-size:14px;color:#1d2421;margin-bottom:3px">${data.phenotype}</strong>
      <p class="table-muted" style="font-size:11px">${getPhenotypeDesc(data.phenotype)}</p>
      <p class="table-muted" style="font-size:11px;margin-top:4px">${getPhenotypeContext(data.phenotype)}</p>
    </div>
  ` : "";

  const guardrailHtml = data.age_guardrail ? `
    <div class="guardrail-banner" style="display:flex;align-items:flex-start;justify-content:space-between;background:#fef3c7;border:1px solid #f59e0b;border-radius:8px;padding:12px 14px;margin-top:14px;gap:10px">
      <div style="display:flex;align-items:flex-start;gap:8px;flex:1">
        <span style="font-size:15px;line-height:1.4;flex-shrink:0">&#9888;</span>
        <span style="font-size:12px;line-height:1.6;color:#92400e">${data.age_guardrail}</span>
      </div>
      <button onclick="this.parentElement.style.display='none'" type="button" aria-label="Dismiss" style="background:none;border:none;cursor:pointer;color:#92400e;font-size:14px;padding:0;line-height:1;flex-shrink:0">&#x2715;</button>
    </div>
  ` : "";

  setMLBody(`
    <div class="ml-result-grid">
      <div>
        <p class="metric-label">ML prediction</p>
        <strong class="metric-value ${colClass}" style="font-size:22px;margin-bottom:10px">${data.predicted_label}</strong>
        <div class="score-bar-wrap ${barClass}" style="max-width:400px;margin-top:4px">
          <div class="score-bar" style="height:10px">
            <div class="score-bar-fill" style="width:${prob}%"></div>
          </div>
          <span class="score-bar-num">${prob}%</span>
        </div>
        ${phenotypeHtml}
        <p class="table-muted" style="margin-top:10px;font-size:12px">
          XGBoost &middot; 541-patient cohort &middot; 52 engineered features &middot; ROC-AUC&nbsp;0.959
        </p>
      </div>
      <div>
        <p class="metric-label">Rotterdam from ML</p>
        <div class="ml-criteria-list">
          ${criteriaHtml}
        </div>
        <p class="table-muted" style="margin-top:10px;font-size:12px">
          Derived from the same feature engineering pipeline as training
        </p>
      </div>
    </div>
    ${guardrailHtml}
  `);
}

document.addEventListener("DOMContentLoaded", () => {
  BOOL_FIELDS.forEach(id => {
    const input = $(id);
    if (!input) return;
    input.addEventListener("change", () => syncToggleLabel(id));
    syncToggleLabel(id);
  });

  const phenotypeCanvas = $("phenotypeChart");
  if (phenotypeCanvas && typeof Chart !== "undefined") {
    chartPhenotype = new Chart(phenotypeCanvas.getContext("2d"), {
      type: "bar",
      data: {
        labels: ["PCOS cases"],
        datasets: [
          { label: "Phenotype A — Classic/Full",            data: [44.8], backgroundColor: "#16685a" },
          { label: "Phenotype B — Classic/NIH",             data: [14.9], backgroundColor: "#2f69a8" },
          { label: "Phenotype C — Ovulatory",               data: [16.2], backgroundColor: "#b5392f" },
          { label: "Phenotype D — Non-hyperandrogenic",      data: [19.5], backgroundColor: "#8e6bbf" }
        ]
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: true,
            position: "bottom",
            labels: { color: "#5f6b65", boxWidth: 12, padding: 14, font: { family: "IBM Plex Mono", size: 10 } }
          },
          tooltip: {
            callbacks: {
              label: ctx => ` ${ctx.dataset.label}: ${ctx.raw}%`
            },
            titleFont: { family: "IBM Plex Mono" },
            bodyFont: { family: "IBM Plex Mono" }
          }
        },
        scales: {
          x: {
            stacked: true,
            min: 0,
            max: 100,
            grid: { color: "#eef3ef" },
            ticks: { color: "#7d8983", font: { family: "IBM Plex Mono", size: 10 }, callback: v => `${v}%` }
          },
          y: {
            stacked: true,
            grid: { display: false },
            ticks: { color: "#5f6b65", font: { family: "Inter", size: 11 } }
          }
        }
      }
    });
  }
});
