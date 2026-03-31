"""
Improved XGBoost Pipeline — Longitudinal Features
===================================================
Key improvements over previous version:
  1. Uses ALL prior visits per patient to build trajectory features
     (slope, mean, std, first score, last score, visit count, time span)
  2. Temporal train/test split — model never sees future patients
  3. Rolling delta and acceleration features capture rate of change
  4. Medication state at last prior visit included as a feature
  5. Deduplication of same-year visits avoids linregress crash

Run from project root:  python3 src/xgboost_longitudinal.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import linregress
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

DATA_DIR         = "./data/PPMI_data"
OUTPUT_DIR       = "./ppmi_outputs"
RANDOM_SEED      = 42
MIN_PRIOR_VISITS = 2   # need at least this many distinct prior visit years

os.makedirs(OUTPUT_DIR, exist_ok=True)

VISIT_MAP = {
    "SC": -0.5, "BL": 0.0, "V04": 1.0, "V06": 2.0,
    "V08": 3.0, "V10": 4.0, "V12": 5.0, "V14": 6.0,
    "V16": 7.0, "V18": 8.0, "V20": 9.0,
}

# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 72)
print("  PARKINSON XGBOOST — LONGITUDINAL TRAJECTORY MODEL")
print("  Target = UPDRS at next visit, using all prior visits as input")
print("=" * 72)

updrs   = pd.read_csv(os.path.join(DATA_DIR, "MDS-UPDRS_Part_III_12Mar2026.csv"),                        low_memory=False)
demo    = pd.read_csv(os.path.join(DATA_DIR, "Demographics_12Mar2026.csv"),                               low_memory=False)
moca    = pd.read_csv(os.path.join(DATA_DIR, "Montreal_Cognitive_Assessment__MoCA__12Mar2026.csv"),       low_memory=False)
age_df  = pd.read_csv(os.path.join(DATA_DIR, "Age_at_visit_12Mar2026.csv"),                               low_memory=False)
sbr     = pd.read_csv(os.path.join(DATA_DIR, "Xing_Core_Lab_-_Quant_SBR_12Mar2026.csv"),                 low_memory=False)
updrs1  = pd.read_csv(os.path.join(DATA_DIR, "MDS-UPDRS_Part_I_12Mar2026.csv"),                          low_memory=False)
updrs2  = pd.read_csv(os.path.join(DATA_DIR, "MDS_UPDRS_Part_II__Patient_Questionnaire_12Mar2026.csv"),  low_memory=False)
ppmi_prs= pd.read_csv(os.path.join(DATA_DIR, "PPMI_Project_9001_20250624_12Mar2026.csv"),                low_memory=False)
part    = pd.read_csv(os.path.join(DATA_DIR, "Participant_Status_12Mar2026.csv"),                         low_memory=False)

print(f"UPDRS III rows:    {len(updrs):,}")
print(f"UPDRS I rows:      {len(updrs1):,}")
print(f"UPDRS II rows:     {len(updrs2):,}")
print(f"Demographics rows: {len(demo):,}")
print(f"PRS patients:      {len(ppmi_prs):,}")
print(f"Participant rows:  {len(part):,}")

# ============================================================
# CLEAN & MERGE
# ============================================================

updrs["visit_year"] = updrs["EVENT_ID"].map(VISIT_MAP)
updrs_clean = updrs.dropna(subset=["visit_year", "NP3TOT"]).copy()

if "NHY" in updrs_clean.columns:
    updrs_clean["NHY"] = updrs_clean["NHY"].replace(101.0, np.nan)

updrs_clean = updrs_clean.merge(
    age_df[["PATNO", "EVENT_ID", "AGE_AT_VISIT"]],
    on=["PATNO", "EVENT_ID"], how="left"
)

demo_sub = demo[["PATNO", "SEX"]].drop_duplicates("PATNO")
updrs_clean = updrs_clean.merge(demo_sub, on="PATNO", how="left")

# Diagnose SEX encoding before mapping — PPMI uses different encodings across cohorts
sex_vals = updrs_clean["SEX"].dropna().unique()
print(f"\nSEX column unique values found in data: {sorted(sex_vals)}")

if set(sex_vals).issubset({1.0, 2.0, 1, 2}):
    # Classic PPMI export: 1=Male, 2=Female
    updrs_clean["is_male"] = updrs_clean["SEX"].map({1: 1, 2: 0, 1.0: 1, 2.0: 0})
    print("SEX encoding: 1=Male, 2=Female")
elif set(sex_vals).issubset({0.0, 1.0, 0, 1}):
    # Newer PPMI export: 0=Female, 1=Male
    updrs_clean["is_male"] = updrs_clean["SEX"].map({1: 1, 0: 0, 1.0: 1, 0.0: 0})
    print("SEX encoding: 0=Female, 1=Male")
else:
    # Fallback: string-based
    updrs_clean["is_male"] = (
        updrs_clean["SEX"].astype(str).str.strip().str.upper()
        .map({"MALE": 1, "M": 1, "1": 1, "1.0": 1,
              "FEMALE": 0, "F": 0, "2": 0, "2.0": 0, "0": 0, "0.0": 0})
    )
    print("SEX encoding: string/mixed — mapped via string lookup")

n_missing = updrs_clean["is_male"].isna().sum()
n_total   = len(updrs_clean)
print(f"is_male distribution: {updrs_clean['is_male'].value_counts(dropna=False).to_dict()}")
print(f"is_male missing: {n_missing}/{n_total} ({100*n_missing/n_total:.1f}%)")
if n_missing / n_total > 0.5:
    print("WARNING: >50% of is_male values are NaN — sex feature will have low impact!")

moca_sub = moca[["PATNO", "EVENT_ID", "MCATOT"]].dropna(subset=["MCATOT"])
updrs_clean = updrs_clean.merge(moca_sub, on=["PATNO", "EVENT_ID"], how="left")

sbr["putamen_sbr"] = (sbr["PUTAMEN_L_REF_CWM"] + sbr["PUTAMEN_R_REF_CWM"]) / 2
sbr["caudate_sbr"] = (sbr["CAUDATE_L_REF_CWM"] + sbr["CAUDATE_R_REF_CWM"]) / 2
sbr_sub = sbr[["PATNO", "EVENT_ID", "putamen_sbr", "caudate_sbr"]].dropna()
updrs_clean = updrs_clean.merge(sbr_sub, on=["PATNO", "EVENT_ID"], how="left")

updrs_clean["PDSTATE"]       = updrs_clean["PDSTATE"].fillna("MISSING")
updrs_clean["is_on_med"]     = (updrs_clean["PDSTATE"] == "ON").astype(int)
updrs_clean["pdmedyn_clean"] = updrs_clean["PDMEDYN"].fillna(0)
state_map = {"OFF": 0, "ON": 1, "ON_WITHOUT_DOPA": 2, "MISSING": 3}
updrs_clean["pdstate_enc"]   = updrs_clean["PDSTATE"].map(state_map).fillna(4)

# Merge UPDRS Part I (non-motor symptoms total)
updrs1_sub = updrs1[["PATNO", "EVENT_ID", "NP1RTOT"]].dropna(subset=["NP1RTOT"])
updrs_clean = updrs_clean.merge(updrs1_sub, on=["PATNO", "EVENT_ID"], how="left")

# Merge UPDRS Part II (patient self-reported motor total)
updrs2_sub = updrs2[["PATNO", "EVENT_ID", "NP2PTOT"]].dropna(subset=["NP2PTOT"])
updrs_clean = updrs_clean.merge(updrs2_sub, on=["PATNO", "EVENT_ID"], how="left")

# Merge cohort from participant status (PATNO only — stable per patient)
part_sub = part[["PATNO", "COHORT"]].drop_duplicates("PATNO")
updrs_clean = updrs_clean.merge(part_sub, on="PATNO", how="left")
# Encode: 1=PD, 2=Healthy Control, 3=SWEDD, 4=Prodromal
updrs_clean["cohort"] = updrs_clean["COHORT"].fillna(-1).astype(int)

# Merge PRS genetic score (PATNO only — genetic, so time-invariant)
# Use PRS88 as primary score (0 nulls, best coverage)
prs_sub = ppmi_prs[["PATNO", "Genetic_PRS_PRS88"]].drop_duplicates("PATNO")
prs_sub = prs_sub.rename(columns={"Genetic_PRS_PRS88": "prs_score"})
updrs_clean = updrs_clean.merge(prs_sub, on="PATNO", how="left")

updrs_clean = updrs_clean.sort_values(["PATNO", "visit_year"]).reset_index(drop=True)

print(f"Rows after merge:  {len(updrs_clean):,}")
print(f"Unique patients:   {updrs_clean['PATNO'].nunique():,}")
print(f"NP1RTOT coverage:  {100*updrs_clean['NP1RTOT'].notna().mean():.1f}%")
print(f"NP2PTOT coverage:  {100*updrs_clean['NP2PTOT'].notna().mean():.1f}%")
print(f"PRS coverage:      {100*updrs_clean['prs_score'].notna().mean():.1f}%")
print(f"Cohort coverage:   {100*(updrs_clean['cohort'] != -1).mean():.1f}%")
print(f"Cohort dist: {updrs_clean['cohort'].value_counts().to_dict()}")

# ============================================================
# HELPER: trajectory features from prior visits
# ============================================================

def trajectory_features(prior_rows):
    """
    Summarise all prior visits into scalar trajectory features.

    Deduplicates by visit_year first (averages scores within the same
    visit year) so linregress never receives identical x values.
    """
    deduped = (prior_rows
               .groupby("visit_year", as_index=False)["NP3TOT"]
               .mean()
               .sort_values("visit_year"))

    scores = deduped["NP3TOT"].values.astype(float)
    times  = deduped["visit_year"].values.astype(float)
    n      = len(scores)

    feats = {
        "traj_n_visits":    float(n),
        "traj_first_score": float(scores[0]),
        "traj_last_score":  float(scores[-1]),
        "traj_mean_score":  float(np.mean(scores)),
        "traj_std_score":   float(np.std(scores)) if n > 1 else 0.0,
        "traj_min_score":   float(np.min(scores)),
        "traj_max_score":   float(np.max(scores)),
        "traj_time_span":   float(times[-1] - times[0]),
        "traj_last_time":   float(times[-1]),
        "traj_last_delta":  float(scores[-1] - scores[-2]) if n >= 2 else 0.0,
        "traj_last_dt":     float(times[-1]  - times[-2])  if n >= 2 else 0.0,
    }

    times_vary = float(np.ptp(times)) > 0
    if n >= 2 and times_vary:
        slope, intercept, r, _, _ = linregress(times, scores)
        feats["traj_slope"]     = float(slope)
        feats["traj_intercept"] = float(intercept)
        feats["traj_r2"]        = float(r ** 2)
    else:
        feats["traj_slope"]     = 0.0
        feats["traj_intercept"] = float(scores[0])
        feats["traj_r2"]        = 0.0

    if n >= 3 and times_vary:
        deltas      = np.diff(scores)
        dt          = np.diff(times)
        dt[dt == 0] = 1e-6
        rates       = deltas / dt
        mid_times   = (times[:-1] + times[1:]) / 2
        if float(np.ptp(mid_times)) > 0:
            acc_slope, *_ = linregress(mid_times, rates)
            feats["traj_accel"] = float(acc_slope)
        else:
            feats["traj_accel"] = 0.0
    else:
        feats["traj_accel"] = 0.0

    # ---- v2: advanced trajectory features ----

    # Exponential moving average (alpha=0.5, heavier weight on recent visits)
    alpha = 0.5
    ema = float(scores[0])
    for s in scores[1:]:
        ema = alpha * float(s) + (1 - alpha) * ema
    feats["traj_ema_score"] = ema

    # Recency-weighted slope: recent visits count more
    if n >= 2 and times_vary:
        weights   = np.arange(1, n + 1, dtype=float)
        w_mean_t  = np.average(times,  weights=weights)
        w_mean_s  = np.average(scores, weights=weights)
        cov       = np.sum(weights * (times - w_mean_t) * (scores - w_mean_s))
        var       = np.sum(weights * (times - w_mean_t) ** 2)
        feats["traj_weighted_slope"] = float(cov / var) if var > 0 else 0.0
    else:
        feats["traj_weighted_slope"] = 0.0

    # Quadratic fit 2nd-order coefficient (captures curvature / acceleration)
    if n >= 3 and times_vary:
        coeffs = np.polyfit(times, scores, 2)
        feats["traj_quad_coef"] = float(coeffs[0])
    else:
        feats["traj_quad_coef"] = 0.0

    # Net change / total time = average progression speed
    time_span = float(times[-1] - times[0])
    feats["traj_score_velocity"] = (
        float((scores[-1] - scores[0]) / time_span) if times_vary else 0.0
    )

    # Rate of change between the last two deduplicated visits
    if n >= 2:
        last_dt = float(times[-1] - times[-2])
        feats["traj_recent_rate"] = (
            float((scores[-1] - scores[-2]) / last_dt) if last_dt > 0 else 0.0
        )
    else:
        feats["traj_recent_rate"] = 0.0

    # Visit density: how frequently the patient is being seen
    feats["traj_visits_per_year"] = (
        float(n / time_span) if times_vary else float(n)
    )

    # Score range: captures how variable the trajectory has been
    feats["traj_score_range"] = float(np.max(scores) - np.min(scores))

    # Ratio of recent rate to overall slope:
    # >1 = accelerating, <1 = decelerating relative to historical trend
    overall_slope = feats["traj_slope"]
    recent_rate   = feats["traj_recent_rate"]
    feats["traj_recent_vs_slope"] = float(
        recent_rate / (abs(overall_slope) + 1e-6)
    )

    return feats


# ============================================================
# BUILD LONGITUDINAL DATASET
# Each sample = (all visits 0..k) -> predict visit k+1
# ============================================================

print("\nBuilding longitudinal prediction pairs...")

records = []

for patno, patient_df in updrs_clean.groupby("PATNO"):
    patient_df   = patient_df.sort_values("visit_year").reset_index(drop=True)
    unique_years = sorted(patient_df["visit_year"].unique())

    if len(unique_years) < MIN_PRIOR_VISITS + 1:
        continue

    for t_idx in range(MIN_PRIOR_VISITS, len(unique_years)):
        prior_cutoff = unique_years[t_idx - 1]
        target_year  = unique_years[t_idx]

        prior_rows  = patient_df[patient_df["visit_year"] <= prior_cutoff]
        target_rows = patient_df[patient_df["visit_year"] == target_year]

        if prior_rows.empty or target_rows.empty:
            continue

        target_updrs = float(target_rows["NP3TOT"].mean())
        traj         = trajectory_features(prior_rows)
        last         = prior_rows.sort_values("visit_year").iloc[-1]

        rec = {
            "PATNO":            patno,
            "target_visit_yr":  float(target_year),
            "target_updrs3":    target_updrs,
            "age_at_last":      last.get("AGE_AT_VISIT", np.nan),
            "is_male":          last.get("is_male", np.nan),
            "moca_last":        last.get("MCATOT", np.nan),
            "hoehn_yahr_last":  last.get("NHY", np.nan),
            "putamen_sbr_last": last.get("putamen_sbr", np.nan),
            "caudate_sbr_last": last.get("caudate_sbr", np.nan),
            "is_on_med_last":   last.get("is_on_med", np.nan),
            "pdmedyn_last":     last.get("pdmedyn_clean", np.nan),
            "pdstate_enc_last": last.get("pdstate_enc", np.nan),
            "dt_to_target":     float(target_year - prior_cutoff),
            # New features
            "updrs1_last":      last.get("NP1RTOT", np.nan),
            "updrs2_last":      last.get("NP2PTOT", np.nan),
            "cohort":           last.get("cohort", -1),
            "prs_score":        last.get("prs_score", np.nan),
        }
        rec.update(traj)
        records.append(rec)

pairs_df = pd.DataFrame(records).dropna(subset=["target_updrs3"]).reset_index(drop=True)

print(f"Prediction pairs:  {len(pairs_df):,}")
print(f"Patients in pairs: {pairs_df['PATNO'].nunique():,}")

# ============================================================
# DERIVED INTERACTION FEATURES
# ============================================================

# Linear projection: last_score + slope * time_to_target
pairs_df["projected_score"] = (
    pairs_df["traj_last_score"] + pairs_df["traj_slope"] * pairs_df["dt_to_target"]
)

# EMA-based projection using recency-weighted slope
pairs_df["projected_ema"] = (
    pairs_df["traj_ema_score"] + pairs_df["traj_weighted_slope"] * pairs_df["dt_to_target"]
)

# How far the last score sits above/below the fitted trend line
pairs_df["last_score_resid"] = (
    pairs_df["traj_last_score"]
    - (pairs_df["traj_intercept"] + pairs_df["traj_slope"] * pairs_df["traj_last_time"])
)

# Total UPDRS burden across all three parts at last visit
pairs_df["total_updrs_last"] = (
    pairs_df["traj_last_score"].fillna(0)
    + pairs_df["updrs1_last"].fillna(0)
    + pairs_df["updrs2_last"].fillna(0)
)

print(f"\nNew feature coverage in pairs_df:")
print(f"  updrs1_last:   {100*pairs_df['updrs1_last'].notna().mean():.1f}%")
print(f"  updrs2_last:   {100*pairs_df['updrs2_last'].notna().mean():.1f}%")
print(f"  prs_score:     {100*pairs_df['prs_score'].notna().mean():.1f}%")
print(f"  cohort dist:   {pairs_df['cohort'].value_counts().to_dict()}")

# ============================================================
# FEATURE COLUMNS
# ============================================================

FEATURE_COLS = [
    # Trajectory features
    "traj_n_visits", "traj_first_score", "traj_last_score",
    "traj_mean_score", "traj_std_score", "traj_min_score", "traj_max_score",
    "traj_time_span", "traj_last_time", "traj_last_delta", "traj_last_dt",
    "traj_slope", "traj_intercept", "traj_r2", "traj_accel",
    # Advanced trajectory (v2)
    "traj_ema_score", "traj_weighted_slope", "traj_quad_coef",
    "traj_score_velocity", "traj_recent_rate", "traj_visits_per_year",
    "traj_score_range", "traj_recent_vs_slope",
    # Clinical / demographic
    "age_at_last", "is_male", "moca_last", "hoehn_yahr_last",
    "putamen_sbr_last", "caudate_sbr_last",
    "is_on_med_last", "pdmedyn_last", "pdstate_enc_last",
    # Prediction horizon
    "dt_to_target",
    # New data sources
    "updrs1_last", "updrs2_last", "cohort", "prs_score",
    # Derived interactions
    "projected_score", "projected_ema", "last_score_resid",
    "total_updrs_last",
]

X      = pairs_df[FEATURE_COLS].copy()
y      = pairs_df["target_updrs3"].copy()
groups = pairs_df["PATNO"].copy()

print(f"\nFinal dataset: {X.shape[0]:,} rows x {X.shape[1]} features")

print("\nFeature completeness:")
for col in FEATURE_COLS:
    pct = 100 * X[col].notna().mean()
    print(f"  {col:25s}: {pct:6.1f}%")

# ============================================================
# TEMPORAL TRAIN / TEST SPLIT (80/20 by patient median year)
# ============================================================

print("\nTemporal train/test split (by patient median visit year)...")

patient_med_year = (pairs_df.groupby("PATNO")["target_visit_yr"]
                             .median()
                             .sort_values())
cutoff_idx     = int(len(patient_med_year) * 0.80)
train_patients = set(patient_med_year.index[:cutoff_idx])
test_patients  = set(patient_med_year.index[cutoff_idx:])

train_mask = groups.isin(train_patients)
test_mask  = groups.isin(test_patients)

X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]
pairs_train = pairs_df[train_mask].copy().reset_index(drop=True)
pairs_test  = pairs_df[test_mask].copy().reset_index(drop=True)

print(f"Train rows: {len(X_train):,} | patients: {len(train_patients):,}")
print(f"Test rows:  {len(X_test):,}  | patients: {len(test_patients):,}")

# ============================================================
# BASELINE: predict last observed score
# ============================================================

print("\nBaseline: predict last observed score (no change)")

baseline_preds = pairs_test["traj_last_score"].values
bl_mae  = mean_absolute_error(y_test, baseline_preds)
bl_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))
bl_r2   = r2_score(y_test, baseline_preds)

print(f"  MAE:  {bl_mae:.2f}")
print(f"  RMSE: {bl_rmse:.2f}")
print(f"  R2:   {bl_r2:.3f}")

# ============================================================
# XGBOOST MODEL
# ============================================================

print("\nTraining XGBoost (longitudinal)...")

xgb_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model",   XGBRegressor(
        objective        = "reg:squarederror",
        n_estimators     = 600,
        max_depth        = 4,
        learning_rate    = 0.025,
        subsample        = 0.8,
        colsample_bytree = 0.75,
        min_child_weight = 5,
        reg_alpha        = 0.2,
        reg_lambda       = 1.5,
        random_state     = RANDOM_SEED,
        n_jobs           = -1,
    ))
])

xgb_pipeline.fit(X_train, y_train)
preds = xgb_pipeline.predict(X_test)

mae  = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2   = r2_score(y_test, preds)

print("\nXGBoost (longitudinal) results:")
print(f"  MAE:         {mae:.2f}  (baseline: {bl_mae:.2f})")
print(f"  RMSE:        {rmse:.2f}  (baseline: {bl_rmse:.2f})")
print(f"  R2:          {r2:.3f}  (baseline: {bl_r2:.3f})")
print(f"  MAE improve: {100*(bl_mae - mae)/bl_mae:.1f}% over baseline")

# ============================================================
# STORE TEST RESULTS
# ============================================================

pairs_test["actual"]    = y_test.values
pairs_test["predicted"] = preds
pairs_test["residual"]  = preds - y_test.values
pairs_test["abs_error"] = np.abs(pairs_test["residual"])

pairs_test.to_csv(
    os.path.join(OUTPUT_DIR, "xgboost_longitudinal_predictions.csv"), index=False
)

model        = xgb_pipeline.named_steps["model"]
importances  = model.feature_importances_
sorted_idx   = np.argsort(importances)
importance_df = (pd.DataFrame({"feature": FEATURE_COLS, "importance": importances})
                   .sort_values("importance", ascending=False))
importance_df.to_csv(
    os.path.join(OUTPUT_DIR, "xgb_long_feature_importance.csv"), index=False
)

# helper: avoid plt.close clobbering figure state
def save(fname):
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")

# ============================================================
# PLOT 1: Predicted vs Actual
# ============================================================

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, preds, alpha=0.35, s=18, color="steelblue")
lims = [min(float(y_test.min()), float(preds.min())) - 1,
        max(float(y_test.max()), float(preds.max())) + 1]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Fullkomin spá")
ax.set_xlabel("Raunverulegt UPDRS-III (næsta heimsókn)", fontsize=12)
ax.set_ylabel("Spáð UPDRS-III", fontsize=12)
ax.set_title(f"XGBoost: Spáð vs Raunverulegt\n"
             f"MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.3f}", fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
save("xgb_long_predicted_vs_actual.png")

# ============================================================
# PLOT 2: Feature Importance
# ============================================================

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(np.array(FEATURE_COLS)[sorted_idx],
               importances[sorted_idx], color="steelblue", edgecolor="white")
for bar, val in zip(bars, importances[sorted_idx]):
    ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=8)
ax.set_xlabel("Vægi einkenna (e.gain)", fontsize=12)
ax.set_title("XGBoost: Vægi einkenna", fontsize=13)
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
save("xgb_long_feature_importance.png")

# ============================================================
# PLOT 3: Residual histogram
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(pairs_test["residual"], bins=45, edgecolor="white", color="steelblue")
ax.axvline(0, linestyle="--", color="red", linewidth=2, label="Zero error")
ax.axvline(pairs_test["residual"].mean(), linestyle="--", color="orange",
           linewidth=1.5, label=f"Mean = {pairs_test['residual'].mean():.2f}")
ax.set_xlabel("Residual (Predicted - Actual)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Residual Distribution — XGBoost Longitudinal", fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
save("xgb_long_residuals.png")

# ============================================================
# PLOT 4: Absolute error histogram + cumulative CDF
# ============================================================

abs_errs = pairs_test["abs_error"].values
pct_within_5  = 100 * float(np.mean(abs_errs <= 5))
pct_within_10 = 100 * float(np.mean(abs_errs <= 10))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(abs_errs, bins=45, edgecolor="white", color="steelblue")
axes[0].axvline(5,  linestyle="--", color="orange", linewidth=1.5, label="5-pt þröskuldur")
axes[0].axvline(10, linestyle="--", color="red",    linewidth=1.5, label="10-pt þröskuldur")
axes[0].axvline(float(np.median(abs_errs)), linestyle=":", color="green", linewidth=1.5,
                label=f"Miðgildi = {float(np.median(abs_errs)):.1f}")
axes[0].set_xlabel("Töluleg skekkja (MAE) ", fontsize=12)
axes[0].set_ylabel("Fjöldi", fontsize=12)
axes[0].set_title("Heildar dreifing skekkju", fontsize=13)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

sorted_errs = np.sort(abs_errs)
cdf = np.arange(1, len(sorted_errs) + 1) / len(sorted_errs)
axes[1].plot(sorted_errs, cdf * 100, color="steelblue", linewidth=2)
axes[1].axvline(5,  linestyle="--", color="orange", linewidth=1.5, label="5 pts")
axes[1].axvline(10, linestyle="--", color="red",    linewidth=1.5, label="10 pts")
axes[1].annotate(f"{pct_within_5:.1f}% innan 5 pts",
                 xy=(5, pct_within_5), xytext=(7, max(pct_within_5 - 10, 5)),
                 arrowprops=dict(arrowstyle="->"), fontsize=9)
axes[1].annotate(f"{pct_within_10:.1f}% innan 10 pts",
                 xy=(10, pct_within_10), xytext=(12, max(pct_within_10 - 10, 5)),
                 arrowprops=dict(arrowstyle="->"), fontsize=9)
axes[1].set_xlabel("Töluleg skekkja - þröskuldur", fontsize=12)
axes[1].set_ylabel("% af spám", fontsize=12)
axes[1].set_title("Uppsöfnuð dreifing skekkju", fontsize=13)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle(f"Skekkja:  MAE={mae:.2f}, RMSE={rmse:.2f}", fontsize=13)
plt.tight_layout()
save("xgb_long_error_analysis.png")

# ============================================================
# PLOT 5: Model vs Baseline bar chart
# ============================================================

metrics   = ["MAE", "RMSE", "R2"]
base_vals = [bl_mae, bl_rmse, bl_r2]
xgb_vals  = [mae,    rmse,    r2]

fig, axes = plt.subplots(1, 3, figsize=(13, 5))
colors = ["#4878cf", "#e87c2b"]
for idx, (metric, bv, xv) in enumerate(zip(metrics, base_vals, xgb_vals)):
    vals   = [bv, xv]
    labels = ["Grunnlíkan", "XGBoost"]
    bars   = axes[idx].bar(labels, vals, color=colors, width=0.5, edgecolor="white")
    for bar, v in zip(bars, vals):
        axes[idx].text(bar.get_x() + bar.get_width() / 2,
                       bar.get_height() + 0.02 * abs(bar.get_height()) + 0.05,
                       f"{v:.3f}", ha="center", va="bottom",
                       fontsize=11, fontweight="bold")
    axes[idx].set_title(metric, fontsize=13)
    axes[idx].grid(True, alpha=0.3, axis="y")
    if metric == "R2":
        axes[idx].set_ylim(min(0, min(vals)) - 0.05, 1.05)
    else:
        axes[idx].set_ylim(0, max(vals) * 1.3)

fig.suptitle("XGBoost vs Grunnlíkan", fontsize=14)
plt.tight_layout()
save("xgb_long_model_vs_baseline.png")

# ============================================================
# PLOT 6: Score distributions (actual vs predicted) + scatter by horizon
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(y_test.values, bins=35, alpha=0.7, color="steelblue",
             edgecolor="white", label="Raunverulegt")
axes[0].hist(preds, bins=35, alpha=0.7, color="coral",
             edgecolor="white", label="Spáð")
axes[0].axvline(float(y_test.mean()), color="steelblue", linestyle="--",
                linewidth=1.5, label=f"Raunverulegt meðalgildi={float(y_test.mean()):.1f}")
axes[0].axvline(float(preds.mean()), color="coral", linestyle="--",
                linewidth=1.5, label=f"Spáð meðalgildi={float(preds.mean()):.1f}")
axes[0].set_xlabel("UPDRS-III Hreyfiskor", fontsize=12)
axes[0].set_ylabel("Fjöldi", fontsize=12)
axes[0].set_title("Raunverulegt vs spáð - dreifing niðurstaða. (prófunar sett)", fontsize=12)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

sc = axes[1].scatter(y_test, preds,
                     c=pairs_test["dt_to_target"].values,
                     cmap="viridis", alpha=0.5, s=18)
lims = [min(float(y_test.min()), float(preds.min())) - 1,
        max(float(y_test.max()), float(preds.max())) + 1]
axes[1].plot(lims, lims, "r--", linewidth=1.5)
plt.colorbar(sc, ax=axes[1], label="Years to target visit")
axes[1].set_xlabel("Actual UPDRS-III", fontsize=12)
axes[1].set_ylabel("Predicted UPDRS-III", fontsize=12)
axes[1].set_title("Prediction Quality Coloured by Forecast Horizon", fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.suptitle("Score Distribution & Prediction Quality", fontsize=13)
plt.tight_layout()
save("xgb_long_score_distributions.png")

# ============================================================
# PLOT 7: Five example patients
# ============================================================

patient_counts = pairs_test.groupby("PATNO").size()
selected_pats  = (patient_counts[patient_counts >= 3]
                  .sort_values(ascending=False)
                  .index[:5]
                  .tolist())
if len(selected_pats) < 5:
    selected_pats = patient_counts.sort_values(ascending=False).index[:5].tolist()

fig, axes = plt.subplots(3, 2, figsize=(14, 14))
axes = axes.flatten()

for i, patno in enumerate(selected_pats):
    ax = axes[i]
    pt = pairs_test[pairs_test["PATNO"] == patno].sort_values("target_visit_yr")

    target_years     = pt["target_visit_yr"].values
    actual_future    = pt["actual"].values
    predicted_future = pt["predicted"].values
    mae_pt           = float(np.mean(np.abs(predicted_future - actual_future)))

    first_time  = float(pt["traj_last_time"].iloc[0])
    first_score = float(pt["traj_last_score"].iloc[0])

    all_t = np.concatenate([[first_time],  target_years])
    all_s = np.concatenate([[first_score], actual_future])

    ax.plot(all_t, all_s, marker="o", linewidth=2.5,
            color="steelblue", label="Mælt", zorder=3)
    ax.plot(target_years, predicted_future, marker="s", linestyle="--",
            linewidth=2, color="coral", label="Spáð", zorder=3)
    ax.plot([first_time, target_years[0]], [first_score, predicted_future[0]],
            ":", color="coral", linewidth=1.5, alpha=0.6)

    for x, s in zip(all_t, all_s):
        ax.annotate(f"{s:.1f}", (x, s),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8, color="steelblue")
    for x, s in zip(target_years, predicted_future):
        ax.annotate(f"{s:.1f}", (x, s),
                    textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=8, color="coral")

    ax.set_title(f"Sjúklingur {patno}  |  MAE = {mae_pt:.2f}", fontsize=11)
    ax.set_xlabel("Ár heimsóknar", fontsize=10)
    ax.set_ylabel("UPDRS-III", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

for j in range(len(selected_pats), len(axes)):
    fig.delaxes(axes[j])

fig.suptitle(
    "Prófun á fimm sjúklingum - Mælt vs sáð framtíðar UPDRS-III\n", fontsize=13
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
save("xgb_long_five_patients.png")


# ============================================================
# PLOT 9: Average trajectory — all test patients
# x-axis = years since each patient's first prediction point
# one row per patient per relative year
# ============================================================

pairs_test_plot = pairs_test.copy()

# Relative time: first target visit for each patient becomes year 0
pairs_test_plot["relative_year"] = (
    pairs_test_plot["target_visit_yr"]
    - pairs_test_plot.groupby("PATNO")["target_visit_yr"].transform("min")
)

# Keep only one row per patient per relative year
pairs_test_plot_unique = (
    pairs_test_plot
    .sort_values(["PATNO", "target_visit_yr"])
    .groupby(["PATNO", "relative_year"], as_index=False)
    .first()
)

# Average measured and predicted score by relative year
avg = (pairs_test_plot_unique.groupby("relative_year")
                           .agg(
                               mean_actual    = ("actual", "mean"),
                               mean_predicted = ("predicted", "mean"),
                               n              = ("PATNO", "nunique"),
                           )
                           .reset_index()
                           .sort_values("relative_year"))

# Anchor point: average last-known score at the first prediction point
first_mask = pairs_test_plot_unique["relative_year"] == avg["relative_year"].iloc[0]
anchor_year  = 0.0
anchor_score = float(pairs_test_plot_unique.loc[first_mask, "traj_last_score"].mean())

years     = avg["relative_year"].values
act_mean  = avg["mean_actual"].values
pred_mean = avg["mean_predicted"].values
n_pts     = avg["n"].values

fig, ax = plt.subplots(figsize=(11, 6))

# Mean lines
ax.plot(years, act_mean, marker="o", linewidth=2.5, markersize=7,
        color="steelblue", label="Mælt meðalgildi")
ax.plot(years, pred_mean, marker="s", linewidth=2.5, markersize=7,
        linestyle="--", color="coral", label="Spáð meðalgildi")

# Dotted bridge from anchor to first prediction point
ax.plot([anchor_year, years[0]], [anchor_score, pred_mean[0]],
        ":", color="coral", linewidth=1.5, alpha=0.6)
ax.plot([anchor_year, years[0]], [anchor_score, act_mean[0]],
        ":", color="steelblue", linewidth=1.5, alpha=0.6)

# Anchor dot
#ax.scatter([anchor_year], [anchor_score], color="gray", zorder=5, s=60,
           #label=f"Meðaltal fyrsta þekkts gildis ({anchor_score:.1f})")

# Annotate mean values and patient count
for yr, am, pm, n in zip(years, act_mean, pred_mean, n_pts):
    ax.annotate(f"{pm:.1f}", (yr, pm),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=8, color="coral")
    ax.annotate(f"{am:.1f}", (yr, am),
                textcoords="offset points", xytext=(0, -22),
                ha="center", fontsize=8, color="steelblue")

# Overall MAE annotation
overall_mae = float(pairs_test_plot_unique["abs_error"].mean())
ax.text(0.02, 0.97, f"Heildar MAE = {overall_mae:.2f}",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

ax.set_xlabel("Ár frá fyrstu spáheimsókn", fontsize=12)
ax.set_ylabel("UPDRS-III skor (hærra = verra)", fontsize=12)
ax.set_title(
    "Meðal UPDRS-III prófunarsjúklinga\n"
    "Mælt vs spáð miðað við tíma frá fyrstu spáheimsókn",
    fontsize=13
)

ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
save("xgb_long_average_trajectory.png")

# ============================================================
# PLOT 8: Error by prediction horizon
# ============================================================

pairs_test["dt_bin"] = pd.cut(
    pairs_test["dt_to_target"],
    bins=[0, 0.75, 1.25, 2.25, 3.5, 100],
    labels=["<9 mo", "~1 yr", "~2 yr", "~3 yr", ">3 yr"]
)

horizon_stats = (pairs_test.groupby("dt_bin", observed=True)["abs_error"]
                            .agg(["mean", "median", "count"])
                            .reset_index())

fig, ax = plt.subplots(figsize=(9, 5))
x     = np.arange(len(horizon_stats))
width = 0.35
b1 = ax.bar(x - width/2, horizon_stats["mean"],   width,
            label="Mean AE",   color="steelblue", edgecolor="white")
b2 = ax.bar(x + width/2, horizon_stats["median"], width,
            label="Median AE", color="coral",     edgecolor="white")
for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{bar.get_height():.1f}", ha="center", fontsize=9)
for bar in b2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{bar.get_height():.1f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([f"{row['dt_bin']}\n(n={row['count']})"
                    for _, row in horizon_stats.iterrows()])
ax.set_xlabel("Prediction horizon (time to target visit)", fontsize=12)
ax.set_ylabel("Absolute Error (UPDRS points)", fontsize=12)
ax.set_title("Prediction Error by Forecast Horizon", fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
save("xgb_long_error_by_horizon.png")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 72)
print("FINAL SUMMARY")
print("=" * 72)
print(f"{'Metric':<10} {'Baseline':>12} {'XGBoost Long.':>15} {'Improvement':>14}")
print("-" * 55)
print(f"{'MAE':<10} {bl_mae:>12.2f} {mae:>15.2f}  {100*(bl_mae-mae)/bl_mae:>12.1f}%")
print(f"{'RMSE':<10} {bl_rmse:>12.2f} {rmse:>15.2f}  {100*(bl_rmse-rmse)/bl_rmse:>12.1f}%")
print(f"{'R2':<10} {bl_r2:>12.3f} {r2:>15.3f}")
print(f"\n% predictions within  5 pts: {pct_within_5:.1f}%")
print(f"% predictions within 10 pts: {pct_within_10:.1f}%")
print("\nTop 10 features by importance:")
print(importance_df.head(10).to_string(index=False))
print("\nAll outputs saved to:", OUTPUT_DIR)
print("Done.")