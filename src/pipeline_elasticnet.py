"""
=============================================================================
PPMI - ElasticNet Mobility Prediction Pipeline
Target: MDS-UPDRS Part III Motor Score (1 year ahead)
Features: baseline UPDRS, visit year, age, sex, MoCA,
          Hoehn & Yahr stage, putamen SBR, caudate SBR
=============================================================================

Run from project root:
    python3 src/pipeline_elasticnet.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR           = "./data/PPMI_data"
OUTPUT_DIR         = "./ppmi_outputs"
PREDICTION_HORIZON = 1
RANDOM_SEED        = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

VISIT_MAP = {
    "SC": -0.5, "BL": 0, "V04": 1, "V06": 2, "V08": 3,
    "V10": 4,   "V12": 5, "V14": 6, "V16": 7, "V18": 8, "V20": 9,
}

# =============================================================================
# SECTION 1: LOAD DATA
# =============================================================================

print("\n" + "="*60)
print("  PPMI ElasticNet Mobility Prediction Pipeline")
print("  Features: 5 original + NHY + Putamen SBR + Caudate SBR")
print("="*60)
print("\n[Step 1] Loading data...")

updrs  = pd.read_csv(os.path.join(DATA_DIR, "MDS-UPDRS_Part_III_12Mar2026.csv"), low_memory=False)
demo   = pd.read_csv(os.path.join(DATA_DIR, "Demographics_12Mar2026.csv"), low_memory=False)
moca   = pd.read_csv(os.path.join(DATA_DIR, "Montreal_Cognitive_Assessment__MoCA__12Mar2026.csv"), low_memory=False)
age_df = pd.read_csv(os.path.join(DATA_DIR, "Age_at_visit_12Mar2026.csv"), low_memory=False)
sbr    = pd.read_csv(os.path.join(DATA_DIR, "Xing_Core_Lab_-_Quant_SBR_12Mar2026.csv"), low_memory=False)

print(f"  UPDRS-III:    {len(updrs):,} rows")
print(f"  Demographics: {len(demo):,} rows")
print(f"  MoCA:         {len(moca):,} rows")
print(f"  Age at visit: {len(age_df):,} rows")
print(f"  DaTscan SBR:  {len(sbr):,} rows")

# =============================================================================
# SECTION 2: FILTER AND CLEAN
# =============================================================================

print("\n[Step 2] Filtering...")

updrs_clean = updrs[
    (updrs["PDSTATE"] == "ON") |
    (updrs["PDSTATE"].isna() & (updrs["PDMEDYN"] == 1))
].copy()

updrs_clean["visit_year"] = updrs_clean["EVENT_ID"].map(VISIT_MAP)
updrs_clean = updrs_clean.dropna(subset=["visit_year", "NP3TOT"])

# Clean Hoehn & Yahr — recode 101.0 (not applicable = healthy controls) to NaN
updrs_clean["NHY"] = updrs_clean["NHY"].replace(101.0, np.nan)

print(f"  Rows after filtering: {len(updrs_clean):,}")
print(f"  Unique patients:      {updrs_clean['PATNO'].nunique():,}")

# =============================================================================
# SECTION 3: MERGE FEATURES
# =============================================================================

print("\n[Step 3] Merging features...")

# Age at each visit
updrs_clean = updrs_clean.merge(
    age_df[["PATNO", "EVENT_ID", "AGE_AT_VISIT"]],
    on=["PATNO", "EVENT_ID"], how="left"
)

# Sex (1=Male, 0=Female)
demo_sub = demo[["PATNO", "SEX"]].drop_duplicates("PATNO")
updrs_clean = updrs_clean.merge(demo_sub, on="PATNO", how="left")

# MoCA cognitive score
moca_sub = moca[["PATNO", "EVENT_ID", "MCATOT"]].dropna(subset=["MCATOT"])
updrs_clean = updrs_clean.merge(moca_sub, on=["PATNO", "EVENT_ID"], how="left")

# DaTscan SBR — putamen and caudate dopamine levels
# Use mean of left and right for each region
sbr["putamen_sbr"] = (sbr["PUTAMEN_L_REF_CWM"] + sbr["PUTAMEN_R_REF_CWM"]) / 2
sbr["caudate_sbr"] = (sbr["CAUDATE_L_REF_CWM"] + sbr["CAUDATE_R_REF_CWM"]) / 2
sbr_sub = sbr[["PATNO", "EVENT_ID", "putamen_sbr", "caudate_sbr"]].dropna()
updrs_clean = updrs_clean.merge(sbr_sub, on=["PATNO", "EVENT_ID"], how="left")

print(f"  Shape after merging: {updrs_clean.shape}")
print(f"  SBR coverage: {updrs_clean['putamen_sbr'].notna().sum():,} rows have DaTscan data")

# =============================================================================
# SECTION 4: BUILD PREDICTION PAIRS
# =============================================================================

print(f"\n[Step 4] Building prediction pairs (horizon = {PREDICTION_HORIZON} year)...")

records = []
for patno, patient_df in updrs_clean.groupby("PATNO"):
    patient_df = patient_df.sort_values("visit_year").reset_index(drop=True)
    for _, row in patient_df.iterrows():
        target_year = row["visit_year"] + PREDICTION_HORIZON
        future = patient_df[
            (patient_df["visit_year"] >= target_year - 0.5) &
            (patient_df["visit_year"] <= target_year + 0.5) &
            (patient_df["visit_year"] > row["visit_year"])
        ]
        if future.empty:
            continue
        future_row = future.iloc[0]
        records.append({
            "PATNO":             patno,
            "baseline_visit_yr": row["visit_year"],
            "baseline_updrs3":   row["NP3TOT"],
            "age_at_visit":      row["AGE_AT_VISIT"],
            "is_male":           row["SEX"],
            "moca_score":        row.get("MCATOT", np.nan),
            "hoehn_yahr":        row["NHY"],
            "putamen_sbr":       row.get("putamen_sbr", np.nan),
            "caudate_sbr":       row.get("caudate_sbr", np.nan),
            "target_updrs3":     future_row["NP3TOT"],
        })

pairs_df = pd.DataFrame(records)
print(f"  Prediction pairs: {len(pairs_df):,}")
print(f"  Unique patients:  {pairs_df['PATNO'].nunique():,}")
print(f"  Target mean ± std: {pairs_df['target_updrs3'].mean():.1f} ± {pairs_df['target_updrs3'].std():.1f}")

# Feature coverage report
print("\n  Feature coverage:")
FEATURE_COLS = [
    "baseline_updrs3",
    "baseline_visit_yr",
    "age_at_visit",
    "is_male",
    "moca_score",
    "hoehn_yahr",
    "putamen_sbr",
    "caudate_sbr",
]
for col in FEATURE_COLS:
    pct = pairs_df[col].notna().mean() * 100
    print(f"    {col:25s}: {pct:.0f}% complete")

# =============================================================================
# SECTION 5: PREPARE FEATURES
# =============================================================================

X = pairs_df[FEATURE_COLS].copy()
y = pairs_df["target_updrs3"].copy()
groups = pairs_df["PATNO"].copy()

mask = y.notna()
X, y, groups = X[mask], y[mask], groups[mask]

print(f"\n[Step 5] Feature matrix: {X.shape[0]:,} samples × {X.shape[1]} features")

# =============================================================================
# SECTION 6: TRAIN/TEST SPLIT BY PATIENT
# =============================================================================

print("\n[Step 6] Train/test split by patient (no data leakage)...")

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print(f"  Train: {len(X_train):,} rows ({groups.iloc[train_idx].nunique()} patients)")
print(f"  Test:  {len(X_test):,} rows ({groups.iloc[test_idx].nunique()} patients)")

preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

# =============================================================================
# SECTION 7: BASELINE MODEL
# =============================================================================

print("\n[Step 7] Baseline model (predict no change)...")

baseline_preds = X_test["baseline_updrs3"].values
baseline_mae   = mean_absolute_error(y_test, baseline_preds)
baseline_rmse  = np.sqrt(mean_squared_error(y_test, baseline_preds))
baseline_r2    = r2_score(y_test, baseline_preds)

print(f"  MAE  = {baseline_mae:.2f}")
print(f"  RMSE = {baseline_rmse:.2f}")
print(f"  R²   = {baseline_r2:.3f}")

# =============================================================================
# SECTION 8: ELASTICNET
# =============================================================================

print("\n[Step 8] Training ElasticNet with cross-validation...")

elastic_cv = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    alphas=np.logspace(-3, 2, 50),
    cv=5,
    max_iter=10000,
    random_state=RANDOM_SEED,
)

elastic_cv.fit(X_train_proc, y_train)

print(f"  Best alpha:    {elastic_cv.alpha_:.4f}")
print(f"  Best l1_ratio: {elastic_cv.l1_ratio_:.2f}")

# =============================================================================
# SECTION 9: EVALUATE
# =============================================================================

preds = elastic_cv.predict(X_test_proc)
mae   = mean_absolute_error(y_test, preds)
rmse  = np.sqrt(mean_squared_error(y_test, preds))
r2    = r2_score(y_test, preds)

print("\n[Step 9] Results:")
print(f"  Baseline MAE:   {baseline_mae:.2f} UPDRS points")
print(f"  ElasticNet MAE: {mae:.2f} UPDRS points")
print(f"  RMSE:           {rmse:.2f}")
print(f"  R²:             {r2:.3f}")
print(f"  Improvement over baseline: {((baseline_mae - mae) / baseline_mae * 100):.1f}%")

# =============================================================================
# SECTION 10: FEATURE COEFFICIENTS
# =============================================================================

print("\n[Step 10] Feature coefficients:")
for name, coef in zip(FEATURE_COLS, elastic_cv.coef_):
    status = "  ← ELIMINATED" if coef == 0.0 else ""
    print(f"  {name:25s}: {coef:+.4f}{status}")

# =============================================================================
# SECTION 11: PLOTS
# =============================================================================

print("\n[Step 11] Generating plots...")

# Plot 1: Predicted vs Actual
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, preds, alpha=0.35, color="#4C72B0", edgecolors="none", s=25)
lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
ax.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
ax.set_xlabel("Actual UPDRS-III Score")
ax.set_ylabel("Predicted UPDRS-III Score")
ax.set_title(f"ElasticNet — Predicted vs Actual\nMAE={mae:.2f}, R²={r2:.3f}")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "elasticnet_predicted_vs_actual.png"), dpi=150)
plt.close()

# Plot 2: Feature coefficients
coefs      = elastic_cv.coef_
colors     = ["#4C72B0" if c != 0 else "#CCCCCC" for c in coefs]
sorted_idx = np.argsort(np.abs(coefs))[::-1]

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(
    [FEATURE_COLS[i] for i in sorted_idx],
    [coefs[i] for i in sorted_idx],
    color=[colors[i] for i in sorted_idx],
    edgecolor="white"
)
ax.invert_yaxis()
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Coefficient (positive = higher future score)")
ax.set_title(f"ElasticNet Feature Coefficients\n"
             f"α={elastic_cv.alpha_:.4f}, l1_ratio={elastic_cv.l1_ratio_:.2f}")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "elasticnet_coefficients.png"), dpi=150)
plt.close()

# Plot 3: Baseline vs ElasticNet
fig, ax = plt.subplots(figsize=(7, 5))
models  = ["Baseline\n(no change)", "ElasticNet\n(5 features)", "ElasticNet\n(8 features)"]
maes    = [baseline_mae, 6.93, mae]   # 6.93 = previous run with 5 features
colors2 = ["#AAAAAA", "#88AACC", "#4C72B0"]

bars = ax.bar(models, maes, color=colors2, edgecolor="white", width=0.5)
for bar, val in zip(bars, maes):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.1,
            f"{val:.2f}", ha="center", fontsize=12, fontweight="bold")

ax.set_ylabel("MAE (UPDRS points)")
ax.set_title("Model Comparison — Effect of Adding Features\n(lower is better)")
ax.set_ylim(0, max(maes) * 1.25)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "elasticnet_feature_improvement.png"), dpi=150)
plt.close()

print("  Saved all plots to ppmi_outputs/")

# =============================================================================
# SUMMARY
# =============================================================================

surviving  = [FEATURE_COLS[i] for i, c in enumerate(elastic_cv.coef_) if c != 0]
eliminated = [FEATURE_COLS[i] for i, c in enumerate(elastic_cv.coef_) if c == 0]

print("\n" + "="*60)
print("  SUMMARY")
print("="*60)
print(f"  Baseline MAE:          {baseline_mae:.2f}")
print(f"  Previous MAE (5 feat): 6.93")
print(f"  Current MAE  (8 feat): {mae:.2f}")
print(f"  R²:                    {r2:.3f}")
print(f"  Improvement vs baseline: {((baseline_mae - mae) / baseline_mae * 100):.1f}%")
print(f"  Surviving features:  {surviving}")
print(f"  Eliminated features: {eliminated}")
print("="*60)