"""
=============================================================================
PPMI - Extended Analysis Plots
Generates 4 additional diagnostic plots for the ElasticNet model
=============================================================================

Run from project root:
    python3 src/analysis_plots.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
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

# =============================================================================
# LOAD + PREPARE DATA (same as pipeline_elasticnet.py)
# =============================================================================

print("\n[Setup] Loading and preparing data...")

updrs  = pd.read_csv(os.path.join(DATA_DIR, "MDS-UPDRS_Part_III_12Mar2026.csv"), low_memory=False)
demo   = pd.read_csv(os.path.join(DATA_DIR, "Demographics_12Mar2026.csv"), low_memory=False)
moca   = pd.read_csv(os.path.join(DATA_DIR, "Montreal_Cognitive_Assessment__MoCA__12Mar2026.csv"), low_memory=False)
age_df = pd.read_csv(os.path.join(DATA_DIR, "Age_at_visit_12Mar2026.csv"), low_memory=False)
sbr    = pd.read_csv(os.path.join(DATA_DIR, "Xing_Core_Lab_-_Quant_SBR_12Mar2026.csv"), low_memory=False)

updrs_clean = updrs[
    (updrs["PDSTATE"] == "ON") |
    (updrs["PDSTATE"].isna() & (updrs["PDMEDYN"] == 1))
].copy()
updrs_clean["visit_year"] = updrs_clean["EVENT_ID"].map(VISIT_MAP)
updrs_clean = updrs_clean.dropna(subset=["visit_year", "NP3TOT"])

updrs_clean = updrs_clean.merge(
    age_df[["PATNO", "EVENT_ID", "AGE_AT_VISIT"]], on=["PATNO", "EVENT_ID"], how="left"
)
demo_sub = demo[["PATNO", "SEX"]].drop_duplicates("PATNO")
updrs_clean = updrs_clean.merge(demo_sub, on="PATNO", how="left")
moca_sub = moca[["PATNO", "EVENT_ID", "MCATOT"]].dropna(subset=["MCATOT"])
updrs_clean = updrs_clean.merge(moca_sub, on=["PATNO", "EVENT_ID"], how="left")

# Clean Hoehn & Yahr
updrs_clean["NHY"] = updrs_clean["NHY"].replace(101.0, np.nan)

# DaTscan SBR
sbr["putamen_sbr"] = (sbr["PUTAMEN_L_REF_CWM"] + sbr["PUTAMEN_R_REF_CWM"]) / 2
sbr["caudate_sbr"] = (sbr["CAUDATE_L_REF_CWM"] + sbr["CAUDATE_R_REF_CWM"]) / 2
sbr_sub = sbr[["PATNO", "EVENT_ID", "putamen_sbr", "caudate_sbr"]].dropna()
updrs_clean = updrs_clean.merge(sbr_sub, on=["PATNO", "EVENT_ID"], how="left")

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
X = pairs_df[FEATURE_COLS].copy()
y = pairs_df["target_updrs3"].copy()
groups = pairs_df["PATNO"].copy()
mask = y.notna()
X, y, groups, pairs_df = X[mask], y[mask], groups[mask], pairs_df[mask]

# Train/test split by patient
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
pairs_test = pairs_df.iloc[test_idx].copy()

preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

# Train ElasticNet
elastic_cv = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    alphas=np.logspace(-3, 2, 50),
    cv=5, max_iter=10000, random_state=RANDOM_SEED,
)
elastic_cv.fit(X_train_proc, y_train)
preds = elastic_cv.predict(X_test_proc)
residuals = preds - y_test.values

pairs_test = pairs_test.reset_index(drop=True)
pairs_test["actual"]    = y_test.values
pairs_test["predicted"] = preds
pairs_test["residual"]  = residuals
pairs_test["abs_error"] = np.abs(residuals)

print(f"  Model ready. MAE = {np.abs(residuals).mean():.2f}")

# =============================================================================
# PLOT 1: RESIDUAL PLOT
# =============================================================================
# Residual = predicted - actual
# If the model is unbiased, residuals should scatter randomly around 0.
# A pattern (e.g. negative at high scores) = systematic error.

print("\n[Plot 1] Residual plot...")

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(y_test, residuals, alpha=0.35, color="#4C72B0", edgecolors="none", s=25)
ax.axhline(0, color="red", linestyle="--", linewidth=1.5, label="Zero error")
ax.axhline(residuals.mean(), color="orange", linestyle="--", linewidth=1.2,
           label=f"Mean residual = {residuals.mean():.2f}")

# Shade the ±5 point clinical zone
ax.axhspan(-5, 5, alpha=0.08, color="green", label="±5 pts (clinical threshold)")

ax.set_xlabel("Actual UPDRS-III Score (1 year later)")
ax.set_ylabel("Residual (Predicted − Actual)")
ax.set_title("Residual Plot — ElasticNet\n"
             "Points above 0 = overpredicted, below 0 = underpredicted")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "analysis_1_residuals.png"), dpi=150)
plt.close()
print("  Saved: analysis_1_residuals.png")

# =============================================================================
# PLOT 2: ERROR DISTRIBUTION HISTOGRAM
# =============================================================================

print("[Plot 2] Error distribution...")

within_5  = (np.abs(residuals) <= 5).mean() * 100
within_10 = (np.abs(residuals) <= 10).mean() * 100

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: residual histogram
axes[0].hist(residuals, bins=35, color="#4C72B0", edgecolor="white", alpha=0.85)
axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5)
axes[0].axvline(residuals.mean(), color="orange", linestyle="--", linewidth=1.2,
                label=f"Mean = {residuals.mean():.2f}")
axes[0].set_xlabel("Residual (Predicted − Actual)")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution of Prediction Errors")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: cumulative % within N points
thresholds = np.arange(0, 31)
pct_within = [(np.abs(residuals) <= t).mean() * 100 for t in thresholds]
axes[1].plot(thresholds, pct_within, color="#4C72B0", linewidth=2.5)
axes[1].axvline(5, color="green", linestyle="--", linewidth=1.5,
                label=f"5 pts: {within_5:.0f}% of predictions")
axes[1].axvline(10, color="orange", linestyle="--", linewidth=1.5,
                label=f"10 pts: {within_10:.0f}% of predictions")
axes[1].set_xlabel("Error Threshold (UPDRS points)")
axes[1].set_ylabel("% of Predictions Within Threshold")
axes[1].set_title("Cumulative Prediction Accuracy")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0, 100)

plt.suptitle(f"Prediction Error Analysis — ElasticNet  (MAE={np.abs(residuals).mean():.2f})",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "analysis_2_error_distribution.png"), dpi=150)
plt.close()
print(f"  Saved: analysis_2_error_distribution.png")
print(f"  {within_5:.0f}% of predictions within 5 UPDRS points")
print(f"  {within_10:.0f}% of predictions within 10 UPDRS points")

# =============================================================================
# PLOT 3: MAE BY DISEASE STAGE
# =============================================================================
# Split patients into Mild / Moderate / Severe based on their baseline score.
# UPDRS-III severity thresholds (commonly used in literature):
#   Mild:     0–20
#   Moderate: 21–40
#   Severe:   41+

print("[Plot 3] MAE by disease stage...")

def get_stage(score):
    if score <= 20:
        return "Mild\n(0–20)"
    elif score <= 40:
        return "Moderate\n(21–40)"
    else:
        return "Severe\n(41+)"

pairs_test["stage"] = pairs_test["baseline_updrs3"].apply(get_stage)

stage_order = ["Mild\n(0–20)", "Moderate\n(21–40)", "Severe\n(41+)"]
stage_stats = pairs_test.groupby("stage").agg(
    n=("abs_error", "count"),
    mae=("abs_error", "mean"),
    within5=("abs_error", lambda x: (x <= 5).mean() * 100)
).reindex(stage_order)

print(f"  Stage breakdown:\n{stage_stats}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

colors = ["#55A868", "#DD8452", "#C44E52"]

# Left: MAE per stage
bars = axes[0].bar(stage_order, stage_stats["mae"], color=colors, edgecolor="white", width=0.5)
axes[0].axhline(5, color="black", linestyle="--", linewidth=1.2, label="Clinical threshold (5 pts)")
for bar, (stage, row) in zip(bars, stage_stats.iterrows()):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.15,
                 f"MAE={row['mae']:.1f}\n(n={int(row['n'])})",
                 ha="center", fontsize=9)
axes[0].set_ylabel("Mean Absolute Error (UPDRS points)")
axes[0].set_title("Prediction Error by Disease Stage")
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis="y")
axes[0].set_ylim(0, stage_stats["mae"].max() * 1.35)

# Right: % within 5 points per stage
bars2 = axes[1].bar(stage_order, stage_stats["within5"], color=colors, edgecolor="white", width=0.5)
axes[1].axhline(50, color="black", linestyle="--", linewidth=1.2, label="50% benchmark")
for bar, (stage, row) in zip(bars2, stage_stats.iterrows()):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 row["within5"] + 1.5,
                 f"{row['within5']:.0f}%",
                 ha="center", fontsize=10, fontweight="bold")
axes[1].set_ylabel("% of Predictions Within 5 UPDRS Points")
axes[1].set_title("Clinical Accuracy by Disease Stage")
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis="y")
axes[1].set_ylim(0, 100)

plt.suptitle("Model Performance by Disease Severity", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "analysis_3_by_stage.png"), dpi=150)
plt.close()
print("  Saved: analysis_3_by_stage.png")

# =============================================================================
# PLOT 4: INDIVIDUAL PATIENT TRAJECTORY PLOTS (one file per patient)
# =============================================================================

print("[Plot 4] Individual patient trajectory plots...")

patient_counts = pairs_test.groupby("PATNO").size()
top_patients   = patient_counts[patient_counts >= 3].index[:6]
if len(top_patients) < 6:
    top_patients = patient_counts.nlargest(6).index

all_visits = updrs_clean[["PATNO", "visit_year", "NP3TOT"]].dropna()

for patno in top_patients:
    pt = pairs_test[pairs_test["PATNO"] == patno].sort_values("baseline_visit_yr")

    mae_pt   = pt["abs_error"].mean()
    within5  = (pt["abs_error"] <= 5).mean() * 100

    # Build two clean series:
    # actual   = score recorded at year+1
    # predicted = what the model said year+1 would be
    years_pred  = pt["baseline_visit_yr"].values + 1
    actual_vals = pt["actual"].values
    pred_vals   = pt["predicted"].values

    fig, ax = plt.subplots(figsize=(10, 5))

    # Actual line
    ax.plot(years_pred, actual_vals, "o-",
            color="#4C72B0", linewidth=2.5, markersize=8,
            label="Actual UPDRS-III (recorded)")

    # Predicted line
    ax.plot(years_pred, pred_vals, "s--",
            color="#DD8452", linewidth=2.5, markersize=8,
            label="Predicted UPDRS-III (model)")

    # Shade the gap between them
    ax.fill_between(years_pred, actual_vals, pred_vals,
                    alpha=0.12, color="#DD8452")

    # Annotate each error simply
    for yr, act, pred in zip(years_pred, actual_vals, pred_vals):
        err = act - pred
        color = "#C44E52" if abs(err) > 5 else "#55A868"
        ax.annotate(f"{err:+.1f}",
                    xy=(yr, (act + pred) / 2),
                    fontsize=9, color=color,
                    ha="left", va="center",
                    xytext=(6, 0), textcoords="offset points")

    ax.set_xlabel("Years from Baseline", fontsize=12)
    ax.set_ylabel("UPDRS-III Score (higher = worse)", fontsize=12)
    ax.set_title(
        f"Patient {patno} — Actual vs Predicted Motor Score\n"
        f"MAE = {mae_pt:.1f} pts  |  {within5:.0f}% of predictions within 5 pts",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"analysis_4_patient_{patno}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: analysis_4_patient_{patno}.png  (MAE={mae_pt:.1f})")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*60)
print("  ANALYSIS SUMMARY")
print("="*60)
print(f"  MAE overall:              {np.abs(residuals).mean():.2f} UPDRS points")
print(f"  Within 5 pts (clinical):  {within_5:.0f}%")
print(f"  Within 10 pts:            {within_10:.0f}%")
print(f"  Mean residual (bias):     {residuals.mean():.2f} (0 = unbiased)")
print(f"\n  MAE by stage:")
for stage, row in stage_stats.iterrows():
    print(f"    {stage.replace(chr(10), ' '):20s}: {row['mae']:.2f} pts  ({row['within5']:.0f}% within 5)")
print("="*60)
print("\n  Plots saved to ppmi_outputs/:")
print("    analysis_1_residuals.png")
print("    analysis_2_error_distribution.png")
print("    analysis_3_by_stage.png")
print("    analysis_4_trajectories.png")