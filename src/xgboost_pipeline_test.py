import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "./data/PPMI_data"
OUTPUT_DIR = "./ppmi_outputs"
RANDOM_SEED = 42

# predict about 1 year ahead
PREDICTION_HORIZON = 1.0
TIME_TOLERANCE = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

VISIT_MAP = {
    "SC": -0.5,
    "BL": 0.0,
    "V04": 1.0,
    "V06": 2.0,
    "V08": 3.0,
    "V10": 4.0,
    "V12": 5.0,
    "V14": 6.0,
    "V16": 7.0,
    "V18": 8.0,
    "V20": 9.0,
}

# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 72)
print("  PARKINSON XGBOOST PIPELINE WITH MEDICATION FEATURES")
print("  Each visit = one datapoint")
print("  Target = real observed UPDRS about 1 year later")
print("=" * 72)

updrs = pd.read_csv(
    os.path.join(DATA_DIR, "MDS-UPDRS_Part_III_12Mar2026.csv"),
    low_memory=False
)
demo = pd.read_csv(
    os.path.join(DATA_DIR, "Demographics_12Mar2026.csv"),
    low_memory=False
)
moca = pd.read_csv(
    os.path.join(DATA_DIR, "Montreal_Cognitive_Assessment__MoCA__12Mar2026.csv"),
    low_memory=False
)
age_df = pd.read_csv(
    os.path.join(DATA_DIR, "Age_at_visit_12Mar2026.csv"),
    low_memory=False
)
sbr = pd.read_csv(
    os.path.join(DATA_DIR, "Xing_Core_Lab_-_Quant_SBR_12Mar2026.csv"),
    low_memory=False
)

print(f"UPDRS rows:         {len(updrs):,}")
print(f"Demographics rows:  {len(demo):,}")
print(f"MoCA rows:          {len(moca):,}")
print(f"Age rows:           {len(age_df):,}")
print(f"SBR rows:           {len(sbr):,}")

# ============================================================
# CLEAN AND MERGE
# IMPORTANT: KEEP ALL MEDICATION STATES
# ============================================================

updrs["visit_year"] = updrs["EVENT_ID"].map(VISIT_MAP)

# Keep all rows with valid visit + valid UPDRS score
updrs_clean = updrs.dropna(subset=["visit_year", "NP3TOT"]).copy()

# Clean Hoehn & Yahr if present
if "NHY" in updrs_clean.columns:
    updrs_clean["NHY"] = updrs_clean["NHY"].replace(101.0, np.nan)

# Merge age
updrs_clean = updrs_clean.merge(
    age_df[["PATNO", "EVENT_ID", "AGE_AT_VISIT"]],
    on=["PATNO", "EVENT_ID"],
    how="left"
)

# Merge sex
demo_sub = demo[["PATNO", "SEX"]].drop_duplicates("PATNO")
updrs_clean = updrs_clean.merge(demo_sub, on="PATNO", how="left")

# Merge MoCA
moca_sub = moca[["PATNO", "EVENT_ID", "MCATOT"]].dropna(subset=["MCATOT"])
updrs_clean = updrs_clean.merge(moca_sub, on=["PATNO", "EVENT_ID"], how="left")

# Merge DaTscan SBR
sbr["putamen_sbr"] = (sbr["PUTAMEN_L_REF_CWM"] + sbr["PUTAMEN_R_REF_CWM"]) / 2
sbr["caudate_sbr"] = (sbr["CAUDATE_L_REF_CWM"] + sbr["CAUDATE_R_REF_CWM"]) / 2
sbr_sub = sbr[["PATNO", "EVENT_ID", "putamen_sbr", "caudate_sbr"]].dropna()
updrs_clean = updrs_clean.merge(sbr_sub, on=["PATNO", "EVENT_ID"], how="left")

# ============================================================
# MEDICATION FEATURES
# ============================================================

# Basic medication indicators
updrs_clean["pdmedyn_clean"] = updrs_clean["PDMEDYN"].fillna(0)

# Keep raw PDSTATE as string for inspection
updrs_clean["PDSTATE"] = updrs_clean["PDSTATE"].fillna("MISSING")

# Simple ON/OFF indicator
updrs_clean["is_on_medication"] = (updrs_clean["PDSTATE"] == "ON").astype(int)
updrs_clean["is_off_medication"] = (updrs_clean["PDSTATE"] == "OFF").astype(int)

# Encoded medication state
# Keep a few common categories and send others to "OTHER"
state_map = {
    "OFF": 0,
    "ON": 1,
    "ON_WITHOUT_DOPA": 2,
    "MISSING": 3,
}
updrs_clean["pdstate_encoded"] = updrs_clean["PDSTATE"].map(state_map).fillna(4)

# Medication interaction with current severity
updrs_clean["med_x_updrs"] = updrs_clean["is_on_medication"] * updrs_clean["NP3TOT"]

# Sex coding
# PPMI usually: 1 = male, 2 = female
updrs_clean["is_male"] = updrs_clean["SEX"].map({1: 1, 2: 0})

print(f"\nRows after merge: {len(updrs_clean):,}")
print(f"Unique patients:  {updrs_clean['PATNO'].nunique():,}")

print("\nMedication state distribution:")
print(updrs_clean["PDSTATE"].value_counts(dropna=False).sort_index())

# ============================================================
# BUILD ONE-STEP-AHEAD PAIRS
# Each observed visit becomes a candidate input
# ============================================================

print("\nBuilding visit-to-future pairs...")

records = []

for patno, patient_df in updrs_clean.groupby("PATNO"):
    patient_df = patient_df.sort_values("visit_year").reset_index(drop=True)

    for _, row in patient_df.iterrows():
        current_year = row["visit_year"]
        target_year = current_year + PREDICTION_HORIZON

        # Find a real future visit close to 1 year later
        future = patient_df[
            (patient_df["visit_year"] > current_year) &
            (patient_df["visit_year"] >= target_year - TIME_TOLERANCE) &
            (patient_df["visit_year"] <= target_year + TIME_TOLERANCE)
        ]

        if future.empty:
            continue

        future_row = future.iloc[0]

        records.append({
            "PATNO": patno,
            "baseline_event_id": row["EVENT_ID"],
            "baseline_visit_yr": row["visit_year"],
            "target_event_id": future_row["EVENT_ID"],
            "target_visit_yr": future_row["visit_year"],

            "baseline_updrs3": row["NP3TOT"],
            "age_at_visit": row.get("AGE_AT_VISIT", np.nan),
            "is_male": row.get("is_male", np.nan),
            "moca_score": row.get("MCATOT", np.nan),
            "hoehn_yahr": row.get("NHY", np.nan),
            "putamen_sbr": row.get("putamen_sbr", np.nan),
            "caudate_sbr": row.get("caudate_sbr", np.nan),

            # medication features
            "is_on_medication": row.get("is_on_medication", np.nan),
            "is_off_medication": row.get("is_off_medication", np.nan),
            "pdmedyn_clean": row.get("pdmedyn_clean", np.nan),
            "pdstate_encoded": row.get("pdstate_encoded", np.nan),
            "med_x_updrs": row.get("med_x_updrs", np.nan),

            "target_updrs3": future_row["NP3TOT"],
        })

pairs_df = pd.DataFrame(records)

print(f"Prediction pairs: {len(pairs_df):,}")
print(f"Patients in pairs: {pairs_df['PATNO'].nunique():,}")

# ============================================================
# FEATURE SET
# ============================================================

FEATURE_COLS = [
    "baseline_updrs3",
    "baseline_visit_yr",
    "age_at_visit",
    "is_male",
    "moca_score",
    "hoehn_yahr",
    "putamen_sbr",
    "caudate_sbr",
    "is_on_medication",
    "is_off_medication",
    "pdmedyn_clean",
    "pdstate_encoded",
    "med_x_updrs",
]

X = pairs_df[FEATURE_COLS].copy()
y = pairs_df["target_updrs3"].copy()
groups = pairs_df["PATNO"].copy()

mask = y.notna()
X = X[mask]
y = y[mask]
groups = groups[mask]
pairs_df = pairs_df[mask].reset_index(drop=True)

print(f"\nFinal dataset: {X.shape[0]:,} rows x {X.shape[1]} features")

print("\nFeature completeness:")
for col in FEATURE_COLS:
    pct = 100 * X[col].notna().mean()
    print(f"  {col:20s}: {pct:6.1f}%")

print("\nMedication feature variation:")
for col in ["is_on_medication", "is_off_medication", "pdmedyn_clean", "pdstate_encoded"]:
    print(f"\n{col}:")
    print(X[col].value_counts(dropna=False).head(10))

# ============================================================
# TRAIN / TEST SPLIT BY PATIENT
# ============================================================

print("\nSplitting by patient...")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=RANDOM_SEED
)

train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()
y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()

pairs_train = pairs_df.iloc[train_idx].copy().reset_index(drop=True)
pairs_test = pairs_df.iloc[test_idx].copy().reset_index(drop=True)

print(f"Train rows: {len(X_train):,} | patients: {pairs_train['PATNO'].nunique():,}")
print(f"Test rows:  {len(X_test):,} | patients: {pairs_test['PATNO'].nunique():,}")

# ============================================================
# BASELINE MODEL
# ============================================================

print("\nBaseline model: predict no change")

baseline_preds = X_test["baseline_updrs3"].values
baseline_mae = mean_absolute_error(y_test, baseline_preds)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))
baseline_r2 = r2_score(y_test, baseline_preds)

print(f"Baseline MAE:  {baseline_mae:.2f}")
print(f"Baseline RMSE: {baseline_rmse:.2f}")
print(f"Baseline R^2:  {baseline_r2:.3f}")

# ============================================================
# XGBOOST MODEL
# ============================================================

print("\nTraining XGBoost...")

xgb_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    ))
])

xgb_pipeline.fit(X_train, y_train)
preds = xgb_pipeline.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print("\nXGBoost results:")
print(f"MAE:         {mae:.2f}")
print(f"RMSE:        {rmse:.2f}")
print(f"R^2:         {r2:.3f}")
print(f"Improvement: {100 * (baseline_mae - mae) / baseline_mae:.1f}% vs baseline MAE")

# ============================================================
# STORE TEST RESULTS
# ============================================================

pairs_test["actual"] = y_test.values
pairs_test["predicted"] = preds
pairs_test["residual"] = pairs_test["predicted"] - pairs_test["actual"]
pairs_test["abs_error"] = np.abs(pairs_test["residual"])

pairs_test.to_csv(
    os.path.join(OUTPUT_DIR, "xgboost_test_predictions_with_medication.csv"),
    index=False
)

# ============================================================
# PLOT 1: PREDICTED VS ACTUAL
# ============================================================

plt.figure(figsize=(6.5, 6))
plt.scatter(y_test, preds, alpha=0.4)
lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
plt.plot(lims, lims, "r--", linewidth=1.5)
plt.xlabel("Actual UPDRS-III")
plt.ylabel("Predicted UPDRS-III")
plt.title(f"XGBoost: Predicted vs Actual\nMAE={mae:.2f}, R^2={r2:.3f}")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "xgboost_predicted_vs_actual_with_medication.png"), dpi=180)
plt.close()

# ============================================================
# PLOT 2: FEATURE IMPORTANCE
# ============================================================

model = xgb_pipeline.named_steps["model"]
importances = model.feature_importances_
sorted_idx = np.argsort(importances)

plt.figure(figsize=(9, 6))
plt.barh(np.array(FEATURE_COLS)[sorted_idx], importances[sorted_idx])
plt.xlabel("Feature importance")
plt.title("XGBoost Feature Importance (with medication features)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "xgboost_feature_importance_with_medication.png"), dpi=180)
plt.close()

# ============================================================
# PLOT 3: MEDICATION FEATURE IMPORTANCE ONLY
# ============================================================

med_feature_names = [
    "is_on_medication",
    "is_off_medication",
    "pdmedyn_clean",
    "pdstate_encoded",
    "med_x_updrs",
]
med_importances = np.array([importances[FEATURE_COLS.index(f)] for f in med_feature_names])

sorted_med_idx = np.argsort(med_importances)

plt.figure(figsize=(8, 5))
plt.barh(np.array(med_feature_names)[sorted_med_idx], med_importances[sorted_med_idx])
plt.xlabel("Feature importance")
plt.title("Medication Feature Importance Only")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "xgboost_medication_feature_importance.png"), dpi=180)
plt.close()

# ============================================================
# PLOT 4: RESIDUAL HISTOGRAM
# ============================================================

plt.figure(figsize=(7, 5))
plt.hist(pairs_test["residual"], bins=35, edgecolor="white")
plt.axvline(0, linestyle="--")
plt.xlabel("Residual (Predicted - Actual)")
plt.ylabel("Count")
plt.title("Residual Distribution (XGBoost)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "xgboost_residual_histogram_with_medication.png"), dpi=180)
plt.close()

# ============================================================
# PLOT 5: ABSOLUTE ERROR HISTOGRAM
# ============================================================

plt.figure(figsize=(7, 5))
plt.hist(pairs_test["abs_error"], bins=35, edgecolor="white")
plt.axvline(5, linestyle="--", label="5-point threshold")
plt.axvline(10, linestyle="--", label="10-point threshold")
plt.xlabel("Absolute error")
plt.ylabel("Count")
plt.title("Absolute Error Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "xgboost_absolute_error_histogram_with_medication.png"), dpi=180)
plt.close()

# ============================================================
# PLOT 6: FIVE PATIENTS WITH FIRST INPUT POINT
# ============================================================

patient_counts = pairs_test.groupby("PATNO").size()

selected_patients = patient_counts[patient_counts >= 2].index.tolist()
if len(selected_patients) < 5:
    selected_patients = patient_counts.sort_values(ascending=False).index.tolist()
selected_patients = selected_patients[:5]

fig, axes = plt.subplots(3, 2, figsize=(13, 12))
axes = axes.flatten()

for i, patno in enumerate(selected_patients):
    ax = axes[i]
    pt = pairs_test[pairs_test["PATNO"] == patno].sort_values("target_visit_yr")

    target_years = pt["target_visit_yr"].values
    actual_future = pt["actual"].values
    predicted_future = pt["predicted"].values
    mae_pt = np.mean(np.abs(predicted_future - actual_future))

    first_input_year = pt["baseline_visit_yr"].iloc[0]
    first_input_score = pt["baseline_updrs3"].iloc[0]

    measured_years = np.concatenate([[first_input_year], target_years])
    measured_scores = np.concatenate([[first_input_score], actual_future])

    ax.plot(
        measured_years,
        measured_scores,
        marker="o",
        linewidth=2,
        label="Measured"
    )

    ax.plot(
        target_years,
        predicted_future,
        marker="s",
        linestyle="--",
        linewidth=2,
        label="Predicted"
    )

    ax.plot(
        [first_input_year, target_years[0]],
        [first_input_score, predicted_future[0]],
        linestyle=":",
        linewidth=1.5,
        alpha=0.6
    )

    for x, yv in zip(measured_years, measured_scores):
        ax.annotate(
            f"{yv:.1f}",
            (x, yv),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8
        )

    for x, yv in zip(target_years, predicted_future):
        ax.annotate(
            f"{yv:.1f}",
            (x, yv),
            textcoords="offset points",
            xytext=(0, -12),
            ha="center",
            fontsize=8
        )

    ax.set_title(
        f"Patient {patno} (start year {first_input_year:.1f})\nMAE={mae_pt:.2f}"
    )
    ax.set_xlabel("Visit year")
    ax.set_ylabel("UPDRS-III")
    ax.grid(True, alpha=0.3)

if len(axes) > len(selected_patients):
    for j in range(len(selected_patients), len(axes)):
        fig.delaxes(axes[j])

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2)
fig.suptitle(
    "Five Example Patients: Input Point, Measured Future Scores, and Predicted Future Scores",
    fontsize=14
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(
    os.path.join(OUTPUT_DIR, "xgboost_five_patients_with_input_point_and_medication.png"),
    dpi=180
)
plt.close()

# ============================================================
# SAVE IMPORTANCE TABLE
# ============================================================

importance_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": importances
}).sort_values("importance", ascending=False)

importance_df.to_csv(
    os.path.join(OUTPUT_DIR, "xgboost_feature_importance_table.csv"),
    index=False
)

# ============================================================
# OPTIONAL DEBUG EXPORT
# ============================================================

if len(selected_patients) > 0:
    debug_patno = selected_patients[0]
    debug_df = pairs_test[pairs_test["PATNO"] == debug_patno][[
        "PATNO",
        "baseline_event_id",
        "baseline_visit_yr",
        "target_event_id",
        "target_visit_yr",
        "baseline_updrs3",
        "is_on_medication",
        "is_off_medication",
        "pdstate_encoded",
        "actual",
        "predicted",
        "abs_error"
    ]].sort_values("target_visit_yr")

    debug_path = os.path.join(OUTPUT_DIR, f"debug_patient_{debug_patno}_with_medication.csv")
    debug_df.to_csv(debug_path, index=False)
    print(f"\nSaved debug table for patient {debug_patno}: {debug_path}")

# ============================================================
# SUMMARY
# ============================================================

print("\nTop feature importances:")
print(importance_df.head(10).to_string(index=False))

print("\nSaved files:")
print(" - xgboost_test_predictions_with_medication.csv")
print(" - xgboost_predicted_vs_actual_with_medication.png")
print(" - xgboost_feature_importance_with_medication.png")
print(" - xgboost_medication_feature_importance.png")
print(" - xgboost_residual_histogram_with_medication.png")
print(" - xgboost_absolute_error_histogram_with_medication.png")
print(" - xgboost_five_patients_with_input_point_and_medication.png")
print(" - xgboost_feature_importance_table.csv")

print("\nDone.")