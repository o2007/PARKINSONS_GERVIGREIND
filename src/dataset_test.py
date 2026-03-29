import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "./data/PPMI_data"
OUTPUT_DIR = "./ppmi_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VISIT_MAP = {
    "SC": -0.5, "BL": 0, "V04": 1, "V06": 2, "V08": 3,
    "V10": 4, "V12": 5, "V14": 6, "V16": 7, "V18": 8, "V20": 9,
}

# --------------------------------------------------
# Load files
# --------------------------------------------------
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

# --------------------------------------------------
# Basic preprocessing
# --------------------------------------------------
updrs["visit_year"] = updrs["EVENT_ID"].map(VISIT_MAP)

# Merge useful columns
demo_sub = demo[["PATNO", "SEX"]].drop_duplicates("PATNO")
df = updrs.merge(demo_sub, on="PATNO", how="left")
df = df.merge(
    age_df[["PATNO", "EVENT_ID", "AGE_AT_VISIT"]],
    on=["PATNO", "EVENT_ID"],
    how="left"
)
df = df.merge(
    moca[["PATNO", "EVENT_ID", "MCATOT"]],
    on=["PATNO", "EVENT_ID"],
    how="left"
)

# Keep rows with valid visit year + UPDRS score
df = df.dropna(subset=["visit_year", "NP3TOT"]).copy()

# Helpful subsets
baseline_all = df[df["EVENT_ID"] == "BL"].copy()
on_df = df[df["PDSTATE"] == "ON"].copy()
off_df = df[df["PDSTATE"] == "OFF"].copy()

# PPMI often uses SEX=1 male, 2 female
sex_counts = baseline_all["SEX"].value_counts().sort_index()

# --------------------------------------------------
# Helper for saving
# --------------------------------------------------
def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, name), dpi=180, bbox_inches="tight")
    plt.close()

# --------------------------------------------------
# 1. Dataset size summary bar chart
# --------------------------------------------------
plt.figure(figsize=(8, 5))
labels = ["UPDRS rows", "Unique patients", "Baseline rows", "ON rows", "OFF rows"]
values = [
    len(df),
    df["PATNO"].nunique(),
    len(baseline_all),
    len(on_df),
    len(off_df),
]
bars = plt.bar(labels, values)
plt.title("Dataset Size Overview")
plt.ylabel("Count")
plt.xticks(rotation=20)
for b, v in zip(bars, values):
    plt.text(b.get_x() + b.get_width()/2, b.get_height(), f"{v:,}",
             ha="center", va="bottom")
savefig("dataset_size_overview.png")

# --------------------------------------------------
# 2. Histogram: baseline UPDRS-III
# --------------------------------------------------
plt.figure(figsize=(8, 5))
plt.hist(baseline_all["NP3TOT"], bins=30, edgecolor="white")
plt.axvline(baseline_all["NP3TOT"].mean(), linestyle="--",
            label=f"Mean = {baseline_all['NP3TOT'].mean():.1f}")
plt.title("Baseline UPDRS-III Distribution")
plt.xlabel("UPDRS-III score")
plt.ylabel("Number of patients")
plt.legend()
plt.grid(alpha=0.25)
savefig("hist_baseline_updrs3.png")

# --------------------------------------------------
# 3. Histograms by medication state: ON vs OFF
# --------------------------------------------------
med_subset = df[df["PDSTATE"].isin(["ON", "OFF"])].copy()

plt.figure(figsize=(8, 5))
plt.hist(
    med_subset[med_subset["PDSTATE"] == "ON"]["NP3TOT"],
    bins=30,
    alpha=0.6,
    label="ON",
    edgecolor="white"
)
plt.hist(
    med_subset[med_subset["PDSTATE"] == "OFF"]["NP3TOT"],
    bins=30,
    alpha=0.6,
    label="OFF",
    edgecolor="white"
)
plt.title("UPDRS-III Distribution by Medication State")
plt.xlabel("UPDRS-III score")
plt.ylabel("Number of assessments")
plt.legend()
plt.grid(alpha=0.25)
savefig("hist_updrs_by_medication_state.png")

# --------------------------------------------------
# 4. Boxplot: ON vs OFF medication
# --------------------------------------------------
plt.figure(figsize=(7, 5))
plot_data = [
    med_subset[med_subset["PDSTATE"] == "ON"]["NP3TOT"].dropna(),
    med_subset[med_subset["PDSTATE"] == "OFF"]["NP3TOT"].dropna(),
]
plt.boxplot(plot_data, labels=["ON", "OFF"], showmeans=True)
plt.title("UPDRS-III by Medication State")
plt.ylabel("UPDRS-III score")
plt.grid(alpha=0.25, axis="y")
savefig("boxplot_updrs_on_vs_off.png")

# --------------------------------------------------
# 5. Medication-state availability / history overview
# --------------------------------------------------
pdstate_counts = df["PDSTATE"].fillna("Missing").value_counts()

plt.figure(figsize=(8, 5))
bars = plt.bar(pdstate_counts.index.astype(str), pdstate_counts.values)
plt.title("Medication State Labels in UPDRS Data")
plt.xlabel("PDSTATE")
plt.ylabel("Number of assessments")
for b, v in zip(bars, pdstate_counts.values):
    plt.text(b.get_x() + b.get_width()/2, b.get_height(), f"{v:,}",
             ha="center", va="bottom")
plt.grid(alpha=0.25, axis="y")
savefig("medication_state_counts.png")

# --------------------------------------------------
# 6. Mean UPDRS over time for ON medication
# --------------------------------------------------
on_stats = on_df.groupby("visit_year")["NP3TOT"].agg(["mean", "std", "count"]).reset_index()

plt.figure(figsize=(9, 5))
plt.plot(on_stats["visit_year"], on_stats["mean"], marker="o", linewidth=2)
plt.fill_between(
    on_stats["visit_year"],
    on_stats["mean"] - on_stats["std"],
    on_stats["mean"] + on_stats["std"],
    alpha=0.2,
    label="±1 std"
)
for _, row in on_stats.iterrows():
    plt.annotate(
        f"{row['mean']:.1f}\n(n={int(row['count'])})",
        (row["visit_year"], row["mean"]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=8
    )
plt.title("Mean UPDRS-III Over Time (ON Medication)")
plt.xlabel("Years from baseline")
plt.ylabel("UPDRS-III score")
plt.legend()
plt.grid(alpha=0.25)
savefig("mean_updrs_over_time_on.png")

# --------------------------------------------------
# 7. Mean UPDRS over time: ON vs OFF
# --------------------------------------------------
state_time = med_subset.groupby(["visit_year", "PDSTATE"])["NP3TOT"].mean().unstack()

plt.figure(figsize=(9, 5))
for state in ["ON", "OFF"]:
    if state in state_time.columns:
        plt.plot(state_time.index, state_time[state], marker="o", linewidth=2, label=state)
plt.title("Mean UPDRS-III Over Time by Medication State")
plt.xlabel("Years from baseline")
plt.ylabel("UPDRS-III score")
plt.legend()
plt.grid(alpha=0.25)
savefig("mean_updrs_over_time_on_vs_off.png")

# --------------------------------------------------
# 8. Histogram: age at baseline
# --------------------------------------------------
baseline_age = baseline_all["AGE_AT_VISIT"].dropna()

plt.figure(figsize=(8, 5))
plt.hist(baseline_age, bins=25, edgecolor="white")
plt.axvline(baseline_age.mean(), linestyle="--",
            label=f"Mean = {baseline_age.mean():.1f}")
plt.title("Age Distribution at Baseline")
plt.xlabel("Age")
plt.ylabel("Number of patients")
plt.legend()
plt.grid(alpha=0.25)
savefig("hist_age_baseline.png")

# --------------------------------------------------
# 9. Histogram: MoCA at baseline
# --------------------------------------------------
baseline_moca = baseline_all["MCATOT"].dropna()

plt.figure(figsize=(8, 5))
plt.hist(baseline_moca, bins=25, edgecolor="white")
plt.axvline(baseline_moca.mean(), linestyle="--",
            label=f"Mean = {baseline_moca.mean():.1f}")
plt.title("MoCA Distribution at Baseline")
plt.xlabel("MoCA score")
plt.ylabel("Number of patients")
plt.legend()
plt.grid(alpha=0.25)
savefig("hist_moca_baseline.png")

# --------------------------------------------------
# 10. Histogram: assessments per patient
# --------------------------------------------------
assessments_per_patient = df.groupby("PATNO").size()

plt.figure(figsize=(8, 5))
plt.hist(assessments_per_patient, bins=20, edgecolor="white")
plt.axvline(assessments_per_patient.median(), linestyle="--",
            label=f"Median = {assessments_per_patient.median():.0f}")
plt.title("Assessments Per Patient")
plt.xlabel("Number of assessments")
plt.ylabel("Number of patients")
plt.legend()
plt.grid(alpha=0.25)
savefig("hist_assessments_per_patient.png")

# --------------------------------------------------
# 11. Sex distribution at baseline
# --------------------------------------------------
plt.figure(figsize=(7, 5))
sex_labels = []
sex_values = []

for code, count in sex_counts.items():
    if code == 1:
        sex_labels.append("Male")
    elif code == 2:
        sex_labels.append("Female")
    else:
        sex_labels.append(str(code))
    sex_values.append(count)

bars = plt.bar(sex_labels, sex_values)
plt.title("Sex Distribution at Baseline")
plt.ylabel("Number of patients")
for b, v in zip(bars, sex_values):
    plt.text(b.get_x() + b.get_width()/2, b.get_height(), f"{v:,}",
             ha="center", va="bottom")
plt.grid(alpha=0.25, axis="y")
savefig("sex_distribution_baseline.png")

# --------------------------------------------------
# 12. Missingness of key variables
# --------------------------------------------------
key_cols = ["NP3TOT", "PDSTATE", "AGE_AT_VISIT", "MCATOT", "SEX"]
missing_pct = df[key_cols].isnull().mean() * 100

plt.figure(figsize=(8, 5))
bars = plt.bar(key_cols, missing_pct.values)
plt.title("Missingness of Key Variables")
plt.ylabel("Missing data (%)")
plt.xticks(rotation=20)
for b, v in zip(bars, missing_pct.values):
    plt.text(b.get_x() + b.get_width()/2, b.get_height(), f"{v:.1f}%",
             ha="center", va="bottom")
plt.grid(alpha=0.25, axis="y")
savefig("missingness_key_variables.png")

print("Saved plots to:", OUTPUT_DIR)
print("""
Created files:
- dataset_size_overview.png
- hist_baseline_updrs3.png
- hist_updrs_by_medication_state.png
- boxplot_updrs_on_vs_off.png
- medication_state_counts.png
- mean_updrs_over_time_on.png
- mean_updrs_over_time_on_vs_off.png
- hist_age_baseline.png
- hist_moca_baseline.png
- hist_assessments_per_patient.png
- sex_distribution_baseline.png
- missingness_key_variables.png
""")