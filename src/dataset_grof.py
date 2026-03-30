import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# File paths
# =========================
base = Path(".")

files = {
    "demo": base / "Demographics_12Mar2026.csv",
    "age": base / "Age_at_visit_12Mar2026.csv",
    "status": base / "Participant_Status_12Mar2026.csv",
    "updrs3": base / "MDS-UPDRS_Part_III_12Mar2026.csv",
    "moca": base / "Montreal_Cognitive_Assessment__MoCA__12Mar2026.csv",
    "ledd": base / "LEDD_Concomitant_Medication_Log_12Mar2026.csv",
    "datscan": base / "Xing_Core_Lab_-_Quant_SBR_12Mar2026.csv",
}

# =========================
# Load data
# =========================
dfs = {}
for k, f in files.items():
    dfs[k] = pd.read_csv(f)
    print(f"\n--- {k} ---")
    print("shape:", dfs[k].shape)
    print("columns:", list(dfs[k].columns[:20]))

# =========================
# Helper: choose likely ID / visit columns
# =========================
def find_col(df, candidates):
    cols_upper = {c.upper(): c for c in df.columns}
    for cand in candidates:
        for c in df.columns:
            if c.upper() == cand.upper():
                return c
    for cand in candidates:
        for c in df.columns:
            if cand.upper() in c.upper():
                return c
    return None

# Common likely keys
id_candidates = ["PATNO", "SUBJECT_ID", "Participant_ID", "ID"]
visit_candidates = ["EVENT_ID", "EVENT_ID.", "VISIT_ID", "VISIT", "INFODT", "EXAMDATE"]

# Detect keys
key_info = {}
for name, df in dfs.items():
    id_col = find_col(df, id_candidates)
    visit_col = find_col(df, visit_candidates)
    key_info[name] = {"id": id_col, "visit": visit_col}
    print(name, key_info[name])

# =========================
# Build a merged clinical dataframe
# =========================
demo = dfs["demo"].copy()
age = dfs["age"].copy()
status = dfs["status"].copy()
updrs3 = dfs["updrs3"].copy()
moca = dfs["moca"].copy()
ledd = dfs["ledd"].copy()
datscan = dfs["datscan"].copy()

id_col = key_info["demo"]["id"] or key_info["updrs3"]["id"]

# Merge demographics + age + status by patient
merged = demo.copy()

for extra_name in ["age", "status"]:
    extra = dfs[extra_name].copy()
    extra_id = key_info[extra_name]["id"]
    common = [c for c in [extra_id] if c in extra.columns and c in merged.columns]
    if common:
        merged = merged.merge(extra, on=common, how="left", suffixes=("", f"_{extra_name}"))

# For longitudinal tables, merge on patient + visit if possible
def merge_longitudinal(left, right, left_name, right_name):
    left_id = key_info[left_name]["id"]
    right_id = key_info[right_name]["id"]
    left_visit = key_info[left_name]["visit"]
    right_visit = key_info[right_name]["visit"]

    on_cols = []
    if left_id and right_id and left_id in left.columns and right_id in right.columns:
        if left_id == right_id:
            on_cols.append(left_id)
    if left_visit and right_visit and left_visit in left.columns and right_visit in right.columns:
        if left_visit == right_visit:
            on_cols.append(left_visit)

    if len(on_cols) >= 1:
        return left.merge(right, on=on_cols, how="left", suffixes=("", f"_{right_name}"))
    return left

# Start from UPDRS III as anchor for clinical progression
clinical = updrs3.copy()
clinical = merge_longitudinal(clinical, moca, "updrs3", "moca")
clinical = merge_longitudinal(clinical, age, "updrs3", "age")
clinical = merge_longitudinal(clinical, status, "updrs3", "status")
clinical = merge_longitudinal(clinical, datscan, "updrs3", "datscan")

# Merge demographics by patient only
demo_id = key_info["demo"]["id"]
updrs_id = key_info["updrs3"]["id"]
if demo_id and updrs_id and demo_id == updrs_id:
    demo_small = demo.copy()
    clinical = clinical.merge(demo_small, on=demo_id, how="left", suffixes=("", "_demo"))

print("\nMerged clinical shape:", clinical.shape)

# =========================
# Identify useful columns automatically
# =========================
def find_first_matching(df, patterns):
    for p in patterns:
        for c in df.columns:
            if p.upper() == c.upper():
                return c
    for p in patterns:
        for c in df.columns:
            if p.upper() in c.upper():
                return c
    return None

col_pd = find_first_matching(clinical, ["APPRDX", "COHORT", "DIAGNOSIS", "ENROLL_CAT", "STATUS"])
col_sex = find_first_matching(clinical, ["SEX", "GENDER"])
col_age = find_first_matching(clinical, ["AGE_AT_VISIT", "AGE", "CURRENT_AGE"])
col_visit = find_first_matching(clinical, ["EVENT_ID", "VISIT", "VISIT_ORDER", "SC", "BL", "V01"])
col_updrs3 = find_first_matching(clinical, ["NP3TOT", "UPDRS3", "NUPDRS3", "MDS_UPDRS_III_TOTAL"])
col_moca = find_first_matching(clinical, ["MCATOT", "MOCA", "MOCA_TOTAL"])
col_datscan = find_first_matching(clinical, ["SBR", "CAUDATE", "PUTAMEN", "STRIATUM"])
col_patient = key_info["updrs3"]["id"]

print("\nSelected columns:")
print("patient:", col_patient)
print("pd group:", col_pd)
print("sex:", col_sex)
print("age:", col_age)
print("visit:", col_visit)
print("updrs3:", col_updrs3)
print("moca:", col_moca)
print("datscan:", col_datscan)

# =========================
# Clean numeric columns
# =========================
for c in [col_age, col_updrs3, col_moca, col_datscan]:
    if c and c in clinical.columns:
        clinical[c] = pd.to_numeric(clinical[c], errors="coerce")

# Try to summarize LEDD by patient if available
ledd_id = key_info["ledd"]["id"]
col_ledd_val = find_first_matching(ledd, ["LEDD", "TOTAL_LEDD", "LEDD_TOTAL"])
ledd_summary = None
if ledd_id and col_ledd_val:
    ledd[col_ledd_val] = pd.to_numeric(ledd[col_ledd_val], errors="coerce")
    ledd_summary = ledd.groupby(ledd_id, as_index=False)[col_ledd_val].sum()
    if col_patient and ledd_id == col_patient:
        clinical = clinical.merge(ledd_summary, on=col_patient, how="left", suffixes=("", "_leddsum"))

# =========================
# Make output folder
# =========================
outdir = Path("figures")
outdir.mkdir(exist_ok=True)

# =========================
# 1. Cohort overview
# =========================
if col_pd and col_pd in clinical.columns:
    plt.figure(figsize=(8, 5))
    clinical[col_pd].astype(str).value_counts().plot(kind="bar")
    plt.title("Cohort composition")
    plt.xlabel("Group")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outdir / "01_cohort_composition.png", dpi=300)
    plt.close()

if col_age and col_age in clinical.columns:
    plt.figure(figsize=(8, 5))
    clinical[col_age].dropna().plot(kind="hist", bins=25)
    plt.title("Age distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outdir / "02_age_distribution.png", dpi=300)
    plt.close()

# =========================
# 2. Disease progression over visits
# =========================
if col_patient and col_updrs3:
    # create simple visit order if none exists
    prog = clinical[[c for c in [col_patient, col_visit, col_updrs3] if c is not None]].copy()
    prog = prog.dropna(subset=[col_updrs3])

    if col_visit:
        # Mean score per visit label
        mean_prog = prog.groupby(col_visit, as_index=False)[col_updrs3].mean()
        plt.figure(figsize=(10, 5))
        plt.plot(mean_prog[col_visit].astype(str), mean_prog[col_updrs3], marker="o")
        plt.title("Mean UPDRS Part III across visits")
        plt.xlabel("Visit")
        plt.ylabel("UPDRS III")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(outdir / "03_updrs_progression_mean.png", dpi=300)
        plt.close()

    # sample individual trajectories
    sample_ids = prog[col_patient].dropna().astype(str).unique()[:20]
    plt.figure(figsize=(10, 6))
    for pid in sample_ids:
        sub = prog[prog[col_patient].astype(str) == pid]
        if len(sub) >= 2:
            x = np.arange(len(sub))
            y = sub[col_updrs3].values
            plt.plot(x, y, alpha=0.6)
    plt.title("Example individual UPDRS III trajectories")
    plt.xlabel("Visit index")
    plt.ylabel("UPDRS III")
    plt.tight_layout()
    plt.savefig(outdir / "04_updrs_individual_trajectories.png", dpi=300)
    plt.close()

# =========================
# 3. MoCA vs UPDRS III
# =========================
if col_moca and col_updrs3:
    tmp = clinical[[col_moca, col_updrs3]].dropna()
    if len(tmp) > 10:
        plt.figure(figsize=(7, 6))
        plt.scatter(tmp[col_moca], tmp[col_updrs3], alpha=0.5)
        z = np.polyfit(tmp[col_moca], tmp[col_updrs3], 1)
        p = np.poly1d(z)
        xs = np.linspace(tmp[col_moca].min(), tmp[col_moca].max(), 100)
        plt.plot(xs, p(xs))
        plt.title("MoCA vs UPDRS Part III")
        plt.xlabel("MoCA score")
        plt.ylabel("UPDRS III")
        plt.tight_layout()
        plt.savefig(outdir / "05_moca_vs_updrs3.png", dpi=300)
        plt.close()

# =========================
# 4. DaTscan vs UPDRS III
# =========================
if col_datscan and col_updrs3:
    tmp = clinical[[col_datscan, col_updrs3]].dropna()
    if len(tmp) > 10:
        plt.figure(figsize=(7, 6))
        plt.scatter(tmp[col_datscan], tmp[col_updrs3], alpha=0.5)
        z = np.polyfit(tmp[col_datscan], tmp[col_updrs3], 1)
        p = np.poly1d(z)
        xs = np.linspace(tmp[col_datscan].min(), tmp[col_datscan].max(), 100)
        plt.plot(xs, p(xs))
        plt.title("DaTscan biomarker vs UPDRS Part III")
        plt.xlabel(col_datscan)
        plt.ylabel("UPDRS III")
        plt.tight_layout()
        plt.savefig(outdir / "06_datscan_vs_updrs3.png", dpi=300)
        plt.close()

# =========================
# 5. LEDD distribution / relationship
# =========================
if ledd_summary is not None and col_ledd_val:
    plt.figure(figsize=(8, 5))
    ledd_summary[col_ledd_val].dropna().plot(kind="hist", bins=30)
    plt.title("LEDD distribution")
    plt.xlabel("Total LEDD")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outdir / "07_ledd_distribution.png", dpi=300)
    plt.close()

    if col_patient and col_updrs3 and col_ledd_val in clinical.columns:
        tmp = clinical[[col_ledd_val, col_updrs3]].dropna()
        if len(tmp) > 10:
            plt.figure(figsize=(7, 6))
            plt.scatter(tmp[col_ledd_val], tmp[col_updrs3], alpha=0.5)
            z = np.polyfit(tmp[col_ledd_val], tmp[col_updrs3], 1)
            p = np.poly1d(z)
            xs = np.linspace(tmp[col_ledd_val].min(), tmp[col_ledd_val].max(), 100)
            plt.plot(xs, p(xs))
            plt.title("Medication burden vs UPDRS Part III")
            plt.xlabel("LEDD")
            plt.ylabel("UPDRS III")
            plt.tight_layout()
            plt.savefig(outdir / "08_ledd_vs_updrs3.png", dpi=300)
            plt.close()

# =========================
# 6. Correlation heatmap without seaborn
# =========================
numeric_candidates = [col_age, col_updrs3, col_moca, col_datscan]
if ledd_summary is not None and col_ledd_val in clinical.columns:
    numeric_candidates.append(col_ledd_val)

numeric_candidates = [c for c in numeric_candidates if c and c in clinical.columns]
corr_df = clinical[numeric_candidates].apply(pd.to_numeric, errors="coerce")

if corr_df.shape[1] >= 2:
    corr = corr_df.corr()

    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr, interpolation="nearest")
    plt.colorbar(im)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig(outdir / "09_correlation_heatmap.png", dpi=300)
    plt.close()

print("\nSaved figures to:", outdir.resolve())