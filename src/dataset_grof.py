import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# Paths
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "PPMI_data"
FIG_DIR = BASE_DIR / "ppmi_outputs" / "grof"

FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 300

VISIT_MONTH_MAP = {
    "SC": -6,
    "BL": 0,
    "V01": 3, "V02": 6, "V03": 9, "V04": 12, "V05": 18, "V06": 24,
    "V07": 30, "V08": 36, "V09": 42, "V10": 48, "V11": 54, "V12": 60,
    "V13": 72, "V14": 84, "V15": 96, "V16": 108, "V17": 120, "V18": 132,
    "V19": 144, "V20": 156,
    "R01": 6, "R04": 18, "R06": 30, "R08": 42, "R10": 54, "R12": 66,
    "R13": 78, "R14": 90, "R15": 102, "R16": 114, "R17": 126,
    "R18": 138, "R19": 150, "R20": 162, "R21": 174,
}

# =========================================================
# Load data
# =========================================================
if not DATA_DIR.exists():
    raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

demo = pd.read_csv(DATA_DIR / "Demographics_12Mar2026.csv", low_memory=False)
age = pd.read_csv(DATA_DIR / "Age_at_visit_12Mar2026.csv", low_memory=False)
status = pd.read_csv(DATA_DIR / "Participant_Status_12Mar2026.csv", low_memory=False)
updrs1 = pd.read_csv(DATA_DIR / "MDS-UPDRS_Part_I_12Mar2026.csv", low_memory=False)
updrs2 = pd.read_csv(DATA_DIR / "MDS_UPDRS_Part_II__Patient_Questionnaire_12Mar2026.csv", low_memory=False)
updrs3 = pd.read_csv(DATA_DIR / "MDS-UPDRS_Part_III_12Mar2026.csv", low_memory=False)
moca = pd.read_csv(DATA_DIR / "Montreal_Cognitive_Assessment__MoCA__12Mar2026.csv", low_memory=False)
ledd = pd.read_csv(DATA_DIR / "LEDD_Concomitant_Medication_Log_12Mar2026.csv", low_memory=False)
sbr = pd.read_csv(DATA_DIR / "Xing_Core_Lab_-_Quant_SBR_12Mar2026.csv", low_memory=False)

# =========================================================
# Basic cleaning
# =========================================================
for df in [demo, age, status, updrs1, updrs2, updrs3, moca, ledd, sbr]:
    if "PATNO" in df.columns:
        df["PATNO"] = pd.to_numeric(df["PATNO"], errors="coerce")
    if "EVENT_ID" in df.columns:
        df["EVENT_ID"] = df["EVENT_ID"].astype(str).str.strip()

numeric_cols = {
    "age": ["AGE_AT_VISIT"],
    "updrs1": ["NP1COG", "NP1HALL", "NP1DPRS", "NP1ANXS", "NP1APAT", "NP1DDS", "NP1RTOT"],
    "updrs2": ["NP2PTOT"],
    "updrs3": ["NP3TOT", "NHY"],
    "moca": ["MCATOT"],
    "ledd": ["LEDD"],
    "sbr": ["STRIATUM_REF_CWM", "CAUDATE_REF_CWM", "PUTAMEN_REF_CWM"],
}

for c in numeric_cols["age"]:
    if c in age.columns:
        age[c] = pd.to_numeric(age[c], errors="coerce")
for c in numeric_cols["updrs1"]:
    if c in updrs1.columns:
        updrs1[c] = pd.to_numeric(updrs1[c], errors="coerce")
for c in numeric_cols["updrs2"]:
    if c in updrs2.columns:
        updrs2[c] = pd.to_numeric(updrs2[c], errors="coerce")
for c in numeric_cols["updrs3"]:
    if c in updrs3.columns:
        updrs3[c] = pd.to_numeric(updrs3[c], errors="coerce")
for c in numeric_cols["moca"]:
    if c in moca.columns:
        moca[c] = pd.to_numeric(moca[c], errors="coerce")
for c in numeric_cols["ledd"]:
    if c in ledd.columns:
        ledd[c] = pd.to_numeric(ledd[c], errors="coerce")
for c in numeric_cols["sbr"]:
    if c in sbr.columns:
        sbr[c] = pd.to_numeric(sbr[c], errors="coerce")


def dedupe_by_key(df, keys, agg_map):
    return df.groupby(keys, dropna=False, as_index=False).agg(agg_map)


def clean_clinical_ranges(df):
    df = df.copy()
    if "AGE_AT_VISIT" in df.columns:
        df.loc[(df["AGE_AT_VISIT"] < 18) | (df["AGE_AT_VISIT"] > 100), "AGE_AT_VISIT"] = np.nan
    if "ENROLL_AGE" in df.columns:
        df.loc[(df["ENROLL_AGE"] < 18) | (df["ENROLL_AGE"] > 100), "ENROLL_AGE"] = np.nan
    if "MCATOT" in df.columns:
        df.loc[(df["MCATOT"] < 0) | (df["MCATOT"] > 30), "MCATOT"] = np.nan
    if "NHY" in df.columns:
        df["NHY"] = df["NHY"].replace(101.0, np.nan)
        df.loc[~df["NHY"].isin([0, 1, 2, 3, 4, 5]) & df["NHY"].notna(), "NHY"] = np.nan
    if "NP3TOT" in df.columns:
        df.loc[(df["NP3TOT"] < 0) | (df["NP3TOT"] > 132), "NP3TOT"] = np.nan
    for c in ["STRIATUM_REF_CWM", "CAUDATE_REF_CWM", "PUTAMEN_REF_CWM"]:
        if c in df.columns:
            df.loc[df[c] < 0, c] = np.nan
    return df

# =========================================================
# Participant-level data
# =========================================================
demo_cols = ["PATNO", "SEX", "HANDED"]
demo_cols = [c for c in demo_cols if c in demo.columns]

status_cols = ["PATNO", "COHORT", "ENROLL_AGE"]
status_cols = [c for c in status_cols if c in status.columns]

participants = (
    demo[demo_cols]
    .drop_duplicates(subset=["PATNO"])
    .merge(status[status_cols].drop_duplicates(subset=["PATNO"]),
           on="PATNO", how="left")
)

# Participant-level LEDD summary
ledd_summary = (
    ledd.groupby("PATNO", as_index=False)["LEDD"]
    .sum()
    .rename(columns={"LEDD": "TOTAL_LEDD"})
)

participants = participants.merge(ledd_summary, on="PATNO", how="left")
if "ENROLL_AGE" in participants.columns:
    participants["ENROLL_AGE"] = pd.to_numeric(participants["ENROLL_AGE"], errors="coerce")
participants = clean_clinical_ranges(participants)

# =========================================================
# Visit-level clinical data
# =========================================================
clinical = (
    dedupe_by_key(
        updrs3[["PATNO", "EVENT_ID", "NP3TOT", "NHY", "PDSTATE"]],
        ["PATNO", "EVENT_ID"],
        {"NP3TOT": "mean", "NHY": "first", "PDSTATE": "first"},
    )
    .merge(
        dedupe_by_key(age[["PATNO", "EVENT_ID", "AGE_AT_VISIT"]], ["PATNO", "EVENT_ID"], {"AGE_AT_VISIT": "mean"}),
        on=["PATNO", "EVENT_ID"], how="left"
    )
    .merge(
        dedupe_by_key(moca[["PATNO", "EVENT_ID", "MCATOT"]], ["PATNO", "EVENT_ID"], {"MCATOT": "mean"}),
        on=["PATNO", "EVENT_ID"], how="left"
    )
    .merge(
        dedupe_by_key(updrs1[["PATNO", "EVENT_ID", "NP1RTOT"]], ["PATNO", "EVENT_ID"], {"NP1RTOT": "mean"}),
        on=["PATNO", "EVENT_ID"], how="left"
    )
    .merge(
        dedupe_by_key(updrs2[["PATNO", "EVENT_ID", "NP2PTOT"]], ["PATNO", "EVENT_ID"], {"NP2PTOT": "mean"}),
        on=["PATNO", "EVENT_ID"], how="left"
    )
    .merge(status[["PATNO", "COHORT"]].drop_duplicates(subset=["PATNO"]), on="PATNO", how="left")
    .merge(demo[["PATNO", "SEX"]].drop_duplicates(subset=["PATNO"]), on="PATNO", how="left")
    .merge(ledd_summary, on="PATNO", how="left")
    .merge(
        dedupe_by_key(
            sbr[["PATNO", "EVENT_ID", "STRIATUM_REF_CWM", "CAUDATE_REF_CWM", "PUTAMEN_REF_CWM"]],
            ["PATNO", "EVENT_ID"],
            {"STRIATUM_REF_CWM": "mean", "CAUDATE_REF_CWM": "mean", "PUTAMEN_REF_CWM": "mean"},
        ),
        on=["PATNO", "EVENT_ID"],
        how="left"
    )
)
clinical = clean_clinical_ranges(clinical)
clinical["visit_month"] = clinical["EVENT_ID"].map(VISIT_MONTH_MAP)

# =========================================================
# Helper functions
# =========================================================
def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close()

def safe_value_counts(series):
    return series.astype(str).fillna("Vantar").value_counts()

def robust_limits(series, lower_q=0.01, upper_q=0.99, pad_ratio=0.06, floor_zero=False):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return (0.0, 1.0)
    lo = float(s.quantile(lower_q))
    hi = float(s.quantile(upper_q))
    if lo == hi:
        lo = float(s.min())
        hi = float(s.max())
    span = max(hi - lo, 1.0)
    pad = span * pad_ratio
    lo -= pad
    hi += pad
    if floor_zero:
        lo = max(0.0, lo)
    return lo, hi


def plot_hist(series, title, xlabel, filename, bins=25, lower_q=0.01, upper_q=0.99, floor_zero=False):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return
    lo, hi = robust_limits(s, lower_q=lower_q, upper_q=upper_q, floor_zero=floor_zero)
    shown = s[s.between(lo, hi)]
    if len(shown) < max(20, int(0.5 * len(s))):
        shown = s
        lo, hi = float(s.min()), float(s.max())
    plt.figure(figsize=(8, 5))
    counts, _, _ = plt.hist(shown, bins=bins, edgecolor="white", color="steelblue")
    plt.xlim(lo, hi)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Fjöldi")
    plt.ylim(0, float(np.max(counts)) * 1.12)
    plt.grid(alpha=0.25, axis="y")
    savefig(filename)

def plot_bar(series, title, xlabel, ylabel, filename, max_bars=12):
    vc = safe_value_counts(series).head(max_bars)
    plt.figure(figsize=(8, 5))
    bars = plt.bar(vc.index.astype(str), vc.values, color="steelblue", edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, float(vc.max()) * 1.12)
    plt.grid(alpha=0.25, axis="y")
    for bar, value in zip(bars, vc.values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{int(value)}",
                 ha="center", va="bottom", fontsize=8)
    savefig(filename)

def boxplot_by_group(df, group_col, value_col, title, ylabel, filename, max_groups=8):
    tmp = df[[group_col, value_col]].dropna().copy()
    if len(tmp) == 0:
        return
    top_groups = tmp[group_col].astype(str).value_counts().index[:max_groups]
    tmp = tmp[tmp[group_col].astype(str).isin(top_groups)]
    data = [tmp.loc[tmp[group_col].astype(str) == g, value_col].values for g in top_groups]
    plt.figure(figsize=(10, 5))
    plt.boxplot(data, labels=top_groups, showfliers=False)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    lo, hi = robust_limits(tmp[value_col], floor_zero=True)
    plt.ylim(lo, hi)
    plt.grid(alpha=0.25, axis="y")
    savefig(filename)


def plot_scatter_with_trend(df, x_col, y_col, title, xlabel, ylabel, filename,
                            x_floor_zero=False, y_floor_zero=False):
    tmp = df[[x_col, y_col]].dropna().copy()
    if len(tmp) <= 10:
        return
    x_lo, x_hi = robust_limits(tmp[x_col], floor_zero=x_floor_zero)
    y_lo, y_hi = robust_limits(tmp[y_col], floor_zero=y_floor_zero)
    tmp = tmp[tmp[x_col].between(x_lo, x_hi) & tmp[y_col].between(y_lo, y_hi)]
    plt.figure(figsize=(7, 6))
    plt.scatter(tmp[x_col], tmp[y_col], alpha=0.35, s=16, color="steelblue")
    z = np.polyfit(tmp[x_col], tmp[y_col], 1)
    p = np.poly1d(z)
    xs = np.linspace(tmp[x_col].min(), tmp[x_col].max(), 100)
    plt.plot(xs, p(xs), color="coral", linewidth=2)
    corr = tmp[x_col].corr(tmp[y_col])
    plt.title(f"{title} (r = {corr:.2f})")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(x_lo, x_hi)
    plt.ylim(y_lo, y_hi)
    plt.grid(alpha=0.25)
    savefig(filename)

# =========================================================
# 1. Cohort composition
# =========================================================
plot_bar(
    participants["COHORT"],
    "Samsetning hópa",
    "Hópur",
    "Fjöldi þátttakenda",
    "01_cohort_composition.png"
)

# =========================================================
# 2. Sex distribution
# =========================================================
sex_map = {0: "Kona", 1: "Karl"}
sex_numeric = pd.to_numeric(participants["SEX"], errors="coerce")
sex_labels = sex_numeric.map(sex_map).dropna()
plot_bar(
    sex_labels,
    "Kynjahlutfall",
    "Kyn",
    "Fjöldi þátttakenda",
    "02_sex_distribution.png"
)

# =========================================================
# 3. Enrollment age distribution
# =========================================================
if "ENROLL_AGE" in participants.columns:
    plot_hist(
        participants["ENROLL_AGE"],
        "Dreifing aldurs við byrjun rannsóknar",
        "Aldur við byrjun rannsóknar",
        "03_enrollment_age_distribution.png",
        bins=25,
        floor_zero=True
    )
elif "AGE_AT_VISIT" in clinical.columns:
    plot_hist(
        clinical["AGE_AT_VISIT"],
        "Dreifing aldurs",
        "Aldur",
        "03_enrollment_age_distribution.png",
        bins=25,
        floor_zero=True
    )

# =========================================================
# 4. Age at visit distribution
# =========================================================
plot_hist(
    clinical["AGE_AT_VISIT"],
    "Dreifing aldurs við heimsókn",
    "Aldur við heimsókn",
    "04_age_at_visit_distribution.png",
    bins=25,
    floor_zero=True
)

# =========================================================
# 5. Visits per participant
# =========================================================
visits_per_patient = clinical.groupby("PATNO")["EVENT_ID"].nunique()
plot_hist(
    visits_per_patient,
    "Heimsóknir á hvern þátttakanda",
    "Fjöldi heimsókna",
    "05_visits_per_participant.png",
    bins=20,
    floor_zero=True
)

# =========================================================
# 6. UPDRS III distribution
# =========================================================
plot_hist(
    clinical["NP3TOT"],
    "Dreifing á MDS-UPDRS hluta III",
    "UPDRS hluti III stig",
    "06_updrs3_distribution.png",
    bins=30,
    floor_zero=True
)

# =========================================================
# 7. MoCA distribution
# =========================================================
plot_hist(
    clinical["MCATOT"],
    "Dreifing á MoCA stiga",
    "MoCA stig",
    "07_moca_distribution.png",
    bins=20,
    floor_zero=True
)

# =========================================================
# 8. LEDD distribution
# =========================================================
plot_hist(
    participants["TOTAL_LEDD"],
    "Dreifing heildar-LEDD á hvern þátttakanda (Lyfjagjöf)",
    "Heildar-LEDD",
    "08_ledd_distribution.png",
    bins=30,
    floor_zero=True
)

# =========================================================
# 9. Mean UPDRS III across visits (PD cohort only, grouped by visit year)
# =========================================================
tmp9 = clinical[["visit_month", "NP3TOT", "COHORT"]].dropna(subset=["visit_month", "NP3TOT"]).copy()
tmp9 = tmp9[tmp9["COHORT"] == "PD"].copy()
tmp9 = tmp9[tmp9["visit_month"].between(0, 156)].copy()

if len(tmp9) > 0:
    tmp9["visit_year"] = (tmp9["visit_month"] / 12).round(1)

    avg9 = (tmp9.groupby("visit_year")["NP3TOT"]
               .agg(mean_score="mean", n="count")
               .reset_index()
               .sort_values("visit_year"))
    avg9 = avg9[avg9["n"] >= 10]

    years9 = avg9["visit_year"].values
    means9 = avg9["mean_score"].values
    ns9    = avg9["n"].values

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(years9, means9, marker="o", linewidth=2.5, markersize=7, color="steelblue")

    for yr, mn, n in zip(years9, means9, ns9):
        ax.annotate(f"{mn:.1f}\n(n={int(n)})", (yr, mn),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=7.5, color="steelblue")

    ax.set_title("Meðalframvinda MDS-UPDRS hluta III — PD hópur", fontsize=13)
    ax.set_xlabel("Ár frá fyrstu heimsókn (BL = 0)", fontsize=11)
    ax.set_ylabel("Meðaltal UPDRS hluta III stiga", fontsize=11)
    ax.set_xlim(-0.2, years9.max() + 0.5)
    y_lo, y_hi = robust_limits(avg9["mean_score"], floor_zero=True, pad_ratio=0.15)
    ax.set_ylim(y_lo, y_hi)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig("09_mean_updrs3_across_visits.png")

# =========================================================
# 10. Example individual UPDRS trajectories
# =========================================================
prog = clinical[["PATNO", "EVENT_ID", "visit_month", "NP3TOT"]].dropna().copy()
if len(prog) > 0:
    prog = prog[prog["visit_month"].between(0, 156)].copy()
    sample_ids = (prog.groupby("PATNO")["visit_month"].nunique()
                     .sort_values(ascending=False)
                     .index[:12])

    plt.figure(figsize=(10, 6))
    for pid in sample_ids:
        sub = prog[prog["PATNO"] == pid].sort_values("visit_month")
        if len(sub) >= 2:
            plt.plot(sub["visit_month"], sub["NP3TOT"], alpha=0.55, linewidth=1.6)
    plt.title("Dæmi um einstaklingaskokr í UPDRS hluta III")
    plt.xlabel("Mánuðir frá fyrstu heimsókn")
    plt.ylabel("UPDRS hluti III")
    y_lo, y_hi = robust_limits(prog["NP3TOT"], floor_zero=True)
    plt.ylim(y_lo, y_hi)
    plt.xlim(-2, 158)
    plt.grid(alpha=0.25)
    savefig("10_individual_updrs3_trajectories.png")

# =========================================================
# 11. MoCA vs UPDRS III
# =========================================================
plot_scatter_with_trend(
    clinical, "MCATOT", "NP3TOT",
    "MoCA á móti UPDRS hluta III",
    "MoCA", "UPDRS hluti III",
    "11_moca_vs_updrs3.png",
    x_floor_zero=True, y_floor_zero=True
)

# =========================================================
# 12. LEDD vs UPDRS III
# =========================================================
plot_scatter_with_trend(
    clinical, "TOTAL_LEDD", "NP3TOT",
    "LEDD á þátttakandastigi á móti UPDRS hluta III",
    "Heildar-LEDD", "UPDRS hluti III",
    "12_ledd_vs_updrs3.png",
    x_floor_zero=True, y_floor_zero=True
)

# =========================================================
# 13. DaTscan SBR vs UPDRS III
# =========================================================
plot_scatter_with_trend(
    clinical, "STRIATUM_REF_CWM", "NP3TOT",
    "DaTscan striatum SBR á móti UPDRS hluta III",
    "Striatum SBR", "UPDRS hluti III",
    "13_sbr_vs_updrs3.png",
    x_floor_zero=True, y_floor_zero=True
)

# =========================================================
# 14. Missingness bar chart
# =========================================================
important_cols = [
    "AGE_AT_VISIT", "NP1RTOT", "NP2PTOT", "NP3TOT", "NHY",
    "MCATOT", "TOTAL_LEDD", "STRIATUM_REF_CWM", "CAUDATE_REF_CWM", "PUTAMEN_REF_CWM"
]

important_cols = [c for c in important_cols if c in clinical.columns]
missing_pct = clinical[important_cols].isna().mean().sort_values(ascending=False) * 100

# Readable Icelandic labels for report figures
label_map = {
    "AGE_AT_VISIT": "Aldur við heimsókn",
    "NP1RTOT": "UPDRS I heildarskor",
    "NP2PTOT": "UPDRS II heildarskor",
    "NP3TOT": "UPDRS III heildarskor",
    "NHY": "Hoehn og Yahr stig",
    "MCATOT": "MoCA heildarskor",
    "TOTAL_LEDD": "Heildar-LEDD",
    "STRIATUM_REF_CWM": "DaTscan: Striatum SBR",
    "CAUDATE_REF_CWM": "DaTscan: Caudate SBR",
    "PUTAMEN_REF_CWM": "DaTscan: Putamen SBR",
}
plot_labels = [label_map.get(c, c) for c in missing_pct.index]

plt.figure(figsize=(10, 5))
bars = plt.bar(plot_labels, missing_pct.values, color="steelblue", edgecolor="white")
plt.title("Auð gögn eftir breytum")
plt.ylabel("Vantar (%)")
plt.xticks(rotation=45, ha="right")
plt.ylim(0, min(100, float(missing_pct.max()) * 1.12))
plt.grid(alpha=0.25, axis="y")
for bar, value in zip(bars, missing_pct.values):
    plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}%",
             ha="center", va="bottom", fontsize=8)
savefig("14_missingness_by_variable.png")

# =========================================================
# 15. Correlation heatmap
# =========================================================
corr_cols = [
    "AGE_AT_VISIT", "NP1RTOT", "NP2PTOT", "NP3TOT", "NHY",
    "MCATOT", "TOTAL_LEDD", "STRIATUM_REF_CWM", "CAUDATE_REF_CWM", "PUTAMEN_REF_CWM"
]

corr_cols = [c for c in corr_cols if c in clinical.columns]
corr_df = clinical[corr_cols].apply(pd.to_numeric, errors="coerce")
corr = corr_df.corr()

plt.figure(figsize=(9, 7))
im = plt.imshow(corr, interpolation="nearest")
plt.colorbar(im, fraction=0.046, pad=0.04)
plot_corr_labels = [label_map.get(c, c) for c in corr.columns]
plt.xticks(range(len(corr.columns)), plot_corr_labels, rotation=45, ha="right")
plt.yticks(range(len(corr.columns)), plot_corr_labels)
plt.title("Fylgnikort helstu klínísku breyta")
savefig("15_correlation_heatmap.png")

# =========================================================
# 16. Boxplots by cohort
# =========================================================
boxplot_by_group(
    clinical, "COHORT", "NP3TOT",
    "UPDRS hluti III eftir hópum",
    "UPDRS hluti III",
    "16_boxplot_updrs3_by_cohort.png"
)

boxplot_by_group(
    clinical, "COHORT", "MCATOT",
    "MoCA eftir hópum",
    "MoCA",
    "17_boxplot_moca_by_cohort.png"
)

boxplot_by_group(
    participants, "COHORT", "TOTAL_LEDD",
    "Heildar-LEDD eftir hópum",
    "Heildar-LEDD",
    "18_boxplot_ledd_by_cohort.png"
)

print(f"Saved figures to: {FIG_DIR}")
