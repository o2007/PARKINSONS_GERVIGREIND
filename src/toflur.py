import pandas as pd
import numpy as np
from pathlib import Path

# =========================================================
# 1. Load data
# =========================================================
DATA_DIR = Path("data/PPMI_data")

demo = pd.read_csv(DATA_DIR / "Demographics_12Mar2026.csv")
age = pd.read_csv(DATA_DIR / "Age_at_visit_12Mar2026.csv")
status = pd.read_csv(DATA_DIR / "Participant_Status_12Mar2026.csv")
updrs1 = pd.read_csv(DATA_DIR / "MDS-UPDRS_Part_I_12Mar2026.csv")
updrs2 = pd.read_csv(DATA_DIR / "MDS_UPDRS_Part_II__Patient_Questionnaire_12Mar2026.csv")
updrs3 = pd.read_csv(DATA_DIR / "MDS-UPDRS_Part_III_12Mar2026.csv")
moca = pd.read_csv(DATA_DIR / "Montreal_Cognitive_Assessment__MoCA__12Mar2026.csv")
ledd = pd.read_csv(DATA_DIR / "LEDD_Concomitant_Medication_Log_12Mar2026.csv")
datscan = pd.read_csv(DATA_DIR / "DaTscan_Imaging_12Mar2026.csv")
sbr = pd.read_csv(DATA_DIR / "Xing_Core_Lab_-_Quant_SBR_12Mar2026.csv")

# =========================================================
# 2. Create participant-level summary table
# =========================================================
participants = (
    status[["PATNO", "COHORT", "ENROLL_AGE"]]
    .drop_duplicates(subset=["PATNO"])
    .merge(demo[["PATNO", "SEX", "HANDED", "HISPLAT", "RAWHITE", "RABLACK", "RAASIAN"]].drop_duplicates(subset=["PATNO"]),
           on="PATNO", how="left")
)

n_participants = participants["PATNO"].nunique()

group_counts = participants["COHORT"].value_counts(dropna=False)

male_pct = (participants["SEX"] == 1).mean() * 100 if participants["SEX"].notna().any() else np.nan
mean_enroll_age = participants["ENROLL_AGE"].mean()

participant_summary = pd.DataFrame({
    "Metric": [
        "Number of unique participants",
        "Mean enrollment age",
        "Male participants (%)"
    ],
    "Value": [
        n_participants,
        round(mean_enroll_age, 2),
        round(male_pct, 2)
    ]
})

print("\n" + "="*60)
print("PARTICIPANT-LEVEL SUMMARY")
print("="*60)
print(participant_summary.to_string(index=False))

print("\nCohort counts:")
print(group_counts.to_string())

# =========================================================
# 3. Create visit-level merged clinical dataset
# =========================================================
clinical = (
    updrs3[["PATNO", "EVENT_ID", "NP3TOT", "NHY", "PDSTATE"]]
    .merge(age[["PATNO", "EVENT_ID", "AGE_AT_VISIT"]], on=["PATNO", "EVENT_ID"], how="left")
    .merge(moca[["PATNO", "EVENT_ID", "MCATOT"]], on=["PATNO", "EVENT_ID"], how="left")
    .merge(updrs1[["PATNO", "EVENT_ID", "NP1COG", "NP1HALL", "NP1DPRS", "NP1ANXS", "NP1APAT", "NP1DDS", "NP1RTOT"]],
           on=["PATNO", "EVENT_ID"], how="left")
    .merge(updrs2[["PATNO", "EVENT_ID", "NP2PTOT"]], on=["PATNO", "EVENT_ID"], how="left")
    .merge(status[["PATNO", "COHORT"]], on="PATNO", how="left")
    .merge(demo[["PATNO", "SEX", "HANDED"]], on="PATNO", how="left")
)

# LEDD summarized per PATNO + EVENT_ID
ledd_summary = (
    ledd.groupby(["PATNO", "EVENT_ID"], as_index=False)["LEDD"]
    .sum()
    .rename(columns={"LEDD": "TOTAL_LEDD"})
)

clinical = clinical.merge(ledd_summary, on=["PATNO", "EVENT_ID"], how="left")

# Imaging SBR: select a few key columns
sbr_cols = [
    "PATNO", "EVENT_ID",
    "STRIATUM_REF_CWM",
    "CAUDATE_REF_CWM",
    "PUTAMEN_REF_CWM"
]
clinical = clinical.merge(sbr[sbr_cols], on=["PATNO", "EVENT_ID"], how="left")

# Numeric conversion
numeric_cols = [
    "NP3TOT", "NHY", "AGE_AT_VISIT", "MCATOT", "NP1RTOT", "NP2PTOT",
    "TOTAL_LEDD", "STRIATUM_REF_CWM", "CAUDATE_REF_CWM", "PUTAMEN_REF_CWM"
]

for col in numeric_cols:
    if col in clinical.columns:
        clinical[col] = pd.to_numeric(clinical[col], errors="coerce")

# =========================================================
# 4. Visit-level summary table
# =========================================================
n_rows = len(clinical)
n_unique_pat_visit = clinical[["PATNO", "EVENT_ID"]].drop_duplicates().shape[0]
mean_age_visit = clinical["AGE_AT_VISIT"].mean()
mean_updrs3 = clinical["NP3TOT"].mean()
mean_moca = clinical["MCATOT"].mean()
mean_ledd = clinical["TOTAL_LEDD"].mean()

visit_summary = pd.DataFrame({
    "Metric": [
        "Number of visit-level rows",
        "Number of unique participant-visit observations",
        "Mean age at visit",
        "Mean UPDRS Part III",
        "Mean MoCA",
        "Mean total LEDD"
    ],
    "Value": [
        n_rows,
        n_unique_pat_visit,
        round(mean_age_visit, 2),
        round(mean_updrs3, 2),
        round(mean_moca, 2),
        round(mean_ledd, 2)
    ]
})

print("\n" + "="*60)
print("VISIT-LEVEL SUMMARY")
print("="*60)
print(visit_summary.to_string(index=False))

# =========================================================
# 5. Availability / missingness table
# =========================================================
availability = pd.DataFrame({
    "Variable": [
        "UPDRS Part III (NP3TOT)",
        "Hoehn and Yahr (NHY)",
        "MoCA (MCATOT)",
        "UPDRS Part I total proxy (NP1RTOT)",
        "UPDRS Part II total (NP2PTOT)",
        "LEDD (TOTAL_LEDD)",
        "DaTscan striatum SBR",
        "DaTscan caudate SBR",
        "DaTscan putamen SBR"
    ],
    "Non-missing count": [
        clinical["NP3TOT"].notna().sum(),
        clinical["NHY"].notna().sum(),
        clinical["MCATOT"].notna().sum(),
        clinical["NP1RTOT"].notna().sum(),
        clinical["NP2PTOT"].notna().sum(),
        clinical["TOTAL_LEDD"].notna().sum(),
        clinical["STRIATUM_REF_CWM"].notna().sum(),
        clinical["CAUDATE_REF_CWM"].notna().sum(),
        clinical["PUTAMEN_REF_CWM"].notna().sum()
    ],
    "Missing (%)": [
        round(clinical["NP3TOT"].isna().mean() * 100, 2),
        round(clinical["NHY"].isna().mean() * 100, 2),
        round(clinical["MCATOT"].isna().mean() * 100, 2),
        round(clinical["NP1RTOT"].isna().mean() * 100, 2),
        round(clinical["NP2PTOT"].isna().mean() * 100, 2),
        round(clinical["TOTAL_LEDD"].isna().mean() * 100, 2),
        round(clinical["STRIATUM_REF_CWM"].isna().mean() * 100, 2),        
        round(clinical["CAUDATE_REF_CWM"].isna().mean() * 100, 2),
        round(clinical["PUTAMEN_REF_CWM"].isna().mean() * 100, 2)
    ]
})

print("\n" + "="*60)
print("VARIABLE AVAILABILITY")
print("="*60)
print(availability.to_string(index=False))

# =========================================================
# 6. Numeric descriptive statistics table
# =========================================================
desc_cols = [
    "AGE_AT_VISIT",
    "NP3TOT",
    "NHY",
    "MCATOT",
    "NP1RTOT",
    "NP2PTOT",
    "TOTAL_LEDD",
    "STRIATUM_REF_CWM",
    "CAUDATE_REF_CWM",
    "PUTAMEN_REF_CWM"
]

desc_table = clinical[desc_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]]
desc_table = desc_table.rename(columns={"50%": "median"})
desc_table = desc_table.round(2)

print("\n" + "="*60)
print("NUMERIC DESCRIPTIVE STATISTICS")
print("="*60)
print(desc_table.to_string())

# =========================================================
# 7. Visits per participant table
# =========================================================
visits_per_patient = clinical.groupby("PATNO")["EVENT_ID"].nunique()

visit_dist = pd.DataFrame({
    "Metric": [
        "Mean visits per participant",
        "Median visits per participant",
        "Minimum visits per participant",
        "Maximum visits per participant"
    ],
    "Value": [
        round(visits_per_patient.mean(), 2),
        round(visits_per_patient.median(), 2),
        visits_per_patient.min(),
        visits_per_patient.max()
    ]
})

print("\n" + "="*60)
print("VISIT DISTRIBUTION PER PARTICIPANT")
print("="*60)
print(visit_dist.to_string(index=False))
