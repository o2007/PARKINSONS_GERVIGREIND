import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 1. LOAD FILES
# =========================================================
updrs3 = pd.read_csv(
    "data/PPMI_data/MDS-UPDRS_Part_III_12Mar2026.csv",
    low_memory=False
)
moca = pd.read_csv(
    "data/PPMI_data/Montreal_Cognitive_Assessment__MoCA__12Mar2026.csv",
    low_memory=False
)
demo = pd.read_csv(
    "data/PPMI_data/Demographics_12Mar2026.csv",
    low_memory=False
)
age = pd.read_csv(
    "data/PPMI_data/Age_at_visit_12Mar2026.csv",
    low_memory=False
)
status = pd.read_csv(
    "data/PPMI_data/Participant_Status_12Mar2026.csv",
    low_memory=False
)
ledd = pd.read_csv(
    "data/PPMI_data/LEDD_Concomitant_Medication_Log_12Mar2026.csv",
    low_memory=False
)
updrs1 = pd.read_csv(
    "data/PPMI_data/MDS-UPDRS_Part_I_12Mar2026.csv",
    low_memory=False
)
updrs2 = pd.read_csv(
    "data/PPMI_data/MDS_UPDRS_Part_II__Patient_Questionnaire_12Mar2026.csv",
    low_memory=False
)


# =========================================================
# 2. SELECT / CLEAN COLUMNS
# =========================================================

# ---------- UPDRS III ----------
updrs3_cols = [
    "PATNO", "EVENT_ID", "INFODT", "PDSTATE", "NP3TOT", "NHY",
    "NP3SPCH", "NP3FACXP", "NP3RIGN", "NP3RIGRU", "NP3RIGLU",
    "NP3RIGRL", "NP3RIGLL", "NP3FTAPR", "NP3FTAPL", "NP3HMOVR",
    "NP3HMOVL", "NP3PRSPR", "NP3PRSPL", "NP3TTAPR", "NP3TTAPL",
    "NP3LGAGR", "NP3LGAGL", "NP3RISNG", "NP3GAIT", "NP3FRZGT",
    "NP3PSTBL", "NP3POSTR", "NP3BRADY", "NP3PTRMR", "NP3PTRML",
    "NP3KTRMR", "NP3KTRML", "NP3RTARU", "NP3RTALU", "NP3RTARL",
    "NP3RTALL", "NP3RTALJ", "NP3RTCON"
]
updrs3_cols = [c for c in updrs3_cols if c in updrs3.columns]
updrs3 = updrs3[updrs3_cols].copy()

# Keep one exam state per patient/visit.
# Preference: OFF -> ON -> Missing -> everything else
pdstate_rank = {"OFF": 0, "ON": 1, "Missing": 2}
updrs3["PDSTATE_rank"] = updrs3["PDSTATE"].map(pdstate_rank).fillna(3)

updrs3 = (
    updrs3.sort_values(["PATNO", "EVENT_ID", "PDSTATE_rank"])
          .drop_duplicates(subset=["PATNO", "EVENT_ID"], keep="first")
          .drop(columns=["PDSTATE_rank"])
)

updrs3 = updrs3.rename(columns={"NP3TOT": "baseline_updrs3"})


# ---------- MoCA ----------
moca_cols = [c for c in ["PATNO", "EVENT_ID", "MCATOT"] if c in moca.columns]
moca = moca[moca_cols].copy()
if "MCATOT" in moca.columns:
    moca = moca.rename(columns={"MCATOT": "moca_score"})


# ---------- Demographics ----------
demo_cols = [
    c for c in ["PATNO", "SEX", "HANDED", "RAWHITE", "RABLACK", "RAASIAN", "HISPLAT"]
    if c in demo.columns
]
demo = demo[demo_cols].copy()

if "SEX" in demo.columns:
    demo["is_male"] = (demo["SEX"] == "M").astype(int)
    demo = demo.drop(columns=["SEX"])


# ---------- Age ----------
age_cols = [c for c in ["PATNO", "EVENT_ID", "AGE_AT_VISIT"] if c in age.columns]
age = age[age_cols].copy()
if "AGE_AT_VISIT" in age.columns:
    age = age.rename(columns={"AGE_AT_VISIT": "approx_age"})


# ---------- Participant status ----------
status_cols = [c for c in ["PATNO", "COHORT", "ENROLL_AGE"] if c in status.columns]
status = status[status_cols].copy()

if "COHORT" in status.columns:
    status["is_pd"] = (
        status["COHORT"].astype(str).str.contains("PD", case=False, na=False).astype(int)
    )


# ---------- LEDD ----------
ledd_cols = [c for c in ["PATNO", "EVENT_ID", "LEDD", "LEDDOSE", "TOTDDA"] if c in ledd.columns]
ledd = ledd[ledd_cols].copy()

# Convert medication columns to numeric before aggregation
for col in ["LEDD", "LEDDOSE", "TOTDDA"]:
    if col in ledd.columns:
        ledd[col] = pd.to_numeric(ledd[col], errors="coerce")

# Aggregate possible multiple medication rows per visit
agg_dict = {}
for col in ["LEDD", "LEDDOSE", "TOTDDA"]:
    if col in ledd.columns:
        agg_dict[col] = "sum"

if agg_dict:
    ledd = (
        ledd.groupby(["PATNO", "EVENT_ID"], as_index=False)
            .agg(agg_dict)
    )


# ---------- UPDRS I ----------
updrs1_keep = [
    "PATNO", "EVENT_ID", "NP1COG", "NP1HALL", "NP1DPRS",
    "NP1ANXS", "NP1APAT", "NP1DDS", "NP1RTOT"
]
updrs1_keep = [c for c in updrs1_keep if c in updrs1.columns]
updrs1 = updrs1[updrs1_keep].copy()


# ---------- UPDRS II ----------
updrs2_keep = [
    "PATNO", "EVENT_ID",
    "NP2SPCH", "NP2SALV", "NP2SWAL", "NP2EAT", "NP2DRES",
    "NP2HYGN", "NP2HWRT", "NP2HOBB", "NP2TURN", "NP2TRMR",
    "NP2RISE", "NP2WALK", "NP2FREZ", "NP2PTOT"
]
updrs2_keep = [c for c in updrs2_keep if c in updrs2.columns]
updrs2 = updrs2[updrs2_keep].copy()


# =========================================================
# 3. MERGE TABLES
# =========================================================
df = updrs3.merge(moca, on=["PATNO", "EVENT_ID"], how="left")
df = df.merge(age, on=["PATNO", "EVENT_ID"], how="left")
df = df.merge(demo, on="PATNO", how="left")
df = df.merge(status, on="PATNO", how="left")
df = df.merge(ledd, on=["PATNO", "EVENT_ID"], how="left")
df = df.merge(updrs1, on=["PATNO", "EVENT_ID"], how="left")
df = df.merge(updrs2, on=["PATNO", "EVENT_ID"], how="left")


# =========================================================
# 4. VISIT ORDER / TIME
# =========================================================
def event_to_order(event_id):
    if pd.isna(event_id):
        return np.nan

    e = str(event_id).strip().upper()

    if e == "SC":
        return -1
    if e == "BL":
        return 0

    if e.startswith("V") and e[1:].isdigit():
        return int(e[1:])

    # Some PPMI datasets may contain V04A, U01, etc.
    # Try to extract the numeric part after V if it begins with V.
    if e.startswith("V"):
        digits = "".join(ch for ch in e[1:] if ch.isdigit())
        if digits != "":
            return int(digits)

    return np.nan


df["visit_order"] = df["EVENT_ID"].apply(event_to_order)
df = df.dropna(subset=["visit_order"]).copy()
df["visit_order"] = df["visit_order"].astype(int)

df = df.sort_values(["PATNO", "visit_order"]).copy()


# =========================================================
# 5. CREATE TARGET = NEXT VISIT UPDRS III
# =========================================================
df["target_updrs3"] = df.groupby("PATNO")["baseline_updrs3"].shift(-1)
df["next_visit_order"] = df.groupby("PATNO")["visit_order"].shift(-1)
df["visit_gap"] = df["next_visit_order"] - df["visit_order"]

# Keep only rows that have a future target
df = df.dropna(subset=["target_updrs3"]).copy()

# Important: remove rows missing the core score variables
df = df.dropna(subset=["baseline_updrs3", "target_updrs3"]).copy()


# =========================================================
# 6. DROP USELESS ALL-NaN COLUMNS
# =========================================================
for col in ["LEDDOSE", "TOTDDA"]:
    if col in df.columns and df[col].isna().all():
        df = df.drop(columns=[col])

# Ensure LEDD-family columns are numeric if present
for col in ["LEDD", "LEDDOSE", "TOTDDA"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# =========================================================
# 7. FEATURE SET
# =========================================================
feature_cols = [
    # Core
    "baseline_updrs3",
    "visit_order",
    "approx_age",
    "is_male",
    "is_pd",
    "moca_score",
    "NHY",

    # Medication
    "PDSTATE",
    "LEDD",
    "LEDDOSE",
    "TOTDDA",

    # Demographics
    "HANDED",
    "RAWHITE",
    "RABLACK",
    "RAASIAN",
    "HISPLAT",
    "ENROLL_AGE",

    # UPDRS Part I
    "NP1COG",
    "NP1HALL",
    "NP1DPRS",
    "NP1ANXS",
    "NP1APAT",
    "NP1DDS",
    "NP1RTOT",

    # UPDRS Part II
    "NP2SPCH",
    "NP2SALV",
    "NP2SWAL",
    "NP2EAT",
    "NP2DRES",
    "NP2HYGN",
    "NP2HWRT",
    "NP2HOBB",
    "NP2TURN",
    "NP2TRMR",
    "NP2RISE",
    "NP2WALK",
    "NP2FREZ",
    "NP2PTOT",

    # UPDRS Part III subscores
    "NP3SPCH",
    "NP3FACXP",
    "NP3RIGN",
    "NP3RIGRU",
    "NP3RIGLU",
    "NP3RIGRL",
    "NP3RIGLL",
    "NP3FTAPR",
    "NP3FTAPL",
    "NP3HMOVR",
    "NP3HMOVL",
    "NP3PRSPR",
    "NP3PRSPL",
    "NP3TTAPR",
    "NP3TTAPL",
    "NP3LGAGR",
    "NP3LGAGL",
    "NP3RISNG",
    "NP3GAIT",
    "NP3FRZGT",
    "NP3PSTBL",
    "NP3POSTR",
    "NP3BRADY",
    "NP3PTRMR",
    "NP3PTRML",
    "NP3KTRMR",
    "NP3KTRML",
    "NP3RTARU",
    "NP3RTALU",
    "NP3RTARL",
    "NP3RTALL",
    "NP3RTALJ",
    "NP3RTCON",
]

feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols].copy()
y = df["target_updrs3"].copy()
groups = df["PATNO"].copy()

print("Final dataframe shape:", df.shape)
print("Number of features:", len(feature_cols))
print("Features used:")
print(feature_cols)


# =========================================================
# 8. NUMERIC / CATEGORICAL FEATURES
# =========================================================
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

print("\nNumeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))
print("Categorical columns:", categorical_features)


# =========================================================
# 9. PATIENT-WISE TRAIN / TEST SPLIT
# =========================================================
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()
y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Train patients:", df.iloc[train_idx]["PATNO"].nunique())
print("Test patients:", df.iloc[test_idx]["PATNO"].nunique())


# =========================================================
# 10. PREPROCESSING
# =========================================================
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# =========================================================
# 11. ELASTIC NET MODEL
# =========================================================
elastic_model = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
    alphas=np.logspace(-4, 2, 100),
    cv=5,
    max_iter=20000,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", elastic_model)
])


# =========================================================
# 12. TRAIN
# =========================================================
pipeline.fit(X_train, y_train)

best_model = pipeline.named_steps["model"]

print("\nBest alpha:", best_model.alpha_)
print("Best l1_ratio:", best_model.l1_ratio_)


# =========================================================
# 13. PREDICT + EVALUATE
# =========================================================
y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nElastic Net Results")
print("-------------------")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R^2  : {r2:.4f}")


# =========================================================
# 14. BASELINE 1: PREDICT NEXT = CURRENT baseline_updrs3
# =========================================================
if "baseline_updrs3" in X_test.columns:
    baseline_df = pd.DataFrame({
        "y_true": y_test.values,
        "baseline_pred": X_test["baseline_updrs3"].values
    }).dropna()

    baseline_mae = mean_absolute_error(
        baseline_df["y_true"],
        baseline_df["baseline_pred"]
    )
    baseline_rmse = np.sqrt(mean_squared_error(
        baseline_df["y_true"],
        baseline_df["baseline_pred"]
    ))
    baseline_r2 = r2_score(
        baseline_df["y_true"],
        baseline_df["baseline_pred"]
    )

    print("\nBaseline Results (predict next = current UPDRS)")
    print("------------------------------------------------")
    print(f"MAE  : {baseline_mae:.4f}")
    print(f"RMSE : {baseline_rmse:.4f}")
    print(f"R^2  : {baseline_r2:.4f}")
    print(f"Rows used for baseline: {len(baseline_df)} / {len(y_test)}")
else:
    print("\nCould not compute baseline: 'baseline_updrs3' not found.")


# =========================================================
# 15. BASELINE 2: PREDICT TRAINING MEAN
# =========================================================
mean_pred = np.full(len(y_test), y_train.mean(), dtype=float)

mean_mae = mean_absolute_error(y_test, mean_pred)
mean_rmse = np.sqrt(mean_squared_error(y_test, mean_pred))
mean_r2 = r2_score(y_test, mean_pred)

print("\nMean Baseline Results (predict train mean)")
print("------------------------------------------")
print(f"MAE  : {mean_mae:.4f}")
print(f"RMSE : {mean_rmse:.4f}")
print(f"R^2  : {mean_r2:.4f}")


# =========================================================
# 16. PLOTS
# =========================================================
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual target_updrs3")
plt.ylabel("Predicted target_updrs3")
plt.title("Elastic Net: Predicted vs Actual")

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--")
plt.tight_layout()
plt.show()

residuals = y_test.values - y_pred

plt.figure(figsize=(7, 5))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted target_updrs3")
plt.ylabel("Residuals")
plt.title("Elastic Net: Residual Plot")
plt.tight_layout()
plt.show()


# =========================================================
# 17. FEATURE COEFFICIENTS
# =========================================================
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
coefficients = pipeline.named_steps["model"].coef_

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients,
    "abs_coefficient": np.abs(coefficients)
}).sort_values("abs_coefficient", ascending=False)

print("\nTop 30 coefficients:")
print(coef_df.head(30).to_string(index=False))


# =========================================================
# 18. SHOW 7 RANDOM TEST PATIENTS
# =========================================================
test_df = df.iloc[test_idx].copy()

test_df = test_df.assign(
    y_true=y_test.values,
    elastic_pred=y_pred
)

if "baseline_updrs3" in X_test.columns:
    test_df = test_df.assign(
        baseline_pred=X_test["baseline_updrs3"].values
    )
    test_df["elastic_abs_error"] = np.abs(test_df["y_true"] - test_df["elastic_pred"])
    test_df["baseline_abs_error"] = np.abs(test_df["y_true"] - test_df["baseline_pred"])
else:
    test_df["elastic_abs_error"] = np.abs(test_df["y_true"] - test_df["elastic_pred"])

sample_7 = test_df.sample(n=min(7, len(test_df)), random_state=42)

cols_to_show = [
    c for c in [
        "PATNO",
        "EVENT_ID",
        "visit_order",
        "baseline_updrs3",
        "y_true",
        "elastic_pred",
        "baseline_pred",
        "elastic_abs_error",
        "baseline_abs_error"
    ]
    if c in sample_7.columns
]

print("\n7 Random Test Patients")
print("----------------------")
print(sample_7[cols_to_show].to_string(index=False))


# =========================================================
# 19. PLOT 7 RANDOM TEST PATIENTS
# =========================================================
plt.figure(figsize=(10, 5))

x = np.arange(len(sample_7))

plt.plot(x, sample_7["y_true"].values, marker="o", label="Actual")
plt.plot(x, sample_7["elastic_pred"].values, marker="o", label="Elastic Net")

if "baseline_pred" in sample_7.columns:
    plt.plot(x, sample_7["baseline_pred"].values, marker="o", label="Baseline")

labels = [
    f'{int(p)}-{e}'
    for p, e in zip(sample_7["PATNO"], sample_7["EVENT_ID"])
]

plt.xticks(x, labels, rotation=45)
plt.ylabel("UPDRS III score")
plt.title("7 Random Patients from Test Set")
plt.legend()
plt.tight_layout()
plt.show()


# =========================================================
# 20. SAVE OUTPUTS
# =========================================================
coef_df.to_csv("elastic_net_coefficients.csv", index=False)

pred_df = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": y_pred,
    "residual": residuals
})
pred_df.to_csv("elastic_net_predictions.csv", index=False)

print("\nSaved files:")
print("- elastic_net_coefficients.csv")
print("- elastic_net_predictions.csv")