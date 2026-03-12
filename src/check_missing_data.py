"""
Plot 2: Missing data analysis across all key PPMI files
Run from project root: python3 src/check_missing_data.py
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load all key files
updrs = pd.read_csv('./data/PPMI_data/MDS-UPDRS_Part_III_12Mar2026.csv', low_memory=False)
demo  = pd.read_csv('./data/PPMI_data/Demographics_12Mar2026.csv', low_memory=False)
moca  = pd.read_csv('./data/PPMI_data/Montreal_Cognitive_Assessment__MoCA__12Mar2026.csv', low_memory=False)
age   = pd.read_csv('./data/PPMI_data/Age_at_visit_12Mar2026.csv', low_memory=False)

files = {
    'UPDRS-III':    updrs,
    'Demographics': demo,
    'MoCA':         moca,
    'Age at visit': age,
}

print("=" * 50)
for name, df in files.items():
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    pct = (missing / len(df) * 100).round(1)
    print(f"\n{name} — {len(df):,} rows total")
    if missing.empty:
        print("  No missing values")
    else:
        for col in missing.index:
            print(f"  {col}: {missing[col]} missing ({pct[col]}%)")
print("=" * 50)

# Visual: missing data heatmap for UPDRS-III (our main file)
key_cols = ['NP3TOT', 'PDSTATE', 'EVENT_ID', 'PATNO', 'NP3SPCH',
            'NP3FACXP', 'NP3GAIT', 'NP3BRADY', 'NP3PSTBL']
key_cols = [c for c in key_cols if c in updrs.columns]

missing_pct = updrs[key_cols].isnull().mean() * 100

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(key_cols, missing_pct, color='steelblue', edgecolor='white')
ax.set_xlabel('% Missing')
ax.set_title('Missing Data — Key UPDRS-III Columns')
ax.set_xlim(0, max(missing_pct.max() + 5, 10))
for bar, val in zip(bars, missing_pct):
    ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', fontsize=9)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('./ppmi_outputs/missing_data.png', dpi=150)
plt.show()
print("Saved to ppmi_outputs/missing_data.png")
