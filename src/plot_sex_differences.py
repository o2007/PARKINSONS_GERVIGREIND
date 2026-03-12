"""
Plot 4: Sex differences in motor score progression
Run from project root: python3 src/plot_sex_differences.py
"""

import pandas as pd
import matplotlib.pyplot as plt

updrs = pd.read_csv('./data/PPMI_data/MDS-UPDRS_Part_III_12Mar2026.csv', low_memory=False)
demo  = pd.read_csv('./data/PPMI_data/Demographics_12Mar2026.csv', low_memory=False)

VISIT_MAP = {'BL': 0, 'V04': 1, 'V06': 2, 'V08': 3, 'V10': 4, 'V12': 5, 'V14': 6}
updrs['visit_year'] = updrs['EVENT_ID'].map(VISIT_MAP)

# Merge sex from demographics (SEX: 1=Male, 2=Female in PPMI)
df = updrs.merge(demo[['PATNO', 'SEX']], on='PATNO')
df = df[df['PDSTATE'] == 'ON'].dropna(subset=['visit_year', 'NP3TOT', 'SEX'])

print("Patient counts by sex:")
print(df.groupby('SEX')['PATNO'].nunique())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: progression over time by sex
for sex_code, label, color in [(1, 'Male', 'steelblue'), (0, 'Female', 'coral')]:
    group = df[df['SEX'] == sex_code].groupby('visit_year')['NP3TOT']
    means = group.mean()
    stds  = group.std()
    axes[0].plot(means.index, means.values, marker='o', label=label,
                 color=color, linewidth=2)
    axes[0].fill_between(means.index,
                         means - stds, means + stds,
                         alpha=0.15, color=color)

axes[0].set_title('Mean UPDRS-III Over Time by Sex\n(ON-medication, ±1 std dev)')
axes[0].set_xlabel('Years from Baseline')
axes[0].set_ylabel('UPDRS-III Score (higher = worse)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: baseline score distribution by sex
baseline = df[df['visit_year'] == 0]
male_bl   = baseline[baseline['SEX'] == 1]['NP3TOT']
female_bl = baseline[baseline['SEX'] == 0]['NP3TOT']

axes[1].hist(male_bl,   bins=25, alpha=0.6, color='steelblue',
             label=f'Male (n={len(male_bl)})',   edgecolor='white')
axes[1].hist(female_bl, bins=25, alpha=0.6, color='coral',
             label=f'Female (n={len(female_bl)})', edgecolor='white')
axes[1].axvline(male_bl.mean(),   color='steelblue', linestyle='--', linewidth=1.5)
axes[1].axvline(female_bl.mean(), color='coral',     linestyle='--', linewidth=1.5)
axes[1].set_title('Baseline Score Distribution by Sex')
axes[1].set_xlabel('UPDRS-III Score at Baseline')
axes[1].set_ylabel('Number of Patients')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

print(f"\nMean baseline score — Male: {male_bl.mean():.1f}, Female: {female_bl.mean():.1f}")

plt.tight_layout()
plt.savefig('./ppmi_outputs/sex_differences.png', dpi=150)
plt.show()
print("Saved to ppmi_outputs/sex_differences.png")
