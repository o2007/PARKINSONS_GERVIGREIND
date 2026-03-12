"""
Plot 1: Distribution of baseline UPDRS-III scores
Run from project root: python3 src/plot_baseline_distribution.py
"""

import pandas as pd
import matplotlib.pyplot as plt

updrs = pd.read_csv('./data/PPMI_data/MDS-UPDRS_Part_III_12Mar2026.csv', low_memory=False)
baseline = updrs[updrs['EVENT_ID'] == 'BL'].dropna(subset=['NP3TOT'])

print(f"Patients at baseline: {len(baseline)}")
print(f"Mean score:  {baseline['NP3TOT'].mean():.1f}")
print(f"Std dev:     {baseline['NP3TOT'].std():.1f}")
print(f"Min / Max:   {baseline['NP3TOT'].min()} / {baseline['NP3TOT'].max()}")

plt.figure(figsize=(9, 5))
plt.hist(baseline['NP3TOT'], bins=30, color='steelblue', edgecolor='white')
plt.axvline(baseline['NP3TOT'].mean(), color='red', linestyle='--', label=f"Mean = {baseline['NP3TOT'].mean():.1f}")
plt.title('Distribution of Baseline UPDRS-III Scores')
plt.xlabel('UPDRS-III Score (higher = worse)')
plt.ylabel('Number of Patients')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./ppmi_outputs/baseline_distribution.png', dpi=150)
plt.show()
print("Saved to ppmi_outputs/baseline_distribution.png")
