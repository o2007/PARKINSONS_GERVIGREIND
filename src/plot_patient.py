import pandas as pd
import matplotlib.pyplot as plt

# Load UPDRS data
updrs = pd.read_csv('./data/PPMI_data/MDS-UPDRS_Part_III_12Mar2026.csv', low_memory=False)

# Visit code to year mapping
VISIT_MAP = {
    'SC': -0.5, 'BL': 0, 'V04': 1, 'V06': 2,
    'V08': 3, 'V10': 4, 'V12': 5, 'V14': 6,
}
updrs['visit_year'] = updrs['EVENT_ID'].map(VISIT_MAP)

# Pick one patient — let's use the first one we find with 3+ visits
patient_counts = updrs.dropna(subset=['visit_year', 'NP3TOT']).groupby('PATNO').size()
patno = patient_counts[patient_counts >= 3].index[0]

# Filter to that patient
patient = (updrs[updrs['PATNO'] == patno]
           .dropna(subset=['visit_year', 'NP3TOT'])
           .sort_values('visit_year'))

print(f"Patient {patno} — {len(patient)} visits:")
print(patient[['EVENT_ID', 'visit_year', 'NP3TOT']])

# Plot
plt.figure(figsize=(9, 5))
plt.plot(patient['visit_year'], patient['NP3TOT'], marker='o',
         linewidth=2, markersize=8, color='steelblue')

for _, row in patient.iterrows():
    plt.annotate(f"{int(row['NP3TOT'])}",
                 (row['visit_year'], row['NP3TOT']),
                 textcoords="offset points", xytext=(0, 10), ha='center')

plt.title(f'MDS-UPDRS III Motor Score Over Time — Patient {patno}', fontsize=13)
plt.xlabel('Years from Baseline')
plt.ylabel('UPDRS-III Score (higher = worse)')
plt.xticks(patient['visit_year'])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./ppmi_outputs/patient_mobility_plot.png', dpi=150)
plt.show()
print("Saved to ppmi_outputs/patient_mobility_plot.png")