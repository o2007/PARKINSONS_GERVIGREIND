import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load UPDRS data
updrs = pd.read_csv('./data/PPMI_data/MDS-UPDRS_Part_III_12Mar2026.csv', low_memory=False)

# Map visit codes to years
VISIT_MAP = {
    'SC': -0.5, 'BL': 0, 'V04': 1, 'V06': 2,
    'V08': 3, 'V10': 4, 'V12': 5, 'V14': 6,
}
updrs['visit_year'] = updrs['EVENT_ID'].map(VISIT_MAP)

# Filter to ON-medication only + valid scores
updrs = updrs[updrs['PDSTATE'] == 'ON']
updrs = updrs.dropna(subset=['visit_year', 'NP3TOT'])

# Calculate mean and standard deviation per visit year
stats = updrs.groupby('visit_year')['NP3TOT'].agg(['mean', 'std', 'count']).reset_index()
print(stats)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

# Shaded area = ±1 standard deviation
ax.fill_between(
    stats['visit_year'],
    stats['mean'] - stats['std'],
    stats['mean'] + stats['std'],
    alpha=0.2, color='steelblue', label='±1 std dev'
)

# Mean line
ax.plot(stats['visit_year'], stats['mean'],
        marker='o', linewidth=2.5, markersize=8,
        color='steelblue', label='Mean score')

# Annotate each point with mean value and patient count
for _, row in stats.iterrows():
    ax.annotate(
        f"{row['mean']:.1f}\n(n={int(row['count'])})",
        (row['visit_year'], row['mean']),
        textcoords="offset points", xytext=(0, 12),
        ha='center', fontsize=8
    )

ax.set_title('Average MDS-UPDRS III Motor Score Over Time\nAll Patients (ON-medication)', fontsize=13)
ax.set_xlabel('Years from Baseline')
ax.set_ylabel('UPDRS-III Score (higher = worse)')
ax.set_xticks(stats['visit_year'])
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./ppmi_outputs/average_mobility_plot.png', dpi=150)
plt.show()
print("Saved to ppmi_outputs/average_mobility_plot.png")