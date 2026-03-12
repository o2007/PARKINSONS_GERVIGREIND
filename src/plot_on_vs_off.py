"""
Plot 3: ON vs OFF medication score comparison
Run from project root: python3 src/plot_on_vs_off.py
"""

import pandas as pd
import matplotlib.pyplot as plt

updrs = pd.read_csv('./data/PPMI_data/MDS-UPDRS_Part_III_12Mar2026.csv', low_memory=False)
updrs = updrs.dropna(subset=['NP3TOT', 'PDSTATE'])

# Summary stats
print("UPDRS-III scores by medication state:")
print(updrs.groupby('PDSTATE')['NP3TOT'].agg(['mean', 'std', 'count']).round(2))

# Filter to just ON and OFF for the plot
on_off = updrs[updrs['PDSTATE'].isin(['ON', 'OFF'])]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: boxplot comparison
on_off.boxplot(column='NP3TOT', by='PDSTATE', ax=axes[0],
               boxprops=dict(color='steelblue'),
               medianprops=dict(color='red', linewidth=2))
axes[0].set_title('Score Distribution: ON vs OFF Medication')
axes[0].set_xlabel('Medication State')
axes[0].set_ylabel('UPDRS-III Score')
plt.sca(axes[0])
plt.title('Score Distribution: ON vs OFF')

# Right: mean scores over time by state
VISIT_MAP = {'BL': 0, 'V04': 1, 'V06': 2, 'V08': 3, 'V10': 4, 'V12': 5, 'V14': 6}
on_off = on_off.copy()
on_off['visit_year'] = on_off['EVENT_ID'].map(VISIT_MAP)
on_off = on_off.dropna(subset=['visit_year'])

for state, color in [('ON', 'steelblue'), ('OFF', 'coral')]:
    g = on_off[on_off['PDSTATE'] == state].groupby('visit_year')['NP3TOT'].mean()
    axes[1].plot(g.index, g.values, marker='o', label=state, color=color, linewidth=2)

axes[1].set_title('Mean UPDRS-III Over Time: ON vs OFF')
axes[1].set_xlabel('Years from Baseline')
axes[1].set_ylabel('UPDRS-III Score')
axes[1].legend(title='Medication State')
axes[1].grid(True, alpha=0.3)

plt.suptitle('')
plt.tight_layout()
plt.savefig('./ppmi_outputs/on_vs_off_medication.png', dpi=150)
plt.show()
print("Saved to ppmi_outputs/on_vs_off_medication.png")
