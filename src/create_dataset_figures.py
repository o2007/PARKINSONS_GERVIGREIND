"""
=============================================================================
PPMI - Comprehensive Dataset Visualization Suite
Creates publication-quality figures showcasing the dataset
=============================================================================

Run from project root with venv activated:
    python3 src/create_dataset_figures.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from scipy import stats
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = "./data/PPMI_data"
OUTPUT_DIR = "./ppmi_outputs"
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

VISIT_MAP = {
    "SC": -0.5, "BL": 0, "V04": 1, "V06": 2, "V08": 3,
    "V10": 4, "V12": 5, "V14": 6, "V16": 7, "V18": 8, "V20": 9,
}

# Consistent color scheme
COLORS = {
    'mild': '#2ecc71',      # Green
    'moderate': '#f39c12',  # Orange
    'severe': '#e74c3c',    # Red
    'primary': '#3498db',   # Blue
    'secondary': '#9b59b6', # Purple
    'male': '#3498db',      # Blue
    'female': '#e74c3c',    # Coral
}

print("\n" + "="*70)
print("  PPMI DATASET VISUALIZATION SUITE")
print("="*70)

# =============================================================================
# LOAD DATA
# =============================================================================

print("\n[1/8] Loading PPMI data files...")

try:
    updrs = pd.read_csv(os.path.join(DATA_DIR, "MDS-UPDRS_Part_III_12Mar2026.csv"), low_memory=False)
    demo = pd.read_csv(os.path.join(DATA_DIR, "Demographics_12Mar2026.csv"), low_memory=False)
    moca = pd.read_csv(os.path.join(DATA_DIR, "Montreal_Cognitive_Assessment__MoCA__12Mar2026.csv"), low_memory=False)
    age_df = pd.read_csv(os.path.join(DATA_DIR, "Age_at_visit_12Mar2026.csv"), low_memory=False)
    
    print(f"  ✓ UPDRS-III: {len(updrs):,} records")
    print(f"  ✓ Demographics: {len(demo):,} patients")
    print(f"  ✓ MoCA: {len(moca):,} assessments")
    print(f"  ✓ Age data: {len(age_df):,} records")
    
except Exception as e:
    print(f"  ✗ Error loading data: {e}")
    print("  Please check that data files are in ./data/PPMI_data/")
    exit(1)

# Process UPDRS data
updrs['visit_year'] = updrs['EVENT_ID'].map(VISIT_MAP)
updrs_clean = updrs[updrs['PDSTATE'] == 'ON'].copy()
updrs_clean = updrs_clean.dropna(subset=['visit_year', 'NP3TOT'])

# Merge demographics
demo_sub = demo[['PATNO', 'SEX']].drop_duplicates('PATNO')
df_full = updrs_clean.merge(demo_sub, on='PATNO', how='left')

# Merge age
df_full = df_full.merge(age_df[['PATNO', 'EVENT_ID', 'AGE_AT_VISIT']], 
                        on=['PATNO', 'EVENT_ID'], how='left')

# Merge MoCA
moca_sub = moca[['PATNO', 'EVENT_ID', 'MCATOT']].dropna(subset=['MCATOT'])
df_full = df_full.merge(moca_sub, on=['PATNO', 'EVENT_ID'], how='left')

n_patients = df_full['PATNO'].nunique()
n_assessments = len(df_full)
baseline = df_full[df_full['EVENT_ID'] == 'BL'].dropna(subset=['NP3TOT'])

print(f"\n  Dataset summary:")
print(f"    Total patients: {n_patients:,}")
print(f"    Total assessments: {n_assessments:,}")
print(f"    Baseline assessments: {len(baseline):,}")
print(f"    Follow-up years: 0 to {df_full['visit_year'].max():.0f}")

# =============================================================================
# FIGURE 1: HERO FIGURE - COMPREHENSIVE OVERVIEW (8 PANELS)
# =============================================================================

print("\n[2/8] Creating HERO FIGURE - Comprehensive Overview...")

fig = plt.figure(figsize=(20, 12))
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35,
                       left=0.06, right=0.98, top=0.94, bottom=0.06)

# Main title
fig.text(0.5, 0.98, 
         'PPMI Parkinson\'s Disease Dataset: Comprehensive Overview',
         ha='center', va='top', fontsize=18, fontweight='bold')
fig.text(0.5, 0.965,
         f'{n_patients:,} patients • {n_assessments:,} assessments • {df_full["visit_year"].max():.0f}-year longitudinal study',
         ha='center', va='top', fontsize=12, style='italic', color='gray')

# Panel A: Sample size across visits
ax1 = fig.add_subplot(gs[0, 0:2])
visit_counts = df_full.groupby('visit_year').size()
bars = ax1.bar(visit_counts.index, visit_counts.values, width=0.8,
               color=COLORS['primary'], edgecolor='black', linewidth=1, alpha=0.85)
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 50,
             f'{int(height):,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax1.set_xlabel('Years from Baseline', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Assessments', fontsize=12, fontweight='bold')
ax1.set_title('A. Study Timeline and Retention', fontsize=13, fontweight='bold', loc='left', pad=10)
ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
ax1.set_axisbelow(True)

# Panel B: Baseline distribution with severity zones
ax2 = fig.add_subplot(gs[0, 2:4])
mean_val = baseline['NP3TOT'].mean()
median_val = baseline['NP3TOT'].median()
std_val = baseline['NP3TOT'].std()

n, bins, patches = ax2.hist(baseline['NP3TOT'], bins=35, 
                             edgecolor='white', alpha=0.7, density=True, linewidth=1.5)

# Color by severity
for i, patch in enumerate(patches):
    if bins[i] < 15:
        patch.set_facecolor(COLORS['mild'])
    elif bins[i] < 25:
        patch.set_facecolor(COLORS['moderate'])
    else:
        patch.set_facecolor(COLORS['severe'])
    patch.set_alpha(0.7)

ax2.axvline(mean_val, color='red', linestyle='--', linewidth=3, 
            label=f'Mean = {mean_val:.1f}', alpha=0.8)
ax2.axvline(median_val, color='darkblue', linestyle='--', linewidth=3,
            label=f'Median = {median_val:.1f}', alpha=0.8)

# Normal distribution overlay
x = np.linspace(baseline['NP3TOT'].min(), baseline['NP3TOT'].max(), 200)
ax2.plot(x, stats.norm.pdf(x, mean_val, std_val), 'k-', 
         linewidth=2.5, alpha=0.6, label='Normal distribution')

ax2.set_xlabel('UPDRS-III Motor Score', fontsize=12, fontweight='bold')
ax2.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
ax2.set_title('B. Baseline Motor Symptom Distribution', 
              fontsize=13, fontweight='bold', loc='left', pad=10)
ax2.legend(fontsize=10, frameon=True, fancybox=True, shadow=True, loc='upper right')
ax2.grid(True, alpha=0.3, linestyle='--')

# Stats box
textstr = f'n = {len(baseline):,}\nSD = {std_val:.2f}\nRange: {baseline["NP3TOT"].min():.0f}–{baseline["NP3TOT"].max():.0f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.text(0.98, 0.97, textstr, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='right', bbox=props, fontweight='bold')

# Panel C: Disease progression over time
ax3 = fig.add_subplot(gs[1, 0:2])
stats_time = df_full.groupby('visit_year')['NP3TOT'].agg(['mean', 'std', 'sem', 'count'])

ci_95 = 1.96 * stats_time['sem']
ax3.fill_between(stats_time.index,
                  stats_time['mean'] - ci_95,
                  stats_time['mean'] + ci_95,
                  alpha=0.2, color=COLORS['primary'], label='95% CI')

ax3.fill_between(stats_time.index,
                  stats_time['mean'] - stats_time['std'],
                  stats_time['mean'] + stats_time['std'],
                  alpha=0.15, color=COLORS['primary'], label='±1 SD')

ax3.plot(stats_time.index, stats_time['mean'], 'o-',
         color=COLORS['primary'], linewidth=4, markersize=12,
         markeredgecolor='black', markeredgewidth=2,
         label='Mean progression', zorder=10)

# Annotate points
for idx, row in stats_time.iterrows():
    ax3.annotate(f'{row["mean"]:.1f}',
                 (idx, row['mean']),
                 textcoords="offset points", xytext=(0, 15),
                 ha='center', fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                          alpha=0.7, edgecolor='black'))

# Linear trend
z = np.polyfit(stats_time.index, stats_time['mean'], 1)
p = np.poly1d(z)
ax3.plot(stats_time.index, p(stats_time.index), 'r--', 
         linewidth=3, alpha=0.7, label=f'Linear trend: +{z[0]:.2f} pts/year')

ax3.set_xlabel('Years from Baseline', fontsize=12, fontweight='bold')
ax3.set_ylabel('Mean UPDRS-III Score', fontsize=12, fontweight='bold')
ax3.set_title('C. Longitudinal Disease Progression (ON-medication)', 
              fontsize=13, fontweight='bold', loc='left', pad=10)
ax3.legend(fontsize=10, loc='upper left', frameon=True, fancybox=True, shadow=True)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_axisbelow(True)

# Panel D: Individual trajectories
ax4 = fig.add_subplot(gs[1, 2:4])
patient_sample = df_full['PATNO'].unique()[:100]
for patno in patient_sample:
    patient_data = df_full[df_full['PATNO'] == patno].dropna(subset=['visit_year', 'NP3TOT']).sort_values('visit_year')
    if len(patient_data) >= 3:
        baseline_score = patient_data[patient_data['visit_year'] == 0]['NP3TOT'].values
        if len(baseline_score) > 0:
            if baseline_score[0] < 15:
                color, alpha = COLORS['mild'], 0.3
            elif baseline_score[0] < 25:
                color, alpha = COLORS['moderate'], 0.3
            else:
                color, alpha = COLORS['severe'], 0.3
        else:
            color, alpha = 'gray', 0.2
        ax4.plot(patient_data['visit_year'], patient_data['NP3TOT'],
                 alpha=alpha, linewidth=1.2, color=color)

ax4.plot(stats_time.index, stats_time['mean'], 'k-', 
         linewidth=4, label='Population mean', zorder=100)

ax4.set_xlabel('Years from Baseline', fontsize=12, fontweight='bold')
ax4.set_ylabel('UPDRS-III Score', fontsize=12, fontweight='bold')
ax4.set_title('D. Individual Patient Heterogeneity (n=100 sample)', 
              fontsize=13, fontweight='bold', loc='left', pad=10)
ax4.grid(True, alpha=0.3, linestyle='--')

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLORS['mild'], alpha=0.5, label='Mild (0-15)'),
    Patch(facecolor=COLORS['moderate'], alpha=0.5, label='Moderate (16-25)'),
    Patch(facecolor=COLORS['severe'], alpha=0.5, label='Severe (26+)')
]
ax4.legend(handles=legend_elements, title='Baseline Severity',
           fontsize=9, loc='upper right', frameon=True)

# Panel E: Medication effect
ax5 = fig.add_subplot(gs[2, 0])
on_off_data = updrs[updrs['PDSTATE'].isin(['ON', 'OFF'])].dropna(subset=['NP3TOT', 'PDSTATE'])
data_to_plot = [
    on_off_data[on_off_data['PDSTATE'] == 'ON']['NP3TOT'],
    on_off_data[on_off_data['PDSTATE'] == 'OFF']['NP3TOT']
]

bp = ax5.boxplot(data_to_plot, labels=['ON', 'OFF'], widths=0.5,
                 patch_artist=True, showmeans=True, meanline=True,
                 boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=2),
                 medianprops=dict(color='red', linewidth=3),
                 meanprops=dict(color='green', linewidth=3, linestyle='--'),
                 whiskerprops=dict(linewidth=2),
                 capprops=dict(linewidth=2),
                 flierprops=dict(marker='o', markersize=4, alpha=0.5))

on_mean = data_to_plot[0].mean()
off_mean = data_to_plot[1].mean()
difference = off_mean - on_mean
pct_diff = (difference / on_mean) * 100

ax5.set_ylabel('UPDRS-III Score', fontsize=11, fontweight='bold')
ax5.set_title('E. Medication Effect', fontsize=12, fontweight='bold', loc='left', pad=10)
ax5.grid(True, alpha=0.3, axis='y', linestyle='--')

y_max = max(data_to_plot[0].max(), data_to_plot[1].max())
ax5.plot([1, 2], [y_max * 1.05, y_max * 1.05], 'k-', linewidth=2)
ax5.text(1.5, y_max * 1.08, f'Δ = {difference:.1f} ({pct_diff:.1f}%)',
         ha='center', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# Panel F: Sex differences
ax6 = fig.add_subplot(gs[2, 1])
on_sex = df_full[df_full['SEX'].isin([0, 1])].dropna(subset=['visit_year', 'NP3TOT'])

for sex_code, label, color, marker in [(1, 'Male', COLORS['male'], 'o'), 
                                        (0, 'Female', COLORS['female'], 's')]:
    sex_group = on_sex[on_sex['SEX'] == sex_code].groupby('visit_year')['NP3TOT'].mean()
    ax6.plot(sex_group.index, sex_group.values, marker=marker, label=label,
             color=color, linewidth=3, markersize=8, markeredgecolor='black',
             markeredgewidth=1.5)

ax6.set_xlabel('Years', fontsize=11, fontweight='bold')
ax6.set_ylabel('Mean Score', fontsize=11, fontweight='bold')
ax6.set_title('F. Sex Differences', fontsize=12, fontweight='bold', loc='left', pad=10)
ax6.legend(fontsize=10, loc='upper left')
ax6.grid(True, alpha=0.3, linestyle='--')

# Panel G: Progression rate distribution
ax7 = fig.add_subplot(gs[2, 2])
slopes = []
for patno in df_full['PATNO'].unique():
    patient_data = df_full[df_full['PATNO'] == patno].dropna(subset=['visit_year', 'NP3TOT'])
    if len(patient_data) >= 3:
        x = patient_data['visit_year'].values
        y = patient_data['NP3TOT'].values
        if len(x) > 1 and x.std() > 0:
            slope, _, _, _, _ = stats.linregress(x, y)
            slopes.append(slope)

slopes = np.array(slopes)
ax7.hist(slopes, bins=40, color=COLORS['secondary'], edgecolor='black',
         linewidth=1, alpha=0.8)
ax7.axvline(0, color='red', linestyle='--', linewidth=3, label='No change', alpha=0.8)
ax7.axvline(np.median(slopes), color='orange', linestyle='--', linewidth=3,
            label=f'Median = {np.median(slopes):.2f}', alpha=0.8)

ax7.set_xlabel('Rate (pts/year)', fontsize=11, fontweight='bold')
ax7.set_ylabel('# Patients', fontsize=11, fontweight='bold')
ax7.set_title('G. Progression Rates', fontsize=12, fontweight='bold', loc='left', pad=10)
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3, axis='y', linestyle='--')

# Panel H: Prediction challenge
ax8 = fig.add_subplot(gs[2, 3])
baseline_scores = df_full[df_full['EVENT_ID'] == 'BL'][['PATNO', 'NP3TOT']].rename(
    columns={'NP3TOT': 'baseline'})
year4_scores = df_full[df_full['visit_year'] == 4][['PATNO', 'NP3TOT']].rename(
    columns={'NP3TOT': 'year4'})
pred_data = baseline_scores.merge(year4_scores, on='PATNO').dropna()

if len(pred_data) > 0:
    ax8.scatter(pred_data['baseline'], pred_data['year4'],
                alpha=0.5, s=30, c=COLORS['primary'], edgecolors='black', linewidth=0.5)
    
    max_val = max(pred_data['baseline'].max(), pred_data['year4'].max())
    ax8.plot([0, max_val], [0, max_val], 'r--', linewidth=2.5, 
             label='Perfect prediction', alpha=0.7)
    
    if len(pred_data) > 5:
        z = np.polyfit(pred_data['baseline'], pred_data['year4'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(pred_data['baseline'].min(), pred_data['baseline'].max(), 100)
        r2 = np.corrcoef(pred_data['baseline'], pred_data['year4'])[0,1]**2
        ax8.plot(x_line, p(x_line), 'g-', linewidth=2.5,
                 label=f'Actual fit\n(R² = {r2:.3f})')

ax8.set_xlabel('Baseline Score', fontsize=11, fontweight='bold')
ax8.set_ylabel('Year 4 Score', fontsize=11, fontweight='bold')
ax8.set_title('H. Prediction Challenge', fontsize=12, fontweight='bold', loc='left', pad=10)
ax8.legend(fontsize=9, loc='upper left')
ax8.grid(True, alpha=0.3, linestyle='--')

plt.savefig(os.path.join(OUTPUT_DIR, 'HERO_FIGURE_comprehensive.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✓ Saved: HERO_FIGURE_comprehensive.png")

# =============================================================================
# FIGURE 2: TEMPORAL PATTERNS (4 PANELS)
# =============================================================================

print("\n[3/8] Creating Figure 2 - Temporal Patterns...")

fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Spaghetti plot
ax = axes[0, 0]
patient_sample = df_full['PATNO'].unique()[:50]
for patno in patient_sample:
    patient_data = df_full[df_full['PATNO'] == patno].dropna(subset=['visit_year', 'NP3TOT']).sort_values('visit_year')
    if len(patient_data) >= 3:
        ax.plot(patient_data['visit_year'], patient_data['NP3TOT'], 
                alpha=0.3, linewidth=0.8, color='gray')

ax.plot(stats_time.index, stats_time['mean'], 'r-', linewidth=3, 
        label='Population Mean', zorder=10)
ax.set_xlabel('Years from Baseline', fontsize=11)
ax.set_ylabel('UPDRS-III Score', fontsize=11)
ax.set_title('A. Individual Trajectories vs Population Mean', fontweight='bold', loc='left')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# Panel B: Rate of progression distribution
ax = axes[0, 1]
ax.hist(slopes, bins=40, color=COLORS['secondary'], edgecolor='white', alpha=0.7)
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
ax.axvline(np.median(slopes), color='orange', linestyle='--', linewidth=2, 
           label=f'Median = {np.median(slopes):.2f} pts/year')
ax.set_xlabel('Rate of Change (UPDRS points/year)', fontsize=11)
ax.set_ylabel('Number of Patients', fontsize=11)
ax.set_title('B. Distribution of Disease Progression Rates', fontweight='bold', loc='left')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Panel C: Progression by baseline severity
ax = axes[1, 0]

def categorize_severity(score):
    if score <= 15:
        return 'Mild'
    elif score <= 25:
        return 'Moderate'
    else:
        return 'Severe'

baseline_with_severity = baseline.copy()
baseline_with_severity['severity'] = baseline_with_severity['NP3TOT'].apply(categorize_severity)

future_data = df_full[df_full['visit_year'] >= 2].merge(
    baseline_with_severity[['PATNO', 'severity']], on='PATNO', how='left'
).dropna(subset=['severity', 'NP3TOT', 'visit_year'])

for severity, color in [('Mild', COLORS['mild']), 
                        ('Moderate', COLORS['moderate']), 
                        ('Severe', COLORS['severe'])]:
    subset = future_data[future_data['severity'] == severity].groupby('visit_year')['NP3TOT'].mean()
    ax.plot(subset.index, subset.values, 'o-', label=f'{severity} (0-15)' if severity == 'Mild' else 
            f'{severity} (16-25)' if severity == 'Moderate' else f'{severity} (26+)',
            color=color, linewidth=2, markersize=7)

ax.set_xlabel('Years from Baseline', fontsize=11)
ax.set_ylabel('Mean UPDRS-III Score', fontsize=11)
ax.set_title('C. Progression by Baseline Severity', fontweight='bold', loc='left')
ax.legend(title='Baseline Severity', loc='upper left')
ax.grid(True, alpha=0.3)

# Panel D: Visit-to-visit variability
ax = axes[1, 1]
visit_changes = []
for patno in df_full['PATNO'].unique():
    patient_data = df_full[df_full['PATNO'] == patno].dropna(subset=['visit_year', 'NP3TOT']).sort_values('visit_year')
    if len(patient_data) >= 2:
        changes = patient_data['NP3TOT'].diff().dropna()
        visit_changes.extend(changes.values)

ax.hist(visit_changes, bins=50, color='teal', edgecolor='white', alpha=0.7)
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
ax.axvline(np.mean(visit_changes), color='orange', linestyle='--', linewidth=2,
           label=f'Mean = {np.mean(visit_changes):.2f}')
ax.set_xlabel('Visit-to-Visit Change in UPDRS-III', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('D. Short-term Score Variability', fontweight='bold', loc='left')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Temporal Patterns in Disease Progression', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure2_temporal_patterns.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✓ Saved: figure2_temporal_patterns.png")

# =============================================================================
# FIGURE 3: DATA QUALITY AND COMPLETENESS
# =============================================================================

print("\n[4/8] Creating Figure 3 - Data Quality...")

fig3, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Missing data by column
ax = axes[0, 0]
key_cols = ['NP3TOT', 'PDSTATE', 'AGE_AT_VISIT', 'NHY', 'MCATOT']
key_cols = [c for c in key_cols if c in df_full.columns]
missing_pct = df_full[key_cols].isnull().mean() * 100

bars = ax.barh(range(len(key_cols)), missing_pct, color='coral', 
               edgecolor='black', linewidth=0.5, alpha=0.8)
ax.set_yticks(range(len(key_cols)))
ax.set_yticklabels(key_cols)
ax.set_xlabel('Missing Data (%)', fontsize=11)
ax.set_title('A. Data Completeness by Variable', fontweight='bold', loc='left')
ax.grid(True, alpha=0.3, axis='x')

for i, (bar, val) in enumerate(zip(bars, missing_pct)):
    ax.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=9)

# Panel B: Assessments per patient
ax = axes[0, 1]
assessments_per_patient = df_full.groupby('PATNO').size()
ax.hist(assessments_per_patient, bins=20, color=COLORS['primary'], 
        edgecolor='white', alpha=0.7)
ax.axvline(assessments_per_patient.median(), color='red', linestyle='--', 
           linewidth=2, label=f'Median = {assessments_per_patient.median():.0f}')
ax.set_xlabel('Number of Assessments per Patient', fontsize=11)
ax.set_ylabel('Number of Patients', fontsize=11)
ax.set_title('B. Follow-up Completeness', fontweight='bold', loc='left')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Panel C: Age distribution
ax = axes[1, 0]
if 'AGE_AT_VISIT' in df_full.columns:
    baseline_age = df_full[df_full['EVENT_ID'] == 'BL'].dropna(subset=['AGE_AT_VISIT'])
    if len(baseline_age) > 0:
        ax.hist(baseline_age['AGE_AT_VISIT'], bins=25, color='mediumseagreen', 
                edgecolor='white', alpha=0.7)
        ax.axvline(baseline_age['AGE_AT_VISIT'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean = {baseline_age["AGE_AT_VISIT"].mean():.1f} years')
        ax.set_xlabel('Age at Baseline (years)', fontsize=11)
        ax.set_ylabel('Number of Patients', fontsize=11)
        ax.set_title('C. Age Distribution at Study Entry', fontweight='bold', loc='left')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

# Panel D: Score distribution by visit
ax = axes[1, 1]
visit_data = df_full[df_full['visit_year'].isin([0, 2, 4, 6])].dropna(subset=['visit_year', 'NP3TOT'])
if len(visit_data) > 0:
    visit_groups = [visit_data[visit_data['visit_year'] == v]['NP3TOT'] for v in [0, 2, 4, 6]]
    bp = ax.violinplot(visit_groups, positions=[0, 2, 4, 6], widths=1.5,
                       showmeans=True, showmedians=True)
    ax.set_xlabel('Years from Baseline', fontsize=11)
    ax.set_ylabel('UPDRS-III Score', fontsize=11)
    ax.set_title('D. Score Distribution Evolution', fontweight='bold', loc='left')
    ax.set_xticks([0, 2, 4, 6])
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Data Quality and Completeness Analysis', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure3_data_quality.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✓ Saved: figure3_data_quality.png")

# =============================================================================
# FIGURE 4: CLINICAL INSIGHTS
# =============================================================================

print("\n[5/8] Creating Figure 4 - Clinical Insights...")

fig4, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Cognitive vs Motor
ax = axes[0, 0]
try:
    # Get baseline MoCA data
    moca_baseline = moca[moca['EVENT_ID'] == 'BL'][['PATNO', 'MCATOT']].dropna(subset=['MCATOT'])
    baseline_cog = baseline.merge(moca_baseline, on='PATNO', how='left')
    baseline_cog = baseline_cog.dropna(subset=['NP3TOT', 'MCATOT'])
    
    if len(baseline_cog) > 10:
        ax.scatter(baseline_cog['MCATOT'], baseline_cog['NP3TOT'], 
                   alpha=0.5, s=30, color=COLORS['primary'], 
                   edgecolors='black', linewidth=0.5)
        
        z = np.polyfit(baseline_cog['MCATOT'], baseline_cog['NP3TOT'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(baseline_cog['MCATOT'].min(), baseline_cog['MCATOT'].max(), 100)
        r = baseline_cog['MCATOT'].corr(baseline_cog['NP3TOT'])
        ax.plot(x_line, p(x_line), "r--", linewidth=2, 
                label=f'Linear fit: r={r:.3f}')
        
        ax.set_xlabel('MoCA Score (cognitive function)', fontsize=11)
        ax.set_ylabel('UPDRS-III Score (motor symptoms)', fontsize=11)
        ax.set_title('A. Motor vs Cognitive Function', fontweight='bold', loc='left')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Insufficient MoCA data\n(n<10 patients)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('A. Motor vs Cognitive Function', fontweight='bold', loc='left')
except Exception as e:
    ax.text(0.5, 0.5, 'MoCA data not available', 
            ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_title('A. Motor vs Cognitive Function', fontweight='bold', loc='left')

# Panel B: Age vs symptom severity
ax = axes[0, 1]
try:
    baseline_age_corr = baseline.dropna(subset=['NP3TOT', 'AGE_AT_VISIT'])
    
    if len(baseline_age_corr) > 10:
        ax.scatter(baseline_age_corr['AGE_AT_VISIT'], baseline_age_corr['NP3TOT'],
                   alpha=0.5, s=30, color='coral', edgecolors='black', linewidth=0.5)
        
        z = np.polyfit(baseline_age_corr['AGE_AT_VISIT'], baseline_age_corr['NP3TOT'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(baseline_age_corr['AGE_AT_VISIT'].min(), 
                            baseline_age_corr['AGE_AT_VISIT'].max(), 100)
        r = baseline_age_corr['AGE_AT_VISIT'].corr(baseline_age_corr['NP3TOT'])
        ax.plot(x_line, p(x_line), "r--", linewidth=2,
                label=f'Linear fit: r={r:.3f}')
        
        ax.set_xlabel('Age at Baseline (years)', fontsize=11)
        ax.set_ylabel('UPDRS-III Score', fontsize=11)
        ax.set_title('B. Age vs Motor Symptom Severity', fontweight='bold', loc='left')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Insufficient age data\n(n<10 patients)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('B. Age vs Motor Symptom Severity', fontweight='bold', loc='left')
except Exception as e:
    ax.text(0.5, 0.5, 'Age data not available', 
            ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_title('B. Age vs Motor Symptom Severity', fontweight='bold', loc='left')

# Panel C: Severity transitions (baseline to year 4)
ax = axes[1, 0]
baseline_year4 = df_full[df_full['visit_year'].isin([0, 4])].copy()
baseline_year4['severity'] = baseline_year4['NP3TOT'].apply(categorize_severity)

transitions = baseline_year4.pivot_table(
    index='PATNO', columns='visit_year', values='severity', aggfunc='first'
).dropna()

if len(transitions) > 0:
    transition_counts = pd.crosstab(transitions[0], transitions[4])
    
    # Create bar chart
    x_pos = np.arange(len(transition_counts))
    width = 0.25
    
    for i, severity_to in enumerate(['Mild', 'Moderate', 'Severe']):
        if severity_to in transition_counts.columns:
            values = [transition_counts.loc[sev, severity_to] if sev in transition_counts.index 
                     and severity_to in transition_counts.columns else 0 
                     for sev in ['Mild', 'Moderate', 'Severe']]
            ax.bar(x_pos + i*width, values, width, 
                   label=f'To {severity_to}',
                   color=[COLORS['mild'], COLORS['moderate'], COLORS['severe']][i],
                   alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Baseline Severity', fontsize=11)
    ax.set_ylabel('Number of Patients', fontsize=11)
    ax.set_title('C. Disease Severity Transitions (Baseline → Year 4)', 
                 fontweight='bold', loc='left')
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(['Mild', 'Moderate', 'Severe'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

# Panel D: Patient variability
ax = axes[1, 1]
patient_std = []
patient_mean = []
for patno in df_full['PATNO'].unique():
    patient_data = df_full[df_full['PATNO'] == patno].dropna(subset=['NP3TOT'])
    if len(patient_data) >= 4:
        patient_std.append(patient_data['NP3TOT'].std())
        patient_mean.append(patient_data['NP3TOT'].mean())

ax.scatter(patient_mean, patient_std, alpha=0.5, s=40, 
          c=patient_mean, cmap='RdYlGn_r', edgecolors='black', linewidth=0.5)
ax.set_xlabel('Mean UPDRS-III Score', fontsize=11)
ax.set_ylabel('Standard Deviation', fontsize=11)
ax.set_title('D. Patient-Level Variability', fontweight='bold', loc='left')
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(ax.collections[0], ax=ax, label='Severity')

plt.suptitle('Clinical Insights and Patterns', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure4_clinical_insights.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✓ Saved: figure4_clinical_insights.png")

# =============================================================================
# FIGURE 5: RIDGELINE PLOT
# =============================================================================

print("\n[6/8] Creating Figure 5 - Ridgeline Distribution...")

fig5, ax = plt.subplots(figsize=(12, 10))

visit_years_sorted = sorted([v for v in VISIT_MAP.values() if v >= 0 and v <= 6])
colors_ridge = plt.cm.viridis(np.linspace(0, 1, len(visit_years_sorted)))

y_offset = 0
density_scale = 0.15

for i, year in enumerate(visit_years_sorted):
    data = df_full[df_full['visit_year'] == year].dropna(subset=['NP3TOT'])['NP3TOT']
    
    if len(data) > 10:
        kde = gaussian_kde(data)
        x_range = np.linspace(0, 80, 300)
        density = kde(x_range)
        
        density_scaled = density * density_scale
        y_values = density_scaled + y_offset
        
        ax.fill_between(x_range, y_offset, y_values, alpha=0.7, color=colors_ridge[i], 
                        edgecolor='black', linewidth=1.5)
        
        median_val = data.median()
        median_height = kde(median_val) * density_scale + y_offset
        ax.plot([median_val, median_val], [y_offset, median_height], 
               'r-', linewidth=2, alpha=0.8)
        
        ax.text(-5, y_offset + density_scale/2, f'Year {int(year)}', 
               fontsize=10, va='center', ha='right', fontweight='bold')
        
        y_offset += density_scale * 1.3

ax.set_xlabel('UPDRS-III Score', fontsize=12)
ax.set_title('Distribution Evolution Over Study Timeline (Ridgeline Plot)', 
            fontweight='bold', fontsize=14)
ax.set_yticks([])
ax.set_xlim(-10, 80)
ax.grid(True, alpha=0.3, axis='x')
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure5_ridgeline_distribution.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✓ Saved: figure5_ridgeline_distribution.png")

# =============================================================================
# FIGURE 6: PATIENT TRAJECTORY HEATMAP
# =============================================================================

print("\n[7/8] Creating Figure 6 - Patient Trajectory Heatmap...")

patient_visits = df_full.dropna(subset=['visit_year', 'NP3TOT']).groupby('PATNO')['visit_year'].apply(list)
complete_patients = patient_visits[patient_visits.apply(lambda x: len(x) >= 5)].index[:100]

visit_years = sorted([v for v in VISIT_MAP.values() if v >= 0 and v <= 6])
matrix_data = []

for patno in complete_patients:
    patient_data = df_full[df_full['PATNO'] == patno].set_index('visit_year')['NP3TOT']
    row = [patient_data.get(year, np.nan) for year in visit_years]
    matrix_data.append(row)

if len(matrix_data) > 0:
    matrix = np.array(matrix_data)
    baseline_scores = matrix[:, 0]
    sort_idx = np.argsort(baseline_scores)
    matrix_sorted = matrix[sort_idx]
    
    fig6, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix_sorted, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_xlabel('Years from Baseline', fontsize=12)
    ax.set_ylabel('Patients (sorted by baseline severity)', fontsize=12)
    ax.set_title('Individual Patient Trajectories Heatmap (100 patients with complete follow-up)', 
                 fontweight='bold', fontsize=13)
    ax.set_xticks(range(len(visit_years)))
    ax.set_xticklabels([f'{int(y)}' for y in visit_years])
    
    cbar = plt.colorbar(im, ax=ax, label='UPDRS-III Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure6_trajectory_heatmap.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ Saved: figure6_trajectory_heatmap.png")

# =============================================================================
# STATISTICS SUMMARY FILE
# =============================================================================

print("\n[8/8] Generating summary statistics...")

summary_stats = {
    'Total Patients': n_patients,
    'Total Assessments': n_assessments,
    'Study Duration (years)': df_full['visit_year'].max(),
    'Baseline Mean Score': baseline['NP3TOT'].mean(),
    'Baseline Std Dev': baseline['NP3TOT'].std(),
    'Baseline Median': baseline['NP3TOT'].median(),
    'ON Medication Mean': on_off_data[on_off_data['PDSTATE'] == 'ON']['NP3TOT'].mean(),
    'OFF Medication Mean': on_off_data[on_off_data['PDSTATE'] == 'OFF']['NP3TOT'].mean(),
    'Medication Effect (Δ)': difference,
    'Median Progression Rate (pts/yr)': np.median(slopes),
    'Patients with 5+ visits': len(complete_patients),
}

with open(os.path.join(OUTPUT_DIR, 'dataset_summary_statistics.txt'), 'w') as f:
    f.write("PPMI DATASET SUMMARY STATISTICS\n")
    f.write("="*60 + "\n\n")
    for key, value in summary_stats.items():
        if isinstance(value, float):
            f.write(f"{key:.<45} {value:>10.2f}\n")
        else:
            f.write(f"{key:.<45} {value:>10,}\n")
    f.write("\n" + "="*60 + "\n")

print("  ✓ Saved: dataset_summary_statistics.txt")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*70)
print("  ✅ ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
print("="*70)
print(f"\n  Files saved in {OUTPUT_DIR}/:")
print("    • HERO_FIGURE_comprehensive.png (THE MAIN FIGURE)")
print("    • figure2_temporal_patterns.png")
print("    • figure3_data_quality.png")
print("    • figure4_clinical_insights.png")
print("    • figure5_ridgeline_distribution.png")
print("    • figure6_trajectory_heatmap.png")
print("    • dataset_summary_statistics.txt")
print("\n  Next steps:")
print("    1. Review the figures")
print("    2. Choose the best ones for your progress report")
print("    3. Add them to your document with captions")
print("="*70)
print()