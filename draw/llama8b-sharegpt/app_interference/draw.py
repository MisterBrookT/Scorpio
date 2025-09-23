import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Style settings
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.titlesize'] = 48
plt.rcParams['axes.labelsize'] = 48
plt.rcParams['lines.linewidth'] = 8
plt.rcParams['lines.markersize'] = 30
plt.rcParams['xtick.labelsize'] = 38
plt.rcParams['ytick.labelsize'] = 38
plt.rcParams['font.family'] = 'serif'
plt.rcParams['legend.fontsize'] = 24
plt.rcParams['figure.dpi'] = 600

# Read the CSV file
df = pd.read_csv(os.path.join(script_dir, 'summary.csv'))

# Separate data for with and without interference
df_w = df[df['policy'].str.contains('w_interference')]
df_wo = df[df['policy'].str.contains('wo_interference')]

# Get request rates for x-axis
request_rates = df_w['request_rate'].unique()

# Prepare data for plotting
goodput_w = df_w['goodput'].values
goodput_wo = df_wo['goodput'].values
slo_w = df_w['slo_adherence'].values
slo_wo = df_wo['slo_adherence'].values

# Normalize the data within each QPS cluster
def normalize_cluster(w_data, wo_data):
    w_norm = []
    wo_norm = []
    for w, wo in zip(w_data, wo_data):
        min_val = min(w, wo)
        w_norm.append(w / min_val)
        wo_norm.append(wo / min_val)
    return np.array(w_norm), np.array(wo_norm)

goodput_w_norm, goodput_wo_norm = normalize_cluster(goodput_w, goodput_wo)
slo_w_norm, slo_wo_norm = normalize_cluster(slo_w, slo_wo)

# Set width of bars
barWidth = 0.35

# Set positions of the bars on X axis
r1 = np.arange(len(request_rates))
r2 = [x + barWidth for x in r1]

# Create first figure (Goodput)
plt.figure(figsize=(12, 8))
ax1 = plt.gca()

# Create the goodput plot
ax1.bar(r1, goodput_w_norm, width=barWidth, label='w/ Interference', 
        color='#ec6446', edgecolor='black', linewidth=1.5)
ax1.bar(r2, goodput_wo_norm, width=barWidth, label='w/o Interference', 
        color='#836ed5', edgecolor='black', linewidth=1.5)

# Add labels and title for goodput plot
ax1.set_xlabel('QPS')
ax1.set_ylabel('Normed Goodput')
ax1.set_title('LLaMA-8B ShareGPT')
ax1.set_xticks([r + barWidth/2 for r in range(len(request_rates))])
ax1.set_xticklabels(request_rates)
ax1.legend()
ax1.set_ylim(0.8, 1.25)  # Set y-axis limits
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'sharegpt_goodput_comparison.pdf'), format='pdf', dpi=600, bbox_inches='tight')
plt.close()

# Create second figure (SLO Adherence)
plt.figure(figsize=(12, 8))
ax2 = plt.gca()

# Create the SLO adherence plot
ax2.bar(r1, slo_w_norm, width=barWidth, label='w/ Interference', 
        color='#ec6446', edgecolor='black', linewidth=1.5)
ax2.bar(r2, slo_wo_norm, width=barWidth, label='w/o Interference', 
        color='#836ed5', edgecolor='black', linewidth=1.5)

# Add labels and title for SLO adherence plot
ax2.set_xlabel('QPS')
ax2.set_ylabel('Normed Adherence')
ax2.set_title('LLaMA-8B ShareGPT')
ax2.set_xticks([r + barWidth/2 for r in range(len(request_rates))])
ax2.set_xticklabels(request_rates)
ax2.legend()
ax2.set_ylim(0.8, 1.25)  # Set y-axis limits

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'sharegpt_slo_adherence_comparison.pdf'), format='pdf', dpi=600, bbox_inches='tight')
plt.close()
