import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Load data
df = pd.read_csv('Multicalibration/Projected_PIT_calibration/csv-files/sensitivity-t.csv')

# Unique preranks and tau values
preranks = ['marginal', 'mean', 'variance', 'dependency', 'pca', 'density', 'cdf']
labels = ["Marginal", "Mean", "Var.", "Dep.", "PCA", "Density", "CDF"]
taus = [50, 300]  # Comparing extremes for ablation

gap = 2.0      # space between prerank groups
delta = 0.7    # spacing between tau values

data = []
positions = []
xticks = []

for i, p in enumerate(preranks):
    base = i * gap
    xticks.append(base + delta / 2)
    for j, t in enumerate(taus):
        positions.append(base + j * delta)
        val = df[(df['prerank'] == p) & (df['tau'] == t)]['pce'].values
        data.append(val)

plt.figure(figsize=(8, 4))
box = plt.boxplot(data, positions=positions, widths=0.5, patch_artist=True, zorder=2,
                  flierprops=dict(marker='o', markersize=3, alpha=0.3),
                  medianprops=dict(color='black'))

colors = ["coral", "skyblue"]
for i, patch in enumerate(box['boxes']):
    patch.set_facecolor(colors[i % len(taus)])

# Scatter points
for i, d in enumerate(data):
    plt.scatter([positions[i]] * len(d), d, color='black', s=10, zorder=3, alpha=0.3)

plt.xticks(xticks, labels)
plt.ylabel("PCE (Calibration Error)")
plt.legend(handles=[mpatches.Patch(color=colors[0], label='tau=50'),
                   mpatches.Patch(color=colors[1], label='tau=300')],
           loc='upper right', frameon=False)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("tau_sensitivity_ablation_pce.png", dpi=300)
print("Plot saved as tau_sensitivity_ablation_pce.png")
