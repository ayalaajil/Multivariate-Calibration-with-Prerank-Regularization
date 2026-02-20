import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Define paths
base_path = 'Multicalibration/Projected_PIT_calibration/csv-files'
final_results_path = os.path.join(base_path, 'FINAL-results')

csv_files = {
    "Before": os.path.join(base_path, 'metrics-from-model-without-reg.csv'),
    "PCA (Indiv)": os.path.join(base_path, 'metrics-from-model-trained-on-train_prerank=pca.csv'),
    "PCA+ALL": os.path.join(final_results_path, 'metrics_PCA+ALL_lambda=10.csv'),
    "Marg+ALL": os.path.join(final_results_path, 'metrics_marginal+ALL_lambda=10.csv')
}

# Projection methods to compare
projections = ['marginal', 'mean', 'variance', 'dependency', 'pca', 'density', 'cdf']
labels = ["Marginal", "Mean", "Var.", "Dep.", "PCA", "Density", "CDF"]
models = ["Before", "PCA (Indiv)", "PCA+ALL", "Marg+ALL"]
colors = ["lightgrey", "coral", "skyblue", "lightgreen"]

# Load data
dfs = {}
for model_name, path in csv_files.items():
    if os.path.exists(path):
        dfs[model_name] = pd.read_csv(path)
    else:
        print(f"Warning: {path} not found.")

plt.figure(figsize=(14, 6))

gap = 3.5
delta = 0.6
positions = []
xticks = []
plot_data = []

for i, proj in enumerate(projections):
    base = i * gap
    xticks.append(base + 1.5 * delta)

    col_name = f"pce_{proj}"

    for j, model in enumerate(models):
        if model in dfs:
            # Extract PCE values for this projection and model
            data = dfs[model][col_name].dropna().values
            plot_data.append(data)
            positions.append(base + j * delta)

# Create boxplot
box = plt.boxplot(plot_data, positions=positions, widths=0.5, patch_artist=True, zorder=2,
                  flierprops=dict(marker='o', markersize=3, alpha=0.2),
                  medianprops=dict(color='black'))

# Color boxes
for i, patch in enumerate(box['boxes']):
    patch.set_facecolor(colors[i % len(models)])

# Scatter points for individual datasets/seeds
for i, d in enumerate(plot_data):
    plt.scatter([positions[i]] * len(d), d, color='black', s=5, zorder=3, alpha=0.15)

plt.xticks(xticks, labels)
plt.ylabel("PCE (Calibration Error)")
plt.title("Comparison of Model Performance across Projections (CSV Data)")

legend_patches = [mpatches.Patch(color=colors[i], label=models[i]) for i in range(len(models))]
plt.legend(handles=legend_patches, loc='upper right', frameon=False)

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("csv_model_comparison_boxplot.png", dpi=300)
print("Plot saved as csv_model_comparison_boxplot.png")
