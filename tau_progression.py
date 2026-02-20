import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('csv-files/sensitivity-t.csv')

# Unique preranks and tau values
preranks = ['marginal', 'mean', 'variance', 'dependency', 'pca', 'density', 'cdf']
labels = ["Marginal", "Mean", "Var.", "Dep.", "PCA", "Density", "CDF"]
prerank_to_label = dict(zip(preranks, labels))

# Group by tau and prerank to get mean and std across datasets
grouped = df.groupby(['tau', 'prerank'])['pce'].agg(['mean', 'std']).reset_index()

plt.figure(figsize=(10, 6))

for p in preranks:
    data = grouped[grouped['prerank'] == p]
    plt.plot(data['tau'], data['mean'], marker='o', label=prerank_to_label[p])
    plt.fill_between(data['tau'], data['mean'] - data['std'], data['mean'] + data['std'], alpha=0.1)

plt.xlabel("Tau")
plt.ylabel("PCE (Calibration Error)")
plt.title("PCE Progression vs Tau for Different Preranks")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.savefig("tau_progression_pce.png", dpi=300)
print("Plot saved to tau_progression_pce.png")
