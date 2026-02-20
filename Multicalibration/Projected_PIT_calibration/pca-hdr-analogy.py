#!/usr/bin/env python3
"""
HDR-PCA Analogy Simulation (Gaussian dependence miscalibration)

This script illustrates a clean link between:
  (i) HDR calibration (highest density region / HPD-based calibration), and
 (ii) PCA pre-ranks (projections onto principal components of the predictive covariance).

Key fact for a multivariate Gaussian prediction N(0, Σ̂):
- The density is a monotone function of the squared Mahalanobis distance:
    d̂^2(y) = y^T Σ̂^{-1} y
- With eigendecomposition Σ̂ = V Λ V^T, we have
    d̂^2(y) = ||Λ^{-1/2} V^T y||^2 = Σ_k ( (v_k^T y)^2 / λ_k )
so the Gaussian HDR / HPD statistic is a *radial* function in (whitened) PC coordinates.

Simulation design:
- True Y ~ N(0, Σ_true) with correlation rho_true.
- Forecast assumes Y ~ N(0, Σ_pred) with correlation rho_pred.
- Both have unit marginal variances, so marginal PITs look calibrated;
  only the dependence structure is wrong.

Outputs (saved to current working directory by default):
- hdr_pca_analogy_histograms.png
- hdr_pca_analogy_geometry.png
- hdr_pca_analogy_decomposition_heatmap.png

Dependencies: numpy, scipy, matplotlib
"""

import argparse
import os
import numpy as np
from scipy.stats import multivariate_normal, chi2, norm, spearmanr
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000, help="number of observations")
    ap.add_argument("--rho_true", type=float, default=0.8, help="true correlation")
    ap.add_argument("--rho_pred", type=float, default=0.2, help="predicted correlation")
    ap.add_argument("--seed", type=int, default=0, help="random seed")
    ap.add_argument("--outdir", type=str, default=".", help="output directory")
    ap.add_argument("--p_hdr", type=float, default=0.9, help="HDR coverage for ellipse plot")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    Sigma_true = np.array([[1.0, args.rho_true], [args.rho_true, 1.0]])
    Sigma_pred = np.array([[1.0, args.rho_pred], [args.rho_pred, 1.0]])

    # Observations from the true process
    Y = rng.multivariate_normal(mean=np.zeros(2), cov=Sigma_true, size=args.n)

    # PCA of the predictive covariance
    eigvals, eigvecs = np.linalg.eigh(Sigma_pred)  # ascending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # PC scores and whitened PC scores
    T = Y @ eigvecs
    Z = T / np.sqrt(eigvals)

    # Squared Mahalanobis distance under the predictive model
    d2 = np.sum(Z**2, axis=1)

    # HDR/HPD PIT: HPD(y) = P( f(Yhat) >= f(y) ) = P( d2(Yhat) <= d2(y) ) = F_{chi2}(d2(y))
    u_hdr = chi2.cdf(d2, df=2)

    # Univariate PITs for marginals (these should look ~uniform because we matched marginals)
    u_m1 = norm.cdf(Y[:, 0], loc=0, scale=1.0)
    u_m2 = norm.cdf(Y[:, 1], loc=0, scale=1.0)

    # Univariate PITs for PCA projections (PC1, PC2 under the predictive Gaussian)
    u_pc1 = norm.cdf(T[:, 0], loc=0, scale=np.sqrt(eigvals[0]))
    u_pc2 = norm.cdf(T[:, 1], loc=0, scale=np.sqrt(eigvals[1]))

    os.makedirs(args.outdir, exist_ok=True)

    # 1) PIT histograms
    hist_path = os.path.join(args.outdir, "hdr_pca_analogy_histograms.png")
    plt.figure(figsize=(11, 7))
    bins = 40
    panels = [
        ("Marginal PIT $U_{m,1}$", u_m1),
        ("Marginal PIT $U_{m,2}$", u_m2),
        ("PCA PIT $U_{\\mathrm{PC1}}$", u_pc1),
        ("PCA PIT $U_{\\mathrm{PC2}}$", u_pc2),
        ("HDR/HPD PIT $U_{\\mathrm{HDR}}$", u_hdr),
    ]
    for i, (title, data) in enumerate(panels, start=1):
        ax = plt.subplot(2, 3, i)
        ax.hist(data, bins=bins, density=True)
        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 1, 2, 3])
        # ax.set_yticklabels([])
    plt.tight_layout()
    plt.savefig(hist_path, dpi=200)
    plt.close()

    # 2) Geometry: predicted HDR ellipse + PC axes
    geom_path = os.path.join(args.outdir, "hdr_pca_analogy_geometry.png")
    plt.figure(figsize=(7, 7))
    ax = plt.gca()
    # Subsample for plotting clarity
    idx_plot = rng.choice(Y.shape[0], size=min(1500, Y.shape[0]), replace=False)
    ax.scatter(Y[idx_plot, 0], Y[idx_plot, 1], s=5, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("$y_1$")
    ax.set_ylabel("$y_2$")
    ax.set_title("PC axes align with predicted Gaussian HDR ellipses")

    # Ellipse boundary: d2(y) = chi2_ppf(p)
    c = chi2.ppf(args.p_hdr, df=2)
    theta = np.linspace(0, 2 * np.pi, 400)
    circle = np.vstack([np.cos(theta), np.sin(theta)])  # unit circle
    ellipse_pc = np.diag(np.sqrt(eigvals)) @ (np.sqrt(c) * circle)
    ellipse_xy = (eigvecs @ ellipse_pc).T
    ax.plot(ellipse_xy[:, 0], ellipse_xy[:, 1], linewidth=2,
            label=f"Predicted {int(args.p_hdr*100)}% HDR ellipse")

    # PC axes
    scale = 4.0
    v1, v2 = eigvecs[:, 0], eigvecs[:, 1]
    ax.plot([0, scale * v1[0]], [0, scale * v1[1]], linewidth=2, label="PC1 axis")
    ax.plot([0, scale * v2[0]], [0, scale * v2[1]], linewidth=2, label="PC2 axis")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    plt.tight_layout()
    plt.savefig(geom_path, dpi=200)
    plt.close()

    # 3) "Decomposition" heatmap: show that -log density is a monotone function of PC-radius d2
    decomp_path = os.path.join(args.outdir, "hdr_pca_analogy_decomposition_heatmap.png")
    logpdf = multivariate_normal(mean=np.zeros(2), cov=Sigma_pred).logpdf(Y)
    z1sq = Z[:, 0] ** 2
    z2sq = Z[:, 1] ** 2
    vars_ = np.vstack([z1sq, z2sq, d2, -logpdf]).T
    names = ["z1^2", "z2^2", "D^2", "-log f(y)"]
    corr = np.zeros((len(names), len(names)))
    for i in range(len(names)):
        for j in range(len(names)):
            corr[i, j] = spearmanr(vars_[:, i], vars_[:, j]).correlation

    plt.figure(figsize=(6.5, 5.5))
    ax = plt.gca()
    im = ax.imshow(corr, vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_title("Spearman correlations: HDR (density) decomposes in PC coordinates")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(decomp_path, dpi=200)
    plt.close()

    print("Saved:")
    print(" -", os.path.abspath(hist_path))
    print(" -", os.path.abspath(geom_path))
    print(" -", os.path.abspath(decomp_path))
    print()
    print("Predictive covariance Σ_pred =", Sigma_pred.tolist())
    print("Predictive eigenvalues λ =", eigvals.tolist())
    print("Predictive eigenvectors (columns) V =\n", eigvecs)


if __name__ == "__main__":
    main()
