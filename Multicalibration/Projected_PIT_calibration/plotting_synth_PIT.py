import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal
from matplotlib.backends.backend_pdf import PdfPages
from moc.metrics.distribution_metrics import calculate_PIT, calculate_PIT_density

device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_model_samples_per_y = 100
n_bins = 10
n_true_samples=1000

def build_multivariate_normal(d, sigma2, tau, mean):
    indices = torch.arange(d).unsqueeze(0)
    covariance_matrix = sigma2 * torch.exp(-torch.abs(indices.T - indices) / tau)
    return MultivariateNormal(mean, covariance_matrix)

def plot_pit_for_distributions(true_dist, model_dist, prerank='identity'):
    with torch.no_grad():
        y = true_dist.sample((n_true_samples,)).to(device)
        samples = model_dist.sample((y.shape[0] * n_model_samples_per_y,)).reshape(y.shape[0], n_model_samples_per_y, -1)
        print(samples.shape)

        if prerank == 'density':
            pit_values = calculate_PIT_density(model_dist, y)
        else:
            pit_values = calculate_PIT(samples, y, prerank) 
        total_pits_np = pit_values.detach().cpu().numpy()

    dimension = len(pit_values)
    cols = 3
    rows = int(np.ceil(dimension / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axs = np.array(axs).reshape(rows, cols)

    for i in range(dimension):
        row, col = divmod(i, cols)
        ax = axs[row, col]
        ax.hist(total_pits_np[i], bins=n_bins, density=True, alpha=0.6, color='g')
        ax.set_title(f"PIT - Projection {i+1}")
        ax.set_xlabel("Projected PIT Value")
        ax.set_ylabel("Density")

    for i in range(dimension, rows * cols):
        fig.delaxes(axs.flatten()[i])

    fig.suptitle(f"PIT histograms for with preranl: {prerank}", fontsize=16)

    textstr = (
        f"Number of true samples (y): {n_true_samples}\n"
        f"Number of samples per y from model: {n_model_samples_per_y}\n"
        f"Number of bins: {n_bins}"
    )
    fig.text(0.5, 0.02, textstr, ha='center', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plots_per_method(method_name):
    
    pdf_filename = f"{method_name}_synth_cfgD.pdf"

    d=10
    synthetic_params = [
        (d, 1.0, 1.0, torch.full((d,), -0.5)),
        (d, 1.0, 1.0, torch.full((d,), 0.5)),
        (d, 0.85, 1.0, torch.zeros(d)),
        (d, 1.25, 1.0, torch.zeros(d)),
        (d, 1.0, 0.5, torch.zeros(d)),
        (d, 1.0, 2.0, torch.zeros(d)),
    ]
    true_dist = build_multivariate_normal(d, 1.0, 1.0, torch.zeros(d))

    with PdfPages(pdf_filename) as pdf:
        for i, (d, sigma2, tau, mean) in enumerate(synthetic_params):
            print(f"Working on synthetic dataset {i} with d={d}, sigma2={sigma2}, tau={tau}")

            model_dist = build_multivariate_normal(d, sigma2, tau, mean)

            mean_val = mean[0].item()
            
            if method_name == "marginal":
                fig = plot_pit_for_distributions(true_dist, model_dist, prerank='identity')
                fig.suptitle(f"PIT histograms with prerank: marginal \n"+
                f"Dataset {i+1}: d={d}, σ²={sigma2}, τ={tau}, " + r"$\mathrm{{mean}} = (%.2f)^{%d}$" % (mean_val, d) ,
                fontsize=14
                )
                pdf.savefig(fig)
                plt.close(fig)
            elif method_name == "PCA":
                fig = plot_pit_for_distributions(true_dist, model_dist, prerank='pca')
                fig.suptitle(f"PIT histograms with prerank:PCA\n"+
                f"Dataset {i+1}: d={d}, σ²={sigma2}, τ={tau}, " + r"$\mathrm{{mean}} = (%.2f)^{%d}$" % (mean_val, d) ,
                fontsize=14
                )
                pdf.savefig(fig)
                plt.close(fig)
            elif method_name == "prerank":
                for prerank in ['mean', 'variance', 'dependency']:
                    fig = plot_pit_for_distributions(true_dist, model_dist, prerank=prerank)
                    fig.suptitle(f"PIT histograms with prerank: {prerank} \n"+
                    f"Dataset {i+1}: d={d}, σ²={sigma2}, τ={tau}, " + r"$\mathrm{{mean}} = (%.2f)^{%d}$" % (mean_val, d) ,
                    fontsize=14
                    )
                    pdf.savefig(fig)
                    plt.close(fig)
            elif method_name == "HDR":
                fig = plot_pit_for_distributions(true_dist, model_dist, prerank='density')
                fig.suptitle(f"PIT histograms with prerank:density \n"+
                f"Dataset {i+1}: d={d}, σ²={sigma2}, τ={tau}, " + r"$\mathrm{{mean}} = (%.2f)^{%d}$" % (mean_val, d) ,
                fontsize=14
                )
                pdf.savefig(fig)
                plt.close(fig)

if __name__ == "__main__":
    for method in ["HDR"]:
        plots_per_method(method)