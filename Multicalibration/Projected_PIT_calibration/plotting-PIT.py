from moc.configs.config import get_config
from moc.utils.run_config import RunConfig
# from moc.models.mqf2.lightning_module import MQF2LightningModule
from moc.models.mixture.mixture_model import MixtureLightningModule
from moc.models.gaussian.gaussian import GaussianLightningModule
from moc.models.trainers.lightning_trainer import get_lightning_trainer
from moc.datamodules.real_datamodule import RealDataModule
import numpy as np
import matplotlib.pyplot as plt
from moc.metrics.distribution_metrics import calculate_PIT
from matplotlib.backends.backend_pdf import PdfPages
import torch

config = get_config()
config.device = 'cpu'
M = 100
alphas = torch.linspace(0, 1, M, device=config.device)

def plot_pit_per_dataset(config, prerank, data_group, data_name):
    rc = RunConfig(config, data_group, data_name)
    seed= 42
    datamodule = RealDataModule(rc, seed=seed)
    p, q = datamodule.input_dim, datamodule.output_dim
    model = MixtureLightningModule(p,q)
    #model = MQF2LightningModule(p, q)
    trainer = get_lightning_trainer(rc)
    trainer.fit(model, datamodule)

    total_pits = []
    for x, y in datamodule.test_dataloader():
        x = x.to(config.device)
        y = y.to(config.device)
        dist = model.predict(x)
        samples = dist.sample((100,)).permute(1, 0, 2) #256,100,4
        c = min(y.shape[1], 3)
        pit_values = calculate_PIT(samples, y, prerank) 
        total_pits.append(pit_values)
    total_pits = torch.cat(total_pits, dim=1) #shape (n_components, number of test data points, 1)

    dimension =  len(pit_values)
    cols = 3  # Fixes 3 plots per row
    rows = int(np.ceil(dimension / cols))  # Number of rows needed

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))  # Dynamically adjust the size

    # Ensure `axs` is always a 2D array to avoid errors
    axs = np.array(axs).reshape(rows, cols)

    for i in range(dimension):
        row, col = divmod(i, cols)  # Find the position in the grid
        ax = axs[row, col]  # Get the corresponding axis
        
        ax.hist(total_pits[i], bins=10, density=True, alpha=0.6, color='g')
        title = f"PIT - Direction according to component ${i+1}$" 
        ax.set_title(title)
        ax.set_xlabel("Projected PIT Value")
        ax.set_ylabel("Density")

    # Remove empty subplots if `dimension` is not a multiple of 3
    for i in range(dimension, rows * cols):
        fig.delaxes(axs.flatten()[i])

    plt.tight_layout()  # Automatically adjust the display
    filename = f"PIT_proj_{data_group}_{data_name}.png"
    plt.tight_layout()
    return fig

def plots_per_method(config, method_name):
    dataset_names = [
     ['camehl', 'households'],
                  ['mulan', 'scm20d'],
                  ['mulan', 'rf2'],
                  ['mulan', 'rf1'],
                  ['mulan', 'scm1d'],
                  ['feldman', 'meps_21'],
                  ['feldman', 'meps_19'],
                  ['feldman', 'meps_20'],
                  ['feldman', 'house'],
                  ['feldman', 'bio'],
                  ['feldman', 'blog_data'],
                 ['del_barrio', 'calcofi'],
                 ['wang', 'taxi']
                 ]
    pdf_filename = f"{method_name}.pdf"
    with PdfPages(pdf_filename) as pdf:
        for dataset in dataset_names:
            data_group, data_name = dataset
            print(f"Working on dataset {data_name}")
            
            if method_name == "marginal":
                fig = plot_pit_per_dataset(config, 'identity', data_group, data_name)
                pdf.savefig(fig)
                plt.close(fig)
            elif method_name == "PCA":
                fig = plot_pit_per_dataset(config, 'pca', data_group, data_name)
                pdf.savefig(fig)
                plt.close(fig)
            elif method_name == "prerank":
                for prerank in ['mean', 'variance', 'dependency']:
                    fig = plot_pit_per_dataset(config, prerank, data_group, data_name)
                    pdf.savefig(fig)
                    plt.close(fig)

def plots(config):
    methods= ["axes", "PCA", "prerank"]
    for method in methods:
        plots_per_method(config, method)

plots(config)