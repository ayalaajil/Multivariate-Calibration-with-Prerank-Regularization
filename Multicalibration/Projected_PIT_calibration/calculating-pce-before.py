from moc.configs.config import get_config
from moc.utils.run_config import RunConfig
# from moc.models.mqf2.lightning_module import MQF2LightningModule
from moc.models.mixture.mixture_model2 import MixtureLightningModule
from moc.models.gaussian.gaussian import GaussianLightningModule
from moc.models.trainers.lightning_trainer import get_lightning_trainer
from moc.datamodules.real_datamodule import RealDataModule
import numpy as np
import matplotlib.pyplot as plt
from moc.metrics.distribution_metrics import pce
import torch
import pickle
import wandb

plt.style.use('seaborn-v0_8')
plt.rcParams.update({
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

def run_prerank_eval(prerank, data_group='mulan', data_name='sf2'):
    config = get_config()
    config.device = 'cuda'
    torch.manual_seed(42)

    seeds = [0, 42, 866, 12, 4]
    pce_over_seeds, cdf_over_seeds = [], []

    for seed in seeds:
        print(f"Dataset: {data_group} {data_name} | Prerank: {prerank} | Seed: {seed}")
        rc = RunConfig(config, data_group, data_name, seed=seed)
        datamodule = RealDataModule(rc, seed=seed, num_workers=8)
        p, q = datamodule.input_dim, datamodule.output_dim
        model = MixtureLightningModule(p, q)
        trainer = get_lightning_trainer(rc)
        trainer.fit(model, datamodule)

        model.to(config.device)
        model.eval()
        pces, cdfs = [], []
        if prerank == 'pca':
            weights = []

        with torch.no_grad():
            for x, y in datamodule.val_dataloader():
                x = x.to(config.device)
                y = y.to(config.device)
                dist = model.predict(x)
                pce_values, cdf_values, extra = pce(dist, y, n_samples=100, prerank=prerank, setup='real', mode='test')
                pces.append(pce_values)
                cdfs.append(cdf_values)
                if prerank == 'pca':
                    explained_var = torch.from_numpy(extra).to(pce_values.device)
                    weights.append(explained_var)

        pce_total = torch.stack(pces).mean(dim=0)
        cdfs_total = torch.stack(cdfs).mean(dim=0)

        if prerank == 'pca':
            weights_total = torch.stack(weights).mean(dim=0)
            pce_total = torch.sum(pce_total * weights_total)
            cdfs_total = cdfs_total.mean(dim=0)
        elif prerank == 'marginal':
            pce_total = pce_total.mean(dim=0)
            cdfs_total = cdfs_total.mean(dim=0)

        pce_over_seeds.append(pce_total.cpu().numpy())
        cdf_over_seeds.append(cdfs_total.cpu().numpy())

    pce_over_seeds = np.stack(pce_over_seeds)
    cdf_over_seeds = np.stack(cdf_over_seeds)

    print(f"[{prerank}] Mean PCE: {pce_over_seeds.mean():.4f}")
    print(f"[{prerank}] Std Error: {pce_over_seeds.std() / np.sqrt(len(seeds)):.4f}")

    alphas = np.linspace(0, 1, 100)
    plt.figure(figsize=(6, 4))
    for s in range(len(seeds)):
        cdf = cdf_over_seeds[s][0] if prerank not in ['pca', 'marginal'] else cdf_over_seeds[s]
        plt.plot(alphas, cdf, color='royalblue', lw=1.5, alpha=0.8)
    plt.plot(alphas, alphas, linestyle='--', color='black')
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\hat{F}_Z(\alpha)$")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figures/rel_plot_{prerank}_{data_name}.png", dpi=300)
    plt.show()

for prerank in ['marginal', 'mean', 'variance', 'dependency', 'pca', 'density']:
    run_prerank_eval(prerank)