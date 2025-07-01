from moc.utils.run_config import RunConfig
from moc.models.mixture.mixture_model2 import MixtureLightningModule
from moc.models.trainers.lightning_trainer import get_lightning_trainer
from moc.datamodules.real_datamodule import RealDataModule
from moc.metrics.distribution_metrics import pce
import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt



from moc.configs.config import get_config
config = get_config()
config.device = 'cpu'

dataset_names = [
                #  ['camehl', 'households'], 
                #  ['cevid', 'air'], ['cevid', 'births1'],
                #  ['cevid', 'births2'], ['cevid', 'wage'], ['mulan', 'scm20d'],
                #  ['mulan', 'rf2'], ['mulan', 'rf1'], ['mulan', 'scm1d'],
                #  ['mulan', 'atp1d'], ['mulan', 'atp7d'], ['mulan', 'oes97'],
                #  ['mulan', 'oes10'], ['mulan', 'jura'], ['mulan', 'sf1'],
                #  ['mulan', 'sf2'], ['mulan', 'wq'], ['mulan', 'enb'],
                #  ['mulan', 'slump'], 
                 ['mulan', 'osales'], 
                #  ['mulan', 'scpf'], 
                #  ['feldman', 'meps_21'], ['feldman', 'meps_19'], ['feldman', 'meps_20'], 
                #  ['feldman', 'house'], ['feldman', 'bio'], ['feldman', 'blog_data'], 
                #  ['del_barrio', 'calcofi'], ['del_barrio', 'ansur2'], ['wang', 'taxi'], 
                #  ['wang', 'energy'],
                 ]


def select_best_lambda(config, data_group, data_name, prerank, seeds, lambda_values):

    results = []
    for lambda_reg in lambda_values:
        pces = []
        for seed in seeds:
            rc = RunConfig(config, data_group, data_name, seed=seed)
            datamodule = RealDataModule(rc, seed=seed, num_workers=8)
            p, q = datamodule.input_dim, datamodule.output_dim
            model = MixtureLightningModule(p, q, lambda_reg=lambda_reg, reg_type='pce-kde', prerank=prerank)
            trainer = get_lightning_trainer(rc)
            trainer.fit(model, datamodule)
            model.to(config.device)
            model.eval()
            batch_pces = []
            with torch.no_grad():
                for x, y in datamodule.val_dataloader():
                    x = x.to(config.device)
                    y = y.to(config.device)
                    dist = model.predict(x)
                    pce_values = pce(dist, y, n_samples=100, prerank=prerank, setup='real')
                    batch_pces.append(pce_values)
            pce_total = torch.stack(batch_pces).mean(dim=0)
            if prerank in ['pca', 'marginal']:
                pce_total = pce_total.mean()
            pces.append(pce_total.item())
        avg_pce = np.mean(pces)
        results.append((lambda_reg, avg_pce))
    best_lambda, best_pce = min(results, key=lambda x: x[1])
    print(f"\nBest lambda selected: {best_lambda:.4f} (Avg PCE: {best_pce:.5f})")
    return best_lambda


def evaluate_and_plot_reliability(config, data_group, data_name, prerank, seeds, lambda_reg, plot_path="reliability_plot.pdf"):

    pce_over_seeds, cdf_over_seeds = [], []
    for seed in seeds:
        print(f"Working on dataset {data_group} {data_name} {prerank} seed {seed}")
        rc = RunConfig(config, data_group, data_name, seed=seed)
        datamodule = RealDataModule(rc, seed=seed, num_workers=8)
        p, q = datamodule.input_dim, datamodule.output_dim
        model = MixtureLightningModule(p, q, lambda_reg=lambda_reg, reg_type='pce-kde', prerank=prerank)
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

    print(f"Mean PCE over seeds: {pce_over_seeds.mean():.5f}")
    print(f"Standard error: {pce_over_seeds.std() / np.sqrt(len(seeds)):.5f}")

    alphas = np.linspace(0, 1, 100)
    plt.figure(figsize=(6, 4))
    for s in range(len(seeds)):
        plt.plot(alphas, cdf_over_seeds[s], color='royalblue', lw=1.5, alpha=0.8)
    plt.plot(alphas, alphas, linestyle='--', color='black')
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\hat{F}_Z(\alpha)$")
    plt.title(f"Reliability diagram – {data_name} ({prerank})")
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Reliability plot saved to: {plot_path}")


# Step 1 : tuning
best_lambda = select_best_lambda(
    config=config,
    data_group='mulan',
    data_name='sf2',
    prerank='cdf',
    seeds=[42],
    lambda_values=[0.001, 0.01, 0.1, 1]
)

data_group='mulan'
data_name='sf2'
prerank = 'cdf'

# Step 2 : analyze and plot
evaluate_and_plot_reliability(
    config=config,
    data_group=data_group,
    data_name= data_name,
    prerank= prerank,
    seeds=[42],
    lambda_reg=best_lambda,
    plot_path=f"reliability_{data_name}_{prerank}.pdf"
)
