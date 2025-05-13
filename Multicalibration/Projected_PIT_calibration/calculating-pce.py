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

# plt.style.use('seaborn-v0_8')
# plt.rcParams.update({
#     'axes.titlesize': 12,
#     'axes.labelsize': 12,
#     'xtick.labelsize': 12,
#     'ytick.labelsize': 12,
#     'legend.fontsize': 12
# })

config = get_config()
config.device = 'cuda'

seeds = [0, 42, 866, 12, 4]
# seeds = [42]
dataset_names = [['camehl', 'households'], 
                 ['cevid', 'air'], ['cevid', 'births1'],
                 ['cevid', 'births2'], ['cevid', 'wage'], ['mulan', 'scm20d'],
                 ['mulan', 'rf2'], ['mulan', 'rf1'], ['mulan', 'scm1d'],
                 ['mulan', 'atp1d'], ['mulan', 'atp7d'], ['mulan', 'oes97'],
                 ['mulan', 'oes10'], ['mulan', 'jura'], ['mulan', 'sf1'],
                 ['mulan', 'sf2'], ['mulan', 'wq'], ['mulan', 'enb'],
                 ['mulan', 'slump'], ['mulan', 'osales'], ['mulan', 'scpf'], 
                 ['feldman', 'meps_21'], ['feldman', 'meps_19'], ['feldman', 'meps_20'], 
                 ['feldman', 'house'], ['feldman', 'bio'], ['feldman', 'blog_data'], 
                 ['del_barrio', 'calcofi'], ['del_barrio', 'ansur2'], ['wang', 'taxi'], 
                 ['wang', 'energy'],
                 ]
prerank = 'variance'
pce_across_datasets = {}
for dataset in dataset_names:
    data_group, data_name = dataset
    pce_over_seeds = []
    for seed in seeds:
        print(f"working on dataset {data_group} {data_name} {prerank} seed {seed}")
        rc = RunConfig(config, data_group, data_name, seed=seed)
        datamodule = RealDataModule(rc, seed=seed, num_workers = 16)
        p, q = datamodule.input_dim, datamodule.output_dim
        # model = GaussianLightningModule(p, q, lambda_reg = 1, reg_type = 'pce-kde', prerank = 'density')
        model = MixtureLightningModule(p, q, prerank = prerank)
        #model = MQF2LightningModule(p, q)
        trainer = get_lightning_trainer(rc)
        trainer.fit(model, datamodule)
        # wandb.finish()
        model.to(config.device)
        model.eval()
        pces = []
        with torch.no_grad():
            for x, y in datamodule.val_dataloader():
                x = x.to(config.device)
                y = y.to(config.device)
                dist = model.predict(x)
                pce_values = pce(dist, y, n_samples = 100, prerank = prerank, setup='real')
                # cdf = reliability_plots(dist, y, n_samples = 100, prerank = 'marginal', setup = 'real')
                pces.append(pce_values)
                # weights.append(_)
        pce_total = torch.stack(pces).mean(dim=0)
        pce_over_seeds.append(pce_total)
    pce_over_seeds = torch.stack(pce_over_seeds)
    avg_pce = pce_over_seeds.mean()
    stderr = pce_over_seeds.std() / np.sqrt(len(seeds))
    print(avg_pce.item(), stderr.item())
    pce_across_datasets[data_name] = (avg_pce, stderr)

filename = f"pkl-files/pce_across_32datasets_mixnll_{prerank}.pkl"
with open(filename, "wb") as f:
    pickle.dump(pce_across_datasets, f)

# alphas = np.linspace(0, 1, 100)
# cdfs_total = cdfs_total.cpu().numpy()
# pce_total = pce_total.cpu().numpy()
# sorted_indices = np.argsort(pce_total)

# colors = [
#     'royalblue',
#     'darkorange',
#     'seagreen',
#     'crimson',
#     'goldenrod',
#     'mediumorchid',
#     'deepskyblue',
#     'saddlebrown'
# ]
# plt.figure(figsize=(6, 4))
# for d in sorted_indices:
#     plt.plot(alphas, cdfs_total[d], color = colors[d], 
#              label=f"d = {d+1}", 
#              lw = 1.5, alpha = 0.8)
# plt.plot(alphas, alphas, linestyle='--', color='black')
# plt.xlabel(r"$\alpha$")
# plt.ylabel(r"$\hat{F}_Z(\alpha)$")
# plt.legend(loc='upper left')
# plt.title(f"{prerank} pre-rank, PCE = {pce_total.mean():.4f}")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(f"figures/after-reg/reliability_plot_{prerank}_bio_mixnll.png", dpi=300)
# plt.show()