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
import wandb

torch.set_printoptions(precision=3, sci_mode=False, threshold=float('inf'), edgeitems=40, linewidth=200)

config = get_config()
config.device = 'cuda'
data_group, data_name = 'mulan', 'sf2'
# hparams = {
#     'model': 'mixture',
#     'prerank': 'marginal',
#     'lambda': 1.0,
# }

rc = RunConfig(config, data_group, data_name)
datamodule = RealDataModule(rc, num_workers=8)
p, q = datamodule.input_dim, datamodule.output_dim #268,16

# prerank_epoch_pairs = {
#     'dependency': 'epoch_0178.ckpt',
#     'pca': "epoch_0185.ckpt",
# }

ckpt_path = "/mnt/default/elnura_workspace/Multivariate-recalibration/Multicalibration/Projected_PIT_calibration/logs/2025-07-09/15-18-14/mulan/sf2/model=mixture,prerank=cdf,lambda=1.0/0/checkpoints/epoch_0273.ckpt"
best_model = MixtureLightningModule.load_from_checkpoint(ckpt_path)
best_model.eval().to(config.device)

test_loader = datamodule.test_dataloader()

pces, cdfs = [], []
prerank = 'marginal'
for x, y in datamodule.test_dataloader():
    x = x.to(config.device)
    y = y.to(config.device)
    dist = best_model.predict(x)
    print(f"model was trained with {best_model.hparams.prerank} prerank and lambda {best_model.hparams.lambda_reg}")
    pce_values, cdf_values = pce(dist, y, n_samples=100, prerank=prerank)
    pces.append(pce_values)
    cdfs.append(cdf_values)

pce_total = torch.stack(pces).mean(dim=0) 
cdfs_total = torch.stack(cdfs).mean(dim=0)  

print(pce_total.shape)
print(cdfs_total.shape)

colors = [
    "#1f77b4",  # muted blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # yellow-green
    "#17becf",  # cyan
    "#aec7e8",  # light blue
    "#ffbb78",  # light orange
    "#98df8a",  # light green
    "#ff9896",  # light red
    "#c5b0d5",  # light purple
    "#c49c94",  # light brown
]

alphas = np.linspace(0, 1, 100)
sorted_indices = torch.argsort(pce_total, descending=True)  # or ascending=False

plt.figure(figsize=(6, 4))
for rank, d in enumerate(sorted_indices):
    d = d.item()
    cdf = cdfs_total[d].cpu().numpy()
    plt.plot(alphas, cdf, color=colors[rank], 
             lw=1.5, alpha=0.8, label=f"d={d+1} ({pce_total[d]:.3f})")

plt.plot(alphas, alphas, linestyle='--', color='black')
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\hat{F}_Z(\alpha)$")
plt.title(f"PCE on {prerank} from model trained on {best_model.hparams.prerank}: {pce_total.mean().item():.3f}")
plt.grid(True)
plt.tight_layout()
plt.legend(loc = 'upper left', fontsize=8)
plt.savefig(f"figures/after-reg/sf2/rel_plot_{prerank}_{data_name}_mixgauss_trained_on_{best_model.hparams.prerank}.png", dpi=300)
plt.show()