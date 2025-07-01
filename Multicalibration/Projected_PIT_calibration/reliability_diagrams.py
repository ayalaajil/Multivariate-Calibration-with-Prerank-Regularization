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
import os
import csv
from itertools import product
import pickle
from scipy.interpolate import make_interp_spline

plt.style.use('seaborn-v0_8')

config = get_config()
config.device = 'cpu'
torch.manual_seed(42)

dataset_names = [
    ['camehl', 'households'], ['cevid', 'air'], ['cevid', 'births1'], ['cevid', 'births2'], ['cevid', 'wage'],
    ['mulan', 'scm20d'], ['mulan', 'rf2'], ['mulan', 'rf1'], ['mulan', 'scm1d'], ['mulan', 'atp1d'],
    ['mulan', 'atp7d'], ['mulan', 'oes97'], ['mulan', 'oes10'], ['mulan', 'jura'], ['mulan', 'sf1'],
    ['mulan', 'sf2'], ['mulan', 'wq'], ['mulan', 'enb'], ['mulan', 'slump'], ['mulan', 'osales'],
    ['mulan', 'scpf'], ['feldman', 'meps_21'], ['feldman', 'meps_19'], ['feldman', 'meps_20'],
    ['feldman', 'house'], ['feldman', 'bio'], ['feldman', 'blog_data'], ['del_barrio', 'calcofi'],
    ['del_barrio', 'ansur2'], ['wang', 'taxi'], ['wang', 'energy']
]

#preranks = ["Marginal", "Loc.", "Scale", "Dep.", "PCA", "HDR"]

preranks =["marginal", "mean", "pca", "dependency", "variance"]

os.makedirs("figures/reliability_diagrams", exist_ok=True)

def evaluate_model(rc, datamodule, p, q, lambda_reg, prerank):
    model = MixtureLightningModule(p, q, lambda_reg=lambda_reg, reg_type='pce-kde', prerank=prerank.lower())
    trainer = get_lightning_trainer(rc)
    trainer.fit(model, datamodule)
    model.to(rc.device)
    model.eval()

    pces, cdfs, explained_vars = [], [], []
    with torch.no_grad():
        for x, y in datamodule.val_dataloader():
            x, y = x.to(rc.device), y.to(rc.device)
            dist = model.predict(x)
            pce_vals, cdf_vals, var_expl = pce(dist, y, n_samples=100, mode='average', prerank=prerank.lower(), setup='real')
            pces.append(pce_vals)
            cdfs.append(cdf_vals)
            explained_vars.append(var_expl)
    return torch.stack(pces).mean(0), torch.stack(cdfs).mean(0), np.stack(explained_vars).mean(0)

def run_experiment(domain, name, prerank, seed=42, lambda_reg=0.033):
    #rc = RunConfig(config=None, domain=domain, name=name)  # Customize as needed
    rc = RunConfig(config,domain, name)
    datamodule = RealDataModule(rc, seed=seed, num_workers=8)
    p, q = datamodule.input_dim, datamodule.output_dim

    pce_reg, cdf_reg, var_reg = evaluate_model(rc, datamodule, p, q, lambda_reg, prerank)
    pce_noreg, cdf_noreg, var_noreg = evaluate_model(rc, datamodule, p, q, 0, prerank)

    pce_reg = (pce_reg.cpu().numpy() * var_reg).sum()
    pce_noreg = (pce_noreg.cpu().numpy() * var_noreg).sum()

    alphas = np.linspace(0, 1, 100)
    plt.figure(figsize=(6, 4))
    for d in range(len(cdf_reg)):
        plt.plot(alphas, cdf_noreg[d], label=f"D{d+1} no-reg", lw=1, alpha=0.6)
        plt.plot(alphas, cdf_reg[d], label=f"D{d+1} reg", lw=1, alpha=0.6)
    plt.plot(alphas, alphas, linestyle='--', color='black')
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\hat{F}_Z(\alpha)$")
    plt.title(f"{domain}/{name} | {prerank} | PCE={pce_reg:.4f}")
    plt.grid(True)
    plt.tight_layout()
    fig_path = f"figures/multicalibration/{domain}_{name}_{prerank}.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    return pce_reg, pce_noreg, fig_path


def get_lambda_from_file(domain, name, prerank):
    folder = "Selected_lambda"
    target_suffix = f"_{prerank}.csv".lower()

    # Find file that ends with _<prerank> (case insensitive)
    matched_files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(target_suffix)
    ]

    if len(matched_files) == 0:
        raise FileNotFoundError(f"No file in {folder} ends with '_{prerank}'")
    if len(matched_files) > 1:
        raise ValueError(f"Multiple files found ending with '_{prerank}' in {folder}: {matched_files}")

    file_path = os.path.join(folder, matched_files[0])
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == domain and row[1] == name:
                try:
                    return float(row[2])  # lambda_reg
                except ValueError:
                    raise ValueError(f"Invalid lambda_reg value for {domain}/{name} in {file_path}")
    
    raise ValueError(f"No entry found for {domain}/{name} in {file_path}")

def run_all():
    results = []
    for (domain, name), prerank in product(dataset_names, preranks):
        try:
            print(f"Running {domain}/{name} with prerank {prerank}...")
            lambda_reg = get_lambda_from_file(domain, name, prerank)
            pce_reg, pce_noreg, fig_path = run_experiment(domain, name, prerank, lambda_reg=lambda_reg)
            results.append({
                "dataset": f"{domain}/{name}",
                "prerank": prerank,
                "pce_reg": pce_reg,
                "pce_noreg": pce_noreg,
                "figure": fig_path
            })
        except Exception as e:
            print(f"Failed on {domain}/{name} with {prerank}: {e}")
    return results



run_all()