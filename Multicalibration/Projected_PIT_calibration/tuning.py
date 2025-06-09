from moc.configs.config import get_config
from moc.utils.run_config import RunConfig
from moc.models.mixture.mixture_model2 import MixtureLightningModule
from moc.models.trainers.lightning_trainer import get_lightning_trainer
from moc.datamodules.real_datamodule import RealDataModule
from moc.metrics.distribution_metrics import pce, multivariate_energy_score
import numpy as np
import torch
import optuna
from functools import partial
import wandb
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def objective(trial, config, data_group, data_name, seed, prerank):
    lambda_reg = trial.suggest_float("lambda_reg", 1e-3, 10.0, log=True)

    rc = RunConfig(config, data_group, data_name, seed=seed)
    datamodule = RealDataModule(rc, seed=seed, num_workers=8)
    p, q = datamodule.input_dim, datamodule.output_dim

    model = MixtureLightningModule(
        p, q,
        lambda_reg=lambda_reg,
        reg_type='pce-kde',
        prerank=prerank,
    )

    trainer = get_lightning_trainer(rc)
    trainer.fit(model, datamodule)

    model.to(config.device)
    model.eval()

    pces, nlls = [], []
    if prerank == 'pca':
        weights = []

    with torch.no_grad():
        for x, y in datamodule.val_dataloader():
            x = x.to(config.device)
            y = y.to(config.device)
            dist = model.predict(x)
            # pce_values, w = pce(dist, y, n_samples=100, prerank=prerank, setup='real')
            if prerank == 'pca':
                pce_values, weight = pce(dist, y, n_samples=100, prerank=prerank, setup='real')
                weights.append(weight)
            else: pce_values = pce(dist, y, n_samples=100, prerank=prerank, setup='real')
            # NLL
            nll = -dist.log_prob(y).mean().item()
            nlls.append(nll)
            pces.append(pce_values)

    pce_total = torch.stack(pces).mean(dim=0)
    nll_mean = np.mean(nlls)
    if prerank =='pca':
        weights_total = torch.stack(weights).mean(dim=0)
        pce_total = torch.sum(pce_total * weights_total)
    elif prerank == 'marginal':
        pce_total = pce_total.mean()
    
    # Log metrics to W&B
    wandb.log({
        "lambda_reg": lambda_reg,
        "pce": pce_total.item(),
        "nll": nll_mean,
    })


    # Log both to the trial
    trial.set_user_attr("nll", nll_mean)
    # trial.set_user_attr("energy", energy_total.item())
    return pce_total.item()

config = get_config()
config.device = 'cuda'
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
data_group, data_name = ['mulan', 'oes10']
seed = 42
prerank = 'dependency'

wandb_run = wandb.init(
    project="multicalibration-hparam-tuning",
    name=f"{data_name}_{prerank}",
    config={"search_space": {"lambda_reg": [1e-4, 10.0]}}
)

wrapped_objective = partial(objective, config=config, data_group=data_group, 
                            data_name=data_name, seed=seed, prerank=prerank)


study = optuna.create_study(direction="minimize")
study.optimize(wrapped_objective, n_trials=40)  # no more callbacks here

wandb_run.finish()

# Post-processing : selection of best compromise PCE + constraint on NLL
all_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

# Find minimal NLL
max_nll, min_nll = max(t.user_attrs["nll"] for t in all_trials), min(t.user_attrs["nll"] for t in all_trials)
nll_threshold = min_nll + 0.5 * (max_nll - min_nll)

admissible_trials = [t for t in all_trials if t.user_attrs["nll"] <= nll_threshold]

best_trial = min(admissible_trials, key=lambda t: t.value)

print("Best lambda_reg (under constraint):", best_trial.params["lambda_reg"])
print("Best PCE (under constraint):", best_trial.value)
print("Corresponding NLL:", best_trial.user_attrs["nll"])

