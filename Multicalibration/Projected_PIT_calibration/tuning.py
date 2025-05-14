from moc.configs.config import get_config
from moc.utils.run_config import RunConfig
from moc.models.mixture.mixture_model2 import MixtureLightningModule
from moc.models.trainers.lightning_trainer import get_lightning_trainer
from moc.datamodules.real_datamodule import RealDataModule
from moc.metrics.distribution_metrics import pce
import numpy as np
import torch
import pickle
import optuna
from functools import partial
import wandb
import csv
import os

class CRPSConstraintCallback:
    def __init__(self):
        self.base_crps = None  # Will store CRPS without regularization

    def __call__(self, trial):
        crps = trial.user_attrs.get("crps")
        if crps is None:
            return

        # Capture baseline CRPS (i.e., with lambda=0 or close to 0)
        lambda_reg = trial.params.get("lambda_reg", None)
        if lambda_reg is not None and lambda_reg < 1e-6:
            if self.base_crps is None or crps < self.base_crps:
                self.base_crps = crps
            return

        # If base_crps is known, enforce constraint
        if self.base_crps is not None:
            threshold = self.base_crps * 1.1
            if crps > threshold:
                trial.set_user_attr("constraint_violation", True)
                trial.report(float('inf'), step=0)
                raise optuna.exceptions.TrialPruned()


def objective(trial, config, data_group, data_name, seed, prerank):
    lambda_reg = trial.suggest_float("lambda_reg", 1e-3, 100.0, log=True)

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

    pces, nlls, crps_vals = [], [], []

    with torch.no_grad():
        for x, y in datamodule.val_dataloader():
            x = x.to(config.device)
            y = y.to(config.device)
            dist = model.predict(x)
            # pce_values, w = pce(dist, y, n_samples=100, prerank=prerank, setup='real')
            pce_values = pce(dist, y, n_samples=100, prerank=prerank, setup='real')
            pces.append(pce_values)

            # NLL
            nll = -dist.log_prob(y).mean().item()
            nlls.append(nll)

            # CRPS
            samples = dist.sample((100,))  # shape (100, batch, d)
            samples = samples.permute(1, 0, 2)  # shape (batch, 100, d)

            if q == 1:
                crps_batch = torch.tensor([
                    crps_ensemble(y[i].cpu().numpy(), samples[i][:, 0].cpu().numpy())
                    for i in range(len(y))
                ])
                crps_vals.append(crps_batch.mean().item())

    pce_total = torch.stack(pces).mean(dim=0)
    if prerank == 'marginal':
        pce_total = pce_total.mean()
    # weights_total = torch.stack(weights).mean(dim=0)
    # weighted_sum = torch.sum(pce_total * weights_total).item()
    nll_mean = np.mean(nlls)
    crps_mean = np.mean(crps_vals) if len(crps_vals) > 0 else float('inf')

    # Log metrics to W&B
    wandb.log({
    "lambda_reg": lambda_reg,
    "pce_weighted_sum": pce_total.item(),
    "nll_mean": nll_mean,
    "crps_mean": crps_mean,
    })


    # Log both to the trial
    trial.set_user_attr("nll", nll_mean)
    trial.set_user_attr("crps", crps_mean)
    return pce_total.item()


dataset_names = [
      ['camehl', 'households'],
                   ['mulan', 'scm20d'],
    #              ['mulan', 'rf2'],
    #              ['mulan', 'rf1'],
    #              ['mulan', 'scm1d'],
    #              ['feldman', 'meps_21'],
    #              ['feldman', 'meps_19'],
    #              ['feldman', 'meps_20'],
    #              ['feldman', 'house'],
    #              ['feldman', 'bio'],
    #              ['feldman', 'blog_data'],
    #             ['del_barrio', 'calcofi'],
    #             ['wang', 'taxi']
                 ]
seeds = [0, 42, 866, 12, 4]

# Chemin vers le fichier CSV
csv_path = "best_lambda.csv"

# Écrire l'en-tête si le fichier n'existe pas encore
if not os.path.exists(csv_path):
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["data_group", "data_name", "best_lambda_reg", "best_PCE", "best_NLL", "best_CRPS"])

for dataset in dataset_names:
    data_group, data_name = dataset
    print(f"Working on dataset {data_name}")

    wandb_run = wandb.init(
        project="multicalibration-hparam-tuning",
        name=f"optuna_tuning_curve_{data_name}",
        config={"search_space": {"lambda_reg": [1e-4, 100.0]}}
    )

    callback = CRPSConstraintCallback()
    config = get_config()
    config.device ="cuda"
    seed = 42
    prerank = 'marginal'

    wrapped_objective = partial(objective, config=config, data_group=data_group, 
                                data_name=data_name, seed=seed, prerank=prerank)
    study = optuna.create_study(direction="minimize")
    study.optimize(wrapped_objective, n_trials=30, callbacks=[callback])

    best_lambda = study.best_params["lambda_reg"]
    best_pce = study.best_value
    best_trial = study.best_trial
    best_nll = best_trial.user_attrs["nll"]
    best_crps = best_trial.user_attrs["crps"]

    print("Best lambda_reg:", best_lambda)
    print("Best PCE:", best_pce)
    print("Corresponding NLL:", best_nll)
    print("Corresponding CRPS:", best_crps)

    # Ajouter les résultats au CSV
    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([data_group, data_name, best_lambda, best_pce, best_nll, best_crps])