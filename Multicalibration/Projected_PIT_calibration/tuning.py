from moc.configs.config import get_config
from moc.utils.run_config import RunConfig
from moc.models.mixture.mixture_model2 import MixtureLightningModule
from moc.models.trainers.lightning_trainer import get_lightning_trainer
from moc.datamodules.real_datamodule import RealDataModule
from moc.metrics.distribution_metrics import pce, multivariate_energy_score
import numpy as np
import torch
import pickle
import optuna
from functools import partial
import wandb

class EnergyConstraintCallback:
    def __init__(self):
        self.best_energy = float('inf')

    def __call__(self, study, trial):
        energy = trial.user_attrs.get("energy")
        if energy is None:
            return

        if energy < self.best_energy:
            self.best_energy = energy

        threshold = self.best_energy * 1.1
        if energy > threshold:
            trial.set_user_attr("constraint_violation", True)
            trial.report(float('inf'), step=0)  # mark as unpromising
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

    # pces, weights, nlls = [], [], []
    pces, nlls, energies = [], [], []

    with torch.no_grad():
        for x, y in datamodule.val_dataloader():
            x = x.to(config.device)
            y = y.to(config.device)
            dist = model.predict(x)
            # pce_values, w = pce(dist, y, n_samples=100, prerank=prerank, setup='real')
            pce_values = pce(dist, y, n_samples=100, prerank=prerank, setup='real')
            nll = -dist.log_prob(y).mean().item()
            energy = multivariate_energy_score(dist, y)
            pces.append(pce_values)
            # weights.append(w)
            nlls.append(nll)
            energies.append(energy)

    pce_total = torch.stack(pces).mean(dim=0)
    energy_total = torch.stack(energies).mean(dim=0)
    if prerank == 'marginal':
        pce_total = pce_total.mean()
    # weights_total = torch.stack(weights).mean(dim=0)
    # weighted_sum = torch.sum(pce_total * weights_total).item()
    nll_mean = np.mean(nlls)

    # Log metrics to W&B
    wandb.log({
    "lambda_reg": lambda_reg,
    "pce": pce_total.item(),
    "nll": nll_mean,
    "energy": energy_total.item()
    })


    # Log both to the trial
    trial.set_user_attr("energy", energy_total.item())
    return pce_total.item()

wandb_run = wandb.init(
    project="multicalibration-hparam-tuning",
    name="optuna_tuning_curve",
    config={"search_space": {"lambda_reg": [1e-4, 100.0]}}
)

callback = EnergyConstraintCallback()
config = get_config()
config.device = 'cuda'
data_group, data_name = ['camehl', 'households']
seed = 42
prerank = 'marginal'
wrapped_objective = partial(objective, config=config, data_group=data_group, 
                            data_name=data_name, seed=seed, prerank=prerank)

study = optuna.create_study(direction="minimize")
study.optimize(wrapped_objective, n_trials=40, callbacks=[callback])
wandb_run.finish()

print("Best lambda_reg:", study.best_params["lambda_reg"])
print("Best PCE:", study.best_value)
print("Corresponding Energy:", study.best_trial.user_attrs["energy"])