from moc.configs.config import get_config
from moc.utils.run_config import RunConfig
# from moc.models.mqf2.lightning_module import MQF2LightningModule
from moc.models.mixture.mixture_model2 import MixtureLightningModule
from moc.models.gaussian.gaussian import GaussianLightningModule
from moc.models.trainers.lightning_trainer import get_lightning_trainer
from moc.datamodules.real_datamodule import RealDataModule
import numpy as np
import pandas as pd
from moc.metrics.distribution_metrics import pce, multivariate_energy_score
import torch
import wandb

# wandb.login(key="9d338bfb8d6dd9ab97384ee89b11f332ae3e12b8") #ELNURA'S KEY

torch.set_printoptions(precision=3, sci_mode=False, threshold=float('inf'), edgeitems=40, linewidth=200)

config = get_config()
config.device = 'cuda'
data_group, data_name = 'mulan', 'sf2'
hparams = {
    'model': 'mixture',
    'prerank': 'density',
    'lambda': 0.0,
} #hparams is useful for the chekcpoint files to have useful names
rc = RunConfig(config, data_group, data_name, hparams = hparams)
datamodule = RealDataModule(rc, num_workers=8)
p, q = datamodule.input_dim, datamodule.output_dim #268,16

preranks = ['marginal', 'mean', 'variance', 'dependency', 'pca', 'density']
# preranks = ['cdf']
lambdas = [0.0, 0.01, 0.1, 1.0, 5.0, 10.0]

results = []

for prerank in preranks:
    for l in lambdas:
        print(f"working on prerank {prerank} and lambda {l}")
        rc.hparams['lambda'] = l
        rc.hparams["prerank"] = prerank

        model = MixtureLightningModule(p, q, lambda_reg=l, reg_type='pce-kde', prerank=prerank)
        trainer = get_lightning_trainer(rc)
        trainer.fit(model, datamodule)

        ckpt_path = trainer.checkpoint_callback.best_model_path
        best_model = MixtureLightningModule.load_from_checkpoint(ckpt_path)
        best_model.eval().to(config.device)

        val_loader = datamodule.val_dataloader()
        all_pce, all_energy = [], []

        for x, y in val_loader:
            x = x.to(config.device)
            y = y.to(config.device)
            dist = best_model.predict(x)
            print(best_model.hparams.prerank)
            print(best_model.hparams.es_num_samples)
            pce_val, _ = pce(dist, y, n_samples=best_model.hparams.es_num_samples, prerank=best_model.hparams.prerank)
            energy_val = multivariate_energy_score(dist, y, n_samples=best_model.hparams.es_num_samples).mean()
            all_pce.append(pce_val.mean().item())
            all_energy.append(energy_val.item())

        avg_pce = np.mean(all_pce)
        avg_energy = np.mean(all_energy)

        results.append({
            "lambda": l,
            "prerank": prerank,
            "pce": avg_pce,
            "energy": avg_energy
        })

       # wandb.finish()
df = pd.DataFrame(results)

best_lambdas = {}

for prerank in preranks:
    subdf = df[df['prerank'] == prerank]
    print(subdf)
    if 0.0 not in subdf['lambda'].values:
        continue
    baseline_energy = subdf[subdf['lambda'] == 0.0]['energy'].values[0]
    valid = subdf[subdf['energy'] <= 1.1 * baseline_energy]
    best = valid.sort_values('pce').iloc[0] if not valid.empty else subdf.sort_values('pce').iloc[0]
    best_lambdas[prerank] = best['lambda']
    
for k, v in best_lambdas.items():
    print(f"{k}: best_lambda = {v}")