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
import os

# wandb.login(key="9d338bfb8d6dd9ab97384ee89b11f332ae3e12b8") #ELNURA'S KEY

torch.set_printoptions(precision=3, sci_mode=False, threshold=float('inf'), edgeitems=40, linewidth=200)

datasets = [['camehl', 'households'], ['cevid', 'air'], ['cevid', 'births1'],
            ['cevid', 'births2'], ['cevid', 'wage'], ['mulan', 'scm20d'],
            ['mulan', 'rf2'], ['mulan', 'rf1'], ['mulan', 'scm1d'],
            # ['mulan', 'sf2'], 
            ['mulan', 'wq'], ['mulan', 'scpf'],
            ['feldman', 'meps_21'], ['feldman', 'meps_19'], ['feldman', 'meps_20'],
            ['feldman', 'house'], ['feldman', 'bio'], ['feldman', 'blog_data'],
            ['del_barrio', 'calcofi'], ['del_barrio', 'ansur2'], ['wang', 'taxi']]

config = get_config()
config.device = 'cuda'

preranks = ['marginal', 'mean', 'variance', 'dependency', 'pca', 'density', 'cdf']
lambdas = [0.0, 0.01, 0.1, 1.0, 5.0, 10.0]

results_path = "tuning-results.csv"
# Load existing results if the file exists
if os.path.exists(results_path):
    df = pd.read_csv(results_path)
else:
    df = pd.DataFrame(columns=["data_name", "lambda", "prerank", "pce", "energy"])

for data_group, data_name in datasets:
    print(f"Loading dataset: {data_group}/{data_name}")
    hparams = {
        'model': 'mixture',
    }
    rc = RunConfig(config, data_group, data_name, hparams=hparams)
    datamodule = RealDataModule(rc, num_workers=8)
    p, q = datamodule.input_dim, datamodule.output_dim

    for prerank in preranks:
        if q < 3 and prerank == 'dependency':
            continue
        for l in lambdas:
            already_ran = (
                (df['lambda'] == l) &
                (df['prerank'] == prerank) &
                (df['data_name'] == data_name)
            ).any()

            if already_ran:
                print(f"Skipping existing pair: {data_group}/{data_name} | prerank={prerank}, lambda={l}")
                continue

            print(f"Working on {data_group}/{data_name} | prerank={prerank}, lambda={l}")
            rc.hparams["lambda"] = l
            rc.hparams["prerank"] = prerank

            model = MixtureLightningModule(p, q, lambda_reg=l, reg_type='pce-kde', prerank=prerank)
            trainer = get_lightning_trainer(rc)

            try:
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
                    pce_val, _ = pce(dist, y, n_samples=best_model.hparams.es_num_samples, prerank=best_model.hparams.prerank)
                    energy_val = multivariate_energy_score(dist, y, n_samples=best_model.hparams.es_num_samples).mean()
                    all_pce.append(pce_val.mean().item())
                    all_energy.append(energy_val.item())

                avg_pce = np.mean(all_pce)
                avg_energy = np.mean(all_energy)

                df = pd.concat([df, pd.DataFrame([{
                    "data_name": data_name,
                    "lambda": l,
                    "prerank": prerank,
                    "pce": avg_pce,
                    "energy": avg_energy
                }])], ignore_index=True)

                df.to_csv(results_path, index=False)
            except Exception as e:
                print(f"Failed on {data_group}/{data_name} | prerank={prerank}, lambda={l} due to {e}")
                continue