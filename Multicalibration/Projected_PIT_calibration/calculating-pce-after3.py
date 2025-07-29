import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from moc.configs.config import get_config
from moc.utils.run_config import RunConfig
from moc.models.mixture.mixture_model2 import MixtureLightningModule
from moc.models.trainers.lightning_trainer import get_lightning_trainer
from moc.datamodules.real_datamodule import RealDataModule
from moc.metrics.distribution_metrics import pce, multivariate_energy_score, mse
import numpy as np
import pandas as pd
import torch

torch.set_printoptions(precision=3, sci_mode=False, threshold=float('inf'), edgeitems=40, linewidth=200)

datasets = [
            ['mulan', 'scm20d'],
            ['mulan', 'scm1d']
            ]

# tuning_all = pd.read_csv('tuning-results.csv')
tuning_all = pd.read_csv("tuning-results-double-reg2.csv")
best_lambdas_all = {}

for dataset in tuning_all['data_name'].unique():
    for prerank in tuning_all.prerank.unique():
        subdf = tuning_all[(tuning_all['data_name'] == dataset) & (tuning_all['prerank'] == prerank)]
        if len(subdf['lambda'].values) == 6:
            if 0.0 not in subdf['lambda'].values:
                continue
            baseline_energy = subdf[subdf['lambda'] == 0.0]['energy'].values[0]
            valid = subdf[subdf['energy'] <= 1.1 * baseline_energy]
            best = valid.sort_values('pce').iloc[0] if not valid.empty else subdf.sort_values('pce').iloc[0]
            best_lambdas_all[(dataset, prerank)] = best['lambda'] 
        else: continue

config = get_config()
config.device = 'cuda'

preranks = ['mean', 'variance', 'dependency', 'density', 'cdf']
seeds = [0, 42, 866, 12, 4]

results_path = "metrics-after-reg-double-reg3.csv"
if os.path.exists(results_path):
    df = pd.read_csv(results_path)
else:
    df = pd.DataFrame(columns=["data_name", "seed", "prerank", "pce_marg", "pce_prerank", "nll", "energy", "mse"])

for data_group, data_name in datasets:
    for prerank in preranks:
        for seed in seeds:
            trained = (
                (df['seed'] == seed) &
                (df['data_name'] == data_name) &
                (df['prerank'] == prerank)
            ).any()

            if trained:
                print(f"Skipping: {data_group}/{data_name} | seed={seed}")
                continue

            print(f"Training: {data_group}/{data_name} | {prerank} | seed={seed}")
            hparams = {
                'model': 'mixture',
                'seed': seed,
                'lambda': best_lambdas_all[(data_name, prerank)],
                'prerank': prerank
            }

            rc = RunConfig(config, data_group, data_name, hparams=hparams, seed = seed)
            datamodule = RealDataModule(rc, num_workers=8, seed = seed)
            p, q = datamodule.input_dim, datamodule.output_dim
            model = MixtureLightningModule(p, q, lambda_reg=hparams['lambda'], reg_type = 'pce-kde', 
                                           prerank=hparams['prerank'], double_reg = True, double_reg_type = 'pca')
            trainer = get_lightning_trainer(rc)

            
            trainer.fit(model, datamodule) #train with one seed and one dataset
            ckpt_path = trainer.checkpoint_callback.best_model_path
            best_model = MixtureLightningModule.load_from_checkpoint(ckpt_path)
            best_model.eval().to(config.device)
            test_loader = datamodule.test_dataloader()

            
            pces_marg, pces_prerank, nlls, energies, mses = [], [], [], [], []

            for x, y, _ in test_loader:
                x = x.to(config.device)
                y = y.to(config.device)
                dist = best_model.predict(x)

                prerank_pce, _ = pce(dist, y, n_samples=best_model.hparams.es_num_samples, prerank=best_model.hparams.prerank)
                marg_pce, _ = pce(dist, y, n_samples=best_model.hparams.es_num_samples, prerank="marginal")

                energy_val = multivariate_energy_score(dist, y, n_samples=best_model.hparams.es_num_samples).mean() #will be the same for all preranks
                nll_value = -dist.log_prob(y).mean() #will be the same for all preranks since it doesn't depend on prerank at all
                mse_val = mse(dist, y)

                pces_marg.append(marg_pce.mean().item())
                pces_prerank.append(prerank_pce.mean().item())
                nlls.append(nll_value.item())
                energies.append(energy_val.item())
                mses.append(mse_val.item())

            avg_pce_marg = np.mean(pces_marg)
            avg_pce_prerank = np.mean(pces_prerank)
            avg_nll = np.mean(nlls)
            avg_energy = np.mean(energies)
            avg_mse = np.mean(mses)

            df = pd.concat([df, pd.DataFrame([{
                "data_name": data_name,
                "seed": seed,
                "prerank": prerank,
                "pce_marg": avg_pce_marg,
                "pce_prerank": avg_pce_prerank,
                "nll": avg_nll,
                "energy": avg_energy,
                "mse": avg_mse
            }])], ignore_index=True)

            df.to_csv(results_path, index=False)