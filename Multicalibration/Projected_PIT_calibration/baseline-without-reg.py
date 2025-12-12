import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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

datasets = [['camehl', 'households'], ['cevid', 'air'], ['cevid', 'births1'],
            ['cevid', 'births2'], ['cevid', 'wage'], ['mulan', 'scm20d'], ['mulan', 'scm1d'],
            ['mulan', 'wq'], ['mulan', 'scpf'], ['feldman', 'meps_21'], ['feldman', 'meps_19'],
            ['feldman', 'meps_20'], ['feldman', 'house'], ['feldman', 'bio'], ['feldman', 'blog_data'],
            ['del_barrio', 'calcofi'], ['del_barrio', 'ansur2'], ['wang', 'taxi']]

config = get_config()
config.device = 'cuda'

test_preranks = ['marginal', 'mean', 'variance', 'dependency', 'pca', 'density', 'cdf']
seeds = [0, 42, 866, 12, 4]

results_path = "csv-files/metrics-from-model-without-reg.csv"
if os.path.exists(results_path):
    df = pd.read_csv(results_path)
else:
    column_names = ["data_name", "seed", "nll", "energy"] + [f"pce_{prerank}" for prerank in test_preranks]
    df = pd.DataFrame(columns=column_names)

for data_group, data_name in datasets:
    for seed in seeds:
        trained = (
            (df['seed'] == seed) &
            (df['data_name'] == data_name)
        ).any()

        if trained:
            print(f"Skipping: {data_group}/{data_name} | seed={seed}")
            continue

        print(f"Training: {data_group}/{data_name} | seed={seed}")
        hparams = {
            'model': 'mixture',
            'seed': seed,
            'prerank': 'none',
            'lambda': 0.0,
        }

        rc = RunConfig(config, data_group, data_name, hparams=hparams, seed = seed)
        datamodule = RealDataModule(rc, num_workers=8, seed = seed)
        p, q = datamodule.input_dim, datamodule.output_dim
        model = MixtureLightningModule(p, q)

        trainer, wandb_logger = get_lightning_trainer(rc)

        trainer.fit(model, datamodule) #train with one seed and one dataset
        wandb_logger.experiment.finish()
        ckpt_path = trainer.checkpoint_callback.best_model_path
        best_model = MixtureLightningModule.load_from_checkpoint(ckpt_path)
        best_model.eval().to(config.device)
        test_loader = datamodule.test_dataloader()

        data_to_store = {}
        data_to_store["data_name"] = data_name
        data_to_store["seed"] = seed
        data_to_store["nll"] = []
        data_to_store["energy"] = []
        data_to_store["mse"] = []

        for prerank in test_preranks:
            data_to_store[f"pce_{prerank}"] = []

        for x, y, _ in test_loader:
            x = x.to(config.device)
            y = y.to(config.device)
            dist = best_model.predict(x)
            
            for prerank in test_preranks:
                pce_prerank, _ = pce(dist, y, n_samples=best_model.hparams.es_num_samples, prerank=prerank)
                data_to_store[f"pce_{prerank}"].append(pce_prerank.mean().item())

            energy_val = multivariate_energy_score(dist, y, n_samples=best_model.hparams.es_num_samples).mean()
            data_to_store["energy"].append(energy_val.item())
            nll_value = -dist.log_prob(y).mean() #will be the same for all preranks since it doesn't depend on prerank at all
            data_to_store["nll"].append(nll_value.item())
            mse_val = mse(dist, y)
            data_to_store["mse"].append(mse_val.item())

        for prerank in test_preranks:
            data_to_store[f"pce_{prerank}"] = np.mean(data_to_store[f"pce_{prerank}"])
        data_to_store["nll"] = np.mean(data_to_store["nll"])
        data_to_store["energy"] = np.mean(data_to_store["energy"])
        data_to_store["mse"] = np.mean(data_to_store["mse"])

        df = pd.concat([df, pd.DataFrame([data_to_store])], ignore_index=True)

        df.to_csv(results_path, index=False)