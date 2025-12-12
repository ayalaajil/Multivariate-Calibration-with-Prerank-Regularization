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

datasets = [['camehl', 'households'], ['cevid', 'air']
            # ['cevid', 'births1'],
            # ['cevid', 'births2'], ['cevid', 'wage'], ['mulan', 'scm20d'], ['mulan', 'scm1d'],
            # ['mulan', 'wq'], ['mulan', 'scpf'], ['feldman', 'meps_21'], ['feldman', 'meps_19'],
            # ['feldman', 'meps_20'], ['feldman', 'house'], ['feldman', 'bio'], ['feldman', 'blog_data'],
            # ['del_barrio', 'calcofi'], ['del_barrio', 'ansur2'], ['wang', 'taxi']
            ]

config = get_config()
config.device = 'cuda'

test_preranks = ['marginal', 'mean', 'variance', 'dependency']
train_preranks = ['marginal', 'mean', 'variance', 'dependency']
seeds = [0, 42, 866, 12, 4]

# results_path = "csv-files/metrics-from-model-without-reg.csv"
# if os.path.exists(results_path):
#     df = pd.read_csv(results_path)
# else:
#     column_names = ["data_name", "seed", "nll", "energy"] + [f"pce_{prerank}" for prerank in test_preranks]
#     df = pd.DataFrame(columns=column_names)

for data_group, data_name in datasets:
    for prerank in train_preranks:
        for seed in seeds:
            print(f"Skipping: {data_group}/{data_name} | prerank = {prerank} | seed={seed}")
            hparams = {
                'model': 'mixture',
                'seed': seed,
                'prerank': prerank,
                'lambda': 0.1,
            }

            rc = RunConfig(config, data_group, data_name, hparams=hparams, seed = seed)
            datamodule = RealDataModule(rc, num_workers=8, seed = seed)
            p, q = datamodule.input_dim, datamodule.output_dim
            model = MixtureLightningModule(p, q, lambda_reg=hparams['lambda'], do_reg = True, prerank=[prerank])

            trainer, _ = get_lightning_trainer(rc)

            trainer.fit(model, datamodule) #train with one seed and one dataset
            # wandb_logger.experiment.finish()
            