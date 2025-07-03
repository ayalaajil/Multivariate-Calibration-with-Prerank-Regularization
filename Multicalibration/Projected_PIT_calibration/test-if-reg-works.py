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

wandb.login(key="9d338bfb8d6dd9ab97384ee89b11f332ae3e12b8") #ELNURA'S KEY

torch.set_printoptions(precision=3, sci_mode=False, threshold=float('inf'), edgeitems=40, linewidth=200)

config = get_config()
config.device = 'cuda'
data_group, data_name = 'mulan', 'sf2'

preranks = ['variance', 'dependency', 'pca', 'density']
for prerank in preranks:
    hparams = {
        'model': 'mixture',
        'prerank': prerank,
        'lambda': 1.0,
    }
    rc = RunConfig(config, data_group, data_name, hparams = hparams)
    datamodule = RealDataModule(rc, num_workers=8)
    p, q = datamodule.input_dim, datamodule.output_dim #268,16

    model = MixtureLightningModule(p, q, lambda_reg = hparams['lambda'], reg_type = 'pce-kde', prerank = hparams['prerank'])
    trainer = get_lightning_trainer(rc)
    trainer.fit(model, datamodule)
    wandb.finish()