import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from moc.configs.config import get_config
from moc.utils.run_config import RunConfig
from moc.models.mixture.mixture_model2 import MixtureLightningModule
from moc.models.trainers.lightning_trainer import get_lightning_trainer
from moc.recalibrators.hdr_recalibrator import HDRRecalibrator
from moc.datamodules.real_datamodule import RealDataModule
from moc.metrics.distribution_metrics import pce, multivariate_energy_score
import numpy as np
import pandas as pd
import torch

torch.set_printoptions(precision=3, sci_mode=False, threshold=float('inf'), edgeitems=40, linewidth=200)


datasets = [
            ['camehl', 'households'], ['cevid', 'air'],
            ['cevid', 'births1'],
            ['cevid', 'births2'],
            ['cevid', 'wage'],
            ['mulan', 'scm20d'],
            ['mulan', 'scm1d'], ['mulan', 'wq'],
            ['mulan', 'scpf'], ['feldman', 'meps_21'], ['feldman', 'meps_19'],
            ['feldman', 'meps_20'], ['feldman', 'house'], ['feldman', 'bio'], ['feldman', 'blog_data'],
            ['del_barrio', 'calcofi'],
            ['del_barrio', 'ansur2'], ['wang', 'taxi']
            ]


config = get_config()
config.device = 'cuda'

preranks = ['marginal', 'mean', 'variance', 'dependency', 'pca', 'density', 'cdf']

results_path = "metrics-hdr.csv"
if os.path.exists(results_path):
    df = pd.read_csv(results_path)
else:
    df = pd.DataFrame(columns=["data_name", "prerank", "pce", "energy", "nll"])

for data_group, data_name in datasets:
    print(f"Training: {data_group}/{data_name} | seed=0")
    hparams = {
        'model': 'mixture',
        'seed': 0,
        'lambda': 0.0,
        'prerank': 'none'
    }

    rc = RunConfig(config, data_group, data_name, hparams=hparams)
    datamodule = RealDataModule(rc, num_workers=8)
    p, q = datamodule.input_dim, datamodule.output_dim
    model = MixtureLightningModule(p, q)
    trainer, wandb_logger = get_lightning_trainer(rc)

    trainer.fit(model, datamodule) #train with one seed and one dataset

    wandb_logger.experiment.finish()

    ckpt_path = trainer.checkpoint_callback.best_model_path
    best_model = MixtureLightningModule.load_from_checkpoint(ckpt_path)
    best_model.eval().to(config.device)

    #HDR recalibration
    calib_loader = datamodule.calib_dataloader()
    recalib = HDRRecalibrator(best_model, calib_loader)

    test_loader = datamodule.test_dataloader()

    for prerank in preranks:
        hdr_pces, hdr_energies, hdr_nlls = [], [], []

        for x, y, _ in test_loader:
            x = x.to(config.device)
            y = y.to(config.device)
            hdr_dist = recalib.predict(x) #the problem is here

            pce_val, _ = pce(hdr_dist, y, n_samples=best_model.hparams.es_num_samples, prerank=prerank)
            energy_val = multivariate_energy_score(hdr_dist, y, n_samples=best_model.hparams.es_num_samples).mean() #will be the same for all preranks

            nll_value = -hdr_dist.log_prob(y).mean() #will be the same for all preranks since it doesn't depend on prerank at all
            hdr_pces.append(pce_val.mean().item())
            hdr_nlls.append(nll_value.item())
            hdr_energies.append(energy_val.item())

        avg_pce = np.mean(hdr_pces)
        avg_nll = np.mean(hdr_nlls)
        avg_energy = np.mean(hdr_energies)

        df = pd.concat([df, pd.DataFrame([{
            "data_name": data_name,
            "prerank": prerank,
            "pce": avg_pce,
            "energy": avg_energy,
            "nll": avg_nll
        }])], ignore_index=True)

        df.to_csv(results_path, index=False)