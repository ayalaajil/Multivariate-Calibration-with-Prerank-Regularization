from moc.configs.config import get_config
from moc.utils.run_config import RunConfig
# from moc.models.mqf2.lightning_module import MQF2LightningModule
from moc.models.mixture.mixture_model import MixtureLightningModule
from moc.models.gaussian.gaussian import GaussianLightningModule
from moc.models.trainers.lightning_trainer import get_lightning_trainer
from moc.datamodules.real_datamodule import RealDataModule
import numpy as np
import matplotlib.pyplot as plt
from moc.metrics.distribution_metrics import pce
import torch
import pickle
import pandas as pd


config = get_config()
config.device = 'cuda'
seeds = [0, 42, 866, 12, 4]

def pce_per_dataset(config, data_group, data_name, seeds, prerank_list):
    rc = RunConfig(config, data_group, data_name)
    pces_by_prerank = {prerank: [] for prerank in prerank_list}

    for seed in seeds:
        datamodule = RealDataModule(rc, seed=seed)
        p, q = datamodule.input_dim, datamodule.output_dim

        #model = GaussianLightningModule(p, q)
        model = MixtureLightningModule(p, q, 5)
        trainer = get_lightning_trainer(rc)
        trainer.fit(model, datamodule)
        model.to(config.device)
        model.eval()

        with torch.no_grad():
            for prerank in prerank_list:
                pces = []
                for x, y in datamodule.val_dataloader():
                    x, y = x.to(config.device), y.to(config.device)
                    dist = model.predict(x)
                    pce_values, _ = pce(dist, y, n_samples=20, mode='all', prerank=prerank, setup='real')
                    pces.append(pce_values)
                mean_pce_for_seed = np.mean(np.array(pces), axis=0)
                pces_by_prerank[prerank].append(mean_pce_for_seed)

    # On calcule moyenne et variance sur les seeds
    stats_by_prerank = {}
    for prerank, pce_list in pces_by_prerank.items():
        pce_array = np.array(pce_list)
        stats_by_prerank[prerank] =  float(pce_array.mean())
           # "var": pce_array.var()
        

    return stats_by_prerank

# Liste des datasets
dataset_names = [
    ['camehl', 'households'],
     ['mulan', 'scm20d'],
     ['mulan', 'rf2'],
     ['mulan', 'rf1'],
     ['mulan', 'scm1d'],
     ['feldman', 'meps_21'],
     ['feldman', 'meps_19'],
     ['feldman', 'meps_20'],
     ['feldman', 'house'],
     ['feldman', 'bio'],
     ['feldman', 'blog_data'],
    #['del_barrio', 'calcofi'],
    ['wang', 'taxi']
]

seeds = [0, 42, 866, 12, 4]
prerank_map = {
    "marginal": "Marginal",
    "mean": "Loc.",
    "variance": "Scale",
    "dependency": "Dep.",
    "pca": "PCA",
    "density": "HDR",
}
preranks = list(prerank_map.keys())

pce_datasets = {label: [] for label in ["Datasets"] + list(prerank_map.values())}

for data_group, data_name in dataset_names:
    print(f"Working on dataset {data_name}")
    pces = pce_per_dataset(config, data_group, data_name, seeds, preranks)
    pce_datasets["Datasets"].append(data_name)
    for raw_key, final_label in prerank_map.items():
        pce_datasets[final_label].append(pces.get(raw_key, np.nan)) 
print(pce_datasets)

df = pd.DataFrame(pce_datasets)

df.to_csv("real_experiments_gaussian_nll.csv", index=False)