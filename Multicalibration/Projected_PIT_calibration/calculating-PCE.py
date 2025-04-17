from moc.configs.config import get_config
from moc.utils.run_config import RunConfig
# from moc.models.mqf2.lightning_module import MQF2LightningModule
from moc.models.mixture.mixture_model import MixtureLightningModule
from moc.models.trainers.lightning_trainer import get_lightning_trainer
from moc.datamodules.real_datamodule import RealDataModule
from moc.metrics.metrics_computer import compute_coverage_indicator, compute_log_region_size
from moc.conformal.conformalizers import L_CP, HDR_H
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import pickle
import os
# import wandb

config = get_config()
config.device = 'cpu'
M = 100
alphas = torch.linspace(0, 1, M, device=config.device)

def ensemble_PIT(samples, y, components = 3):
        pca = PCA(n_components = components) #keeping 3 components by default
        pca.fit(samples.reshape(-1,len(y[0])))
        vectors = pca.components_ 
        pits = []
        for i in range(len(vectors)):
            u = torch.as_tensor(vectors[i], dtype=samples.dtype, device=samples.device)
            sample_proj = torch.matmul(samples, u) # 256,100,1
            y_proj = torch.matmul(y, u) #256,1
            sample_sorted = torch.sort(sample_proj)[0] #256,100 sort across the columns
            n = len(samples[0]) #100
            cdf_values = torch.searchsorted(sample_sorted, y_proj.unsqueeze(-1), side='right') / n #256,1
            pits.append(cdf_values)
        return torch.stack(pits)

def pce_per_dataset(config, data_group, data_name, seeds):
    rc = RunConfig(config, data_group, data_name)
    pce_over_seeds = []
    for seed in seeds:
        datamodule = RealDataModule(rc, seed=seed)
        p, q = datamodule.input_dim, datamodule.output_dim
        model = MixtureLightningModule(p,q)
        #model = MQF2LightningModule(p, q)
        trainer = get_lightning_trainer(rc)
        trainer.fit(model, datamodule)

        total_pits = []
        for x, y in datamodule.test_dataloader():
            x = x.to(config.device)
            y = y.to(config.device)
            dist = model.predict(x)
            samples = dist.sample((100,)).permute(1, 0, 2) #256,100,4
            c = min(y.shape[1], 3)
            pit_values = ensemble_PIT(samples, y, components = c) #shape (4,256,1)
            total_pits.append(pit_values)
        total_pits = torch.cat(total_pits, dim=1) #shape (n_components, number of test data points, 1)

        pces = []
        for d in range(total_pits.shape[0]):
            pits = total_pits[d].view(-1)  # shape: (N,)
            pits_sorted = pits.sort()[0]
            cdf_estimates = torch.searchsorted(pits_sorted, alphas, side='right') / pits_sorted.numel()
            pce = torch.mean(torch.abs(cdf_estimates - alphas)).item()
            pces.append(pce)
        pce_over_seeds.append(pces)
        # pits = total_pits[0].view(-1) #taking the first dimension only, shape (256,)
        # pits_sorted = pits.sort()[0]
        # cdf_estimates = torch.searchsorted(pits_sorted, alphas, side='right') / pits_sorted.numel() #shape (100,)
        # pce = torch.mean(torch.abs(cdf_estimates - alphas)).item()
        # pce_over_seeds.append(pce)
    return pce_over_seeds

dataset_names = [
                 ['camehl', 'households'],
                #  ['mulan', 'scm20d'],
                #  ['mulan', 'rf2'],
                #  ['mulan', 'rf1'],
                #  ['mulan', 'scm1d'],
                #  ['feldman', 'meps_21'],
                #  ['feldman', 'meps_19'],
                #  ['feldman', 'meps_20'],
                #  ['feldman', 'house'],
                #  ['feldman', 'bio'],
                #  ['feldman', 'blog_data'],
                #  ['del_barrio', 'calcofi'],
                #  ['wang', 'taxi']
                 ]
# seeds = [0, 42, 866, 12, 4]
seeds = [42]
pces_across_datasets = {}

for dataset in dataset_names:
    data_group, data_name = dataset
    print(f"Working on dataset {data_name}")
    pce_over_seeds = pce_per_dataset(config, data_group, data_name, seeds)
    pces_across_datasets[f'{data_name}'] = np.array(pce_over_seeds)

    # with open("pce_calcofi_taxi.pkl", "wb") as f:
    #     pickle.dump(pces_across_datasets, f)

