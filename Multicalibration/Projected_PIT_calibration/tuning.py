from moc.configs.config import get_config
from moc.utils.run_config import RunConfig
# from moc.models.mqf2.lightning_module import MQF2LightningModule
# from moc.models.mixture.mixture_model2 import MixtureLightningModule
from moc.models.gaussian.gaussian import GaussianLightningModule
from moc.models.trainers.lightning_trainer import get_lightning_trainer
from moc.datamodules.real_datamodule import RealDataModule
from moc.metrics.distribution_metrics import multivariate_energy_score, pce
import numpy as np
import torch
import pickle

config = get_config()
config.device = 'cuda'
M = 100
alphas = torch.linspace(0, 1, M, device=config.device)
# rc = RunConfig(config, 'mulan', 'rf2')
#rc = RunConfig(config,'feldman', 'bio')
rc = RunConfig(config,'camehl', 'households')
#rc = RunConfig(config,'del_barrio', 'ansur2')
datamodule = RealDataModule(rc, num_workers = 8)
p, q = datamodule.input_dim, datamodule.output_dim

lambdas = np.linspace(0,10,20)
pce_energy_pairs = {}
for l in lambdas:
    print(f"working on lambda {l:.2f}")
    model = GaussianLightningModule(p, q, lambda_reg = l, reg_type = 'pce-kde', prerank = 'dependency')
    trainer = get_lightning_trainer(rc)
    trainer.fit(model, datamodule)
    model.to(config.device)
    model.eval()
    pces, energies = 0.0, 0.0
    with torch.no_grad():
        for x, y in datamodule.val_dataloader():
            x = x.to(config.device)
            y = y.to(config.device)
            dist = model.predict(x)  # Ensure `dist` is on the same device as `x` and `y`
            
            # Calculate metrics (both return floats)
            pce_score, _ = pce(dist, y, n_samples = 20, mode = 'average', 
                               prerank = 'dependency', setup = 'real') 
            energy_score = multivariate_energy_score(dist, y) 
            pces += pce_score
            energies += energy_score

    # Average over batches
    pces /= len(datamodule.val_dataloader())
    energies /= len(datamodule.val_dataloader())
    pce_energy_pairs[l] = (pces, energies.item())

with open('pkl-files/tuning_gaussNLL_households_pce_dep.pkl', 'wb') as f:
    pickle.dump(pce_energy_pairs, f)
