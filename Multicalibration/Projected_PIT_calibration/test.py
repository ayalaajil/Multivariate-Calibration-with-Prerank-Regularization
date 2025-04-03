from moc.configs.config import get_config
from moc.utils.run_config import RunConfig
from moc.models.mqf2.lightning_module import MQF2LightningModule
from moc.models.mixture.mixture_model2 import MixtureLightningModule
from moc.models.trainers.lightning_trainer import get_lightning_trainer
from moc.datamodules.real_datamodule import RealDataModule
from moc.metrics.metrics_computer import compute_coverage_indicator, compute_log_region_size
from moc.conformal.conformalizers import L_CP, HDR_H
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import wandb


wandb.init(project="conformal_regression", name="mixture_model")

# Function to project samples onto the vectors
def proj_for(x_values, u, sample):
    u = torch.tensor(u, dtype=torch.float32)
    sample = torch.stack([torch.tensor(s) for s in sample])  
    sample_proj = torch.matmul(sample, u)
        
    if x_values is None:
        x_values = sample    

    x_values = torch.tensor(x_values)
    x_proj = torch.matmul(x_values, u)

    # Sort each row (each sub-tensor) along the last dimension
    sample_proj_2d = sample_proj.squeeze(-1)
    sample_sorted = torch.sort(sample_proj_2d, dim=-1)[0]
    n = len(sample[0])
    print(n)
    print(len(sample))
    print("len(x_values)")
    print(len(x_values))

    cdf_values = torch.searchsorted(sample_sorted, x_proj.unsqueeze(-1), side='right') / n
    print("len(cdf_values)")
    print(len(cdf_values))
    return x_proj, cdf_values

def calculate_pit(values, u, sample):
        #values = torch.as_tensor(values, dtype=sample.dtype, device=sample.device)
        return proj_for(values, u, sample)[1]


# Function to display PIT plots
def plot_pit(pit_values, dimension, n_samples):

    cols = 3 
    rows = int(np.ceil(dimension / cols)) 

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4)) 

    axs = np.array(axs).reshape(rows, cols)

    for i in range(dimension):
        row, col = divmod(i, cols)  # Find the position in the grid
        ax = axs[row, col]  # Retrieve the corresponding axis
        
        ax.hist(pit_values[i], bins=10, density=True, alpha=0.6, color='g')
        title = f"PIT - Direction along component ${i+1}$" 
        ax.set_title(title)
        ax.set_xlabel("Projected PIT value")
        ax.set_ylabel("Density")

    # Remove empty subplots if `dimension` is not a multiple of 3
    for i in range(dimension, rows * cols):
        fig.delaxes(axs.flatten()[i])

    plt.tight_layout() 
    filename = f"PIT_proj_PCA_{n_samples}_{dataset_name}_{model_name}.png" 
    plt.savefig(filename)
    plt.show()



config = get_config()
config.device = 'cpu'
dataset = ('mulan', 'sf2')
n_samples = 100
rc = RunConfig(config, dataset[0], dataset[1])
dataset_name = dataset[0]+"_"+dataset[1]
#rc = RunConfig(config,'feldman', 'bio')
#rc = RunConfig(config,'camehl', 'households')
#rc = RunConfig(config,'del_barrio', 'ansur2')
datamodule = RealDataModule(rc)
p, q = datamodule.input_dim, datamodule.output_dim 
model = MixtureLightningModule(p,q)
model_name = model.name
#model = MQF2LightningModule(p, q)
trainer = get_lightning_trainer(rc)
trainer.fit(model, datamodule)
test_batch = next(iter(datamodule.test_dataloader()))
data, y_true = test_batch
print("len(data)")
print(len(datamodule.get_data())) 
print(len(datamodule.get_data()[0]))
print(len(data[0]))
print("len(y_true[0])")
print(len(y_true[0]))
if isinstance(data, np.ndarray):
    data = torch.tensor(data, dtype=torch.float32)
#y_pred = model.predict(data).sample((30,)) 
y_pred = [model.predict(data[i].unsqueeze(0)).sample((n_samples,)) for i in range(len(data))]
y_pred = torch.stack(y_pred)  
d = len(y_true[0])
pca = PCA(n_components= d)
pca.fit(y_pred.reshape(-1,d))
vectors = pca.components_
'''d = len(y_true[0])
pca = PCA(n_components= d)
pca.fit(y_true)
vectors = pca.components_'''

# Compute PIT values
pit_values = [calculate_pit(y_true, vectors[i], y_pred) for i in range(d)]

# Plot PIT values
plot_pit(pit_values, d, n_samples)

'''alpha = 0.1
conformalizer = HDR_H(datamodule.calib_dataloader(), model)
test_batch = next(iter(datamodule.test_dataloader()))
x, y = test_batch
coverage = compute_coverage_indicator(conformalizer, alpha, x, y)
volume = compute_log_region_size(conformalizer, model, alpha, x, n_samples=100)'''
wandb.finish()
