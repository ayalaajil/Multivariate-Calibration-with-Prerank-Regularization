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
import optuna


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

    cdf_values = torch.searchsorted(sample_sorted, x_proj.unsqueeze(-1), side='right') / n
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

def ensemble_PIT(samples, y):
        pca = PCA(n_components=len(y[0])) #keeping all components, 4 in this case
        pca.fit(samples.reshape(-1,len(y[0])))
        vectors = pca.components_ #4 by 4
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


def pce(model, dataset):
    total_pits = []
    for x, y in dataset:
        x = x.to(config.device)
        y = y.to(config.device)
        dist = model.predict(x)
        samples = dist.sample((100,)).permute(1, 0, 2) #256,100,4

        pit_values = ensemble_PIT(samples, y) #shape (4,256,1)
        total_pits.append(pit_values)
    total_pits = torch.cat(total_pits, dim=1)

    pits = total_pits[0].view(-1) #taking the first dimension only, shape (256,)
    pits_sorted = pits.sort()[0]
    cdf_estimates = torch.searchsorted(pits_sorted, alphas, side='right') / pits_sorted.numel() #shape (100,)
    pce = torch.mean(torch.abs(cdf_estimates - alphas)).item()
    return pce
     

config = get_config()
config.device = 'cpu'
M = 100
alphas = torch.linspace(0, 1, M, device=config.device)
dataset = ('mulan', 'rf1')
n_samples = 100
rc = RunConfig(config, dataset[0], dataset[1])
dataset_name = dataset[0]+"_"+dataset[1]
#rc = RunConfig(config,'feldman', 'bio')
#rc = RunConfig(config,'camehl', 'households')
#rc = RunConfig(config,'del_barrio', 'ansur2')
datamodule = RealDataModule(rc)
p, q = datamodule.input_dim, datamodule.output_dim
'''best_lambda = 0.1
model = MixtureLightningModule(p,q, best_lambda)
model_name = model.name
#model = MQF2LightningModule(p, q)
trainer = get_lightning_trainer(rc)
trainer.fit(model, datamodule)
best_metric = pce(model, datamodule.train_dataloader())
print(best_metric)'''


def objective(trial):
    lambda_reg = trial.suggest_float('lambda_reg', 1e-6, 50, log=True)
    model = MixtureLightningModule(p,q, lambda_reg)
    trainer = get_lightning_trainer(rc)
    trainer.fit(model, datamodule)
    score = pce(model, datamodule.val_dataloader())
    print(f"Trial {trial.number} - lambda: {lambda_reg:.1e} - PCE: {score:.4f}")
    return score

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)
best_lambda = study.best_trial.params["lambda_reg"]


'''for lambda_reg in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
    model = MixtureLightningModule(p,q, lambda_reg)
    model_name = model.name
    #model = MQF2LightningModule(p, q)
    trainer = get_lightning_trainer(rc)
    trainer.fit(model, datamodule)
    metric = pce(model, datamodule.train_dataloader())
    if metric < best_metric:
                best_metric = metric
                best_lambda = lambda_reg
                print(best_lambda)'''
model = MixtureLightningModule(p,q, best_lambda)
model_name = model.name
#model = MQF2LightningModule(p, q)
trainer = get_lightning_trainer(rc)
trainer.fit(model, datamodule)
test_batch = next(iter(datamodule.test_dataloader()))
data, y_true = test_batch
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
