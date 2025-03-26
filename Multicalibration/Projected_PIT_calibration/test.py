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

# Fonction pour projeter les échantillons sur les vecteurs
def proj_for(x_values, u, sample):
    sample_proj = torch.matmul(sample, u)
        
    if x_values is None:
        x_values = sample

    x_proj = torch.matmul(x_values, u)
    sample_sorted = torch.sort(sample_proj)[0]
    n = len(sample)
    
    cdf_values = torch.searchsorted(sample_sorted.T  , x_proj.unsqueeze(-1), side='right') / n
    return x_proj, cdf_values

def calculate_pit(values, u, sample):
        u = torch.as_tensor(u, dtype=sample.dtype, device=sample.device)
        values = torch.as_tensor(values, dtype=sample.dtype, device=sample.device)
        return proj_for(values, u, sample)[1]


#Fonction pour afficher les graphiques PIT
def plot_pit(pit_values, dimension, n_samples):

    cols = 3  # Fixe 3 graphes par ligne
    rows = int(np.ceil(dimension / cols))  # Nombre de lignes nécessaires

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))  # Ajustement dynamique de la taille

    # S'assurer que `axs` est toujours un tableau 2D pour éviter les erreurs
    axs = np.array(axs).reshape(rows, cols)

    for i in range(dimension):
        row, col = divmod(i, cols)  # Trouver la position dans la grille
        ax = axs[row, col]  # Récupérer l'axe correspondant
        
        ax.hist(pit_values[i], bins=10, density=True, alpha=0.6, color='g')
        title = f"PIT - Direction selon la composante ${i+1}$" 
        ax.set_title(title)
        ax.set_xlabel("Valeur projetée du PIT")
        ax.set_ylabel("Densité")

    # Supprimer les sous-graphiques vides si `dimension` n'est pas un multiple de 3
    for i in range(dimension, rows * cols):
        fig.delaxes(axs.flatten()[i])

    plt.tight_layout()  # Ajuster automatiquement l'affichage
    filename = f"PIT_proj_PCA_{n_samples}_VIctor.png" 
    plt.savefig(filename)
    plt.show()



config = get_config()
config.device = 'cpu'
#rc = RunConfig(config, 'mulan', 'rf2')
#rc = RunConfig(config,'feldman', 'bio')
rc = RunConfig(config,'camehl', 'households')
#rc = RunConfig(config,'del_barrio', 'ansur2')
datamodule = RealDataModule(rc)
p, q = datamodule.input_dim, datamodule.output_dim
model = MixtureLightningModule(p,q)
#model = MQF2LightningModule(p, q)
trainer = get_lightning_trainer(rc)
trainer.fit(model, datamodule)
y_true = datamodule.get_data()[1]  # ou les valeurs réelles à prédire 
data = datamodule.get_data()[0]
if isinstance(data, np.ndarray):
    data = torch.tensor(data, dtype=torch.float32)
y_pred = model.predict(data).sample((30,)) 

'''print(y_true)
print("Y_PRED")
print(y_pred)
d = len(y_pred[0][0])
pca = PCA(n_components= d)
pca.fit(y_pred.reshape(-1,d))
vectors = pca.components_'''
d = len(y_true[0])
pca = PCA(n_components= d)
pca.fit(y_true)
vectors = pca.components_

# Calcul des PIT
pit_values = [calculate_pit(y_true, vectors[i], y_pred) for i in range(d)]

# Tracé des PIT
plot_pit(pit_values, d, 30)

alpha = 0.1
conformalizer = HDR_H(datamodule.calib_dataloader(), model)
test_batch = next(iter(datamodule.test_dataloader()))
x, y = test_batch
coverage = compute_coverage_indicator(conformalizer, alpha, x, y)
volume = compute_log_region_size(conformalizer, model, alpha, x, n_samples=100)
wandb.finish()

