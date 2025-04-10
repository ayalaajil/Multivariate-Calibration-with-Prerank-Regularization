import logging

import torch
from lightning.pytorch import LightningModule
from torch.distributions import MixtureSameFamily, Categorical, MultivariateNormal
import math
from sklearn.decomposition import PCA
import wandb
import numpy as np
import sys
from pathlib import Path
from moc.metrics.distribution_metrics import energy_score

reg_path = Path(__file__).resolve().parents[2]
sys.path.append(str(reg_path))
from regularizers.reguls import rqr_regularization, truncation_regularization
log = logging.getLogger('moc')

class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, num_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(
                input_dim if i == 0 else hidden_size,
                hidden_size if i != num_layers - 1 else output_dim,
            )
            for i in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)


def adjust_scale_tril(scale_tril):
    """
    Sometimes the covariance matrix is not positive definite due to numerical instability.
    This adjusts the covariance matrix to be positive definite.
    """
    cov_matrix = scale_tril @ scale_tril.transpose(-1, -2)
    # Perform eigenvalue decomposition on the covariance matrix
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    failed = False
    for epsilon in torch.logspace(-6, 10, 17):
        epsilon = epsilon.item()
        try:
            adjusted_eigenvalues = torch.clamp(eigenvalues, min=epsilon)
            # Reconstruct the covariance matrix
            adjusted_cov_matrix = eigenvectors @ torch.diag_embed(adjusted_eigenvalues) @ eigenvectors.transpose(-1, -2)
            # Symmetrize the matrix
            adjusted_cov_matrix = (adjusted_cov_matrix + adjusted_cov_matrix.transpose(-1, -2)) / 2
            # Verify positive definiteness
            new_scale_tril = torch.linalg.cholesky(adjusted_cov_matrix)
        except torch.linalg.LinAlgError as e:
            failed = True
        else:
            if failed:
                log.info(f'Covariance matrix adjusted with epsilon={epsilon}.')
            return new_scale_tril
    raise ValueError('Failed to adjust the covariance matrix to be positive definite.')


def create_multivariate_normal_mixture(logits, locs, scale_trils):
    return MixtureSameFamily(
        Categorical(logits=logits),
        MultivariateNormal(loc=locs, scale_tril=adjust_scale_tril(scale_trils)),
    )


def extract_multivariate_normal_mixture_parameters(params, mixture_size, output_dim):
    logits, locs, scale_trils_raw = params
    locs = locs.reshape(-1, mixture_size, output_dim)
    scale_trils = torch.zeros(logits.shape[:-1] + (mixture_size, output_dim, output_dim), device=logits.device)
    tril_indices = torch.tril_indices(row=output_dim, col=output_dim, offset=0, device=logits.device)
    scale_trils_raw = scale_trils_raw.reshape(-1, mixture_size, output_dim * (output_dim + 1) // 2)
    scale_trils[..., tril_indices[0], tril_indices[1]] = scale_trils_raw
    diag_indices = torch.arange(output_dim, device=logits.device)
    scale_trils[..., diag_indices, diag_indices] = torch.nn.functional.softplus(
        scale_trils[..., diag_indices, diag_indices]
    ) + 1e-3
    return logits, locs, scale_trils





class MixtureLightningModule(LightningModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        lambda_reg:float,
        hidden_size: int = 100,
        num_layers: int = 3,
        loss: str = 'nll',
        reg_type: str = 'rqr',
        mixture_size: int = 5,
        es_num_samples: int = 50,
        lr=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        wandb.init(project="conformal_regression")
        # wandb.config.update(self.hparams)


        output_dim = output_dim
        mixture_size = self.hparams.mixture_size
        self.lambda_reg = lambda_reg
        self.reg_type = reg_type
        self.output_shape = (
            mixture_size,
            mixture_size * output_dim,
            mixture_size * output_dim * (output_dim + 1) // 2,
        )
        self.model = MLP(
            input_dim=input_dim,
            output_dim=torch.sum(torch.tensor(self.output_shape)),
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.num_layers,
        )
        self.name = "GaussianMixture"
        self.validation_step_outputs = []
        self.train_step_outputs = []

    def forward(self, x):
        out = self.model(x) #(batch_size, 75)
        out = out.split(self.output_shape, dim=-1) #(batch_size, 5), (batch_size, 20), (batch_size, 50)
        params = extract_multivariate_normal_mixture_parameters(
            out,
            mixture_size=self.hparams.mixture_size,
            output_dim=self.trainer.datamodule.output_dim,
        )
        return create_multivariate_normal_mixture(*params)
    
    def predict(self, x):
        return self(x)
    # def project_along_pca(self, samples, pca_vectors):
    #     return torch.matmul(samples, torch.tensor(pca_vectors, dtype=samples.dtype, device=samples.device))

    

    # def proj_emp_cdf(self, x_values, u, sample):
    #     sample_proj = torch.matmul(sample, u)
    #     if x_values is None:
    #         x_values = sample
    #     x_proj = torch.matmul(x_values, u)
    #     sample_sorted = torch.sort(sample_proj)[0] #256,100 sort across the columns
    #     n = len(sample[0]) #100
    #     cdf_values = torch.searchsorted(sample_sorted, x_proj.unsqueeze(-1), side='right') / n #256,1
    #     return x_proj, cdf_values #returns projected predictions and cdf values which represent where in the sorted projected samples the projected ground truth lies

    # # Fonction pour calculer les PIT
    # def calculate_pit(self,values, u, sample):
    #     u = torch.as_tensor(u, dtype=sample.dtype, device=sample.device)
    #     pits = self.proj_emp_cdf(values, u, sample)[1]
    #     return pits
    
    # def sample(self, dist, num_samples=1000):
    #     return dist.sample((num_samples,)).permute(1, 0, 2)

    # def ensemble_PIT(self, y_hat, y):
    #     pca = PCA(n_components=len(y[0])) #keeping all components, 4 in this case
    #     pca.fit(y_hat.reshape(-1,len(y[0])))
    #     vectors = pca.components_ #4 by 4
    #     return torch.stack([self.calculate_pit(y, vectors[i], y_hat) for i in range(len(vectors))]), vectors
    
    


    def compute_loss(self, dist, y): #with rqr
        """
        Compute the loss with the added regularization term based on PIT values.
        """
        if self.reg_type == 'rqr':
            reg_term = rqr_regularization(dist, y)
        elif self.reg_type == 'truncation':
            reg_term = truncation_regularization(dist, y)
        else:
            reg_term = 0.0
        # rqr = self.rqr_regularization(pit_values, 100, N)
        #rqr = max(rqr, 0.0)
        
        if self.hparams.loss == 'nll':
            loss_term = -dist.log_prob(y).mean()
            reg_loss = loss_term + (self.lambda_reg * reg_term)
            return reg_loss, loss_term, self.lambda_reg*reg_term, reg_term
        elif self.hparams.loss == 'es':
            loss_term = energy_score(dist, y, n_samples=self.hparams.es_num_samples)
            reg_loss = loss_term + (self.lambda_reg * reg_term)
            return reg_loss, loss_term, self.lambda_reg*reg_term, reg_term
        else:
            raise ValueError(f'Invalid loss: {self.hparams.loss}')
    
    
    # def compute_quantiles(self, y_hat, y, quantile_levels):
    #     pit_vals, pca_vectors = self.ensemble_PIT(y_hat, y)
    #     '''y_hat_proj = torch.stack([self.proj_emp_cdf(y_hat, pca[i], sample)[1] for i in range(len(pca))])
    #     y_proj = torch.stack([self.proj_emp_cdf(y, pca[i], sample)[1] for i in range(len(pca))])

    #     above = y_hat_proj>= quantile_levels
    #     print(above)
    #     above = above.squeeze(-1)
    #     has_true = above.any(dim=1)
    #     first_true_indices = torch.where(
    #     has_true,
    #     above.float().argmax(dim=1),
    #     torch.tensor(-1) 
    #     )
    #     print(first_true_indices)
    #     batch_size, num_points = sorted_pit.shape[:2]
    #     safe_indices = torch.where(first_true_indices == -1,
    #                                torch.tensor(num_points - 1, device=first_true_indices.device),
    #                                first_true_indices)
    #     # Create batch indices for gather
    #     batch_indices = torch.arange(batch_size, device=first_true_indices.device)

    #     # Retrieve the quantiles in terms of PIT values
    #     quantiles = sorted_pit[batch_indices, safe_indices]

    #     return torch.stack(quantiles)  # Shape: (8,)'''
    #     quantiles = []
    #     for i in range(len(pca_vectors)):  # For each PCA component
    #         # Project the predictions along the PCA vector
    #         projections = self.project_along_pca(y_hat, pca_vectors[i])
    #         #print(len(projections))#256
    #         #print(len(projections[0]))#100

    #         # Sort the projections along the last dimension (samples)
    #         sorted_proj = torch.sort(projections, dim=-1)[0]
    #         #print(sorted_proj[0])

    #         # Compute the index of the quantile for each sample
    #         n = sorted_proj.shape[-1]  # Number of samples i.e. 100
    #         quantile_index = int(quantile_levels * (n - 1)) #9

    #         # Select the quantile value based on the sorted projections
    #         quantile_values = sorted_proj[..., quantile_index].unsqueeze(-1) #256 values

    #         quantiles.append(quantile_values)
    #     #print(len(quantiles)) 8 (each dimension)

    #     # Stack the quantiles along the component dimension
    #     #print(len(torch.cat(quantiles, dim=0)) ) 2048=256x8
    #     return torch.stack(quantiles, dim=0)

    
    # def truncation_regularization(self, dist, y, alpha=1):
    #     """
    #     Compute the Truncation-based Calibration regularization term.

    #     Args:
    #         dist: MixtureSameFamily distribution.
    #         y: Ground truth values (batch_size, dim).
    #         alpha: Truncation threshold.

    #     Returns:
    #         Regularization value.
    #     """
    #     # Échantillonner les prévisions
    #     num_samples = 100
    #     y_hat = self.sample(dist, num_samples=num_samples)  # Shape: (num_samples, batch_size, dim)
        
    #     # Calculer les PIT
    #     pit_values, pca_vectors = self.ensemble_PIT(y_hat, y) # Shape: (dim, batch_size, num_samples)
    #     #print(len(pit_values)) #3
    #     #print(len(pit_values[0]))#256
        
    #     # Estimation de la CDF du PIT à alpha
    #     F_hat_alpha = (pit_values < alpha).float().mean(dim=-1)  # Moyenne sur les échantillons
    #     #print(len(F_hat_alpha))#3
    #     #print(len(F_hat_alpha[0]))#256

        
    #     # Définition de ρ(x, y) = (y - x) 1(x < y)
    #     def rho(x, y):
    #         return (y - x) * (x < y).float()
        
    #     reg_term = 0 
    #     dim = len(y[0])
    #     # Calcul de la régularisation
    #     if F_hat_alpha.mean() < alpha:
    #         for i in range(dim):
    #             reg_term += rho(self.compute_quantiles(y_hat, y, torch.tensor(alpha, device=y.device))[i], self.project_along_pca(y, pca_vectors[i])).mean()

    #     else:
    #         for i in range(dim):
    #             # print(len(rho(self.project_along_pca(y, pca_vectors[i]), self.compute_quantiles(y_hat, y,torch.tensor(alpha, device=y.device))))) 8
    #             #print(len(self.compute_quantiles(y_hat, y,torch.tensor(alpha, device=y.device))[i]))#256
    #             #print(len(self.compute_quantiles(y_hat, y,torch.tensor(alpha, device=y.device)))) 8
    #             #print(len(self.project_along_pca(y, pca_vectors[i]))) 256
    #             reg_term += rho(self.project_along_pca(y, pca_vectors[i]), self.compute_quantiles(y_hat, y,torch.tensor(alpha, device=y.device))[i]).mean()
    #         reg_term/=dim
        
    #     return reg_term


    # def compute_loss(self, dist, y,  lamda):
    #     # Liste des valeurs alpha pour la régularisation
    #     alphas = np.arange(0.1, 1.1, 0.1) 
        
    #     # Calcul de la régularisation pour chaque alpha et somme des termes
    #     trunc_reg_sum = sum(self.truncation_regularization(dist, y, alpha=alpha) for alpha in alphas)
        
    #     # Retour de la perte totale avec la régularisation ajustée
    #     if self.hparams.loss == 'nll':
    #         loss_term = -dist.log_prob(y).mean()
    #         reg_loss = loss_term + (lamda * trunc_reg_sum)
    #         return reg_loss, loss_term, lamda*trunc_reg_sum, trunc_reg_sum
    #     elif self.hparams.loss == 'es':
    #         loss_term = energy_score(dist, y, n_samples=self.hparams.es_num_samples)
    #         reg_loss = loss_term + (lamda * trunc_reg_sum)
    #         return reg_loss, loss_term, lamda*trunc_reg_sum, trunc_reg_sum
        
    def smooth_indicator(self, a, b, tau=10.0):
        return torch.sigmoid(tau * (b - a))

    def kde_cdf_smoothed(self, z_vals, alphas, tau):
        z_vals = z_vals.unsqueeze(0)     # (1, N)
        alphas = alphas.unsqueeze(1)     # (M, 1)
        smoothed_indicators = torch.sigmoid(tau * (alphas - z_vals))  # (M, N)
        return smoothed_indicators.mean(dim=1)  # Moyenne sur les N, résultat (M,)
    
    tau = 0.5
    p = 1

    def compute_loss_kde(self, dist, y, lamda):
        sample_pred = self.sample(dist) #returns 256,100,4
        pit_values = self.ensemble_PIT(sample_pred, y)[0] #4,256,1
        phi_kde =self.kde_cdf_smoothed(pit_values, alphas, tau=tau)  # (M,)
        reg_kde = (torch.abs(alphas - phi_kde) ** p).mean()
        if self.hparams.loss == 'nll':
            loss_term = -dist.log_prob(y).mean()
            reg_loss = loss_term + (lamda * reg_kde)
            return reg_loss, loss_term, lamda*reg_kde, reg_kde
        elif self.hparams.loss == 'es':
            loss_term = energy_score(dist, y, n_samples=self.hparams.es_num_samples)
            reg_loss = loss_term + (lamda * reg_kde)
            return reg_loss, loss_term, lamda*reg_kde, reg_kde

    '''def compute_loss(self, dist, y):
        if self.hparams.loss == 'nll':
            self.validation_step_outputs.append(-dist.log_prob(y).mean()) 
            return -dist.log_prob(y).mean()
        elif self.hparams.loss == 'es':
            self.validation_step_outputs.append(energy_score(dist, y, n_samples=self.hparams.es_num_samples))
            return energy_score(dist, y, n_samples=self.hparams.es_num_samples)
        else:
            raise ValueError(f'Invalid loss: {self.hparams.loss}')'''

    def step(self, batch):
        x, y = batch
        dist = self(x) #256 distributions in 4D
        reg_loss, loss, lamda_rqr, rqr = self.compute_loss(dist, y, self.lambda_reg)
        return reg_loss, loss, lamda_rqr, rqr

    def training_step(self, batch, batch_idx):
        reg_loss, loss, lamda_rqr, rqr = self.step(batch)
        self.train_step_outputs.append(reg_loss)
        '''wandb.log({"train_reg_loss": reg_loss.item(), "train_loss": loss.item(),
                   "train_lambda_rqr":lamda_rqr, "train_rqr": rqr})'''
        return reg_loss
    

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_step_outputs).mean()  # Moyenne sur tous les batches
        self.log("train_loss", avg_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.train_step_outputs.clear()  # Nettoyer pour la prochaine epoch
        wandb.log({"train/loss_epoch": avg_loss.item()})
        

    def validation_step(self, batch, batch_idx):
        reg_loss, loss, lamda_rqr, rqr = self.step(batch)
        self.validation_step_outputs.append(reg_loss)
        '''wandb.log({"val_reg_loss": reg_loss.item(), "val_loss": loss.item(),
                   "val_lambda_rqr":lamda_rqr, "val_rqr": rqr})'''
        return reg_loss
    

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs).mean()  # Moyenne sur tous les batches
        self.log("val_loss", avg_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.validation_step_outputs.clear()  # Nettoyer pour la prochaine epoch
        wandb.log({"val/loss_epoch": avg_loss.item()})

    @classmethod
    def output_type(cls):
        return 'distribution'
