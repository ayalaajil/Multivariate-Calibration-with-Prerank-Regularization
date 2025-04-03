import logging

import torch
from lightning.pytorch import LightningModule
from torch.distributions import MixtureSameFamily, Categorical, MultivariateNormal
import math
from sklearn.decomposition import PCA
import wandb
import numpy as np

from moc.metrics.distribution_metrics import energy_score


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
        hidden_size: int = 100,
        num_layers: int = 3,
        loss: str = 'nll',
        mixture_size: int = 5,
        es_num_samples: int = 50,
        lr=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        wandb.config.update(self.hparams)


        output_dim = output_dim
        mixture_size = self.hparams.mixture_size
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

#--------------------------------------------------------------------------------------------------------------------------------------------
    


    # Fonction pour projeter les échantillons sur les vecteurs
    def proj_for(self, x_values, u, sample):
        sample_proj = torch.matmul(sample, u)
        
        if x_values is None:
            x_values = sample

        x_proj = torch.matmul(x_values, u)
        sample_sorted = torch.sort(sample_proj)[0]
        n = len(sample[0])
        
        cdf_values = torch.searchsorted(sample_sorted, x_proj.unsqueeze(-1), side='right') / n
        return x_proj, cdf_values

    # Fonction pour calculer les PIT
    def calculate_pit(self,values, u, sample):
        u = torch.as_tensor(u, dtype=sample.dtype, device=sample.device)
        return self.proj_for(values, u, sample)[1]
    
    def sample(self, dist, num_samples=1000):
        return dist.sample((num_samples,)).permute(1, 0, 2)

    def ensemble_PIT(self, y_hat, y):
        pca = PCA(n_components=len(y[0]))
        pca.fit(y_hat.reshape(-1,len(y[0])))
        vectors = pca.components_
        return torch.stack([self.calculate_pit(y, vectors[i], y_hat) for i in range(len(vectors))])
    
    def rqr_regularization(self, Z, k, N):
        """
        Compute the RQR regularization term based on PIT values.
        Z: Tensor of PIT values calculated on the calibration set.
        k: The window size used for the calculation.
        N: The total number of PIT values.
        """
        # Sort the PIT values if necessary (sorting might depend on the context)
        Z_sorted = torch.sort(Z, dim=1)[0]# Sort the PIT values

        # Calculate the regularization term
        rqr = 0
        for j in range(len(Z)):
            for i in range(N - k):
                term = np.absolute(torch.log(((N + 1) / k) * (Z_sorted[j][i + k] - Z_sorted[j][i])))
                rqr += term                                      
        return rqr / (len(Z)*(N - k))
    


    def compute_loss_rqr(self, dist, y): 
        """
        Compute the loss with the added regularization term based on PIT values.
        """
        # Compute PIT values
        sample_pred = self.sample(dist)
        pit_values = self.ensemble_PIT(sample_pred, y)


        # Compute RQR regularization term
        N = len(y)  # Number of samples in the PIT values
        rqr = self.rqr_regularization(pit_values, 100, N)


        if self.hparams.loss == 'nll':
            return -dist.log_prob(y).mean() + 0.5*rqr
        elif self.hparams.loss == 'es':
            return energy_score(dist, y, n_samples=self.hparams.es_num_samples) + 0.5*rqr
        else:
            raise ValueError(f'Invalid loss: {self.hparams.loss}')
    
    def compute_quantiles(self, dist, quantile_levels, num_samples=100):
        """
        Compute the empirical quantiles of a given MixtureSameFamily distribution.

        Args:
            dist (MixtureSameFamily): The mixture distribution.
            quantile_levels (list or tensor): A list of quantile levels (e.g., [0.1, 0.5, 0.9]).
            num_samples (int): Number of samples to draw for estimation.

        Returns:
            Tensor: Estimated quantiles for each level.
        """
        # Generate samples from the distribution
        samples = dist.sample((num_samples,))  # Shape: (num_samples, batch_size, dim)
        
        # Sort samples along the first dimension
        sorted_samples, _ = torch.sort(samples, dim=0)

        # Compute quantiles by selecting the corresponding indices
        quantile_indices = (torch.tensor(quantile_levels, device=samples.device) * (num_samples - 1)).long()
        quantiles = sorted_samples[quantile_indices]

        return quantiles
    
    def truncation_regularization(self, dist, y, alpha=0.1):
        """
        Compute the Truncation-based Calibration regularization term.

        Args:
            dist: MixtureSameFamily distribution.
            y: Ground truth values (batch_size, dim).
            alpha: Truncation threshold.

        Returns:
            Regularization value.
        """
        # Échantillonner les prévisions
        num_samples = 100
        y_hat = self.sample(dist, num_samples=num_samples)  # Shape: (num_samples, batch_size, dim)
        
        # Calculer les PIT
        pit_values = self.ensemble_PIT(y_hat, y)  # Shape: (dim, batch_size, num_samples)
        
        # Estimation de la CDF du PIT à alpha
        F_hat_alpha = (pit_values < alpha).float().mean(dim=-1)  # Moyenne sur les échantillons
        
        # Définition de ρ(x, y) = (y - x) 1(x < y)
        def rho(x, y):
            return (y - x) * (x < y).float()
        
        # Calcul de la régularisation
        if F_hat_alpha.mean() < alpha:
            reg_term = rho(self.compute_quantiles(dist, torch.tensor(alpha, device=y.device)), y).mean()
            print("COUUUUUUUUUUUUUUUUUUUUUCOUUUUUUUUUUUUUU  ")
            print( rho(self.compute_quantiles(dist,torch.tensor(alpha, device=y.device)), y))

        else:
            print("COUUUUUUUUUUUUUUUUUUUUUCOUUUUUUUUUUUUUU  ")
            print(rho(y, self.compute_quantiles(dist,torch.tensor(alpha, device=y.device))))
            reg_term = rho(y, self.compute_quantiles(dist,torch.tensor(alpha, device=y.device))).mean()
        
        return reg_term


    def compute_loss(self, dist, y):
        # Calcul du base loss (perte de base)
        base_loss = -dist.log_prob(y).mean() if self.hparams.loss == 'nll' else energy_score(dist, y, self.hparams.es_num_samples)
        
        # Liste des valeurs alpha pour la régularisation
        alphas = np.arange(0.1, 1.1, 0.1) 
        
        # Calcul de la régularisation pour chaque alpha et somme des termes
        trunc_reg_sum = sum(self.truncation_regularization(dist, y, alpha=alpha) for alpha in alphas)
        
        # Retour de la perte totale avec la régularisation ajustée
        return base_loss + 0.5 * trunc_reg_sum  # Facteur d'équilibrage

    

#-----------------------------------------------------------------------------------------------------------------------------------------------------

    def forward(self, x):
        out = self.model(x)
        out = out.split(self.output_shape, dim=-1)
        params = extract_multivariate_normal_mixture_parameters(
            out,
            mixture_size=self.hparams.mixture_size,
            output_dim=self.trainer.datamodule.output_dim,
        )
        return create_multivariate_normal_mixture(*params)
    
    def predict(self, x):
        return self(x)

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
        dist = self(x)
        loss = self.compute_loss(dist, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        wandb.log({"train/loss": loss.item()})
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log(
            f'val/loss',
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        wandb.log({"val/loss_batch": loss.item()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)
    
    '''def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs).mean()  # Moyenne sur tous les batches
        self.log("val_loss", avg_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.validation_step_outputs.clear()  # Nettoyer pour la prochaine epoch
        wandb.log({"val/loss_epoch": avg_loss.item()})'''

    @classmethod
    def output_type(cls):
        return 'distribution'
