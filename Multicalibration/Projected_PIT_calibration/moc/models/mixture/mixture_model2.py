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
        wandb.init(project="conformal_regression")
        # wandb.config.update(self.hparams)


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

    # Fonction pour projeter les échantillons sur les vecteurs
    def proj_for(self, x_values, u, sample):
        sample_proj = torch.matmul(sample, u)
        if x_values is None:
            x_values = sample
        x_proj = torch.matmul(x_values, u)
<<<<<<< HEAD
        sample_sorted = torch.sort(sample_proj)[0]
        n = len(sample[0])
        
        cdf_values = torch.searchsorted(sample_sorted, x_proj.unsqueeze(-1), side='right') / n
        return x_proj, cdf_values
=======
        sample_sorted = torch.sort(sample_proj)[0] #256,100 sort across the columns
        n = len(sample[0]) #100
        cdf_values = torch.searchsorted(sample_sorted, x_proj.unsqueeze(-1), side='right') / n #256,1
        return x_proj, cdf_values #returns projected predictions and cdf values which represent where in the sorted projected samples the projected ground truth lies
>>>>>>> 124cf8d3e389937a2260f717dc3c7b5a272d6ed9

    # Fonction pour calculer les PIT
    def calculate_pit(self,values, u, sample):
        u = torch.as_tensor(u, dtype=sample.dtype, device=sample.device)
        pits = self.proj_for(values, u, sample)[1]
        return pits
    
    def sample(self, dist, num_samples=1000):
        return dist.sample((num_samples,)).permute(1, 0, 2)

    def ensemble_PIT(self, y_hat, y):
        pca = PCA(n_components=len(y[0])) #keeping all components, 4 in this case
        pca.fit(y_hat.reshape(-1,len(y[0])))
        vectors = pca.components_ #4 by 4
        return torch.stack([self.calculate_pit(y, vectors[i], y_hat) for i in range(len(vectors))])
    
    def rqr_regularization(self, Z, k, N):
        """
        Compute the RQR regularization term based on PIT values.
        Z: Tensor of PIT values calculated on the calibration set.
        k: The window size used for the calculation.
        N: The total number of PIT values.
        """
        # Sort the PIT values if necessary (sorting might depend on the context)
<<<<<<< HEAD
        Z_sorted = torch.sort(Z, dim=1)[0]# Sort the PIT values
=======
        Z_sorted = torch.sort(Z, dim=1)[0]# Sort the PIT values along 256
        # print(len(Z_sorted))
        # print(len(Z_sorted[0]))
>>>>>>> 124cf8d3e389937a2260f717dc3c7b5a272d6ed9

        # Calculate the regularization term
        rqr = 0
        for j in range(len(Z)): #loop over dimensions 4
            uni_rqr = 0.0
            for i in range(N - k):
<<<<<<< HEAD
                term = np.absolute(torch.log(((N + 1) / k) * (Z_sorted[j][i + k] - Z_sorted[j][i])))
                rqr += term                                      
        return rqr / (len(Z)*(N - k))
    


    def compute_loss_rqr(self, dist, y): 
=======
                term = torch.log(((N + 1) / k) * (Z_sorted[j][i + k] - Z_sorted[j][i]))
                uni_rqr += term  
            rqr += uni_rqr/(N-k)                                    
        return rqr/len(Z) #average over dimensions
    


    def compute_loss(self, dist, y, lamda=0.01): #with rqr
>>>>>>> 124cf8d3e389937a2260f717dc3c7b5a272d6ed9
        """
        Compute the loss with the added regularization term based on PIT values.
        """
        # Compute PIT values
        sample_pred = self.sample(dist) #returns 256,100,4
        pit_values = self.ensemble_PIT(sample_pred, y) #4,256,1

        # Compute RQR regularization term
        N = len(y)  # Number of samples in the PIT values, 256
        rqr = self.rqr_regularization(pit_values, 100, N)
        rqr = max(rqr, 0.0)
        
        if self.hparams.loss == 'nll':
<<<<<<< HEAD
            return -dist.log_prob(y).mean() + 0.5*rqr
        elif self.hparams.loss == 'es':
            return energy_score(dist, y, n_samples=self.hparams.es_num_samples) + 0.5*rqr
=======
            loss_term = -dist.log_prob(y).mean()
            reg_loss = loss_term + (lamda * rqr)
            return reg_loss, loss_term, lamda*rqr, rqr
        elif self.hparams.loss == 'es':
            loss_term = energy_score(dist, y, n_samples=self.hparams.es_num_samples)
            reg_loss = loss_term + (lamda * rqr)
            return reg_loss, loss_term, lamda*rqr, rqr
>>>>>>> 124cf8d3e389937a2260f717dc3c7b5a272d6ed9
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
        
<<<<<<< HEAD
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

=======
>>>>>>> 124cf8d3e389937a2260f717dc3c7b5a272d6ed9
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

<<<<<<< HEAD
    '''def compute_loss(self, dist, y):
        if self.hparams.loss == 'nll':
            self.validation_step_outputs.append(-dist.log_prob(y).mean()) 
            return -dist.log_prob(y).mean()
        elif self.hparams.loss == 'es':
            self.validation_step_outputs.append(energy_score(dist, y, n_samples=self.hparams.es_num_samples))
            return energy_score(dist, y, n_samples=self.hparams.es_num_samples)
        else:
            raise ValueError(f'Invalid loss: {self.hparams.loss}')'''

=======
>>>>>>> 124cf8d3e389937a2260f717dc3c7b5a272d6ed9
    def step(self, batch):
        x, y = batch
        dist = self(x) #256 distributions in 4D
        reg_loss, loss, lamda_rqr, rqr = self.compute_loss(dist, y)
        return reg_loss, loss, lamda_rqr, rqr

    def training_step(self, batch, batch_idx):
        reg_loss, loss, lamda_rqr, rqr = self.step(batch)
        wandb.log({"train_reg_loss": reg_loss.item(), "train_loss": loss.item(),
                   "train_lambda_rqr":lamda_rqr, "train_rqr": rqr})
        return reg_loss

    def validation_step(self, batch, batch_idx):
<<<<<<< HEAD
        loss = self.step(batch)
=======
        reg_loss, loss, lamda_rqr, rqr = self.step(batch)
        wandb.log({"val_reg_loss": reg_loss.item(), "val_loss": loss.item(),
                   "val_lambda_rqr":lamda_rqr, "val_rqr": rqr})
>>>>>>> 124cf8d3e389937a2260f717dc3c7b5a272d6ed9
        self.log(
            f'val/loss',
            reg_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
<<<<<<< HEAD
        wandb.log({"val/loss_batch": loss.item()})
        return loss
=======
        return reg_loss
>>>>>>> 124cf8d3e389937a2260f717dc3c7b5a272d6ed9

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
