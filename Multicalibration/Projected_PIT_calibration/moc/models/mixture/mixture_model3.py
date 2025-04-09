import logging

import torch
from lightning.pytorch import LightningModule
from torch.distributions import MixtureSameFamily, Categorical, MultivariateNormal
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
        lambda_reg: float,
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

        # Dimensions du modèle
        output_dim = output_dim
        mixture_size = self.hparams.mixture_size
        self.lambda_reg = lambda_reg
        self.output_shape = (
            mixture_size,
            mixture_size * output_dim,
            mixture_size * output_dim * (output_dim + 1) // 2,
        )

        # Initialisation du modèle MLP
        self.model = MLP(
            input_dim=input_dim,
            output_dim=torch.sum(torch.tensor(self.output_shape)),
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.num_layers,
        )
        self.name = "GaussianMixture"
        self.validation_step_outputs = []

    def project_along_pca(self, samples, pca_vectors):
        """
        Project the samples along the PCA vectors.

        Args:
            samples (Tensor): The sample predictions, shape (batch_size, num_samples, num_features).
            pca_vectors (Tensor): The PCA vectors, shape (num_components, num_features).

        Returns:
            Tensor: The projections along the PCA vectors, shape (num_components, batch_size, num_samples).
        """
        return torch.matmul(samples, pca_vectors.T)

    def compute_empirical_cdf(self, projections, y):
        """
        Compute the empirical CDF for the projected samples.

        Args:
            projections (Tensor): The projected predictions, shape (num_components, batch_size, num_samples).
            y (Tensor): The true values, shape (batch_size, num_features).

        Returns:
            Tensor: The empirical CDF values, shape (num_components, batch_size, 1).
        """
        sorted_proj = torch.sort(projections, dim=-1)[0]  # Sorted projections
        n = len(projections[-1])  # Number of samples
        x_proj = torch.matmul(y, pca_vectors.T)  # Project true values

        cdf_values = torch.searchsorted(sorted_proj, x_proj.unsqueeze(-1), side='right') / n
        return cdf_values

    def calculate_pit(self, values, pca_vectors, sample):
        """
        Calculate the PIT (Probability Integral Transform) values.

        Args:
            values (Tensor): The predicted values, shape (batch_size, num_features).
            pca_vectors (Tensor): The PCA vectors, shape (num_components, num_features).
            sample (Tensor): The sample predictions, shape (num_samples, batch_size, num_features).

        Returns:
            Tensor: The PIT values, shape (num_components, batch_size, 1).
        """
        projections = self.project_along_pca(sample, pca_vectors)
        return self.compute_empirical_cdf(projections, values)

    def sample_from_distribution(self, dist, num_samples=1000):
        """
        Sample from the mixture distribution.

        Args:
            dist (MixtureSameFamily): The distribution to sample from.
            num_samples (int): The number of samples to generate.

        Returns:
            Tensor: The sampled predictions, shape (num_samples, batch_size, num_features).
        """
        return dist.sample((num_samples,)).permute(1, 0, 2)

    def ensemble_PIT(self, y_hat, y):
        """
        Compute the PIT values using PCA.

        Args:
            y_hat (Tensor): The predicted values, shape (batch_size, num_samples, num_features).
            y (Tensor): The true values, shape (batch_size, num_features).

        Returns:
            Tuple: A tuple containing:
                - Tensor: The PIT values for each PCA component, shape (num_components, batch_size, 1).
                - Tensor: The PCA components, shape (num_components, num_features).
        """
        pca = PCA(n_components=len(y[0]))  # Keep all components
        pca.fit(y_hat.reshape(-1, len(y[0])))  # Fit PCA on the predictions
        pca_vectors = torch.tensor(pca.components_, dtype=y_hat.dtype, device=y_hat.device)

        return torch.stack([self.calculate_pit(y, pca_vectors[i], y_hat) for i in range(len(pca_vectors))]), pca_vectors

    def rqr_regularization(self, Z, k, N):
        """
        Compute the RQR regularization term based on PIT values.

        Args:
            Z (Tensor): The PIT values, shape (num_components, batch_size, 1).
            k (int): The window size for the regularization calculation.
            N (int): The number of PIT values.

        Returns:
            Tensor: The RQR regularization term.
        """
        Z_sorted = torch.sort(Z, dim=1)[0]  # Sort the PIT values
        rqr = 0
        for j in range(len(Z)):  # Loop over dimensions
            uni_rqr = 0.0
            for i in range(N - k):
                term = torch.abs(torch.log(((N + 1) / k) * (Z_sorted[j][i + k] - Z_sorted[j][i])))
                uni_rqr += term
            rqr += uni_rqr / (N - k)
        return rqr / len(Z)  # Average over dimensions

    def compute_quantiles(self, y_hat, y, quantile_level):
        """
        Compute the quantile at a given level for each projection.

        Args:
            y_hat (Tensor): The predicted values, shape (batch_size, num_samples, num_features).
            y (Tensor): The true values, shape (batch_size, num_features).
            quantile_level (float): The quantile level to compute, e.g., 0.9 for the 90th percentile.

        Returns:
            Tensor: The computed quantiles for each projection, shape (num_components, batch_size, 1).
        """
        # Calculating PIT for each component using ensemble_PIT
        pit_vals, pca_vectors = self.ensemble_PIT(y_hat, y)

        quantiles = []
        for i in range(len(pca_vectors)):  # For each PCA component
            # Project the predictions along the PCA vector
            projections = self.project_along_pca(y_hat, pca_vectors[i])

            # Sort the projections along the last dimension (samples)
            sorted_proj = torch.sort(projections, dim=-1)[0]

            # Compute the index of the quantile for each sample
            n = sorted_proj.shape[-1]  # Number of samples
            quantile_index = int(quantile_level * (n - 1))

            # Select the quantile value based on the sorted projections
            quantile_values = sorted_proj[..., quantile_index].unsqueeze(-1)
            print(quantile_values)

            quantiles.append(quantile_values)

        # Stack the quantiles along the component dimension
        return torch.cat(quantiles, dim=0)  # Shape: (num_components, batch_size, 1)


    def truncation_regularization(self, dist, y, alpha=1):
        """
        Compute the Truncation-based Calibration regularization term.

        Args:
            dist (MixtureSameFamily): The mixture distribution.
            y (Tensor): The true values, shape (batch_size, num_features).
            alpha (float): The truncation threshold.

        Returns:
            Tensor: The truncation regularization term.
        """
        num_samples = 100
        y_hat = self.sample_from_distribution(dist, num_samples=num_samples)
        pit_values = self.ensemble_PIT(y_hat, y)[0]

        F_hat_alpha = (pit_values < alpha).float().mean(dim=-1)
        def rho(x, y):
            return (y - x) * (x < y).float()

        if F_hat_alpha.mean() < alpha:
            reg_term = rho(self.compute_quantiles(y_hat, y, torch.tensor(alpha, device=y.device)), y).mean()
        else:
            reg_term = rho(y, self.compute_quantiles(y_hat, y, torch.tensor(alpha, device=y.device))).mean()

        return reg_term

    def compute_loss(self, dist, y, lamda):
        """
        Compute the total loss, combining the chosen loss and regularization terms.

        Args:
            dist (MixtureSameFamily): The mixture distribution.
            y (Tensor): The true values, shape (batch_size, num_features).
            lamda (float): The regularization strength.

        Returns:
            Tuple: A tuple containing the total loss, individual loss, and regularization terms.
        """
        alphas = np.arange(0.1, 1.1, 0.1)
        trunc_reg_sum = sum(self.truncation_regularization(dist, y, alpha=alpha) for alpha in alphas)

        if self.hparams.loss == 'nll':
            loss_term = -dist.log_prob(y).mean()
            reg_loss = loss_term + (lamda * trunc_reg_sum)
            return reg_loss, loss_term, lamda * trunc_reg_sum, trunc_reg_sum
        elif self.hparams.loss == 'es':
            loss_term = energy_score(dist, y, n_samples=self.hparams.es_num_samples)
            reg_loss = loss_term + (lamda * trunc_reg_sum)
            return reg_loss, loss_term, lamda * trunc_reg_sum, trunc_reg_sum
        
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
        wandb.log({"train_reg_loss": reg_loss.item(), "train_loss": loss.item(),
                   "train_lambda_rqr":lamda_rqr, "train_rqr": rqr})
        return reg_loss

    def validation_step(self, batch, batch_idx):
        reg_loss, loss, lamda_rqr, rqr = self.step(batch)
        wandb.log({"val_reg_loss": reg_loss.item(), "val_loss": loss.item(),
                   "val_lambda_rqr":lamda_rqr, "val_rqr": rqr})
        self.log(
            f'val/loss',
            reg_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return reg_loss

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
