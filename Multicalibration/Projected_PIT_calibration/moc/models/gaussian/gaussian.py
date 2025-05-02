import logging
import sys
import os
import torch
from lightning.pytorch import LightningModule
from torch.distributions import MultivariateNormal
from moc.metrics.distribution_metrics import multivariate_energy_score, pce
from pathlib import Path

reg_path = Path(__file__).resolve().parents[2]
sys.path.append(str(reg_path))
from regularizers.reguls import truncation_regularization, pce_kde_regularization


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

def create_multivariate_normal(loc, scale_tril):
    return MultivariateNormal(loc=loc, scale_tril=adjust_scale_tril(scale_tril))

def extract_multivariate_normal_parameters(params, output_dim):
    loc, scale_tril_raw = params
    scale_tril = torch.zeros(loc.shape[:-1] + (output_dim, output_dim), device=loc.device)
    tril_indices = torch.tril_indices(row=output_dim, col=output_dim, offset=0, device=loc.device)
    scale_tril[..., tril_indices[0], tril_indices[1]] = scale_tril_raw
    diag_indices = torch.arange(output_dim, device=loc.device)
    scale_tril[..., diag_indices, diag_indices] = torch.nn.functional.softplus(
        scale_tril[..., diag_indices, diag_indices]
    ) + 1e-3
    return loc, scale_tril

class GaussianLightningModule(LightningModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int = 100,
        num_layers: int = 3,
        loss: str = 'nll',
        es_num_samples: int = 100,
        lr=1e-4,
        lambda_reg: float = 0.0,
        reg_type: str = 'none',
        prerank: str = 'none',
    ):
        super().__init__()
        self.save_hyperparameters()

        output_dim = output_dim
        # Output parameters: loc (output_dim) and scale_tril (output_dim * (output_dim + 1) // 2)
        self.output_shape = (
            output_dim,
            output_dim * (output_dim + 1) // 2,
        )
        self.model = MLP(
            input_dim=input_dim,
            output_dim=torch.sum(torch.tensor(self.output_shape)),
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.num_layers,
        )

    def forward(self, x):
        out = self.model(x)
        out = out.split(self.output_shape, dim=-1)
        params = extract_multivariate_normal_parameters(
            out,
            output_dim=self.trainer.datamodule.output_dim,
        )
        return create_multivariate_normal(*params)
    
    def predict(self, x):
        return self(x)

    def compute_loss(self, dist, y):
        reg_val = 0.0  # raw reg
        reg_term = 0.0  # scaled reg
        if self.hparams.reg_type == 'truncation':
            reg_val = truncation_regularization(dist, y)
        elif self.hparams.reg_type == 'pce-kde':
            reg_val = pce_kde_regularization(dist, y, self.hparams.prerank)

        if self.hparams.loss == 'nll':
            loss_term = -dist.log_prob(y).mean()
        elif self.hparams.loss == 'es':
            loss_term = multivariate_energy_score(dist, y, n_samples=self.hparams.es_num_samples).mean()
        else:
            raise ValueError(f'Invalid loss: {self.hparams.loss}')

        # pce_score = pce(dist, y) #return a list of d elements

        reg_term = self.hparams.lambda_reg * reg_val
        total_loss = loss_term + reg_term

        return total_loss, reg_val
    

    def step(self, batch):
        x, y = batch
        dist = self(x)
        total_loss, raw_reg = self.compute_loss(dist, y)
        return total_loss, raw_reg

    def training_step(self, batch, batch_idx):
        total_loss, raw_reg = self.step(batch)
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, raw_reg = self.step(batch)
        self.log(
            f'val/loss',
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)

    @classmethod
    def output_type(cls):
        return 'distribution'