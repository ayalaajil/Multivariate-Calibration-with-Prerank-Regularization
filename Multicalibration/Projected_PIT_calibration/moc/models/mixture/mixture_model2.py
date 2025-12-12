import logging

import torch
from lightning.pytorch import LightningModule
from torch.distributions import MixtureSameFamily, Categorical, MultivariateNormal
from sklearn.decomposition import PCA
import numpy as np
import sys
from pathlib import Path
from moc.metrics.distribution_metrics import multivariate_energy_score, pce

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
        es_num_samples: int = 100,
        lr=1e-4, #was 1e-4 BEFORE
        lambda_reg: float = 0.0,
        do_reg: bool = False,
        prerank: list = [],
        tau: int = 100
    ):
        super().__init__()
        self.save_hyperparameters()
        mixture_size = self.hparams.mixture_size
        self.lambda_reg = lambda_reg
        self.do_reg = do_reg
        # self.tau = tau
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
        self.train_cdfs = []
        self.val_cdfs = []
        self.total_train_cdfs = []
        self.total_val_cdfs = []

    def forward(self, x):
        out = self.model(x) #(batch_size, 75)
        out = out.split(self.output_shape, dim=-1) #(batch_size, 5), (batch_size, 20), (batch_size, 50)
        params = extract_multivariate_normal_mixture_parameters(
            out,
            mixture_size=self.hparams.mixture_size,
            output_dim=self.hparams.output_dim,
        )
        return create_multivariate_normal_mixture(*params)
    
    def predict(self, x):
        return self(x)

    def compute_loss(self, dist, y):

        pces_list = []
        total_pce = torch.zeros(1).to(y.device)

        if self.do_reg:
            if len(self.hparams.prerank)>1: #if there are two or more preranks
                for rho in self.hparams.prerank:
                    pce_value = pce_kde_regularization(dist, y, n_samples = self.hparams.es_num_samples, 
                                                        prerank = rho, tau = self.hparams.tau)
                    pces_list.append(pce_value)
                total_pce = torch.stack(pces_list).sum()
            else: 
                pce = pce_kde_regularization(dist, y, n_samples = self.hparams.es_num_samples, 
                                                    prerank = self.hparams.prerank[0], tau = self.hparams.tau)
                pces_list = [pce]
                total_pce = pce
        
        if self.hparams.loss == 'nll':
            loss_term = -dist.log_prob(y).mean()
        elif self.hparams.loss == 'es':
            loss_term = multivariate_energy_score(dist, y, n_samples=100).mean()
        else:
            raise ValueError(f'Invalid loss: {self.hparams.loss}')
        
        reg_term = self.hparams.lambda_reg * (total_pce)
        total_loss = loss_term + reg_term

        return total_loss, loss_term, total_pce, pces_list

    def step(self, batch):
        x, y, idx = batch
        dist = self(x)

        total_loss, loss_term, total_pce, pces_list = self.compute_loss(dist, y)

        return total_loss, loss_term, total_pce, pces_list
    
    def training_step(self, batch, batch_idx):
        total_loss, loss_term, total_pce, pces_list  = self.step(batch)
        
        if self.global_step == 0:
            print(f"Checking {total_pce.requires_grad}")

        self.log('train/total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/nll', loss_term, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/total_pce', total_pce, on_step=False, on_epoch=True, prog_bar=False)
        if self.do_reg:
            for i in range(len(self.hparams.prerank)):
                self.log(f'train/{self.hparams.prerank[i]}_pce', pces_list[i], on_step=False, on_epoch=True, prog_bar=False)
        

        return total_loss 

    def validation_step(self, batch, batch_idx):
        total_loss, loss_term, total_pce, pces_list = self.step(batch)

        self.log('val/total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/nll', loss_term, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/total_pce', total_pce, on_step=False, on_epoch=True, prog_bar=False)
        if self.do_reg:
            for i in range(len(self.hparams.prerank)):
                self.log(f'val/{self.hparams.prerank[i]}_pce', pces_list[i], on_step=False, on_epoch=True, prog_bar=False)

        return total_loss


    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)
    

    @classmethod
    def output_type(cls):
        return 'distribution'
