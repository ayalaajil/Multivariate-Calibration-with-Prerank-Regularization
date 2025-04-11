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
from regularizers.reguls import rqr_regularization, truncation_regularization, pce_kde_regularization
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
        mixture_size: int = 10,
        es_num_samples: int = 50,
        lr=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        # wandb.init(project="multicalibration", name=f"{reg_type}")

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

    def compute_loss(self, dist, y): #with rqr
        """
        Compute the loss with the added regularization term based on PIT values.
        """
        if self.reg_type == 'rqr':
            reg_term = rqr_regularization(dist, y)
        elif self.reg_type == 'truncation':
            reg_term = truncation_regularization(dist, y)
        elif self.reg_type == 'pce-kde':
            reg_term = pce_kde_regularization(dist, y)
        else:
            reg_term = 0.0
        
        if self.hparams.loss == 'nll':
            loss_term = -dist.log_prob(y).mean()
            reg_loss = loss_term + (self.lambda_reg * reg_term)
            return reg_loss, loss_term, reg_term
        elif self.hparams.loss == 'es':
            loss_term = energy_score(dist, y, n_samples=self.hparams.es_num_samples)
            reg_loss = loss_term + (self.lambda_reg * reg_term)
            return reg_loss, loss_term, reg_term
        else:
            raise ValueError(f'Invalid loss: {self.hparams.loss}')
    
    def step(self, batch):
        x, y = batch
        dist = self(x) #256 distributions in 4D
        reg_loss, loss, reg = self.compute_loss(dist, y)
        return reg_loss, loss, reg

    def training_step(self, batch, batch_idx):
        reg_loss, loss, reg = self.step(batch)
        # self.train_step_outputs.append({
        #     "reg_loss": reg_loss.detach(),
        #     "loss": loss.detach(),
        #     "reg": reg if isinstance(reg, float) else reg.detach(),
        # })
        return reg_loss
    
    # def on_train_epoch_end(self):
    #     reg_losses = torch.stack([x["reg_loss"] for x in self.train_step_outputs])
    #     losses = torch.stack([x["loss"] for x in self.train_step_outputs])
    #     regs = torch.stack([x["reg"] for x in self.train_step_outputs])

    #     wandb.log({
    #         "train_reg_loss": reg_losses.mean().item(),
    #         "train_loss": losses.mean().item(),
    #         "train_reg": regs.mean().item(),
    #         "epoch": self.current_epoch
    #     })

    #     self.train_step_outputs.clear()  # Clear for next epoch
        

    def validation_step(self, batch, batch_idx):
        reg_loss, loss, reg = self.step(batch)
        self.log(
            f'val/loss',
            reg_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        # self.validation_step_outputs.append({
        #     "reg_loss": reg_loss.detach(),
        #     "loss": loss.detach(),
        #     "reg": reg if isinstance(reg, float) else reg.detach(),
        #     })
        return reg_loss
    
    # def on_validation_epoch_end(self):
    #     reg_losses = torch.stack([x["reg_loss"] for x in self.validation_step_outputs])
    #     losses = torch.stack([x["loss"] for x in self.validation_step_outputs])
    #     regs = torch.stack([x["reg"] for x in self.validation_step_outputs])

    #     wandb.log({
    #         "val_reg_loss": reg_losses.mean().item(),
    #         "val_loss": losses.mean().item(),
    #         "val_reg": regs.mean().item(),
    #         "epoch": self.current_epoch
    #     })

    #     self.validation_step_outputs.clear()  # Clear for next epoch

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)
    

    @classmethod
    def output_type(cls):
        return 'distribution'
