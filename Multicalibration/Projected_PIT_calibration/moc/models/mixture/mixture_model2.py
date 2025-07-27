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
        reg_type: str = 'none',
        prerank: str = 'none',
        warmup_epochs: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        mixture_size = self.hparams.mixture_size
        self.lambda_reg = lambda_reg
        self.reg_type = reg_type
        self.warmup_epochs = warmup_epochs
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

        reg_val = torch.zeros(1).to(y.device)
        marg_val = torch.zeros(1).to(y.device)
        prerank_val = torch.zeros(1).to(y.device)
        
        if self.hparams.reg_type == 'truncation':
            reg_val = truncation_regularization(dist, y)
        elif self.hparams.reg_type == 'pce-kde':
            marg_val = pce_kde_regularization(dist, y, n_samples = self.hparams.es_num_samples, prerank = 'marginal')
            prerank_val = pce_kde_regularization(dist, y, n_samples = self.hparams.es_num_samples, prerank = self.hparams.prerank)

        if self.hparams.loss == 'nll':
            loss_per_sample = -dist.log_prob(y)
            loss_term = -dist.log_prob(y).mean()
        elif self.hparams.loss == 'es':
            loss_term = multivariate_energy_score(dist, y, n_samples=self.hparams.es_num_samples).mean()
        else:
            raise ValueError(f'Invalid loss: {self.hparams.loss}')
        
        reg_term = self.hparams.lambda_reg * (marg_val + prerank_val)
        total_loss = loss_term + reg_term

        return total_loss, loss_term, marg_val, prerank_val, loss_per_sample

    def step(self, batch):
        x, y, idx = batch
        dist = self(x)

        total_loss, loss_term, marg_val, prerank_val, loss_per_sample = self.compute_loss(dist, y)

        # pce_val, cdfs = pce(dist, y, n_samples=self.hparams.es_num_samples, prerank=self.hparams.prerank)
        # energy_score = multivariate_energy_score(dist, y, n_samples=self.hparams.es_num_samples).mean()

        return total_loss, loss_term, marg_val, prerank_val, loss_per_sample
    # pce_val, cdfs, energy_score

    def training_step(self, batch, batch_idx):
        total_loss, loss_term, marg_val, prerank_val,loss_per_sample  = self.step(batch)
        # pce_val, cdfs, energy_score 
        if self.global_step == 0:
            print(f"Checking {marg_val.requires_grad}, {prerank_val.requires_grad}")

        self.log('train/total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/nll', loss_term, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/marg_val', marg_val, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/prerank_val', prerank_val, on_step=False, on_epoch=True, prog_bar=False)
        # self.log('train/pce_val', pce_val, on_step=False, on_epoch=True, prog_bar=False)
        # self.train_cdfs.append(cdfs)

        return total_loss 

    def validation_step(self, batch, batch_idx):
        x, y, idx = batch
        total_loss, loss_term, marg_val, prerank_val,loss_per_sample  = self.step(batch)
        '''print("BATCH IDX")
        print(batch_idx)
        print("LOSS")
        print(total_loss)'''
        '''if batch_idx== 3:
            print("== BATCH IDX 2 ==")
            for i, (index, loss_val) in enumerate(zip(idx, loss_per_sample)):
                print(f"Sample index in dataset: {index.item()} | Loss: {loss_val.item()}")'''
        # pce_val, cdfs, energy_score = self.step(batch)

        # self.log('val/total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/nll', loss_term, on_step=False, on_epoch=True, prog_bar=False)
        # self.log('val/raw_reg', raw_reg, on_step=False, on_epoch=True, prog_bar=False)
        # self.log('val/pce_val', pce_val.mean(), on_step=False, on_epoch=True, prog_bar=False)
        # self.log('val/energy_score', energy_score, on_step=False, on_epoch=True, prog_bar=False)
        #add the energy score here
        # self.val_cdfs.append(cdfs)

        return total_loss
    
    '''def test_step(self, batch, batch_idx):
        x, y, idx = batch
        total_loss, loss_term, marg_val, prerank_val, loss_per_sample = self.step(batch)

        print("BATCH IDX")
        print(batch_idx)
        print("LOSS")
        print(total_loss)

        self.log('test/total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('test/nll', loss_term, on_step=False, on_epoch=True, prog_bar=False)
        self.log('test/marg_val', marg_val, on_step=False, on_epoch=True, prog_bar=False)
        self.log('test/prerank_val', prerank_val, on_step=False, on_epoch=True, prog_bar=False)
        # self.log('test/energy_score', energy_score, on_step=False, on_epoch=True, prog_bar=False)

        return total_loss'''

    # def on_train_epoch_end(self):
    #     train_cdfs = torch.cat(self.train_cdfs, dim=0).mean(dim=0)
    #     self.total_train_cdfs.append(train_cdfs)
    #     self.train_cdfs = []

    # def on_validation_epoch_end(self):
    #     val_cdfs = torch.cat(self.val_cdfs, dim=0).mean(dim=0)
    #     self.total_val_cdfs.append(val_cdfs)
    #     self.val_cdfs = []

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)
    

    @classmethod
    def output_type(cls):
        return 'distribution'
