import logging

import torch
from lightning.pytorch import LightningModule
from torch.distributions import MixtureSameFamily, Categorical, MultivariateNormal
from sklearn.decomposition import PCA
import wandb
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
        prerank: str = 'none'
    ):
        super().__init__()
        self.save_hyperparameters()
        # wandb.init(project="multicalibration", entity = 'ryuzaki')

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

        # self.recent_losses = []
        # self.regularization_active = True #HERE
        # self.stabilization_patience = 5
        # self.stabilization_threshold = 1e-1 

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

    def compute_loss(self, dist, y):

        reg_val = 0.0  # raw reg
        # if self.regularization_active:
        if self.hparams.reg_type == 'truncation':
            reg_val = truncation_regularization(dist, y)
        elif self.hparams.reg_type == 'pce-kde':
            reg_val = pce_kde_regularization(dist, y, n_samples = self.hparams.es_num_samples, prerank = self.hparams.prerank)

        if self.hparams.loss == 'nll':
            loss_term = -dist.log_prob(y).mean()
        elif self.hparams.loss == 'es':
            loss_term = multivariate_energy_score(dist, y, n_samples=self.hparams.es_num_samples).mean()
        else:
            raise ValueError(f'Invalid loss: {self.hparams.loss}')
        print(f"Checking {reg_val.requires_grad}, {reg_val.grad_fn}")
        # pce_score = pce(dist, y, n_samples = self.hparams.es_num_samples, prerank = self.hparams.prerank) #return a list of d elements
        # total_loss = loss_term + (self.hparams.lambda_reg * reg_val)
        total_loss = reg_val

        return total_loss, loss_term, reg_val


    def step(self, batch):
        x, y = batch
        dist = self(x) #256 distributions in 4D
        # print(dist.mean.requires_grad) #printed True
        total_loss, loss_term, raw_reg = self.compute_loss(dist, y)
        return total_loss, loss_term, raw_reg

    def training_step(self, batch, batch_idx):
        total_loss, loss_term, raw_reg = self.step(batch)
        self.train_step_outputs.append({
            "total_loss": total_loss.detach().cpu().numpy(), #float
            "raw_reg": raw_reg, #float
            "nll": loss_term.detach().cpu().numpy(), #a list
            # "pce_score": pce_score.detach().cpu().numpy(), #a list
        })
        return total_loss
    
    def on_train_epoch_end(self):
        total_losses = []
        raw_regs = []
        nlls = []
        # pce_scores = []

        for out in self.train_step_outputs:
            total_losses.append(float(out["total_loss"]))
            raw_regs.append(float(out["raw_reg"]))
            nlls.append(np.array(out["nll"]))  # shape: (d,)
            # pce_scores.append(np.array(out["pce_score"]))  # shape: (d,)

        avg_total_loss = np.mean(total_losses)
        avg_raw_reg = np.mean(raw_regs)
        avg_nll = np.mean(nlls)
        # avg_pce = np.mean(pce_scores)  # shape: (d,)

        # Build log dictionary
        log_dict = {
            "train/total_loss": avg_total_loss,
            "train/raw_reg": avg_raw_reg,
            "train/nll": avg_nll,
            # "train/pce": avg_pce,
        }

        # Add each dimension of the PCE score
        # for i, val in enumerate(avg_pce_score):
        #     log_dict[f"train/pce_dim_{i+1}"] = val

        # Log to W&B
        # wandb.log(log_dict)

        # Clear for next epoch
        self.train_step_outputs.clear()
        

    def validation_step(self, batch, batch_idx):
        total_loss, loss_term, raw_reg = self.step(batch)
        self.log(
            f'val/loss',
            float(total_loss),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.validation_step_outputs.append({
            "total_loss": total_loss.detach().cpu().numpy(), #float
            "raw_reg": raw_reg,
            "nll": loss_term.detach().cpu().numpy(),
            # "pce_score": pce_score.detach().cpu().numpy(),
        })
        return total_loss
    
    def on_validation_epoch_end(self):
        total_losses = []
        raw_regs = []
        nlls = []
        # pce_scores = []

        for out in self.validation_step_outputs:
            total_losses.append(float(out["total_loss"]))
            raw_regs.append(float(out["raw_reg"]))
            nlls.append(np.array(out["nll"]))
            # pce_scores.append(np.array(out["pce_score"]))  # shape: (d,)

        avg_total_loss = np.mean(total_losses)
        avg_raw_reg = np.mean(raw_regs)
        avg_nll_score = np.mean(nlls)
        # avg_pce = np.mean(pce_scores)

        # Build log dictionary
        log_dict = {
            "val/total_loss": avg_total_loss,
            "val/raw_reg": avg_raw_reg,
            "val/nll": avg_nll_score,
            # "val/pce": avg_pce,
        }

        # Update recent_losses and check for stabilization
        # self.recent_losses.append(avg_nll_score) 


        # Add each dimension of the PCE score
        # for i, val in enumerate(avg_pce_score):
        #     log_dict[f"val/pce_dim_{i+1}"] = val

        # Log to W&B
        # wandb.log(log_dict)

        # Clear for next epoch
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)
    

    @classmethod
    def output_type(cls):
        return 'distribution'
