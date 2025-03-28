import logging

import torch
from lightning.pytorch import LightningModule
from torch.distributions import MixtureSameFamily, Categorical, MultivariateNormal
import math
from sklearn.decomposition import PCA
import wandb

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

    # Fonction pour projeter les échantillons sur les vecteurs
    def proj_for(self, x_values, u, sample):
        sample_proj = torch.matmul(sample, u)
        if x_values is None:
            x_values = sample
        x_proj = torch.matmul(x_values, u)
        sample_sorted = torch.sort(sample_proj)[0] #256,100 sort across the columns
        n = len(sample[0]) #100
        cdf_values = torch.searchsorted(sample_sorted, x_proj.unsqueeze(-1), side='right') / n
        return x_proj, cdf_values #returns projected predictions and cdf values which represent where in the sorted projected samples the projected predictions lie

    # Fonction pour calculer les PIT
    def calculate_pit(self,values, u, sample):
        u = torch.as_tensor(u, dtype=sample.dtype, device=sample.device)
        pits = self.proj_for(values, u, sample)[1]
        return pits
    
    def sample(self, dist, num_samples=100):
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
        Z_sorted = torch.sort(Z, dim=1)[0]# Sort the PIT values
        # print(len(Z_sorted))
        # print(len(Z_sorted[0]))

        # Calculate the regularization term
        rqr = 0
        for j in range(len(Z)): #loop over dimensions 4
            uni_rqr = 0.0
            for i in range(N - k):
                term = torch.log(((N + 1) / k) * (Z_sorted[j][i + k] - Z_sorted[j][i]))
                uni_rqr += term  
            rqr += uni_rqr/(N-k)                                    
        return rqr/len(Z) #average over dimensions
    


    def compute_loss(self, dist, y, lamda=0.01): #with rqr
        """
        Compute the loss with the added regularization term based on PIT values.
        """
        # Compute PIT values
        sample_pred = self.sample(dist)
        pit_values = self.ensemble_PIT(sample_pred, y) #4,256,1

        # Compute RQR regularization term
        N = len(y)  # Number of samples in the PIT values, 256
        rqr = self.rqr_regularization(pit_values, 100, N)
        
        if self.hparams.loss == 'nll':
            loss_term = -dist.log_prob(y).mean()
            reg_loss = loss_term + (lamda * rqr)
            return reg_loss, loss_term, lamda*rqr, rqr
        elif self.hparams.loss == 'es':
            loss_term = energy_score(dist, y, n_samples=self.hparams.es_num_samples)
            reg_loss = loss_term + (lamda * rqr)
            return reg_loss, loss_term, lamda*rqr, rqr
        else:
            raise ValueError(f'Invalid loss: {self.hparams.loss}')
        
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

    def step(self, batch):
        x, y = batch
        dist = self(x)
        reg_loss, loss, lamda_rqr, rqr = self.compute_loss(dist, y)
        return reg_loss, loss, lamda_rqr, rqr

    def training_step(self, batch, batch_idx):
        reg_loss, loss, lamda_rqr, rqr = self.step(batch)
        wandb.log({"train_reg_loss": reg_loss.item(), "train_loss": loss.item(),
                   "train_lambda_rqr":lamda_rqr.item(), "train_rqr": rqr.item()})
        return reg_loss

    def validation_step(self, batch, batch_idx):
        reg_loss, loss, lamda_rqr, rqr = self.step(batch)
        wandb.log({"val_reg_loss": reg_loss.item(), "val_loss": loss.item(),
                   "val_lambda_rqr":lamda_rqr.item(), "val_rqr": rqr.item()})
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

    @classmethod
    def output_type(cls):
        return 'distribution'
