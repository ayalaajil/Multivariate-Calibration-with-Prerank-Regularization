import torch
from lightning.pytorch import LightningModule
from torch.distributions import MixtureSameFamily, Categorical, MultivariateNormal
import math
from sklearn.decomposition import PCA
import numpy as np


def projected_pit(y, v, samples):
    v = torch.as_tensor(v, dtype=samples.dtype, device=samples.device)
    proj_samples = torch.matmul(samples, v)
    # if x_values is None:
    #     x_values = sample
    proj_y = torch.matmul(y, v)
    sorted_samples = torch.sort(proj_samples)[0] #256,100 sort across the columns
    n = samples.shape[1] #100 or 1000
    pits = torch.searchsorted(sorted_samples, proj_y.unsqueeze(-1), side='right') / n #256,1
    return pits

def calculate_PIT(self, y_hat, y):
    pca = PCA(n_components=len(y[0])) #keeping all components, 4 in this case
    pca.fit(y_hat.reshape(-1,len(y[0])))
    vectors = pca.components_ #4 by 4
    pits = torch.stack([self.projected_pit(y, vectors[i], y_hat) for i in range(len(vectors))])
    return pits, vectors

def rqr_regularization(dist, y, k = 100, num_samples = 1000):
        
    samples = dist.sample((num_samples,)).permute(1, 0, 2) #256,100,4
    pit_values = calculate_PIT(samples, y)[0]
    # Sort the PIT values if necessary (sorting might depend on the context)
    sorted_pits = torch.sort(pit_values, dim=1)[0]# Sort the PIT values

    # Calculate the regularization term
    rqr = 0
    N, d = y.shape[0], y.shape[1]
    for j in range(d): #loop over dimensions 4
        uni_rqr = 0.0
        for i in range(N - k):
            term = np.absolute(torch.log(((N + 1) / k) * (sorted_pits[j][i + k] - sorted_pits[j][i])))
            uni_rqr += term  # Weight proportional to the importance of the component ???
        rqr += uni_rqr/(N-k)                                    
    return rqr/d #average over dimensions

def compute_quantile(y_hat, alpha, vector):
    proj_yhat = torch.matmul(y_hat, vector)
    sorted_proj = torch.sort(proj_yhat, dim=1)[0]
    #print(sorted_proj[0])

    # Compute the index of the quantile for each sample
    n = sorted_proj.shape[1]  # Number of samples i.e. 100
    quantile_index = int(alpha * n)-1 #90

    # Select the quantile value based on the sorted projections
    quantile_values = sorted_proj[:, quantile_index, :] #256 values
    return quantile_values
    

def truncation_regularization(dist, y, num_samples = 1000, M = 100):
    y_hat = dist.sample((num_samples,)).permute(1, 0, 2) 
    pca = PCA(n_components=y.shape[1]) #keeping all components, 4 in this case
    pca.fit(y_hat.reshape(-1,y.shape[1]))
    vectors = pca.components_
    # Shape: (batch_size, number fo samples, dim)
    alphas = torch.linspace(0, 1, M, device=y_hat.device)
    dim = y.shape[1]
    trunc_total = 0.0
    for alpha in alphas:
        trunc_alpha_dim = 0.0
        for d in range(dim):
            vector = torch.as_tensor(vectors[d], dtype=y.dtype, device=y.device)
            quantiles = compute_quantile(y_hat, alpha, vector) #256
            proj_y = torch.matmul(y, vector) #256,1

            F_hat_alpha = (proj_y <= quantiles).float().mean()  # scalar
            if F_hat_alpha < alpha:
                    rho = (proj_y - quantiles) * (quantiles < proj_y)
            else:
                    rho = (quantiles - proj_y) * (proj_y < quantiles)
            trunc_alpha_dim += rho.mean()
        trunc_total += trunc_alpha_dim / dim
    return trunc_total / M

def pce_kde_regularization(dist, y, num_samples = 1000, M = 100, tau = 100, p = 1):
    y_hat = dist.sample((num_samples,)).permute(1, 0, 2) # (256,100,4)
    pit_values = calculate_PIT(y_hat, y)[0] # (4,256,1)
    alphas = torch.linspace(0, 1, M, device=y_hat.device)
    dim = y.shape[1]
    pce_kde = 0.0
    for alpha in alphas:
        phi_kde_dim = 0.0
        for d in range(dim):
            phi_kde = torch.sigmoid(tau * (alpha - pit_values[d,:,:])).mean()
            phi_kde_dim += phi_kde
        pce_kde += torch.abs(alpha - phi_kde_dim/dim)**p
    return pce_kde/M
    
    


# def smooth_indicator(self, a, b, tau=10.0):
#         return torch.sigmoid(tau * (b - a))

# def kde_cdf_smoothed(self, z_vals, alphas, tau):
#     z_vals = z_vals.unsqueeze(0)     # (1, N)
#     alphas = alphas.unsqueeze(1)     # (M, 1)
#     smoothed_indicators = torch.sigmoid(tau * (alphas - z_vals))  # (M, N)
#     return smoothed_indicators.mean(dim=1)  # Moyenne sur les N, résultat (M,)

# tau = 0.5
# p = 1

# def compute_loss_kde(self, dist, y, lamda):
#     sample_pred = self.sample(dist) #returns 256,100,4
#     pit_values = self.ensemble_PIT(sample_pred, y)[0] #4,256,1
#     phi_kde =self.kde_cdf_smoothed(pit_values, alphas, tau=tau)  # (M,)
#     reg_kde = (torch.abs(alphas - phi_kde) ** p).mean()
#     if self.hparams.loss == 'nll':
#         loss_term = -dist.log_prob(y).mean()
#         reg_loss = loss_term + (lamda * reg_kde)
#         return reg_loss, loss_term, lamda*reg_kde, reg_kde
#     elif self.hparams.loss == 'es':
#         loss_term = energy_score(dist, y, n_samples=self.hparams.es_num_samples)
#         reg_loss = loss_term + (lamda * reg_kde)
#         return reg_loss, loss_term, lamda*reg_kde, reg_kde
