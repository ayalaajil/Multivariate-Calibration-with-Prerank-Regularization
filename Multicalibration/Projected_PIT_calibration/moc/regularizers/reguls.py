import torch
from sklearn.decomposition import PCA
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metrics.distribution_metrics import calculate_PIT



# def projected_pit(y, v, samples):
#     v = torch.as_tensor(v, dtype=samples.dtype, device=samples.device)
#     proj_samples = torch.matmul(samples, v)
#     # if x_values is None:
#     #     x_values = sample
#     proj_y = torch.matmul(y, v)
#     sorted_samples = torch.sort(proj_samples)[0] #256,100 sort across the columns
#     n = samples.shape[1] #100 or 1000
#     pits = torch.searchsorted(sorted_samples, proj_y.unsqueeze(-1), side='right') / n #256,1
#     return pits

# def calculate_PIT(y_hat, y):
#     pca = PCA(n_components=len(y[0])) #keeping all components, 4 in this case
#     pca.fit(y_hat.reshape(-1,len(y[0])))
#     vectors = pca.components_ #4 by 4
#     pits = torch.stack([projected_pit(y, vectors[i], y_hat) for i in range(len(vectors))])
#     return pits, vectors

# def rqr_regularization(dist, y, k = 100, num_samples = 1000):
        
#     samples = dist.sample((num_samples,)).permute(1, 0, 2) #256,100,4
#     pit_values = calculate_PIT(samples, y)[0]
#     # Sort the PIT values if necessary (sorting might depend on the context)
#     sorted_pits = torch.sort(pit_values, dim=1)[0]# Sort the PIT values

#     # Calculate the regularization term
#     rqr = 0
#     N, d = y.shape[0], y.shape[1]
#     for j in range(d): #loop over dimensions 4
#         uni_rqr = 0.0
#         for i in range(N - k):
#             term = torch.abs(torch.log(((N + 1) / k) * (sorted_pits[j][i + k] - sorted_pits[j][i])))
#             uni_rqr += term  # Weight proportional to the importance of the component ???
#         rqr += uni_rqr/(N-k)                                    
#     return rqr/d #average over dimensions

def compute_quantile(y_hat, alphas, vector):
    proj_yhat = torch.matmul(y_hat, vector) #(256,1000)
    sorted_proj = torch.sort(proj_yhat, dim=1)[0]
    #print(sorted_proj[0])

    # Compute the index of the quantile for each sample
    n = sorted_proj.shape[1]  # Number of samples i.e. 1000
    alpha_idxs = (alphas * n).long() - 1
    alpha_idxs = torch.clamp(alpha_idxs, min=0)  # to avoid -1

    # Select the quantile value based on the sorted projections
    quantile_values = sorted_proj[:, alpha_idxs] #256,100 values
    return quantile_values
    

def truncation_regularization(dist, y, num_samples = 1000, M = 100):
    y_hat = dist.sample((num_samples,)).permute(1, 0, 2) #256,1000,4
    y_hat_np = y_hat.detach().cpu().numpy().reshape(-1,y.shape[1])
    pca = PCA(n_components=y.shape[1]) #keeping all components, 4 in this case
    pca.fit(y_hat_np)
    vectors = torch.tensor(pca.components_, dtype=y.dtype, device = y.device)
    # Shape: (batch_size, number fo samples, dim)
    alphas = torch.linspace(0, 1, M, device=y_hat.device)
    dim = y.shape[1]
    trunc_total = 0.0
    for d in range(dim):
        vector = vectors[d]
        proj_y = torch.matmul(y, vector) #256,1
        quantiles = compute_quantile(y_hat, alphas, vector) #256,100
        # Broadcast y_proj for comparison
        y_proj_exp = proj_y.unsqueeze(1)  # (256, 1)
        F_hat = (y_proj_exp <= quantiles).float().mean(dim=0)  # (100,)

        # Compute rho for all alphas
        diff = y_proj_exp - quantiles  # (256, 100)
        rho = torch.where(F_hat < alphas,
                          diff * (diff > 0),
                          -diff * (diff < 0))  # (256, 100)

        trunc_alpha = rho.mean(dim=0).mean()  # scalar
        trunc_total += trunc_alpha

    return trunc_total / dim

def pce_kde_regularization(dist, y, prerank, n_samples = 100, M = 100, tau = 100, p = 1):
    pit_values, _ = calculate_PIT(dist, y, n_samples = n_samples, prerank = prerank)  # (4, 256, 1)
    alphas = torch.linspace(0, 1, M, device=pit_values.device)  # (100,)
    dim = pit_values.shape[0]
    pce_kdes = []
    for d in range(dim):
        # Expand for broadcasting
        pit_d = pit_values[d]  # (256, 1)
        pit_exp = pit_d.expand(-1, M)  # (256, 100)
        alphas_exp = alphas.view(1, M)  # (1, 100)
        # Compute phi_kde for all alphas at once
        phi_kde = torch.sigmoid(tau * (alphas_exp - pit_exp)).mean(dim=0)  # (100,)
        pce_kde =  torch.abs(alphas - phi_kde).pow(p).mean()
        
        pce_kdes.append(pce_kde)
    pce_kdes = torch.stack(pce_kdes) #shape (d)
    if prerank == 'pca':
        explained_var = torch.from_numpy(_).to(pce_kdes.device)
        return (pce_kdes * explained_var).sum()
    else: 
        return pce_kdes.mean()