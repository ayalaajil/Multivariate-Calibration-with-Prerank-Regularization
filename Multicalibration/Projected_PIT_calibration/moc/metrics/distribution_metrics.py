import torch
import torch.nn.functional as F
import numpy as np
import torch.nn.functional as F
def nll(model, x, y):
    dist = model.predict(x)
    return -dist.log_prob(y).detach()


def sample(dist, n_samples):
    r"""
        B - batch size,
        K - the number of Gaussian mixture components,
        D - the dimensionality of each Gaussian component (for example 4D or 16D)
        n_samples - number of samples to draw
    """
    if dist.has_rsample: 
        samples = dist.rsample((n_samples,)).permute(1, 0, 2) # [B,n_samples,D] very important to use rsample() here
    else:
        logits = dist.mixture_distribution.logits  # [B, K]
        weights = F.gumbel_softmax(logits.unsqueeze(1).expand(-1, n_samples, -1), tau=1.0, hard=False, dim=-1) #[B, n_samples, K]
        means = dist.component_distribution.loc  # [B, K, D]
        scales = dist.component_distribution.scale_tril  # [B, K, D, D]

        B, K, D = means.shape
        eps = torch.randn(B, n_samples, K, D, device=means.device)
        component_samples = torch.matmul(scales.unsqueeze(1), eps.unsqueeze(-1)).squeeze(-1) + means.unsqueeze(1)  # [B, n_samples, K, D]

        samples = (weights.unsqueeze(-1) * component_samples).sum(dim=2)  # [B, n_samples, D]
    return samples


def multivariate_energy_score(dist, y, n_samples = 100):

    s1 = dist.sample((n_samples,)).permute(1, 0, 2) #256, 100, 16
    s2 = dist.sample((n_samples,)).permute(1, 0, 2) # 256, 100, 16

    y_expanded = y.unsqueeze(1)  # (256, 1, 16)
    term1 = torch.linalg.vector_norm(s1 - y_expanded, dim=-1).mean(dim=1)  # (256,)

    # Second term: 0.5 * E[||X - X'||]
    pairwise_dists = torch.linalg.vector_norm(s1.unsqueeze(2) - s2.unsqueeze(1), dim=-1)  # (256, 100, 100)
    term2 = 0.5 * pairwise_dists.mean(dim=(1, 2))  # (256,)

    return (term1 - term2).mean()

def empirical_cdf(dist, values: torch.Tensor, n_samples=10_000) -> torch.Tensor:
    """
    Args:
        values: (batch_size, d)
        samples: (n_samples, d)

    Returns:
        cdf_vals: (batch_size,) — \hat{F}_n(values[i]) for all i
    """

    samples = sample(dist,n_samples).permute(1, 0, 2) 
    comparison = samples <= values # (batch_size, n_samples, d)
    tau=1
    comparison =  torch.sigmoid(tau *(values-samples)).float() #torch.Size([10000, 256, 3]
    print("haha")
    print(comparison.shape)
    dominated =comparison.prod(dim=2) # (batch_size, n_samples)
    print(dominated.shape)
    counts = dominated.mean(dim=0)      # (batch_size,)
    print(counts.shape)
    cdf_vals = counts.float() / samples.shape[0]
    return cdf_vals

def calculate_PIT(dist, y, n_samples, prerank):
    batch_size, dim = y.shape
    '''if setup == 'simulated':
        samples = dist.sample((batch_size*n_samples,)).reshape(batch_size, n_samples, dim) #10000, 1000, 10  #HEEEERE rsample
        print("1")
    else: 
        samples = dist.sample((n_samples,)).permute(1, 0, 2) #256,20,4 # HEEERE rsample
        print("2")'''
    samples = sample(dist, n_samples)
    pits = []
    explained_var = np.ones(dim) * (1/dim)
    if prerank in ['mean', 'variance', 'dependency']:
        y_proj, samples_proj = get_prerank(y, samples, prerank)
        sorted_samples_proj = torch.sort(samples_proj, dim=1)[0] #256,100
        cdfs = torch.searchsorted(sorted_samples_proj, y_proj.unsqueeze(-1), side='right') / n_samples #256,1
        pits.append(cdfs)
    elif prerank == 'marginal':
        for d in range(dim):
            dsample = samples[:,:,d] #256,100
            dy = y[:,d] #256
            dsample_sorted = torch.sort(dsample, dim=1)[0]
            cdfs = torch.searchsorted(dsample_sorted.contiguous(), 
                                      dy.unsqueeze(-1).contiguous(), side='right') / n_samples #256,1
            pits.append(cdfs)
    elif prerank == 'pca':
        samples_np = samples.detach().cpu().numpy().reshape(-1, y.shape[1])
        pca = PCA(n_components=dim) #keeping all components, 4 in this case
        pca.fit(samples_np) #this will not work in gpu
        vectors = torch.tensor(pca.components_, dtype=samples.dtype, device = samples.device) #4 by 4
        explained_var = pca.explained_variance_ratio_ #array of len 4
        for d in range(dim):
            u = vectors[d]
            sample_proj = torch.matmul(samples, u) # 256,100,1
            y_proj = torch.matmul(y, u) #256,1
            sample_sorted = torch.sort(sample_proj, dim=1)[0] #256,100 sort across the columns
            cdfs = torch.searchsorted(sample_sorted.contiguous(), 
                                      y_proj.unsqueeze(-1).contiguous(), side='right') / n_samples #256,1
            pits.append(cdfs)
    elif prerank =='density':
        #print(samples.shape) #torch.Size([256, 100, 3])
        log_densities_samples = []
        for i in range(n_samples):
            log_density = dist.log_prob(samples[:, i, :]) #torch.Size([256])
            print(log_density.shape)
            log_densities_samples.append(log_density) #256
        
        log_densities_samples = torch.stack(log_densities_samples).permute(1,0) #torch.Size([106, 100]) or torch.Size([256, 100])
        #print(log_densities_samples.shape) #torch.Size([106, 100])
        # log_densities_samples = dist.log_prob(samples)#256,100
        log_densities_y = dist.log_prob(y) #256
        '''cdfs = (log_densities_samples <= log_densities_y.unsqueeze(1)).float().mean(dim=1, keepdim=True) #256,1'''
        tau=100
        cdfs =  torch.sigmoid(tau *(log_densities_y.unsqueeze(1) -log_densities_samples)).float().mean(dim=1, keepdim=True)
        pits.append(cdfs)

    elif prerank == 'cdf':
        # samples: (256, 100, 4)
        # y: (256, 4)
        cdfs_samples = []
        for i in range(n_samples):
            print(samples[:, i, :].shape) # torch.Size([256, 3])
            cdf_sample = empirical_cdf(dist,samples[:, i, :])  # torch.Size([256])
            cdfs_samples.append(cdf_sample)
        cdfs_samples = torch.stack(cdfs_samples).permute(1,0)  # shape: (256, 100)
        cdfs_y = empirical_cdf(dist, y)  # shape: (256,)
        tau = 100
        # Apply sigmoid smoothing around CDF difference
        prerank_cdfs = torch.sigmoid(tau * (cdfs_y.unsqueeze(1) - cdfs_samples)).mean(dim=1, keepdim=True)  # shape: (256,1)
        pits.append(prerank_cdfs)
    else:
        raise ValueError(f"Unknown prerank function: {prerank}")
    return torch.stack(pits), explained_var


def pce(dist, y, n_samples = 100, prerank = 'pca', mode = 'train'):
    alphas = torch.linspace(0, 1, 100, device=y.device)
    pit_values, _ = calculate_PIT(dist, y, n_samples = n_samples, prerank = prerank) #shape (4,256,1) or (1, 256,1)
    dim = pit_values.shape[0]
    pces = []
    cdfs = []
    for d in range(dim):
        pits = pit_values[d].view(-1)  # shape: (256,)
        pits_sorted = pits.sort()[0]
        cdf_estimates = torch.searchsorted(pits_sorted, alphas, side='right') / pits_sorted.numel()
        pce = torch.mean(torch.abs(cdf_estimates - alphas))
        pces.append(pce)
        cdfs.append(cdf_estimates)
    pces = torch.stack(pces)
    cdfs = torch.stack(cdfs)
    if mode == 'train':
        if prerank == 'pca':
            explained_var = torch.from_numpy(_).to(pces.device)
            return (pces * explained_var).sum()
        else: return pces.mean()
    else: return pces, cdfs, _
    