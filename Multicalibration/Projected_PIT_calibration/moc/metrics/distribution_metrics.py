import torch
import torch.nn.functional as F
import numpy as np
import torch.nn.functional as F
from sklearn.decomposition import PCA
from .preranks import get_prerank


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
        # Some distributions (e.g. mixture) expose mixture/component attributes; use
        # them when available, otherwise fall back to generic sampling.
        if hasattr(dist, "mixture_distribution") and hasattr(dist, "component_distribution"):
            logits = dist.mixture_distribution.logits  # [B, K]
            weights = F.gumbel_softmax(logits.unsqueeze(1).expand(-1, n_samples, -1), tau=1.0, hard=False, dim=-1) #[B, n_samples, K]
            means = dist.component_distribution.loc  # [B, K, D]
            scales = dist.component_distribution.scale_tril  # [B, K, D, D]

            B, K, D = means.shape
            eps = torch.randn(B, n_samples, K, D, device=means.device)
            component_samples = torch.matmul(scales.unsqueeze(1), eps.unsqueeze(-1)).squeeze(-1) + means.unsqueeze(1)  # [B, n_samples, K, D]

            samples = (weights.unsqueeze(-1) * component_samples).sum(dim=2)  # [B, n_samples, D]
        else:
            # Fallback for generic distributions (e.g. HDRRecalibratedDistribution)
            samples = dist.sample((n_samples,))
            if samples.dim() >= 2:
                permute_order = [1, 0] + list(range(2, samples.dim()))
                samples = samples.permute(*permute_order)
    return samples

def mse(dist, y):
    means = dist.component_distribution.loc # (B, K, D) 256, 5, 4
    weights = dist.mixture_distribution.probs #(B, K)
    mixture_mean = torch.sum(weights.unsqueeze(-1) * means, dim=1) #B, D (e.g. 256,4)
    # samples = dist.sample((n_samples,)).permute(1, 0, 2)  # [B, n_samples, D]
    mse = ((mixture_mean - y)**2).mean()
    return mse


def multivariate_energy_score(dist, y, n_samples = 100):

    s1 = dist.sample((n_samples,)).permute(1, 0, 2) #256, 100, 16
    s2 = dist.sample((n_samples,)).permute(1, 0, 2) # 256, 100, 16

    y_expanded = y.unsqueeze(1)  # (256, 1, 16)
    term1 = torch.linalg.vector_norm(s1 - y_expanded, dim=-1).mean(dim=1)  # (256,)

    # Second term: 0.5 * E[||X - X'||]
    pairwise_dists = torch.linalg.vector_norm(s1.unsqueeze(2) - s2.unsqueeze(1), dim=-1)  # (256, 100, 100)
    term2 = 0.5 * pairwise_dists.mean(dim=(1, 2))  # (256,)

    return (term1 - term2).mean()

def empirical_cdf(dist, x, n_samples=100, tau = 100) -> torch.Tensor:
    r"""
    Args:
        x: (batch_size, d) 256, 3
    Returns:
        cdf_vals: (batch_size,) — \hat{F}_n(values[i]) for all i
    """

    samples = sample(dist, n_samples) #256, 100, 3
    # samples = dist.sample((n_samples,)).permute(1, 0, 2) # 256, 100, 3
    x = x.unsqueeze(1) # 256, 1, 3
    # comparison = samples <= values # (batch_size, n_samples, d)
    # tau=1
    # comparison =  torch.sigmoid(tau *(values-samples)).float() #torch.Size([10000, 256, 3]
    # dominated =comparison.prod(dim=2) # (batch_size, n_samples)
    # counts = dominated.mean(dim=0)      # (batch_size,)
    # cdf_vals = counts.float() / samples.shape[0]
    cdf_at_x = torch.sigmoid(tau * (x - samples)).prod(dim=2).mean(dim=1)
    return cdf_at_x

def calculate_PIT(dist, y, n_samples, prerank, setup = 'real', tau = 100):
    """
    Calculate projected PIT values.

    `prerank` can be:
      - a single prerank name (str)
      - 'combined' (aliases the default list below)
      - a list/tuple of prerank names to combine
    """
    batch_size, dim = y.shape
    if setup == "simulated":
        samples = dist.sample((n_samples,)).unsqueeze(0)
    else:
        samples = sample(dist, n_samples) #shape (256, 100, 4)
        # samples = dist.sample((n_samples,)).permute(1, 0, 2)

    # Normalise combined/list input
    if prerank == "combined":
        prerank_list = ["marginal", "mean", "variance", "dependency", "pca"]
    elif isinstance(prerank, (list, tuple)):
        prerank_list = list(prerank)
    else:
        prerank_list = [prerank]

    pits = []
    explained_vars = []

    for rho in prerank_list:

        if rho in ['mean', 'variance', 'dependency']:
            y_proj, samples_proj = get_prerank(y, samples, rho)

            cdfs = torch.sigmoid(tau *(y_proj.unsqueeze(-1) - samples_proj)).mean(dim=1, keepdim=True)

            # cdfs = (samples_proj <= y_proj.unsqueeze(1)).float().mean(dim=1, keepdim=True)

            pits.append(cdfs)
            explained_vars.append(np.ones(1))

        elif rho == 'marginal':

            for d in range(dim):
                dsample = samples[:,:,d] #256,100
                dy = y[:,d] #256
                cdfs =  torch.sigmoid(tau *(dy.unsqueeze(-1) - dsample)).mean(dim=1, keepdim=True)

                # cdfs = (dsample <= dy.unsqueeze(1)).float().mean(dim=1, keepdim=True)

                pits.append(cdfs)
                explained_vars.append(np.ones(1))

        elif rho == 'pca':
            samples_np = samples.detach().cpu().numpy().reshape(-1, y.shape[1])
            pca = PCA(n_components=None) #keeping all components, 4 in this case

            #pca = PCA(n_components=min(2, dim))  # keep only top 2 PCs

            pca.fit(samples_np)
            pca_dim = pca.n_components_
            vectors = torch.tensor(pca.components_, dtype=samples.dtype, device = samples.device) #4 by 4
            explained_var = pca.explained_variance_ratio_ #array of len 4

            for d in range(pca_dim):
                u = vectors[d]
                sample_proj = torch.matmul(samples, u) # 256,100
                y_proj = torch.matmul(y, u) #256

                cdfs =  torch.sigmoid(tau *(y_proj.unsqueeze(-1) - sample_proj)).mean(dim=1, keepdim=True)

                # cdfs = (sample_proj <= y_proj.unsqueeze(1)).float().mean(dim=1, keepdim=True)

                pits.append(cdfs)
                explained_vars.append(np.array([explained_var[d]]))

        elif rho =='density':    # it is the HDR calibration
            log_pdf_samples = []
            for i in range(n_samples):
                log_pdf = dist.log_prob(samples[:, i, :])
                log_pdf_samples.append(log_pdf) #256
            log_pdf_samples = torch.stack(log_pdf_samples).permute(1,0)
            log_pdf_y = dist.log_prob(y) #256

            cdfs =  torch.sigmoid(tau *(log_pdf_y.unsqueeze(-1) - log_pdf_samples)).mean(dim=1, keepdim=True)

            # cdfs = (log_pdf_samples <= log_pdf_y.unsqueeze(1)).float().mean(dim=1, keepdim=True)

            pits.append(cdfs)
            explained_vars.append(np.ones(1))

        elif rho == 'cdf':
            #samples 256, 100, 3
            # cdfs = torch.sigmoid(tau * (y.unsqueeze(1) - samples)).prod(dim=2).mean(dim=1, keepdim=True) # 256, 1
            cdfs_samples = []
            for i in range(n_samples):
                cdf_sample = empirical_cdf(dist,samples[:, i, :])  # 256
                cdfs_samples.append(cdf_sample)
            cdfs_samples = torch.stack(cdfs_samples).permute(1,0)  # shape: (256, 100)
            cdfs_y = empirical_cdf(dist, y)
            # Apply sigmoid smoothing around CDF difference

            prerank_cdfs = torch.sigmoid(tau * (cdfs_y.unsqueeze(1) - cdfs_samples)).mean(dim=1, keepdim=True)  # shape: (256,1)
            # prerank_cdfs = (cdfs_samples <= cdfs_y.unsqueeze(1)).float().mean(dim=1, keepdim=True)

            pits.append(prerank_cdfs)
            explained_vars.append(np.ones(1))
        else:
            raise ValueError(f"Unknown prerank function: {rho}")

    # Flatten explained variance list; currently only used for pca weighting when needed.
    if explained_vars:
        explained_var = np.concatenate(explained_vars)
    else:
        explained_var = np.ones(dim) * (1/dim)

    return torch.stack(pits), explained_var


def pce(dist, y, n_samples = 100, prerank = 'pca', setup = 'real'):
    alphas = torch.linspace(0, 1, 100, device=y.device)
    pit_values, _ = calculate_PIT(dist, y, n_samples = n_samples, prerank = prerank, setup=setup) #shape (4,256,1) or (1, 256,1)
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
    # if prerank == 'pca':
    #     explained_var = torch.from_numpy(_).to(pces.device)
    #     return (pces * explained_var).sum(), cdfs
    return pces, cdfs
