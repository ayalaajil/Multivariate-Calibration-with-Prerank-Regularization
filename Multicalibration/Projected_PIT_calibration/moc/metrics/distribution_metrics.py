import torch
import numpy as np
from sklearn.decomposition import PCA
from .preranks import get_prerank
torch.manual_seed(42)

def nll(model, x, y):
    dist = model.predict(x)
    return -dist.log_prob(y).detach()


def kernel_score_from_samples(y, s1, s2, kernel):
    """
    Returns the kernel score evaluated in `y` using the samples `s1` and `s2`.
    `s1` and `s2` are tensors of shape (n_samples, b, d).
    `y` is a tensor of shape (..., b, d), where the first dimensions are arbitrary 
    and will be evaluated for the same batch element.
    `kernel` is a callable that takes two broadcastable tensors of shape (..., d) and returns a tensor of shape (...,).
    """

    n_samples, b, d = s1.shape
    assert s1.shape == s2.shape
    assert y.shape[-2:] == (b, d)

    first_term = kernel(
        s1.unsqueeze(-3),
        s2.unsqueeze(-4),
    ).mean(dim=(-3, -2)) #has shape (256,)

    second_term = kernel(
        s1,
        y.unsqueeze(-3),
    ).mean(dim=-2) #has shape (256,)

    return 0.5 * first_term - second_term #has shape (256,)


def energy_score_from_samples(y, s1, s2, beta):
    def kernel(y1, y2):
        return torch.linalg.vector_norm(y1 - y2, dim=-1) ** beta
    return kernel_score_from_samples(y, s1, s2, kernel)



def sample(dist, n_samples, rsample):
    if rsample:
        return dist.rsample(n_samples)
    else:
        return dist.sample(n_samples)


def energy_score(dist, y, n_samples=100, beta=2., rsample=False):
    s1 = sample(dist, (n_samples,), rsample)
    s2 = sample(dist, (n_samples,), rsample)
    return energy_score_from_samples(y, s1, s2, beta)

def multivariate_energy_score(dist, y, n_samples = 100):

    s1 = dist.sample((n_samples,)).permute(1, 0, 2) #256, 100, 16
    s2 = dist.sample((n_samples,)).permute(1, 0, 2) # 256, 100, 16

    y_expanded = y.unsqueeze(1)  # (256, 1, 16)
    term1 = torch.linalg.vector_norm(s1 - y_expanded, dim=-1).mean(dim=1)  # (256,)

    # Second term: 0.5 * E[||X - X'||]
    pairwise_dists = torch.linalg.vector_norm(s1.unsqueeze(2) - s2.unsqueeze(1), dim=-1)  # (256, 100, 100)
    term2 = 0.5 * pairwise_dists.mean(dim=(1, 2))  # (256,)

    return (term1 - term2).mean()

def calculate_PIT(dist, y, n_samples, setup, prerank, tau = 100):
    batch_size, dim = y.shape
    if setup == 'simulated':
        samples = dist.sample((batch_size*n_samples,)).reshape(batch_size, n_samples, dim) #10000, 1000, 10
    else: 
        samples = dist.rsample((n_samples,)).permute(1, 0, 2) #256,20,4
    pits = []
    explained_var = np.ones(dim) * (1/dim)
    if prerank in ['mean', 'variance', 'dependency']:
        y_proj, samples_proj = get_prerank(y, samples, prerank)
        cdfs = torch.sigmoid(tau *(y_proj.unsqueeze(-1) - samples_proj)).mean(dim=1, keepdim=True)
        pits.append(cdfs)
    elif prerank == 'marginal':
        for d in range(dim):
            dsample = samples[:,:,d] #256,100
            dy = y[:,d] #256
            cdfs =  torch.sigmoid(tau *(dy.unsqueeze(-1) - dsample)).mean(dim=1, keepdim=True)
            pits.append(cdfs)
    elif prerank == 'pca':
        samples_np = samples.detach().cpu().numpy().reshape(-1, y.shape[1])
        pca = PCA(n_components=dim) #keeping all components, 4 in this case
        pca.fit(samples_np) #this will not work in gpu
        vectors = torch.tensor(pca.components_, dtype=samples.dtype, device = samples.device) #4 by 4
        explained_var = pca.explained_variance_ratio_ #array of len 4
        for d in range(dim):
            u = vectors[d]
            sample_proj = torch.matmul(samples, u) # 256,100
            y_proj = torch.matmul(y, u) #256
            cdfs =  torch.sigmoid(tau *(y_proj.unsqueeze(-1) - sample_proj)).mean(dim=1, keepdim=True)
            pits.append(cdfs)
    elif prerank =='density':
        #samples are of shape 256, 100, 4
        log_densities_samples = []
        for i in range(n_samples):
            log_density = dist.log_prob(samples[:, i, :]) #10000,10
            log_densities_samples.append(log_density) #256
        log_densities_samples = torch.stack(log_densities_samples).permute(1,0)
        # log_densities_samples = dist.log_prob(samples)#256,100
        log_densities_y = dist.log_prob(y) #256
        cdfs =  torch.sigmoid(tau *(log_densities_y.unsqueeze(-1) -log_densities_samples)).mean(dim=1, keepdim=True)
        pits.append(cdfs)
    else:
        raise ValueError(f"Unknown prerank function: {prerank}")
    return torch.stack(pits), explained_var

def pce(dist, y, n_samples = 100, prerank = 'pca', setup = 'real', mode = 'train'):
    alphas = torch.linspace(0, 1, 100, device=y.device)
    pit_values, _ = calculate_PIT(dist, y, n_samples = n_samples, setup = setup, prerank = prerank) #shape (4,256,1) or (1, 256,1)
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
            return pces, explained_var
        else: return pces.mean()
    else: return pces, cdfs, _
    