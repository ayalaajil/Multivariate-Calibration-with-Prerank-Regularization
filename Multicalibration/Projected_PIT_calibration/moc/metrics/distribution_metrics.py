import torch
from sklearn.decomposition import PCA

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

    return (term1 - term2).mean()  # (256,)

def calculate_PIT(samples, y, prerank):
    pits = []
    n = samples.shape[1]
    dim = y.shape[1]
    if prerank == 'identity':
        for d in range(dim):
            dsample = samples[:,:,d] #256,100
            dy = y[:,d] #256,1
            dsample_sorted = torch.sort(dsample)[0]
            cdfs = torch.searchsorted(dsample_sorted, dy.unsqueeze(-1), side='right') / n #256,1
            pits.append(cdfs)
    elif prerank == 'pca':
        pca = PCA(n_components=dim) #keeping all components, 4 in this case
        pca.fit(samples.reshape(-1,dim))
        vectors = pca.components_ #4 by 4
        # explained_var = pca.explained_variance_ #array of len 4
        for i in range(dim):
            u = torch.as_tensor(vectors[i], dtype=samples.dtype, device=samples.device)
            sample_proj = torch.matmul(samples, u) # 256,100,1
            y_proj = torch.matmul(y, u) #256,1
            sample_sorted = torch.sort(sample_proj)[0] #256,100 sort across the columns
            cdf_values = torch.searchsorted(sample_sorted, y_proj.unsqueeze(-1), side='right') / n #256,1
            pits.append(cdf_values)
    return torch.stack(pits)


def pce(dist, y, n_samples = 100, alphas = torch.linspace(0, 1, 100), mode = 'all', prerank = 'pca'):
    samples = dist.sample((n_samples,)).permute(1, 0, 2) #256,100,4
    pit_values = calculate_PIT(samples, y, prerank = prerank) #shape (4,256,1)
    dim = pit_values.shape[0]
    pces = []
    for d in range(dim):
        pits = pit_values[d].view(-1)  # shape: (256,)
        pits_sorted = pits.sort()[0]
        cdf_estimates = torch.searchsorted(pits_sorted, alphas, side='right') / pits_sorted.numel()
        pce = torch.mean(torch.abs(cdf_estimates - alphas)).item()
        pces.append(pce)
    if mode == 'average':
        return sum(pces)/len(pces)
    elif mode == 'all':
        return pces #it returns a list with 4 values, pce corresponding to each PCA
