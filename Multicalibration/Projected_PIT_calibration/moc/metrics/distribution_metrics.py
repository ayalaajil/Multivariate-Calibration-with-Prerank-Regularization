import torch


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
