import numpy as np
from scipy.stats import rankdata, multivariate_normal
import torch

def get_prerank(y, x, prerank, **kwargs):

    B, M, d = x.shape
    
    # Check if prerank is a string (built-in function) or callable (custom function)
    if isinstance(prerank, str):
        if prerank == "mean":
            prerank_func = lambda z: torch.mean(z, dim=-1)
        elif prerank == "variance":
            prerank_func = lambda z: torch.var(z, dim=-1, unbiased=False)
        elif prerank == "dependency":
            prerank_func = lambda z: rho_dep(z, **kwargs)
        else:
            raise ValueError(f"Unknown prerank function: {prerank}")
    else:
        # User-defined function
        prerank_func = prerank
    
    # Calculate pre-ranks
    obs_prerank = prerank_func(y) #256 values
    sample_preranks = prerank_func(x) #256,100
    
    return obs_prerank, sample_preranks
    

def rho_dep(z, h=1):
    """
    z: (B, d) tensor of observations or (B, N, d) tensor of samples
    h: lag
    Returns:
        rho_dep: tensor of dependency pre-rank values of shape either (B,) or (B, N)
    """
    d = z.shape[-1]
    if h >= d:
        raise ValueError("Lag h must be less than dimension d")

    # Variogram numerator: mean squared difference at lag h
    def compute_gamma(tensor):  # (B, d) or (B, N, d)
        diff = tensor[..., :-h] - tensor[..., h:]
        gamma = (diff ** 2).mean(dim=-1) / 2  # shape (B) or (B, N)
        return gamma

    # Variance denominator
    var = torch.var(z, dim=-1, unbiased=False)

    return - compute_gamma(z) / var