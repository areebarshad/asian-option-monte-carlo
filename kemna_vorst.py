import numpy as np
from scipy import stats

def analytical_geometric_asian(self) -> float:
    """
    Compute the analytical price of a Geometric Asian Call Option using Kemna-Vorst formula.
    """
    S0 = self.params.S0
    K = self.params.K
    T = self.params.T
    r = self.params.r
    sigma = self.params.sigma
    M = self.params.M

    N = M + 1
    sigma_adj_sq = (sigma ** 2 / 6) * ((N + 1) * (2 * N + 1)) / (N ** 2)
    sigma_adj = np.sqrt(sigma_adj_sq)

    mu_adj = 0.5 * (r - 0.5 * sigma ** 2 + 0.5 * sigma_adj_sq)

    F = S0 * np.exp(mu_adj * T)

    discount = np.exp(-r * T)

    d1 = (np.log(F / K) + 0.5 * sigma_adj_sq * T) / (sigma_adj * np.sqrt(T))
    d2 = d1 - sigma_adj * np.sqrt(T)

    call_price = discount * (F * stats.norm_cdf(d1) - K * stats.norm_cdf(d2))
    return call_price