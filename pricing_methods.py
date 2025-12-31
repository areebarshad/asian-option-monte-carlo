from typing import Tuple, Dict
import numpy as np

def price_monte_carlo(self) -> Tuple[float, float]:
    """
    Price the Arithmetic Asian Call Option using Monte Carlo simulation
    """
    arith_payoff, _ = self.compute_payoffs()
    discounted_payoffs = self.discount_factor * arith_payoff

    price = np.mean(discounted_payoffs)

    std_dev = np.std(discounted_payoffs, ddof = 1)
    std_error = std_dev / np.sqrt(self.params.I)

    return price, std_error

def price_control_variate(self) -> Tuple[float, float, float, Dict[str, float]]:
    """
    Price the Arithmetic Asian Call Option using Control Variate variance reduction
    """
    arith_payoff, geom_payoff = self.compute_payoffs()
    Y = self.discount_factor * arith_payoff
    X = self.discount_factor * geom_payoff
    mu_X = self.analytical_geometric_asian()

    covariance = np.cov(Y, X)[0, 1]
    variance_X = np.var(X, ddof = 1)
    beta_optimal = covariance / variance_X

    Y_CV = Y - beta_optimal * (X - mu_X)
    price = np.mean(Y_CV)
    std_error = np.std(Y_CV, ddof = 1) / np.sqrt(self.params.I)

    correlation = covariance / (np.std(Y, ddof = 1) * np.std(X, ddof = 1))
    variance_reduction = 1 - correlation ** 2

    diagnostics = {
        'beta': beta_optimal,
        'correlation': correlation,
        'variance_reduction_factor': variance_reduction,
        'control_mean': mu_X,
        'crude_variance': np.var(Y, ddof = 1),
        'cv_variance': np.var(Y_CV, ddof = 1)
    }

    return price, std_error, beta_optimal, diagnostics