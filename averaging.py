import numpy as np
from typing import Tuple

def compute_arithmetic_average(self) -> np.ndarray:
    """
    Compute the arithmetic average of the asset prices along each path.
    """
    if self._arithmetic_avg is None:
        self._arithmetic_avg = np.mean(self.paths, axis = 0)
    return self._arithmetic_avg

def compute_geometric_average(self) -> np.ndarray:
    """
    Compute the geometric average of the asset prices along each path.
    """
    if self._geometric_avg is None:
        log_avg = np.mean(np.log(self.paths), axis = 0)
        self._geometric_avg = np.exp(log_avg)
    return self._geometric_avg

def compute_payoffs(self) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the payoffs for arithmetic and geometric average Asian options.
    """
    arith_avg = self.compute_arithmetic_average()
    geom_avg = self.compute_geometric_average()
    
    arith_payoff = np.maximum(arith_avg - self.params.K, 0)
    geom_payoff = np.maximum(geom_avg - self.params.K, 0)

    return arith_payoff, geom_payoff