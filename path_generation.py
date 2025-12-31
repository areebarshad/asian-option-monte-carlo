from definitons import MarketParameters
import numpy as np

class AsianOptionPricer:
    """
    Monte Carlo pricer for Arithmetic Asian Call Options.
    This class implements a vectorized GBM path generator.
    Uses the Geometric Asian Option as a control variate to reduce variance.
    """
    def __init__(self, params: MarketParameters):
        self.params = params
        self.dt = self.params.T / self.params.M
        self.discount_factor = np.exp(-params.r * params.T)

        self.drift = (params.r - 0.5 * params.sigma ** 2) * self.dt
        self.diffusion = params.sigma * np.sqrt(self.dt)

        self._paths = None
        self._arithmetic_avg = None
        self._geometric_avg = None
    
    def generate_paths(self, seed: int = 42) -> np.ndarray:
        """
        Generate asset price paths using vectorized GBM.
        """
        np.random.seed(seed)
        Z = np.random.standard_normal((self.params.M, self.params.I))
        log_returns = self.drift + self.diffusion * Z

        paths = np.zeros((self.params.M + 1, self.params.I))
        paths[0] = self.params.S0

        paths[1:] = self.params.S0 * np.exp(np.cumsum(log_returns, axis=0))

        self._paths = paths
        return paths
    
    def paths(self) -> np.ndarray:
        if self._paths is None:
            self.generate_paths()
        return self._paths