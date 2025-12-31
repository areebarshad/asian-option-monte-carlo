from dataclasses import dataclass

@dataclass
class MarketParameters:
    """
    Attributes:
        S0: Initial asset price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized)
        M: Number of time steps
        I: Number of Monte Carlo simulations
    """
    S0: float
    K: float
    T: float
    r: float
    sigma: float
    M: int
    I: int
    
    def __post_init__(self):
        assert self.S0 > 0, "Initial asset price must be positive."
        assert self.K > 0, "Strike price must be positive." 
        assert self.T > 0, "Time to maturity must be positive."
        assert self.sigma > 0, "Volatility must be positive."
        assert self.M > 0, "Number of time steps must be positive."
        assert self.I > 0, "Number of simulations must be positive."

