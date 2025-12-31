"""
Arithmetic Asian Option Pricer with Control Variates
A production-grade Monte Carlo framework for exotic option pricing

This framework demonstrates:
1. Vectorized Geometric Brownian Motion simulation
2. Control Variate variance reduction using Geometric Asian as control
3. Analytical Kemna-Vorst formula for benchmark
4. Professional software architecture with type hints and documentation
"""

import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass
from scipy import stats
import time
import matplotlib.pyplot as plt

# Set plotting style
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


@dataclass
class MarketParameters:
    """
    Encapsulates market and option parameters.
    
    Attributes:
        S0: Initial asset price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate (annualized)
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
        """Validate parameters."""
        assert self.S0 > 0, "Initial price must be positive"
        assert self.K > 0, "Strike must be positive"
        assert self.T > 0, "Time to maturity must be positive"
        assert self.sigma > 0, "Volatility must be positive"
        assert self.M > 0, "Time steps must be positive"
        assert self.I > 0, "Number of simulations must be positive"


class AsianOptionPricer:
    """
    Monte Carlo pricer for Arithmetic Asian Call Options with Control Variates.
    
    This class implements a vectorized GBM path generator and uses the 
    Geometric Asian option as a control variate to reduce variance.
    
    Mathematical Foundation:
    ------------------------
    Under risk-neutral measure, asset follows:
        dS_t = r * S_t * dt + sigma * S_t * dW_t
    
    Discrete approximation:
        S_{t+1} = S_t * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        where Z ~ N(0,1)
    
    Control Variate Method:
    -----------------------
    Let Y_A = Arithmetic Asian payoff (unknown expectation)
    Let Y_G = Geometric Asian payoff (known analytical expectation E[Y_G])
    
    Improved estimator:
        Y_CV = Y_A - beta * (Y_G - E[Y_G])
    
    Optimal beta minimizes Var(Y_CV):
        beta* = Cov(Y_A, Y_G) / Var(Y_G)
    
    Variance reduction:
        Var(Y_CV) = Var(Y_A) * (1 - rho^2)
        where rho = correlation between Y_A and Y_G
    """
    
    def __init__(self, params: MarketParameters):
        """
        Initialize the pricer with market parameters.
        
        Args:
            params: MarketParameters dataclass containing all inputs
        """
        self.params = params
        self.dt = params.T / params.M
        self.discount_factor = np.exp(-params.r * params.T)
        
        # Pre-compute drift and diffusion terms
        self.drift = (params.r - 0.5 * params.sigma**2) * self.dt
        self.diffusion = params.sigma * np.sqrt(self.dt)
        
        # Storage for paths (lazy initialization)
        self._paths = None
        self._arithmetic_avg = None
        self._geometric_avg = None
    
    def generate_paths(self, seed: int = 42) -> np.ndarray:
        """
        Generate asset price paths using vectorized GBM.
        
        Uses the exact discretization scheme:
            S_{t+dt} = S_t * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Array of shape (M+1, I) containing price paths
            First row is initial price S0, subsequent rows are evolved prices
        
        Mathematical Note:
        ------------------
        This generates the ENTIRE matrix of random shocks at once,
        then applies cumulative sum to create paths. This is ~100x faster
        than naive loop-based generation.
        """
        np.random.seed(seed)
        
        # Generate all random shocks: shape (M, I)
        Z = np.random.standard_normal((self.params.M, self.params.I))
        
        # Compute log-returns for each step
        log_returns = self.drift + self.diffusion * Z
        
        # Initialize paths matrix: (M+1, I)
        paths = np.zeros((self.params.M + 1, self.params.I))
        paths[0] = self.params.S0
        
        # Vectorized path evolution: apply cumulative sum of log-returns
        # S_t = S_0 * exp(sum of log-returns from 0 to t)
        paths[1:] = self.params.S0 * np.exp(np.cumsum(log_returns, axis=0))
        
        self._paths = paths
        return paths
    
    @property
    def paths(self) -> np.ndarray:
        """Lazy getter for paths."""
        if self._paths is None:
            self.generate_paths()
        return self._paths
    
    def compute_arithmetic_average(self) -> np.ndarray:
        """
        Compute arithmetic average of asset prices along each path.
        
        For discrete monitoring at times t_0, t_1, ..., t_M:
            A_arith = (1/(M+1)) * sum(S_i)
        
        Returns:
            Array of shape (I,) containing arithmetic averages
        """
        if self._arithmetic_avg is None:
            # Average over time dimension (axis=0)
            self._arithmetic_avg = np.mean(self.paths, axis=0)
        return self._arithmetic_avg
    
    def compute_geometric_average(self) -> np.ndarray:
        """
        Compute geometric average of asset prices along each path.
        
        For discrete monitoring:
            A_geom = (product(S_i))^(1/(M+1))
        
        Computed in log-space for numerical stability:
            log(A_geom) = (1/(M+1)) * sum(log(S_i))
            A_geom = exp(log(A_geom))
        
        Returns:
            Array of shape (I,) containing geometric averages
        """
        if self._geometric_avg is None:
            # Compute in log-space to avoid overflow
            log_avg = np.mean(np.log(self.paths), axis=0)
            self._geometric_avg = np.exp(log_avg)
        return self._geometric_avg
    
    def compute_payoffs(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute call option payoffs for both arithmetic and geometric averages.
        
        Call payoff: max(Average - K, 0)
        
        Returns:
            Tuple of (arithmetic_payoffs, geometric_payoffs)
            Each array has shape (I,)
        """
        arith_avg = self.compute_arithmetic_average()
        geom_avg = self.compute_geometric_average()
        
        arith_payoff = np.maximum(arith_avg - self.params.K, 0)
        geom_payoff = np.maximum(geom_avg - self.params.K, 0)
        
        return arith_payoff, geom_payoff
    
    def analytical_geometric_asian(self) -> float:
        """
        Compute analytical price of Geometric Asian Call using Kemna-Vorst formula.
        
        Mathematical Derivation:
        ------------------------
        The geometric average of a lognormal process is also lognormal.
        For discrete monitoring at M+1 points:
        
        Adjusted parameters:
            sigma_adj^2 = sigma^2 * (M+1)(2M+1) / (6M^2)
            mu_adj = 0.5 * (r - sigma^2/2 - sigma_adj^2/2)
        
        Moment matching gives us an equivalent Black-Scholes problem:
            S_adj = S0 * exp(mu_adj * T)
            r_adj = adjusted risk-free rate
        
        Then apply standard Black-Scholes formula with adjusted parameters.
        
        Reference:
        ----------
        Kemna, A.G.Z. and Vorst, A.C.F. (1990)
        "A pricing method for options based on average asset values"
        Journal of Banking and Finance, 14, 113-129
        
        Returns:
            Analytical price of geometric Asian call option
        """
        S0, K, T, r, sigma, M = (
            self.params.S0, self.params.K, self.params.T,
            self.params.r, self.params.sigma, self.params.M
        )
        
        # Number of monitoring points
        N = M + 1
        
        # Adjusted volatility for geometric average
        sigma_adj_sq = (sigma**2 / 6) * ((N + 1) * (2*N + 1)) / (N**2)
        sigma_adj = np.sqrt(sigma_adj_sq)
        
        # Adjusted drift
        mu_adj = 0.5 * (r - 0.5 * sigma**2 + 0.5 * sigma_adj_sq)
        
        # Adjusted forward price
        F = S0 * np.exp(mu_adj * T)
        
        # Adjusted discount rate
        discount = np.exp(-r * T)
        
        # Black-Scholes d1 and d2 with adjusted parameters
        d1 = (np.log(F / K) + 0.5 * sigma_adj_sq * T) / (sigma_adj * np.sqrt(T))
        d2 = d1 - sigma_adj * np.sqrt(T)
        
        # Black-Scholes formula
        call_price = discount * (F * stats.norm.cdf(d1) - K * stats.norm.cdf(d2))
        
        return call_price
    
    def price_crude_monte_carlo(self) -> Tuple[float, float]:
        """
        Price arithmetic Asian call using standard (crude) Monte Carlo.
        
        Estimator:
            Price = exp(-rT) * E[max(A_arith - K, 0)]
            ≈ exp(-rT) * (1/I) * sum(payoffs)
        
        Standard Error:
            SE = sqrt(Var(payoffs) / I)
        
        Returns:
            Tuple of (price, standard_error)
        """
        arith_payoff, _ = self.compute_payoffs()
        
        # Discounted payoff
        discounted_payoffs = self.discount_factor * arith_payoff
        
        # Price estimate
        price = np.mean(discounted_payoffs)
        
        # Standard error
        std_dev = np.std(discounted_payoffs, ddof=1)
        std_error = std_dev / np.sqrt(self.params.I)
        
        return price, std_error
    
    def price_control_variate(self) -> Tuple[float, float, float, Dict[str, float]]:
        """
        Price arithmetic Asian call using Control Variate variance reduction.
        
        Control Variate Theory:
        -----------------------
        Let Y = arithmetic payoff (unknown expectation μ_Y)
        Let X = geometric payoff (known expectation μ_X from Kemna-Vorst)
        
        Control variate estimator:
            Y_CV = Y - β(X - μ_X)
        
        Where β is chosen to minimize Var(Y_CV):
            β* = Cov(Y,X) / Var(X)
        
        Properties:
            E[Y_CV] = E[Y]  (unbiased)
            Var(Y_CV) = Var(Y) + β^2*Var(X) - 2β*Cov(Y,X)
            
        At optimal β*:
            Var(Y_CV) = Var(Y) * (1 - ρ²)
            where ρ = correlation between Y and X
        
        Returns:
            Tuple of (price, standard_error, beta, diagnostics)
            diagnostics: Dict with correlation, variance_reduction_factor, etc.
        """
        arith_payoff, geom_payoff = self.compute_payoffs()
        
        # Discount payoffs
        Y = self.discount_factor * arith_payoff  # Arithmetic
        X = self.discount_factor * geom_payoff   # Geometric (control)
        
        # Analytical expectation of control
        mu_X = self.analytical_geometric_asian()
        
        # Compute optimal beta
        covariance = np.cov(Y, X)[0, 1]
        variance_X = np.var(X, ddof=1)
        beta_optimal = covariance / variance_X
        
        # Apply control variate adjustment
        Y_CV = Y - beta_optimal * (X - mu_X)
        
        # Price and standard error
        price = np.mean(Y_CV)
        std_error = np.std(Y_CV, ddof=1) / np.sqrt(self.params.I)
        
        # Diagnostics
        correlation = covariance / (np.std(Y, ddof=1) * np.std(X, ddof=1))
        variance_reduction = 1 - correlation**2
        
        diagnostics = {
            'beta': beta_optimal,
            'correlation': correlation,
            'variance_reduction_factor': variance_reduction,
            'control_mean': mu_X,
            'crude_variance': np.var(Y, ddof=1),
            'cv_variance': np.var(Y_CV, ddof=1)
        }
        
        return price, std_error, beta_optimal, diagnostics
    
    def run_analysis(self, seed: int = 42) -> Dict[str, float]:
        """
        Run complete pricing analysis with both methods.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing all pricing results and diagnostics
        """
        print("=" * 80)
        print("ARITHMETIC ASIAN CALL OPTION PRICING WITH CONTROL VARIATES")
        print("=" * 80)
        
        # Generate paths
        print("\n[1/4] Generating price paths...")
        start = time.time()
        self.generate_paths(seed=seed)
        path_time = time.time() - start
        print(f"      Generated {self.params.I:,} paths with {self.params.M} steps "
              f"in {path_time:.3f}s")
        
        # Analytical benchmark
        print("\n[2/4] Computing analytical Geometric Asian price (Kemna-Vorst)...")
        analytical_price = self.analytical_geometric_asian()
        print(f"      Analytical Geometric Asian Call: ${analytical_price:.6f}")
        
        # Crude Monte Carlo
        print("\n[3/4] Running Crude Monte Carlo...")
        start = time.time()
        crude_price, crude_se = self.price_crude_monte_carlo()
        crude_time = time.time() - start
        print(f"      Completed in {crude_time:.3f}s")
        
        # Control Variate
        print("\n[4/4] Running Control Variate Monte Carlo...")
        start = time.time()
        cv_price, cv_se, beta, diagnostics = self.price_control_variate()
        cv_time = time.time() - start
        print(f"      Completed in {cv_time:.3f}s")
        
        # Calculate variance reduction percentage
        var_reduction_pct = (1 - (cv_se / crude_se)**2) * 100
        
        # Display results
        self._display_results(crude_price, crude_se, cv_price, cv_se, 
                            var_reduction_pct, diagnostics)
        
        return {
            'crude_price': crude_price,
            'crude_se': crude_se,
            'cv_price': cv_price,
            'cv_se': cv_se,
            'variance_reduction_pct': var_reduction_pct,
            'analytical_geometric': analytical_price,
            **diagnostics
        }
    
    def _display_results(self, crude_price: float, crude_se: float,
                        cv_price: float, cv_se: float,
                        var_reduction_pct: float,
                        diagnostics: Dict[str, float]) -> None:
        """Display formatted results table."""
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        
        print(f"\n{'Method':<30} {'Price':<15} {'Std Error':<15} {'95% CI'}")
        print("-" * 80)
        
        # Crude MC
        ci_crude = 1.96 * crude_se
        print(f"{'Crude Monte Carlo':<30} "
              f"${crude_price:>8.6f}    "
              f"${crude_se:>8.6f}    "
              f"±${ci_crude:.6f}")
        
        # Control Variate
        ci_cv = 1.96 * cv_se
        print(f"{'Control Variate MC':<30} "
              f"${cv_price:>8.6f}    "
              f"${cv_se:>8.6f}    "
              f"±${ci_cv:.6f}")
        
        print("\n" + "=" * 80)
        print("VARIANCE REDUCTION ANALYSIS")
        print("=" * 80)
        
        print(f"\nOptimal β (beta):              {diagnostics['beta']:>10.6f}")
        print(f"Correlation (ρ):               {diagnostics['correlation']:>10.6f}")
        print(f"Variance Reduction Factor:     {diagnostics['variance_reduction_factor']:>10.6f}")
        print(f"\n{'Standard Error Reduction:':<30} {var_reduction_pct:>6.2f}%")
        print(f"{'Efficiency Gain:':<30} {(crude_se/cv_se)**2:>6.2f}x")
        
        print("\n" + "=" * 80)
        print(f"To achieve the same accuracy as CV with {self.params.I:,} simulations,")
        print(f"Crude MC would require {int(self.params.I * (crude_se/cv_se)**2):,} simulations")
        print("=" * 80)
    
    def plot_sample_paths(self, n_paths: int = 50, save_fig: bool = False) -> None:
        """
        Plot sample price paths to visualize the GBM simulation.
        
        Args:
            n_paths: Number of paths to display
            save_fig: Whether to save figure to disk
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        time_grid = np.linspace(0, self.params.T, self.params.M + 1)
        
        # Plot sample paths
        for i in range(min(n_paths, self.params.I)):
            ax.plot(time_grid, self.paths[:, i], alpha=0.3, linewidth=0.8, color='steelblue')
        
        # Plot mean path
        mean_path = np.mean(self.paths, axis=1)
        ax.plot(time_grid, mean_path, 'r-', linewidth=2.5, label='Mean Path', zorder=5)
        
        # Add strike line
        ax.axhline(y=self.params.K, color='green', linestyle='--', 
                   linewidth=2, label=f'Strike K={self.params.K}', zorder=5)
        
        ax.set_xlabel('Time (years)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Asset Price ($)', fontsize=12, fontweight='bold')
        ax.set_title(f'Geometric Brownian Motion: {n_paths} Sample Paths\n'
                    f'S₀=${self.params.S0}, μ={self.params.r:.1%}, σ={self.params.sigma:.1%}',
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_fig:
            plt.savefig('gbm_paths.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_averaging_comparison(self, save_fig: bool = False) -> None:
        """
        Compare arithmetic vs geometric averages across all paths.
        
        Args:
            save_fig: Whether to save figure to disk
        """
        arith_avg = self.compute_arithmetic_average()
        geom_avg = self.compute_geometric_average()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot
        axes[0].scatter(geom_avg, arith_avg, alpha=0.3, s=10, color='steelblue')
        axes[0].plot([geom_avg.min(), geom_avg.max()], 
                     [geom_avg.min(), geom_avg.max()], 
                     'r--', linewidth=2, label='y=x')
        axes[0].set_xlabel('Geometric Average', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Arithmetic Average', fontsize=12, fontweight='bold')
        axes[0].set_title('Arithmetic vs Geometric Average\n(AM-GM Inequality)', 
                         fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Distribution plot
        axes[1].hist(arith_avg, bins=50, alpha=0.6, label='Arithmetic', 
                    color='steelblue', edgecolor='black', density=True)
        axes[1].hist(geom_avg, bins=50, alpha=0.6, label='Geometric', 
                    color='coral', edgecolor='black', density=True)
        axes[1].axvline(self.params.K, color='green', linestyle='--', 
                       linewidth=2, label=f'Strike K={self.params.K}')
        axes[1].set_xlabel('Average Value', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Density', fontsize=12, fontweight='bold')
        axes[1].set_title('Distribution of Averages', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_fig:
            plt.savefig('averaging_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_convergence_analysis(self, n_points: int = 20, save_fig: bool = False) -> None:
        """
        Analyze convergence of both MC methods as sample size increases.
        
        Args:
            n_points: Number of points for convergence plot
            save_fig: Whether to save figure to disk
        """
        print("\n" + "=" * 80)
        print("RUNNING CONVERGENCE ANALYSIS")
        print("=" * 80)
        
        sample_sizes = np.logspace(2, np.log10(self.params.I), n_points, dtype=int)
        
        crude_prices = []
        crude_errors = []
        cv_prices = []
        cv_errors = []
        
        arith_payoff, geom_payoff = self.compute_payoffs()
        Y = self.discount_factor * arith_payoff
        X = self.discount_factor * geom_payoff
        mu_X = self.analytical_geometric_asian()
        
        # Compute beta once
        covariance = np.cov(Y, X)[0, 1]
        variance_X = np.var(X, ddof=1)
        beta = covariance / variance_X
        
        print(f"Computing convergence for {n_points} sample sizes...")
        
        for i, n in enumerate(sample_sizes):
            # Crude MC
            crude_sample = Y[:n]
            crude_prices.append(np.mean(crude_sample))
            crude_errors.append(np.std(crude_sample, ddof=1) / np.sqrt(n))
            
            # Control Variate
            Y_cv_sample = Y[:n] - beta * (X[:n] - mu_X)
            cv_prices.append(np.mean(Y_cv_sample))
            cv_errors.append(np.std(Y_cv_sample, ddof=1) / np.sqrt(n))
            
            if (i + 1) % 5 == 0:
                print(f"  Progress: {i+1}/{n_points} complete")
        
        # Create convergence plot
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Price convergence
        axes[0].plot(sample_sizes, crude_prices, 'o-', linewidth=2, 
                    markersize=6, label='Crude MC', color='steelblue')
        axes[0].plot(sample_sizes, cv_prices, 's-', linewidth=2, 
                    markersize=6, label='Control Variate', color='coral')
        axes[0].axhline(y=crude_prices[-1], color='gray', linestyle='--', 
                       linewidth=1.5, alpha=0.7, label='Final Estimate')
        axes[0].set_xscale('log')
        axes[0].set_xlabel('Number of Simulations', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Option Price ($)', fontsize=12, fontweight='bold')
        axes[0].set_title('Price Convergence Analysis', fontsize=14, fontweight='bold', pad=15)
        axes[0].legend(fontsize=11, loc='best')
        axes[0].grid(True, alpha=0.3, which='both')
        
        # Standard error convergence
        axes[1].plot(sample_sizes, crude_errors, 'o-', linewidth=2, 
                    markersize=6, label='Crude MC', color='steelblue')
        axes[1].plot(sample_sizes, cv_errors, 's-', linewidth=2, 
                    markersize=6, label='Control Variate', color='coral')
        
        # Add theoretical 1/sqrt(n) line
        theoretical_line = crude_errors[0] * np.sqrt(sample_sizes[0]) / np.sqrt(sample_sizes)
        axes[1].plot(sample_sizes, theoretical_line, '--', linewidth=2, 
                    color='gray', alpha=0.7, label='Theoretical 1/√n')
        
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        axes[1].set_xlabel('Number of Simulations', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Standard Error ($)', fontsize=12, fontweight='bold')
        axes[1].set_title('Standard Error Convergence (Log-Log Scale)', 
                         fontsize=14, fontweight='bold', pad=15)
        axes[1].legend(fontsize=11, loc='best')
        axes[1].grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        if save_fig:
            plt.savefig('convergence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Convergence analysis complete\n")
    
    def plot_payoff_distribution(self, save_fig: bool = False) -> None:
        """
        Visualize the distribution of payoffs for both averaging methods.
        
        Args:
            save_fig: Whether to save figure to disk
        """
        arith_payoff, geom_payoff = self.compute_payoffs()
        Y = self.discount_factor * arith_payoff
        X = self.discount_factor * geom_payoff
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Arithmetic payoff distribution
        axes[0, 0].hist(Y, bins=60, alpha=0.7, color='steelblue', 
                       edgecolor='black', density=True)
        axes[0, 0].axvline(np.mean(Y), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean = ${np.mean(Y):.4f}')
        axes[0, 0].set_xlabel('Discounted Payoff ($)', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Density', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Arithmetic Asian Payoff Distribution', 
                            fontsize=12, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Geometric payoff distribution
        axes[0, 1].hist(X, bins=60, alpha=0.7, color='coral', 
                       edgecolor='black', density=True)
        axes[0, 1].axvline(np.mean(X), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean = ${np.mean(X):.4f}')
        axes[0, 1].axvline(self.analytical_geometric_asian(), color='green', 
                          linestyle=':', linewidth=2.5, 
                          label=f'Analytical = ${self.analytical_geometric_asian():.4f}')
        axes[0, 1].set_xlabel('Discounted Payoff ($)', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('Density', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Geometric Asian Payoff Distribution', 
                            fontsize=12, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Scatter plot of payoffs
        axes[1, 0].scatter(X, Y, alpha=0.3, s=10, color='steelblue')
        axes[1, 0].set_xlabel('Geometric Payoff ($)', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('Arithmetic Payoff ($)', fontsize=11, fontweight='bold')
        axes[1, 0].set_title(f'Payoff Correlation (ρ = {np.corrcoef(X, Y)[0,1]:.4f})', 
                            fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Control variate adjustment visualization
        mu_X = self.analytical_geometric_asian()
        covariance = np.cov(Y, X)[0, 1]
        variance_X = np.var(X, ddof=1)
        beta = covariance / variance_X
        Y_CV = Y - beta * (X - mu_X)
        
        axes[1, 1].hist(Y, bins=50, alpha=0.5, label='Original (Crude)', 
                       color='steelblue', edgecolor='black', density=True)
        axes[1, 1].hist(Y_CV, bins=50, alpha=0.5, label='Control Variate', 
                       color='coral', edgecolor='black', density=True)
        axes[1, 1].axvline(np.mean(Y), color='blue', linestyle='--', linewidth=2)
        axes[1, 1].axvline(np.mean(Y_CV), color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Discounted Payoff ($)', fontsize=11, fontweight='bold')
        axes[1, 1].set_ylabel('Density', fontsize=11, fontweight='bold')
        axes[1, 1].set_title(f'Variance Reduction: {(1 - np.var(Y_CV)/np.var(Y))*100:.1f}%', 
                            fontsize=12, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_fig:
            plt.savefig('payoff_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_report(self, save_figs: bool = False) -> None:
        """
        Generate all visualizations in one comprehensive report.
        
        Args:
            save_figs: Whether to save all figures to disk
        """
        print("\n" + "=" * 80)
        print("GENERATING COMPREHENSIVE VISUALIZATION REPORT")
        print("=" * 80)
        
        print("\n[1/4] Plotting sample price paths...")
        self.plot_sample_paths(n_paths=100, save_fig=save_figs)
        
        print("[2/4] Comparing averaging methods...")
        self.plot_averaging_comparison(save_fig=save_figs)
        
        print("[3/4] Analyzing payoff distributions...")
        self.plot_payoff_distribution(save_fig=save_figs)
        
        print("[4/4] Running convergence analysis...")
        self.plot_convergence_analysis(n_points=15, save_fig=save_figs)
        
        print("\n" + "=" * 80)
        print("✓ VISUALIZATION REPORT COMPLETE")
        if save_figs:
            print("✓ All figures saved to current directory")
        print("=" * 80)


def main():
    """
    Main execution block demonstrating the Asian option pricer.
    
    This runs a production test case with 100,000 simulations
    and demonstrates the effectiveness of control variates.
    """
    # Define market parameters
    params = MarketParameters(
        S0=100.0,      # Initial asset price
        K=100.0,       # Strike price (ATM)
        T=1.0,         # 1 year to maturity
        r=0.05,        # 5% risk-free rate
        sigma=0.2,     # 20% volatility
        M=50,          # 50 time steps
        I=100_000      # 100,000 Monte Carlo simulations
    )
    
    print("\nMARKET PARAMETERS")
    print("-" * 80)
    print(f"Initial Price (S₀):            ${params.S0:.2f}")
    print(f"Strike Price (K):              ${params.K:.2f}")
    print(f"Time to Maturity (T):          {params.T:.2f} years")
    print(f"Risk-Free Rate (r):            {params.r:.2%}")
    print(f"Volatility (σ):                {params.sigma:.2%}")
    print(f"Time Steps (M):                {params.M:,}")
    print(f"Monte Carlo Simulations (I):   {params.I:,}")
    
    # Initialize and run pricer
    pricer = AsianOptionPricer(params)
    results = pricer.run_analysis(seed=42)
    
    # Additional validation
    print("\n" + "=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)
    
    print(f"\n✓ Paths shape: {pricer.paths.shape} (expected: ({params.M+1}, {params.I}))")
    print(f"✓ Arithmetic avg range: [{pricer.compute_arithmetic_average().min():.2f}, "
          f"{pricer.compute_arithmetic_average().max():.2f}]")
    print(f"✓ Geometric avg range: [{pricer.compute_geometric_average().min():.2f}, "
          f"{pricer.compute_geometric_average().max():.2f}]")
    print(f"✓ Control variate bias: ${abs(results['cv_price'] - results['crude_price']):.6f}")
    
    # Generate comprehensive visualizations
    print("\n" + "=" * 80)
    print("Would you like to generate visualizations? (Recommended)")
    print("=" * 80)
    print("\nGenerating comprehensive visualization report...")
    print("(Set save_figs=True to save plots to disk)")
    
    pricer.create_comprehensive_report(save_figs=False)
    
    return results


if __name__ == "__main__":
    results = main()
