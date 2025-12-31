from typing import Dict
import time

def run_analysis(self, seed: int = 42) -> Dict[str, float]:
        """
        Run complete pricing analysis with both methods.
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