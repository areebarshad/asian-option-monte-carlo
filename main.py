from definitons import MarketParameters
from path_generation import AsianOptionPricer

def main():
    params = MarketParameters(
        S0 = 100.0,
        K = 100.0,
        T = 1.0,
        r = 0.05,
        sigma = 0.2,
        M = 50,
        I = 100_000
    )

    print("\nMARKET PARAMETERS")
    print("-" * 80)
    print(f"Initial Price:                 ${params.S0:.2f}")
    print(f"Strike Price (K):              ${params.K:.2f}")
    print(f"Time to Maturity (T):          {params.T:.2f} years")
    print(f"Risk-Free Rate (r):            {params.r:.2%}")
    print(f"Volatility:                    {params.sigma:.2%}")
    print(f"Time Steps (M):                {params.M:,}")
    print(f"Monte Carlo Simulations (I):   {params.I:,}")
    pricer = AsianOptionPricer(params)
    results = pricer.run_analysis(seed = 42)

    return results

if __name__ == "__main__":
    results = main()
