# Monte Carlo Pricing Engine: Asian Options

A high-performance Python simulation engine that calculates the fair value of **Asian options**. It uses statistical variance reduction techniques to achieve high precesion in milliseconds.

## What is this project?

- In layman terms: **This is a calculator for a complex financial contract where no simple formula exists.**
- Standard European options depend on the stock price at the very end of the year, and can be calculated using the Black-Scholes formula.
- Meanwhile Asian options are different as their payoff depends on the **average** stock price over the entire year. As the average depends on the whole path, there is no simple formula to price them. Therefore, we have to simulate thousands of possible futures to find the answer.

## Key Features:

**Vectorized Simulation**: Instead of simulating one path at a time, the engine simulates 100,000+ paths simultaneously using NumPy, making it incredibly fast.
**Variance Reduction**: Uses a statistical technique called **Control Variate**:-
  - *Why*: Random simulations introduce a high error rate (noisy).
  - *Fix*: We use a simpler option (Geometric Asian) with a known price to calibrate our simulation.
  - *The Result*: This reduces the error rate by **~99%**, giving us the same accuracy with far less computing power.

## Mathematical Framework:

This engine leverages **Stochastic Calculus** and **Statistical Variance Reduction** to price the option.

### 1. Asset Price Dynamics:
We model the underlying stock price $S_t$ using **Geometric Brownian Motion (GBM)** under the Risk-Neutral measure. The stochastic differential equation (SDE) is:

$$dS_t = r S_t dt + \sigma S_t dW_t$$

For the simulation, we use the exact solution to this SDE discretized over time steps $\Delta t$:

$$S_{t+\Delta t} = S_t \exp\left((r - \frac{1}{2}\sigma^2)\Delta t + \sigma\sqrt{\Delta t}Z\right)$$

Where $Z \sim \mathcal{N}(0, 1)$ is a standard normal random variable.

### 2. The Payoff:
Unlike European options (which depend on $S_T$), an **Arithmetic Asian Option** depends on the average price over the path. The payoff function is:

$$Payoff = \max\left( \underbrace{\frac{1}{M} \sum_{i=1}^{M} S_{t_i}}_{\text{Arithmetic Mean}} - K, \;\; 0 \right)$$

Because the sum of log-normal variables (the Arithmetic Mean) has no closed-form distribution, we cannot use a simple formula like Black-Scholes. We must simulate it.

### 3. Variance Reduction (Optimization):
To solve the "noise" problem of Monte Carlo, we implement **Control Variates**.

We define two estimators:
* $Y$: The payoff of the **Arithmetic** Asian Option (Unknown).
* $X$: The payoff of the **Geometric** Asian Option (Known Analytical Solution via *Kemna-Vorst*).

Since $X$ and $Y$ are highly correlated ($\rho \approx 0.99$), we can use the error in $X$ to correct $Y$:

$$Y_{CV} = Y - \beta^* (X - \mathbb{E}[X])$$

Where the optimal coefficient $\beta^*$ is calculated dynamically during the simulation:

$$\beta^* = \frac{Cov(Y, X)}{Var(X)}$$

**The Intuition:** If our simulation accidentally overprices the Geometric option ($X > \mathbb{E}[X]$), it likely overpriced the Arithmetic option ($Y$) too. The formula subtracts this error, locking in the true price.

## Results Summary:

We ran the simulation with **100,000 paths** to compare the standard "Crude" Monte Carlo method against our "Control Variate" engine.

### Simulation Results:
| Metric | Crude Monte Carlo | Control Variate Monte Carlo | Improvement |
| :--- | :--- | :--- | :--- |
| **Estimated Price** | $5.720 | **$5.642** | Converged to True Value |
| **Standard Error** | $0.0250 | **$0.0007** | **~97% Reduction** |
| **95% Confidence Interval** | ± $0.0490 | **± $0.0014** | Extremely Tight Bounds |

### Efficiency Gain: 1,291x
The most critical metric is the **Efficiency Gain**. 

**The Result**: The Control Variate Monte Carlo method achieved a variance reduction factor of `0.0007`.
**The Implication**: To achieve this same level of precision using the old method, you would need to run **129,155,607 simulations**. 
**My Engine**: Achieved this precision with only **100,000 simulations**, making the code **~1,291 times more efficient** than a standard implementation.

### Author: Areeb Arshad | Virginia Tech | Data Science, Statistics, and Mathematics

### Notes:
- This model is for educational and research purposes only.
- It does not constitute financial advice or a production-ready trading system.
