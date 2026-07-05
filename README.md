# Monte Carlo Pricing Engine: Asian Options

A high-performance Python simulation engine that calculates the fair value of **Asian options**. It uses statistical variance reduction techniques to achieve high precision in milliseconds — and is now served as a production REST API.

## What is this project?

**In layman terms: this is a calculator for a complex financial contract where no simple formula exists.**

Standard European options depend on the stock price at the very end of the contract period and can be priced with the closed-form Black-Scholes formula. Asian options are fundamentally different — their payoff depends on the **average** stock price over the entire life of the contract. Because the arithmetic average of log-normal prices has no closed-form distribution, the only rigorous approach is to simulate thousands of possible futures and average the outcomes.

## Key Features

**Vectorized Simulation** — Instead of simulating one path at a time, the engine generates 100,000+ paths simultaneously using NumPy, making it orders of magnitude faster than naive loop-based implementations.

**Control Variate Variance Reduction** — Uses a statistical technique to suppress noise in the simulation:
- *Why*: Random simulations introduce a high standard error.
- *Fix*: A simpler option (the Geometric Asian) has a known analytical price via the Kemna-Vorst formula. We use the error in that known quantity to calibrate our unknown one.
- *Result*: ~97% reduction in standard error — the same accuracy with ~1,291× less compute.

**Production REST API** — The engine is wrapped in a FastAPI service and deployed to Render. Any application can price an option or retrieve Greeks over HTTP with a single POST request.

## Mathematical Framework

### 1. Asset Price Dynamics

The underlying stock price $S_t$ is modelled using **Geometric Brownian Motion (GBM)** under the risk-neutral measure:

$$dS_t = r S_t \, dt + \sigma S_t \, dW_t$$

For simulation, we use the exact log-normal solution discretized over time steps $\Delta t$:

$$S_{t+\Delta t} = S_t \exp\!\left(\!\left(r - \tfrac{1}{2}\sigma^2\right)\!\Delta t + \sigma\sqrt{\Delta t}\,Z\right), \quad Z \sim \mathcal{N}(0,1)$$

The entire path matrix — shape $(M \times I)$ for $M$ steps and $I$ simulations — is generated in a single vectorized operation.

### 2. The Payoff

An **Arithmetic Asian Call** pays off based on the time-averaged price rather than the terminal price:

$$\text{Payoff} = \max\!\left(\underbrace{\frac{1}{M}\sum_{i=1}^{M} S_{t_i}}_{\text{Arithmetic Mean}} - K,\; 0\right)$$

Because the arithmetic mean of log-normal variables has no closed-form distribution, we cannot price it analytically — we must simulate.

### 3. Variance Reduction via Control Variates

To suppress Monte Carlo noise we implement **Control Variates**.

Define two correlated estimators:
- $Y$ — discounted payoff of the **Arithmetic** Asian (target, unknown expectation)
- $X$ — discounted payoff of the **Geometric** Asian (known via Kemna-Vorst: $\mu_X = \mathbb{E}[X]$)

Since $\rho(X, Y) \approx 0.99$, we can correct $Y$ using the measured error in $X$:

$$Y_{CV} = Y - \beta^*(X - \mathbb{E}[X])$$

The optimal coefficient is estimated from the simulation itself:

$$\beta^* = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$$

**Intuition:** If the simulation accidentally overprices the Geometric option ($X > \mathbb{E}[X]$), it almost certainly overpriced the Arithmetic option too. The formula subtracts that correlated error, anchoring the estimate to the known analytical benchmark.

At optimal $\beta^*$, the variance collapses to:

$$\text{Var}(Y_{CV}) = \text{Var}(Y)\left(1 - \rho^2\right)$$

With $\rho \approx 0.99$, this eliminates roughly 98% of the original variance.

## Results

Simulation run with 100,000 paths, $S_0 = 100$, $K = 100$, $T = 1$ yr, $r = 5\%$, $\sigma = 20\%$, $M = 252$ steps.

| Metric | Crude Monte Carlo | Control Variate MC | Improvement |
| :--- | :--- | :--- | :--- |
| **Estimated Price** | $5.720 | **$5.642** | Converged to true value |
| **Standard Error** | $0.0250 | **$0.0007** | ~97% reduction |
| **95% Confidence Interval** | ± $0.0490 | **± $0.0014** | Extremely tight bounds |

**Efficiency Gain: ~1,291×** — To match the Control Variate's precision using Crude MC alone would require **129,155,607 simulations** instead of 100,000.

## REST API

The pricing engine is deployed as a FastAPI service on Render. Three endpoints are available.

### `GET /health`

```
200 OK
{"status": "ok"}
```

### `POST /price`

Prices an Asian option using the Control Variate Monte Carlo method.

**Request**
```json
{
  "S0": 100.0,
  "K": 100.0,
  "T": 1.0,
  "r": 0.05,
  "sigma": 0.2,
  "n_steps": 252,
  "n_simulations": 100000,
  "option_type": "call"
}
```

**Response**
```json
{
  "price": 5.642,
  "std_error": 0.0007,
  "ci_lower": 5.641,
  "ci_upper": 5.643,
  "method": "control_variate"
}
```

### `POST /greeks`

Same request body as `/price`. Returns Delta, Gamma, Vega, Theta, and Rho computed via central finite differences (bump-and-reprice).

**Response**
```json
{
  "delta": 0.52,
  "gamma": 0.03,
  "vega": 0.18,
  "theta": -0.012,
  "rho": 0.09
}
```

Greeks are computed by bumping one parameter at a time and repricing:

| Greek | Parameter | Bump size |
| :--- | :--- | :--- |
| Delta | $S_0$ | ±1% of $S_0$ |
| Gamma | $S_0$ | ±1% of $S_0$ |
| Vega | $\sigma$ | ±0.01 |
| Theta | $T$ | ±1/365 yr |
| Rho | $r$ | ±0.0001 |

All bumped calls share the same random seed so the identical path matrix is reused — the common noise cancels in the difference quotient, yielding accurate numerical derivatives without inflating the simulation count.

## Deployment

The service deploys to [Render](https://render.com) via `render.yaml`.

<br>

#### Author: Areeb Arshad | Virginia Tech | Data Science, Statistics, and Mathematics

<br>

#### Notes
- This model is for educational and research purposes only.
- It does not constitute financial advice or a production-ready trading system.
