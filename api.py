"""
FastAPI REST API wrapper for the Asian Option Monte Carlo Pricer.
Exposes /health, /price, and /greeks endpoints.
"""

# Force headless matplotlib backend before any import touches pyplot
import matplotlib
matplotlib.use("Agg")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

from asian_option_pricer import AsianOptionPricer, MarketParameters


app = FastAPI(
    title="Asian Option Monte Carlo Pricer",
    description="Prices arithmetic Asian options via Control Variate Monte Carlo (Kemna-Vorst).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PriceRequest(BaseModel):
    S0: float = Field(..., gt=0, description="Initial asset price")
    K: float = Field(..., gt=0, description="Strike price")
    T: float = Field(..., gt=0, description="Time to maturity in years")
    r: float = Field(..., description="Continuously compounded risk-free rate")
    sigma: float = Field(..., gt=0, description="Annualised volatility")
    n_steps: int = Field(..., gt=0, description="Number of time steps per path")
    n_simulations: int = Field(
        ..., ge=1000, le=500_000, description="Monte Carlo simulation count"
    )
    option_type: str = Field(
        ..., pattern="^(call|put)$", description="'call' or 'put'"
    )


class PriceResponse(BaseModel):
    price: float
    std_error: float
    ci_lower: float
    ci_upper: float
    method: str


class GreeksResponse(BaseModel):
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


# ---------------------------------------------------------------------------
# Core pricing helper
# ---------------------------------------------------------------------------

def _price_one(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
    n_simulations: int,
    option_type: str,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Price an Asian option with Control Variate Monte Carlo.

    Uses the same Z-matrix (fixed seed) across all calls so that finite-difference
    Greeks cancel the common random noise, giving accurate numerical derivatives.

    Returns
    -------
    (price, std_error)
    """
    mp = MarketParameters(
        S0=S0, K=K, T=T, r=r, sigma=sigma, M=n_steps, I=n_simulations
    )
    pricer = AsianOptionPricer(mp)
    pricer.generate_paths(seed=seed)

    arith_avg = pricer.compute_arithmetic_average()
    geom_avg = pricer.compute_geometric_average()
    discount = pricer.discount_factor

    if option_type == "call":
        Y = discount * np.maximum(arith_avg - K, 0.0)
        X = discount * np.maximum(geom_avg - K, 0.0)
        mu_X = pricer.analytical_geometric_asian()
    else:
        Y = discount * np.maximum(K - arith_avg, 0.0)
        X = discount * np.maximum(K - geom_avg, 0.0)
        # Analytical geometric Asian put via put-call parity on the Kemna-Vorst price:
        # P_geom = C_geom - discount*(F - K)
        # where F = S0 * exp(mu_adj * T) is the Kemna-Vorst adjusted forward.
        call_kv = pricer.analytical_geometric_asian()
        N_pts = mp.M + 1
        sigma_adj_sq = (mp.sigma ** 2 / 6) * ((N_pts + 1) * (2 * N_pts + 1)) / (N_pts ** 2)
        mu_adj = 0.5 * (mp.r - 0.5 * mp.sigma ** 2 + 0.5 * sigma_adj_sq)
        F = mp.S0 * np.exp(mu_adj * mp.T)
        mu_X = call_kv - discount * (F - mp.K)

    cov = np.cov(Y, X)[0, 1]
    var_X = float(np.var(X, ddof=1))

    if var_X < 1e-14:
        # Control variate degenerates (all paths deep OTM); fall back to crude MC.
        price_val = float(np.mean(Y))
        se = float(np.std(Y, ddof=1) / np.sqrt(n_simulations))
    else:
        beta = cov / var_X
        Y_CV = Y - beta * (X - mu_X)
        price_val = float(np.mean(Y_CV))
        se = float(np.std(Y_CV, ddof=1) / np.sqrt(n_simulations))

    return price_val, se


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/price", response_model=PriceResponse)
def price(req: PriceRequest):
    price_val, se = _price_one(
        S0=req.S0,
        K=req.K,
        T=req.T,
        r=req.r,
        sigma=req.sigma,
        n_steps=req.n_steps,
        n_simulations=req.n_simulations,
        option_type=req.option_type,
    )
    return PriceResponse(
        price=round(price_val, 6),
        std_error=round(se, 6),
        ci_lower=round(price_val - 1.96 * se, 6),
        ci_upper=round(price_val + 1.96 * se, 6),
        method="control_variate",
    )


@app.post("/greeks", response_model=GreeksResponse)
def greeks(req: PriceRequest):
    """
    Compute option Greeks via central finite differences (bump-and-reprice).

    All bumped prices share seed=42, so the identical Z-matrix is reused and
    random noise cancels in the difference quotients.

    Bump sizes
    ----------
    Delta / Gamma : ±1 % of S0
    Vega          : ±0.01 (absolute vol)
    Theta         : ±1/365 years  (capped at T/3 for short-dated options)
    Rho           : ±0.0001 (absolute rate)
    """
    h_S = req.S0 * 0.01
    h_sigma = 0.01
    h_r = 0.0001
    # Guard against T - h_T <= 0 for very short-dated options
    h_T = min(1.0 / 365.0, req.T / 3.0)

    def _p(S0=None, K=None, T=None, r=None, sigma=None) -> float:
        return _price_one(
            S0=S0 if S0 is not None else req.S0,
            K=K if K is not None else req.K,
            T=T if T is not None else req.T,
            r=r if r is not None else req.r,
            sigma=sigma if sigma is not None else req.sigma,
            n_steps=req.n_steps,
            n_simulations=req.n_simulations,
            option_type=req.option_type,
        )[0]

    # Compute all bumped prices; p_base is shared by Gamma.
    p_base = _p()
    p_up_S = _p(S0=req.S0 + h_S)
    p_dn_S = _p(S0=req.S0 - h_S)
    p_up_v = _p(sigma=req.sigma + h_sigma)
    p_dn_v = _p(sigma=req.sigma - h_sigma)
    p_up_T = _p(T=req.T + h_T)
    p_dn_T = _p(T=req.T - h_T)
    p_up_r = _p(r=req.r + h_r)
    p_dn_r = _p(r=req.r - h_r)

    delta = (p_up_S - p_dn_S) / (2.0 * h_S)
    gamma = (p_up_S - 2.0 * p_base + p_dn_S) / (h_S ** 2)
    vega = (p_up_v - p_dn_v) / (2.0 * h_sigma)
    # Theta: rate of value change as calendar time passes (T shrinks), hence negative sign.
    theta = -(p_up_T - p_dn_T) / (2.0 * h_T)
    rho = (p_up_r - p_dn_r) / (2.0 * h_r)

    return GreeksResponse(
        delta=round(delta, 6),
        gamma=round(gamma, 6),
        vega=round(vega, 6),
        theta=round(theta, 6),
        rho=round(rho, 6),
    )
