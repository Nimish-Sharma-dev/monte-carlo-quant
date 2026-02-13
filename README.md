----Monte Carlo Quant Engine----
Overview:
    The Monte Carlo Quant Engine is a modular financial simulation system that models the future behavior of financial assets using stochastic processes. It estimates drift and volatility from historical market data, simulates thousands of possible future price paths using Geometric Brownian Motion, and analyzes the resulting distribution to measure risk and price derivatives. The system computes key quantitative metrics such as Value at Risk (VaR), Conditional Value at Risk (CVaR), expected returns, and portfolio loss distributions. It also prices European options via Monte Carlo simulation and validates results against the closed-form solution from the Black-Scholes model. With support for multi-asset correlated simulations, the project functions as a compact risk and derivatives research engine, demonstrating practical quantitative finance modeling, numerical methods, and structured software design.
Mathematical Foundation:
     GBM equation: St+1=St​e(μ−0.5σ2)Δt+σΔtZ
     VaR definition: VaRα=Percentile1−α​(Loss)
     Option payoff: max(ST​−K,0)
Features:
     GBM simulation
     Multi-asset correlated simulation
     VaR & CVaR
     Monte Carlo option pricing
     Black-Scholes validation
     Portfolio risk engine
     Unit tests
Installation:
    pip install -r requirements.txt
Example Usage:
    python main.py
Sample Outputs:
    Simulated paths
    Histogram
    VaR
    Option pricing comparison
Future Enhancements:
    Jump diffusion
    Stochastic volatility
    Variance reduction
    Greeks estimation
