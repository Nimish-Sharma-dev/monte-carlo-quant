----Monte Carlo Quant Engine----<br>
Overview:<br>
    The Monte Carlo Quant Engine is a modular financial simulation system that models the future behavior of financial assets using stochastic processes. It estimates drift and volatility from historical market data, simulates thousands of possible future price paths using Geometric Brownian Motion, and analyzes the resulting distribution to measure risk and price derivatives. The system computes key quantitative metrics such as Value at Risk (VaR), Conditional Value at Risk (CVaR), expected returns, and portfolio loss distributions. It also prices European options via Monte Carlo simulation and validates results against the closed-form solution from the Black-Scholes model. With support for multi-asset correlated simulations, the project functions as a compact risk and derivatives research engine, demonstrating practical quantitative finance modeling, numerical methods, and structured software design.<br>
Mathematical Foundation:<br>
     GBM equation: St+1=St​e(μ−0.5σ2)Δt+σΔtZ<br>
     VaR definition: VaRα=Percentile1−α​(Loss)<br>
     Option payoff: max(ST​−K,0)<br>
Features:<br>
     GBM simulation<br>
     Multi-asset correlated simulation<br>
     VaR & CVaR<br>
     Monte Carlo option pricing<br>
     Black-Scholes validation<br>
     Portfolio risk engine<br>
     Unit tests<br>
Installation:<br>
    pip install -r requirements.txt<br>
Example Usage:<br>
    python main.py<br>
Sample Outputs:<br>
    Simulated paths<br>
    Histogram<br>
    VaR<br>
    Option pricing comparison<br>
Future Enhancements:<br>
    Jump diffusion<br>
    Stochastic volatility<br>
    Variance reduction<br>
    Greeks estimation<br>
