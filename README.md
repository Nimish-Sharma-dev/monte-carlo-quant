# Monte Carlo Quant Engine

A modular Monte Carlo simulation engine for option pricing and risk analysis.

## Overview

This project implements:

- Geometric Brownian Motion (GBM) simulation
- European Call Option Pricing (Monte Carlo)
- Black-Scholes analytical pricing
- Variance Reduction (Control Variates)
- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)
- Confidence Interval estimation
- Convergence diagnostics

The system is structured in a modular research-oriented architecture.

---

## Project Structure


monte-carlo-quant/
│
├── main.py # Orchestrator
├── config.py # Model parameters
├── simulation.py # GBM path generation
├── pricing.py # MC + Black-Scholes pricing
├── risk.py # VaR / CVaR calculations
└── README.md


---

## Installation

Create virtual environment:


python -m venv venv


Activate it:

Windows:

venv\Scripts\activate


Mac/Linux:

source venv/bin/activate


Install dependencies:


pip install numpy scipy pandas matplotlib yfinance


---

## Running the Program

From project root:


python main.py


---

## Output Example

- Estimated μ and σ
- Simulated path shape
- VaR (95%)
- CVaR
- Monte Carlo Call Price
- Standard Error
- 95% Confidence Interval
- Black-Scholes Price
- Control Variate Price (if enabled)

---

## Methods Implemented

### Monte Carlo Pricing
Risk-neutral GBM simulation of terminal stock price.

### Black-Scholes Benchmark
Closed-form analytical solution.

### Variance Reduction
Control variates using discounted stock price.

### Risk Metrics
- VaR (95%)
- CVaR (Expected Shortfall)

---

## Future Extensions

- Antithetic Variates
- Sobol Quasi-Monte Carlo
- Greeks estimation
- Multi-asset simulation
- American option pricing (LSM)