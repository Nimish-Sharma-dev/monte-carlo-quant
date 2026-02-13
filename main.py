import yaml
import numpy as np
import matplotlib.pyplot as plt

from src.data.data_loader import DataLoader
from src.simulation.gbm import GBMSimulator
from src.risk.risk_metrics import RiskMetrics
from src.pricing.monte_carlo_pricer import MonteCarloPricer
from src.pricing.black_scholes import BlackScholes
from src.visualization.plotter import Plotter


# -----------------------------------
# Load YAML Configuration
# -----------------------------------
def load_config():
    with open("config/config.yaml", "r") as file:
        return yaml.safe_load(file)


# -----------------------------------
# Convergence Plot
# -----------------------------------
def convergence_plot(payoffs, bs_price):

    running_prices = []
    simulation_counts = []

    cumulative_sum = 0.0

    for i in range(len(payoffs)):
        cumulative_sum += payoffs[i]
        current_price = cumulative_sum / (i + 1)

        if (i + 1) % 100 == 0:
            running_prices.append(current_price)
            simulation_counts.append(i + 1)

    plt.figure()
    plt.plot(simulation_counts, running_prices)
    plt.axhline(y=bs_price)
    plt.xlabel("Number of Simulations")
    plt.ylabel("Monte Carlo Price")
    plt.title("Monte Carlo Convergence")
    plt.show()


# -----------------------------------
# MAIN EXECUTION
# -----------------------------------
def main():

    config = load_config()

    # -----------------------------------
    # 1. Load Historical Data
    # -----------------------------------
    loader = DataLoader(
        config["ticker"],
        config["start_date"],
        config["end_date"]
    )

    params = loader.estimate_parameters()

    mu = params["mu"]
    sigma = params["sigma"]
    S0 = params["last_price"]

    print(f"Estimated μ: {mu}")
    print(f"Estimated σ: {sigma}")
    print(f"Current Price S0: {S0}")

    # Extract pricing params
    r = config["risk_free_rate"]
    T = config["horizon_years"]
    K = config["strike_price"]

    # -----------------------------------
    # 2. Real-World Simulation (Risk)
    # -----------------------------------
    simulator_real = GBMSimulator(
        S0=S0,
        drift=mu,
        sigma=sigma,
        T=T,
        steps=config["time_steps"],
        simulations=config["simulations"]
    )

    paths_real = simulator_real.simulate()
    print(f"Simulated paths shape: {paths_real.shape}")

    # -----------------------------------
    # 3. Risk Metrics
    # -----------------------------------
    risk = RiskMetrics(config["confidence_level"])
    var = risk.compute_var(paths_real)
    cvar = risk.compute_cvar(paths_real)

    print(f"VaR ({config['confidence_level']*100}%): {var}")
    print(f"CVaR: {cvar}")

    # -----------------------------------
    # 4. Risk-Neutral Simulation (Pricing)
    # -----------------------------------
    simulator_rn = GBMSimulator(
        S0=S0,
        drift=r,  # risk-neutral drift
        sigma=sigma,
        T=T,
        steps=config["time_steps"],
        simulations=config["simulations"]
    )

    paths_rn = simulator_rn.simulate(antithetic=True)


    # Terminal prices (last row)
    terminal_prices = paths_rn[-1, :]

    # Discounted Payoffs
    payoffs = np.exp(-r * T) * np.maximum(terminal_prices - K, 0)

    # -----------------------------------
    # 5. Monte Carlo Pricing
    # -----------------------------------
    mc_pricer = MonteCarloPricer(r, K)
    mc_results = mc_pricer.price_call(terminal_prices, T)

    # -----------------------------------
    # 6. Black-Scholes Pricing
    # -----------------------------------
    bs = BlackScholes(S0, K, r, sigma, T)
    bs_call = bs.call_price()

    print(f"Monte Carlo Call Price: {mc_results['price']}")
    print(f"Standard Error: {mc_results['std_error']}")
    print(f"95% CI: [{mc_results['ci_lower']}, {mc_results['ci_upper']}]")
    print(f"Black-Scholes Call Price: {bs_call}")

    # -----------------------------------
    # 7. Visualizations
    # -----------------------------------
    Plotter.plot_price_paths(paths_real, num_paths=100)
    Plotter.plot_terminal_distribution(terminal_prices, var)
    convergence_plot(payoffs, bs_call)


if __name__ == "__main__":
    main()
