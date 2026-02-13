import yaml
from src.data.data_loader import DataLoader
from src.simulation.gbm import GBMSimulator
from src.risk.risk_metrics import RiskMetrics
from src.pricing.monte_carlo_pricer import MonteCarloPricer
from src.pricing.black_scholes import BlackScholes
from src.visualization.plotter import Plotter


def load_config():
    with open("config/config.yaml", "r") as file:
        return yaml.safe_load(file)


def main():
    config = load_config()

    # Step 1: Load Data
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

    # Step 2: Simulate GBM
       # ----------------------------
    # Real-World Simulation (Risk)
    # ----------------------------
    simulator_real = GBMSimulator(
        S0=S0,
        drift=mu,  # historical drift
        sigma=sigma,
        T=config["horizon_years"],
        steps=config["time_steps"],
        simulations=config["simulations"]
    )

    paths_real = simulator_real.simulate()

    print(f"Simulated paths shape: {paths_real.shape}")


    # Step 3: Risk Metrics
    risk = RiskMetrics(config["confidence_level"])
    var = risk.compute_var(paths_real)
    cvar = risk.compute_cvar(paths_real)
        # ---------------------------------
    # Risk-Neutral Simulation (Pricing)
    # ---------------------------------
    simulator_rn = GBMSimulator(
        S0=S0,
        drift=config["risk_free_rate"],  # risk-neutral drift
        sigma=sigma,
        T=config["horizon_years"],
        steps=config["time_steps"],
        simulations=config["simulations"]
    )

    paths_rn = simulator_rn.simulate()
    terminal_prices = paths_rn[-1]


    print(f"VaR ({config['confidence_level']*100}%): {var}")
    print(f"CVaR: {cvar}")

    # Step 4: Option Pricing
    terminal_prices = paths_rn[-1]


    mc_pricer = MonteCarloPricer(
        config["risk_free_rate"],      
        config["strike_price"]
    ) 

    mc_results = mc_pricer.price_call(terminal_prices, config["horizon_years"])


    bs = BlackScholes(
        S0,
        config["strike_price"],
        config["risk_free_rate"],
        sigma,
        config["horizon_years"]
    )

    bs_call = bs.call_price()

    print(f"Monte Carlo Call Price: {mc_results['price']}")
    print(f"Standard Error: {mc_results['std_error']}")
    print(f"95% CI: [{mc_results['ci_lower']}, {mc_results['ci_upper']}]")
    print(f"Black-Scholes Call Price: {bs_call}")
    # Step 5: Visualization
    Plotter.plot_price_paths(paths_real, num_paths=100)
    Plotter.plot_terminal_distribution(terminal_prices, var)

if __name__ == "__main__":
    main()

 