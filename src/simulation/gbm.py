import numpy as np


class GBMSimulator:
    def __init__(self, S0, drift, sigma, T, steps, simulations):
        self.S0 = S0
        self.drift = drift
        self.sigma = sigma
        self.T = T
        self.steps = steps
        self.simulations = simulations

    def simulate(self):
        dt = self.T / self.steps

        Z = np.random.normal(0, 1, (self.steps, self.simulations))

        drift_term = (self.drift - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt) * Z

        increments = drift_term + diffusion

        log_paths = np.vstack([
            np.zeros(self.simulations),
            np.cumsum(increments, axis=0)
        ])

        paths = self.S0 * np.exp(log_paths)

        return paths
