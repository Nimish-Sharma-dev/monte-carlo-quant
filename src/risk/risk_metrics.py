import numpy as np


class RiskMetrics:
    def __init__(self, confidence_level=0.95):
        self.alpha = confidence_level

    def compute_var(self, portfolio_values):
        losses = portfolio_values[0] - portfolio_values[-1]
        var = np.percentile(losses, self.alpha * 100)
        return var

    def compute_cvar(self, portfolio_values):
        losses = portfolio_values[0] - portfolio_values[-1]
        var = self.compute_var(portfolio_values)
        cvar = losses[losses >= var].mean()
        return cvar
