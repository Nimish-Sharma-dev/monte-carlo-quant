import numpy as np


class MonteCarloPricer:
    def __init__(self, risk_free_rate, strike):
        self.r = risk_free_rate
        self.K = strike

    def price_call(self, terminal_prices, T):
        payoffs = np.maximum(terminal_prices - self.K, 0)
        discounted = np.exp(-self.r * T) * payoffs
        price = discounted.mean()
        return price

    def price_put(self, terminal_prices, T):
        payoffs = np.maximum(self.K - terminal_prices, 0)
        discounted = np.exp(-self.r * T) * payoffs
        price = discounted.mean()
        return price
