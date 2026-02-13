import numpy as np
from scipy.stats import norm


class BlackScholes:
    def __init__(self, S0, K, r, sigma, T):
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T

    def d1(self):
        return (np.log(self.S0 / self.K) + 
                (self.r + 0.5 * self.sigma**2) * self.T) / \
               (self.sigma * np.sqrt(self.T))

    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)

    def call_price(self):
        return (self.S0 * norm.cdf(self.d1()) -
                self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2()))

    def put_price(self):
        return (self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2()) -
                self.S0 * norm.cdf(-self.d1()))
