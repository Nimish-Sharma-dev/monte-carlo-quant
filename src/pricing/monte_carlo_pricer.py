import numpy as np


class MonteCarloPricer:
    def __init__(self, risk_free_rate, strike):
        self.r = risk_free_rate
        self.K = strike

    def price_call(self, terminal_prices, T):
        # Discounted payoffs
        payoffs = np.exp(-self.r * T) * np.maximum(terminal_prices - self.K, 0)

        price = np.mean(payoffs)

        # Standard deviation of discounted payoffs
        std_dev = np.std(payoffs, ddof=1)

        # Standard error
        standard_error = std_dev / np.sqrt(len(payoffs))

        # 95% confidence interval
        ci_lower = price - 1.96 * standard_error
        ci_upper = price + 1.96 * standard_error

        return {
            "price": price,
            "std_error": standard_error,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        }

    def price_put(self, terminal_prices, T):
        payoffs = np.exp(-self.r * T) * np.maximum(self.K - terminal_prices, 0)

        price = np.mean(payoffs)
        std_dev = np.std(payoffs, ddof=1)
        standard_error = std_dev / np.sqrt(len(payoffs))

        ci_lower = price - 1.96 * standard_error
        ci_upper = price + 1.96 * standard_error

        return {
            "price": price,
            "std_error": standard_error,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        }
