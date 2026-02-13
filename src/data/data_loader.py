"""Responsibilities:

Fetch historical prices

Compute log returns

Estimate:

μ (annualized drift)

σ (annualized volatility)"""

import yfinance as yf
import numpy as np
import pandas as pd


class DataLoader:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self):
        data = yf.download(
            self.ticker,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True
        )

        if data.empty:
            raise ValueError("No data downloaded. Check ticker or date range.")

        # If MultiIndex columns exist, flatten them
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        return data["Close"].dropna()


    def estimate_parameters(self):
        prices = self.fetch_data()
        log_returns = np.log(prices / prices.shift(1)).dropna()

        mu_daily = log_returns.mean()
        sigma_daily = log_returns.std()

        mu_annual = mu_daily * 252
        sigma_annual = sigma_daily * np.sqrt(252)

        return {
            "mu": float(mu_annual),
            "sigma": float(sigma_annual),
            "last_price": float(prices.iloc[-1])
        }
