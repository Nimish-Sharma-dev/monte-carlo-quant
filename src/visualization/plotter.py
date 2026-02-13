import matplotlib.pyplot as plt
import numpy as np
import os


class Plotter:

    @staticmethod
    def plot_price_paths(paths, num_paths=100, save_path="outputs/price_paths.png"):
        os.makedirs("outputs", exist_ok=True)

        plt.figure(figsize=(10, 6))

        for i in range(min(num_paths, paths.shape[1])):
            plt.plot(paths[:, i], linewidth=0.8)

        plt.title("Monte Carlo Simulated Price Paths")
        plt.xlabel("Time Steps")
        plt.ylabel("Price")
        plt.grid(True)

        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_terminal_distribution(terminal_prices, var=None, save_path="outputs/terminal_distribution.png"):
        os.makedirs("outputs", exist_ok=True)

        plt.figure(figsize=(10, 6))

        plt.hist(terminal_prices, bins=50, alpha=0.7)
        plt.title("Distribution of Terminal Prices")
        plt.xlabel("Terminal Price")
        plt.ylabel("Frequency")

        if var is not None:
            cutoff_price = np.percentile(terminal_prices, 5)
            plt.axvline(cutoff_price, linestyle="--")

        plt.grid(True)

        plt.savefig(save_path)
        plt.close()
