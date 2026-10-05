"""
Experiments: parameter sweeps over density and noise for the
structural balance geopolitics model.

Produces:
  - balance_over_time.png : balance ratio trajectories for a few example runs
  - noise_sweep.png       : effect of noise on convergence / final bloc count
  - density_sweep.png     : effect of network density on convergence
"""

import numpy as np
import matplotlib.pyplot as plt
from balance_model import run_simulation, detect_blocs


def experiment_balance_trajectories(n_nations=10, seed=1):
    """Show how balance ratio evolves over time for a few noise levels."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for noise in [0.0, 0.01, 0.05]:
        G, history, converged = run_simulation(
            n_nations=n_nations, density=1.0, noise=noise,
            max_steps=60000, converge_patience=1500, seed=seed
        )
        steps, ratios = zip(*history)
        ax.plot(steps, ratios, label=f"noise={noise} ({'converged' if converged else 'not converged'})")

    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Fraction of balanced triads")
    ax.set_title(f"Balance ratio over time (N={n_nations} nations, fully connected)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("balance_over_time.png", dpi=150)
    plt.close(fig)
    print("Saved balance_over_time.png")


def experiment_noise_sweep(n_nations=8, n_trials=8, seed=10):
    """
    For a range of noise levels, measure:
      - probability of convergence within a fixed budget
      - resulting number of blocs (when converged)

    Note: n_nations is kept small (8) because convergence time grows
    very fast with network size (triad count grows ~N^3).
    """
    noise_levels = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    conv_rate = []
    avg_blocs = []

    for noise in noise_levels:
        converged_flags = []
        bloc_counts = []
        for trial in range(n_trials):
            G, history, converged = run_simulation(
                n_nations=n_nations, density=1.0, noise=noise,
                max_steps=60000, converge_patience=1000,
                seed=seed + trial
            )
            converged_flags.append(converged)
            if converged:
                _, n_factions = detect_blocs(G)
                bloc_counts.append(n_factions)

        conv_rate.append(np.mean(converged_flags))
        avg_blocs.append(np.mean(bloc_counts) if bloc_counts else np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(noise_levels, conv_rate, marker='o')
    ax1.set_xlabel("Noise level")
    ax1.set_ylabel("Fraction of trials that converged")
    ax1.set_title("Convergence rate vs. noise")
    ax1.grid(alpha=0.3)

    ax2.plot(noise_levels, avg_blocs, marker='o', color='darkorange')
    ax2.set_xlabel("Noise level")
    ax2.set_ylabel("Avg. number of blocs (when converged)")
    ax2.set_title("Resulting bloc count vs. noise")
    ax2.set_ylim(0, 4)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("noise_sweep.png", dpi=150)
    plt.close(fig)
    print("Saved noise_sweep.png")


def experiment_density_sweep(n_nations=8, n_trials=8, seed=20):
    """For a range of densities (with zero noise), measure convergence."""
    densities = [0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]
    conv_rate = []
    avg_blocs = []

    for density in densities:
        converged_flags = []
        bloc_counts = []
        for trial in range(n_trials):
            G, history, converged = run_simulation(
                n_nations=n_nations, density=density, noise=0.0,
                max_steps=60000, converge_patience=1000,
                seed=seed + trial
            )
            converged_flags.append(converged)
            if converged:
                _, n_factions = detect_blocs(G)
                bloc_counts.append(n_factions)

        conv_rate.append(np.mean(converged_flags))
        avg_blocs.append(np.mean(bloc_counts) if bloc_counts else np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(densities, conv_rate, marker='o', color='seagreen')
    ax1.set_xlabel("Network density")
    ax1.set_ylabel("Fraction of trials that converged")
    ax1.set_title("Convergence rate vs. density")
    ax1.grid(alpha=0.3)

    ax2.plot(densities, avg_blocs, marker='o', color='crimson')
    ax2.set_xlabel("Network density")
    ax2.set_ylabel("Avg. number of blocs (when converged)")
    ax2.set_title("Resulting bloc count vs. density")
    ax2.set_ylim(0, 4)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("density_sweep.png", dpi=150)
    plt.close(fig)
    print("Saved density_sweep.png")


if __name__ == "__main__":
    experiment_balance_trajectories()
    experiment_noise_sweep()
    experiment_density_sweep()