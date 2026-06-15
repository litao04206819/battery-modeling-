"""System-level result plotting and CSV export."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / non-interactive backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def save_system_csv(path: str | Path, columns: dict[str, np.ndarray]) -> Path:
    """Write a dict of equal-length arrays to a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns).to_csv(path, index=False)
    return path


def plot_system_timeseries(path: str | Path, time_s: np.ndarray, *,
                           pack_power_W: np.ndarray,
                           pack_voltage_V: np.ndarray,
                           pack_current_A: np.ndarray,
                           soc: np.ndarray,
                           cell_temperature_K: np.ndarray) -> Path:
    """Create a 5-panel summary figure of the system response."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t_min = time_s / 60.0

    fig, axes = plt.subplots(5, 1, figsize=(9, 12), sharex=True)
    axes[0].plot(t_min, pack_power_W / 1000.0, color="tab:red")
    axes[0].set_ylabel("Pack power [kW]")
    axes[1].plot(t_min, pack_voltage_V, color="tab:blue")
    axes[1].set_ylabel("Bus voltage [V]")
    axes[2].plot(t_min, pack_current_A, color="tab:orange")
    axes[2].set_ylabel("Pack current [A]")
    axes[3].plot(t_min, 100.0 * soc, color="tab:green")
    axes[3].set_ylabel("SOC [%]")
    axes[4].plot(t_min, cell_temperature_K - 273.15, color="tab:purple")
    axes[4].set_ylabel("Cell temp [°C]")
    axes[4].set_xlabel("Time [min]")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("Battery-Motor-Rotor system response")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
