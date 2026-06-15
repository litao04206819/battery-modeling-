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


def plot_soh_curve(path: str | Path, cycle_number: np.ndarray,
                   soh_percent: np.ndarray,
                   loss_breakdown_Ah: dict[str, np.ndarray] | None = None,
                   ) -> Path:
    """Plot the state-of-health life curve and per-mechanism capacity loss."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    have_loss = bool(loss_breakdown_Ah)
    nrows = 2 if have_loss else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(9, 4.5 * nrows), squeeze=False)

    ax = axes[0][0]
    ax.plot(cycle_number, soh_percent, "o-", color="tab:blue", markersize=4)
    ax.set_ylabel("State of health [%]")
    ax.set_title("Capacity fade over mission cycles")
    ax.grid(True, alpha=0.3)
    if not have_loss:
        ax.set_xlabel("Cycle number")

    if have_loss:
        ax2 = axes[1][0]
        for label, vals in loss_breakdown_Ah.items():
            short = label.replace("Loss of capacity to ", "").replace(" [A.h]", "")
            ax2.plot(cycle_number, vals, "o-", markersize=3, label=short)
        ax2.set_xlabel("Cycle number")
        ax2.set_ylabel("Capacity loss [A.h]")
        ax2.set_title("Capacity loss by mechanism")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_efficiency_map(path: str | Path, speed_rpm: np.ndarray,
                        torque_Nm: np.ndarray, efficiency: np.ndarray,
                        field_weakening: np.ndarray | None = None,
                        title: str = "PMSM efficiency map") -> Path:
    """Plot a torque-speed efficiency map (infeasible points masked as NaN).

    ``efficiency`` is a 2D array indexed ``[torque, speed]`` with NaN for
    infeasible operating points. ``field_weakening`` (same shape, boolean)
    overlays the field-weakening boundary if given.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    S, T = np.meshgrid(speed_rpm, torque_Nm)
    eff = np.ma.masked_invalid(efficiency * 100.0)

    fig, ax = plt.subplots(figsize=(9, 6))
    levels = np.linspace(50, 100, 26)
    cf = ax.contourf(S, T, eff, levels=levels, cmap="viridis", extend="min")
    cs = ax.contour(S, T, eff, levels=np.arange(60, 100, 5),
                    colors="white", linewidths=0.6, alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%d%%")

    if field_weakening is not None:
        ax.contour(S, T, field_weakening.astype(float), levels=[0.5],
                   colors="red", linewidths=1.5, linestyles="--")
        ax.plot([], [], "r--", label="field-weakening boundary")
        ax.legend(loc="upper right", fontsize=8)

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Efficiency [%]")
    ax.set_xlabel("Speed [rpm]")
    ax.set_ylabel("Torque [N.m]")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
