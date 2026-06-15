"""Cycle-life / capacity-fade study driven by repeated mission discharges.

Uses the ``life_degradation`` preset (OKane2022 + coupled SEI / lithium plating /
particle cracking / loss of active material). Each cycle is:

    [ mission power drive-cycle discharge ] -> rest -> CCCV charge -> rest

repeated ``n_cycles`` times. PyBaMM accumulates degradation across the cycles
and exposes per-cycle :class:`SummaryVariables`, from which the capacity fade and
the contribution of each mechanism are extracted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pybamm

from bmr.battery.dfn_model import (
    BatteryConfig,
    build_model,
    build_parameter_values,
    build_solver,
)

# Candidate summary-variable keys for the per-mechanism capacity-loss breakdown.
_LOSS_KEYS = [
    "Loss of capacity to negative SEI [A.h]",
    "Loss of capacity to negative SEI on cracks [A.h]",
    "Loss of capacity to negative lithium plating [A.h]",
    "Loss of capacity to positive SEI [A.h]",
]


@dataclass
class DegradationResult:
    cycle_number: np.ndarray
    capacity_Ah: np.ndarray
    soh_percent: np.ndarray
    loss_breakdown_Ah: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def final_soh_percent(self) -> float:
        return float(self.soh_percent[-1])


def run_cycle_life(
    config: BatteryConfig,
    time_s: np.ndarray,
    pack_power_W: np.ndarray,
    n_cycles: int = 5,
    charge_c_rate: float = 0.5,
    v_min: float = 2.5,
    v_max: float = 4.2,
) -> DegradationResult:
    """Run ``n_cycles`` mission-discharge + recharge cycles and report fade."""
    num_cells = config.pack.num_cells
    time_s = np.asarray(time_s, dtype=float)
    cell_power = np.asarray(pack_power_W, dtype=float) / num_cells
    duration = float(time_s[-1] - time_s[0])

    # A constant discharge is solved as a scalar power step (much faster than a
    # multi-point drive-cycle interpolant); a varying profile uses the array.
    if cell_power.size == 1 or np.allclose(cell_power, cell_power.flat[0]):
        discharge_step = pybamm.step.power(
            float(cell_power.flat[0]), duration=duration, termination=f"< {v_min} V")
    else:
        drive = np.column_stack([time_s - time_s[0], cell_power])
        discharge_step = pybamm.step.power(
            drive, duration=duration, termination=f"< {v_min} V")

    cycle = (
        discharge_step,
        pybamm.step.string("Rest for 10 minutes"),
        pybamm.step.string(f"Charge at {charge_c_rate}C until {v_max} V"),
        pybamm.step.string(f"Hold at {v_max} V until C/20"),
        pybamm.step.string("Rest for 10 minutes"),
    )
    experiment = pybamm.Experiment([cycle] * int(n_cycles))

    model = build_model(config.options)
    pv = build_parameter_values(config)
    sim = pybamm.Simulation(
        model, parameter_values=pv, experiment=experiment, solver=build_solver(config.solver)
    )
    solution = sim.solve(initial_soc=config.initial_soc)

    return _extract_degradation(solution)


def _extract_degradation(solution) -> DegradationResult:
    summ = solution.summary_variables

    def get(key: str):
        try:
            return np.asarray(summ[key])
        except (KeyError, TypeError):
            return None

    capacity = get("Capacity [A.h]")
    if capacity is None:
        capacity = get("Measured capacity [A.h]")
    n = len(capacity)
    cycles = get("Cycle number")
    if cycles is None:
        cycles = np.arange(1, n + 1)

    soh = 100.0 * capacity / capacity[0]

    breakdown: dict[str, np.ndarray] = {}
    for key in _LOSS_KEYS:
        vals = get(key)
        if vals is not None:
            breakdown[key] = vals

    return DegradationResult(
        cycle_number=np.asarray(cycles),
        capacity_Ah=capacity,
        soh_percent=soh,
        loss_breakdown_Ah=breakdown,
    )
