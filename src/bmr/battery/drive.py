"""Drive the DFN model with a power/current demand profile.

Two coupling paths are supported (matching the approved plan):

* **Path 1 (profile driven)** :func:`BatterySimulator.run_power_profile` /
  :func:`run_current_profile` build a single PyBaMM drive-cycle step
  (``pybamm.step.power`` / ``pybamm.step.current`` from a ``(time, value)``
  array) and solve the whole mission in one call. No within-solve feedback.

* **Path 2 (feedback co-simulation)** :func:`BatterySimulator.step` advances one
  timestep with ``Simulation.step(dt, inputs=...)`` so an outer loop can
  recompute the motor power demand from the previous step's bus voltage.

Pack scaling
------------
PyBaMM solves ONE representative cell. The public methods take **pack-level**
demand (the motors draw from the whole pack) and return **pack-level** outputs:

    cell_power  = pack_power  / (series * parallel)
    pack_voltage = cell_voltage * series
    pack_current = cell_current * parallel
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybamm

from bmr.battery.dfn_model import (
    BatteryConfig,
    build_model,
    build_parameter_values,
    build_solver,
)

# Canonical PyBaMM output variable names (verified for current versions).
VAR_VOLTAGE = "Voltage [V]"
VAR_CURRENT = "Current [A]"
VAR_TEMPERATURE = "Volume-averaged cell temperature [K]"
VAR_DISCHARGE_CAPACITY = "Discharge capacity [A.h]"
VAR_TIME = "Time [s]"


@dataclass
class BatteryResult:
    """Pack-level time series extracted from a PyBaMM solution."""

    time_s: np.ndarray
    pack_voltage_V: np.ndarray
    pack_current_A: np.ndarray
    cell_voltage_V: np.ndarray
    cell_temperature_K: np.ndarray
    discharge_capacity_Ah: np.ndarray
    soc: np.ndarray

    @property
    def final_soc(self) -> float:
        return float(self.soc[-1])


class BatterySimulator:
    """Build once, then drive the DFN cell with pack-level power/current."""

    def __init__(self, config: BatteryConfig, drive_mode: str = "power"):
        self.config = config
        self.drive_mode = drive_mode
        self.num_cells = config.pack.num_cells
        self.series = config.pack.series
        self.parallel = config.pack.parallel

        self.parameter_values = build_parameter_values(config)
        self.solver = build_solver(config.solver)

        # Nominal cell capacity for SOC bookkeeping.
        self._capacity_Ah = float(
            self.parameter_values["Nominal cell capacity [A.h]"]
        )

        # Lazily-built objects for the stepping (feedback) interface.
        self._step_sim: pybamm.Simulation | None = None
        self._solution = None

    # ------------------------------------------------------------------
    # Path 1: whole-mission drive cycle
    # ------------------------------------------------------------------
    def run_power_profile(
        self, time_s: np.ndarray, pack_power_W: np.ndarray
    ) -> BatteryResult:
        """Solve the full mission given a pack power demand time series."""
        cell_power = np.asarray(pack_power_W, dtype=float) / self.num_cells
        return self._run_profile(time_s, cell_power, kind="power")

    def run_current_profile(
        self, time_s: np.ndarray, pack_current_A: np.ndarray
    ) -> BatteryResult:
        """Solve the full mission given a pack current demand time series."""
        cell_current = np.asarray(pack_current_A, dtype=float) / self.parallel
        return self._run_profile(time_s, cell_current, kind="current")

    def _run_profile(
        self, time_s: np.ndarray, cell_value: np.ndarray, kind: str
    ) -> BatteryResult:
        time_s = np.asarray(time_s, dtype=float)
        drive = np.column_stack([time_s - time_s[0], cell_value])
        duration = float(time_s[-1] - time_s[0])

        if kind == "power":
            step = pybamm.step.power(drive, duration=duration)
        else:
            step = pybamm.step.current(drive, duration=duration)

        model = build_model(self.config.options)
        experiment = pybamm.Experiment([step])
        sim = pybamm.Simulation(
            model,
            parameter_values=self.parameter_values,
            experiment=experiment,
            solver=self.solver,
        )
        solution = sim.solve(initial_soc=self.config.initial_soc)
        return self._extract(solution)

    # ------------------------------------------------------------------
    # Path 2: single-step feedback co-simulation
    # ------------------------------------------------------------------
    def _ensure_step_sim(self) -> pybamm.Simulation:
        if self._step_sim is None:
            options = dict(self.config.options)
            options["operating mode"] = "power"
            model = build_model(options)
            pv = self.parameter_values.copy()
            pv["Power function [W]"] = "[input]"
            # Simulation.step() does not accept initial_soc, so bake the initial
            # state of charge into the parameter values up front.
            if abs(self.config.initial_soc - 1.0) > 1e-9:
                pv.set_initial_stoichiometries(self.config.initial_soc, inplace=True)
            self._step_sim = pybamm.Simulation(
                model, parameter_values=pv, solver=self.solver
            )
        return self._step_sim

    def step(self, dt: float, pack_power_W: float) -> BatteryResult:
        """Advance one timestep with a pack power demand and return the result.

        The previous solution is carried over automatically, so calling this in
        a loop performs the co-simulation. Use :attr:`last_pack_voltage_V` to
        feed the bus voltage back into the motor model before the next call.
        """
        sim = self._ensure_step_sim()
        cell_power = float(pack_power_W) / self.num_cells
        self._solution = sim.step(
            dt,
            inputs={"Power function [W]": cell_power},
            starting_solution=self._solution,
        )
        return self._extract(self._solution)

    @property
    def last_pack_voltage_V(self) -> float:
        if self._solution is None:
            raise RuntimeError("No step has been taken yet.")
        return float(self._solution[VAR_VOLTAGE].entries[-1]) * self.series

    # ------------------------------------------------------------------
    # Output extraction + SOC bookkeeping
    # ------------------------------------------------------------------
    def _extract(self, solution) -> BatteryResult:
        t = solution[VAR_TIME].entries
        cell_v = solution[VAR_VOLTAGE].entries
        cell_i = solution[VAR_CURRENT].entries
        temp = solution[VAR_TEMPERATURE].entries
        q_dis = solution[VAR_DISCHARGE_CAPACITY].entries

        # SOC from coulomb counting relative to nominal capacity.
        soc = self.config.initial_soc - q_dis / self._capacity_Ah

        return BatteryResult(
            time_s=t,
            pack_voltage_V=cell_v * self.series,
            pack_current_A=cell_i * self.parallel,
            cell_voltage_V=cell_v,
            cell_temperature_K=temp,
            discharge_capacity_Ah=q_dis,
            soc=soc,
        )
