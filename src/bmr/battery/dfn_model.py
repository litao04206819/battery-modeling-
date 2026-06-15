"""Build the high-fidelity DFN battery model from a configuration dict.

Wraps :class:`pybamm.lithium_ion.DFN` with the verified thermal and
degradation option keys, and provides the two presets defined in
``configs/battery.yaml``:

* ``high_rate_thermal`` - Chen2020/ORegan2022 + lumped thermal
* ``life_degradation``  - OKane2022 + coupled SEI/plating/cracking/LAM
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pybamm

from bmr.battery.parameters import PackTopology, load_parameter_values


@dataclass
class BatteryConfig:
    """Resolved battery configuration (one preset, flattened)."""

    parameter_set: str
    options: dict[str, str]
    initial_soc: float
    ambient_temperature_K: float
    pack: PackTopology
    overrides: dict[str, Any]
    solver: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "BatteryConfig":
        preset_name = cfg["preset"]
        preset = cfg["presets"][preset_name]
        pack = cfg.get("pack", {})
        return cls(
            parameter_set=preset["parameter_set"],
            options=dict(preset.get("options", {})),
            initial_soc=float(preset.get("initial_soc", 1.0)),
            ambient_temperature_K=float(preset.get("ambient_temperature_K", 298.15)),
            pack=PackTopology(
                series=int(pack.get("series", 1)),
                parallel=int(pack.get("parallel", 1)),
            ),
            overrides=dict(cfg.get("overrides", {})),
            solver=cfg.get("solver", "idaklu"),
        )


def build_solver(name: str) -> pybamm.BaseSolver:
    """Return a solver instance. ``idaklu`` is fastest for thermal+ageing DAEs."""
    name = (name or "idaklu").lower()
    if name == "idaklu":
        try:
            return pybamm.IDAKLUSolver()
        except Exception:  # pragma: no cover - depends on optional native build
            # Fall back to the robust general-purpose solver if IDAKLU is
            # unavailable in this PyBaMM build.
            return pybamm.CasadiSolver(mode="safe")
    if name == "casadi":
        return pybamm.CasadiSolver(mode="safe")
    raise ValueError(f"Unknown solver '{name}' (expected 'idaklu' or 'casadi')")


def build_model(options: dict[str, str]) -> pybamm.lithium_ion.DFN:
    """Construct the DFN model with the given (verified) option keys."""
    return pybamm.lithium_ion.DFN(options=options or None)


def build_parameter_values(config: BatteryConfig) -> pybamm.ParameterValues:
    """Construct the parameter values for the configuration."""
    return load_parameter_values(
        config.parameter_set,
        overrides=config.overrides,
        ambient_temperature_K=config.ambient_temperature_K,
    )
