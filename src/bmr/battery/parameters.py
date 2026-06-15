"""Load PyBaMM parameter sets and apply user overrides.

PyBaMM models a single representative cell. Pack-level series/parallel scaling
is intentionally NOT applied here — it lives in the coupling layer
(:mod:`bmr.coupling`) so the cell parameters stay physically clean.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pybamm


@dataclass
class PackTopology:
    """Series/parallel arrangement of the cells in the pack."""

    series: int = 1
    parallel: int = 1

    @property
    def num_cells(self) -> int:
        return self.series * self.parallel


def load_parameter_values(
    parameter_set: str,
    overrides: dict[str, Any] | None = None,
    ambient_temperature_K: float | None = None,
) -> pybamm.ParameterValues:
    """Return a :class:`pybamm.ParameterValues` for ``parameter_set``.

    Parameters
    ----------
    parameter_set
        A built-in PyBaMM set name, e.g. ``"Chen2020"``, ``"ORegan2022"`` or
        ``"OKane2022"``.
    overrides
        Optional mapping of valid PyBaMM parameter names to values, applied on
        top of the chosen set (e.g. ``{"Nominal cell capacity [A.h]": 5.0}``).
    ambient_temperature_K
        If given, sets both the ambient and initial temperature parameters.
    """
    pv = pybamm.ParameterValues(parameter_set)

    if ambient_temperature_K is not None:
        # These keys exist in all standard lithium-ion sets.
        for key in ("Ambient temperature [K]", "Initial temperature [K]"):
            if key in pv:
                pv[key] = ambient_temperature_K

    if overrides:
        pv.update(overrides, check_already_exists=False)

    return pv
