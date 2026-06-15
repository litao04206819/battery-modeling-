"""Calibrate the BEMT aerofoil polar to measured propeller CT/CQ data.

Manufacturer / wind-tunnel data usually give the non-dimensional thrust and
torque (or power) coefficients of a propeller. Two free parameters of the BEMT
model control them over a wide, well-conditioned range:

* ``pitch_offset`` (collective pitch) -> drives thrust -> CT
* ``cd0``          (parasite drag)     -> drives torque -> CQ

:func:`calibrate_to_coefficients` tunes these two so the BEMT prediction matches
a target (CT, CQ) at a reference operating point, using a bounded least-squares
solve. It mutates the rotor in place and returns the calibration result.

Coefficients use the propeller convention CT = T / (rho n^2 D^4),
CQ = Q / (rho n^2 D^5) with n in rev/s (see ``BEMTRotor.coefficients``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from bmr.rotor.bemt import BEMTRotor


@dataclass
class CalibrationResult:
    pitch_offset_deg: float
    cd0: float
    target_CT: float
    target_CQ: float
    achieved_CT: float
    achieved_CQ: float
    ref_speed_rad_s: float
    success: bool

    @property
    def ct_error(self) -> float:
        return abs(self.achieved_CT - self.target_CT)

    @property
    def cq_error(self) -> float:
        return abs(self.achieved_CQ - self.target_CQ)


def calibrate_to_coefficients(
    rotor: BEMTRotor,
    target_CT: float,
    target_CQ: float,
    ref_speed_rad_s: float = 350.0,
    pitch_offset_bounds_deg: tuple[float, float] = (-12.0, 12.0),
    cd0_bounds: tuple[float, float] = (0.004, 0.06),
) -> CalibrationResult:
    """Tune ``rotor.pitch_offset`` and ``rotor.cd0`` to match (CT, CQ).

    Parameters
    ----------
    rotor
        The rotor to calibrate (modified in place).
    target_CT, target_CQ
        Measured propeller coefficients to match.
    ref_speed_rad_s
        Rotor speed at which the coefficients are evaluated (hover-like point).
    """
    # Normalise residuals by the targets so CT and CQ are weighted comparably.
    def residual(x: np.ndarray) -> np.ndarray:
        rotor.pitch_offset = np.deg2rad(float(x[0]))
        rotor.cd0 = float(x[1])
        ct, cq = rotor.coefficients(ref_speed_rad_s)
        return np.array([
            (ct - target_CT) / target_CT,
            (cq - target_CQ) / target_CQ,
        ])

    x0 = np.array([np.rad2deg(rotor.pitch_offset), rotor.cd0])
    sol = least_squares(
        residual, x0,
        bounds=([pitch_offset_bounds_deg[0], cd0_bounds[0]],
                [pitch_offset_bounds_deg[1], cd0_bounds[1]]),
        xtol=1e-12, ftol=1e-12,
    )

    rotor.pitch_offset = np.deg2rad(float(sol.x[0]))
    rotor.cd0 = float(sol.x[1])
    ct, cq = rotor.coefficients(ref_speed_rad_s)

    return CalibrationResult(
        pitch_offset_deg=float(sol.x[0]),
        cd0=rotor.cd0,
        target_CT=target_CT,
        target_CQ=target_CQ,
        achieved_CT=ct,
        achieved_CQ=cq,
        ref_speed_rad_s=ref_speed_rad_s,
        success=bool(sol.success),
    )
