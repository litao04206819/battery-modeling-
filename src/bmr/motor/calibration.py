"""Calibrate the PMSM loss coefficients to measured efficiency data.

The simple loss model in :class:`bmr.motor.pmsm.PMSMMotor` has four coefficients
that set its efficiency:

* ``phase_resistance_ohm`` (Rs)      -> copper loss  ~ Rs * iq^2
* ``iron_loss_coeff_hyst`` (k_h)     -> iron loss    ~ k_h * f_e
* ``iron_loss_coeff_eddy`` (k_e)     -> iron loss    ~ k_e * f_e^2
* ``mechanical_loss_coeff`` (c_mech) -> mech loss    ~ c_mech * omega^2

Because each loss term has a distinct dependence on torque and speed, the four
coefficients are identifiable from a handful of measured (speed, torque,
efficiency) points spanning the operating plane.
:func:`calibrate_loss_coefficients` fits them with a bounded (non-negative)
least-squares solve, mutating the motor in place.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from bmr.motor.pmsm import PMSMMotor

# Coefficients that can be fitted, mapped to their PMSMMotor attribute names.
_PARAM_ATTRS = {
    "Rs": "Rs",
    "k_h": "k_h",
    "k_e": "k_e",
    "c_mech": "c_mech",
}


@dataclass
class LossCalibrationResult:
    Rs: float
    k_h: float
    k_e: float
    c_mech: float
    rms_efficiency_error: float
    max_efficiency_error: float
    peak_efficiency_before: float
    peak_efficiency_after: float
    success: bool


def _peak_efficiency(motor: PMSMMotor, points, bus_v: float) -> float:
    return max(
        motor.operating_point(tq, spd * 2 * np.pi / 60.0, bus_v).efficiency
        for spd, tq, _ in points
    )


def calibrate_loss_coefficients(
    motor: PMSMMotor,
    points: list[tuple[float, float, float]],
    bus_voltage_V: float,
    fit: tuple[str, ...] = ("Rs", "k_h", "k_e", "c_mech"),
) -> LossCalibrationResult:
    """Fit the chosen loss coefficients to ``points`` = [(speed_rpm, torque, eff)].

    The motor is modified in place. Coefficients are constrained non-negative.
    """
    peak_before = _peak_efficiency(motor, points, bus_voltage_V)

    x0 = np.array([getattr(motor, _PARAM_ATTRS[p]) for p in fit], dtype=float)

    def apply(x: np.ndarray) -> None:
        for name, val in zip(fit, x):
            setattr(motor, _PARAM_ATTRS[name], float(val))

    def residual(x: np.ndarray) -> np.ndarray:
        apply(x)
        res = []
        for spd_rpm, tq, eff_target in points:
            st = motor.operating_point(tq, spd_rpm * 2 * np.pi / 60.0, bus_voltage_V)
            res.append(st.efficiency - eff_target)
        return np.array(res)

    # Non-negative coefficients with Jacobian-based scaling so the wildly
    # different parameter magnitudes (Rs ~ 1e-2, k_e ~ 1e-8) are conditioned.
    sol = least_squares(
        residual, x0, method="trf", bounds=(0.0, np.inf),
        x_scale="jac", xtol=1e-14, ftol=1e-14, gtol=1e-14,
    )
    apply(sol.x)

    final_res = residual(sol.x)
    peak_after = _peak_efficiency(motor, points, bus_voltage_V)

    return LossCalibrationResult(
        Rs=motor.Rs,
        k_h=motor.k_h,
        k_e=motor.k_e,
        c_mech=motor.c_mech,
        rms_efficiency_error=float(np.sqrt(np.mean(final_res**2))),
        max_efficiency_error=float(np.max(np.abs(final_res))),
        peak_efficiency_before=peak_before,
        peak_efficiency_after=peak_after,
        success=bool(sol.success),
    )
