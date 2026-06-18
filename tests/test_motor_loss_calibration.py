import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmr.config import load_yaml
from bmr.motor.calibration import calibrate_loss_coefficients
from bmr.motor.pmsm import PMSMMotor


def _motor_and_targets():
    cfg = load_yaml("motor.yaml")
    motor = PMSMMotor(cfg)
    targets = cfg["efficiency_targets"]
    points = [tuple(p) for p in targets["points"]]
    return motor, points, targets["bus_voltage_V"]


def test_calibration_reduces_efficiency_error():
    motor, points, bus_v = _motor_and_targets()
    # Perturb the coefficients away from the (already calibrated) config values.
    motor.Rs *= 3.0
    motor.k_h *= 3.0
    motor.c_mech *= 3.0

    def rms_error():
        import numpy as np
        res = [motor.operating_point(tq, spd * 2 * 3.14159265 / 60.0, bus_v).efficiency
               - eff for spd, tq, eff in points]
        return float(np.sqrt(np.mean(np.array(res) ** 2)))

    before = rms_error()
    result = calibrate_loss_coefficients(motor, points, bus_v)
    after = rms_error()

    assert after < before
    assert result.rms_efficiency_error < 0.03   # within 3 % RMS of the targets
    assert result.peak_efficiency_after > result.peak_efficiency_before - 1e-6


def test_calibrated_coefficients_non_negative():
    motor, points, bus_v = _motor_and_targets()
    result = calibrate_loss_coefficients(motor, points, bus_v)
    assert result.Rs >= 0
    assert result.k_h >= 0
    assert result.k_e >= 0
    assert result.c_mech >= 0
