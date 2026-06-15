import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmr.config import load_yaml
from bmr.motor.pmsm import PMSMMotor, compute_efficiency_map


def _motor():
    return PMSMMotor(load_yaml("motor.yaml"))


def test_efficiency_map_shape_and_range():
    motor = _motor()
    speed_rpm = np.linspace(500, 6000, 20)
    torque = np.linspace(0.05, 5.0, 15)
    eff, fw = compute_efficiency_map(motor, speed_rpm, torque, bus_voltage_V=44.4)

    assert eff.shape == (len(torque), len(speed_rpm))
    assert fw.shape == eff.shape
    finite = np.isfinite(eff)
    assert finite.any()
    # Feasible efficiencies are physical fractions in (0, 1).
    assert np.all((eff[finite] > 0.0) & (eff[finite] < 1.0))


def test_field_weakening_appears_above_no_load():
    """Above the no-load speed (KV * V) the map must contain FW points."""
    motor = _motor()
    no_load_rpm = motor.KV * 44.4
    speed_rpm = np.linspace(0.5 * no_load_rpm, 1.3 * no_load_rpm, 25)
    torque = np.linspace(0.05, 3.0, 10)
    _, fw = compute_efficiency_map(motor, speed_rpm, torque, bus_voltage_V=44.4)
    assert fw.any()
