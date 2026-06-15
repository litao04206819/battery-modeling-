import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmr.config import load_yaml
from bmr.motor.pmsm import PMSMMotor


def _motor():
    return PMSMMotor(load_yaml("motor.yaml"))


def test_power_balance():
    """Electrical DC power must exceed mechanical output (losses are positive)."""
    motor = _motor()
    torque, speed, bus_v = 0.5, 600.0, 44.4
    st = motor.operating_point(torque, speed, bus_v)
    p_mech = torque * speed
    assert st.dc_power_W > p_mech
    assert st.copper_loss_W >= 0
    assert st.iron_loss_W >= 0
    assert st.mech_loss_W >= 0
    # DC power = mech out + all losses, divided by inverter efficiency.
    expected = (p_mech + st.copper_loss_W + st.iron_loss_W + st.mech_loss_W) / motor.inverter_efficiency
    assert abs(st.dc_power_W - expected) < 1e-6


def test_efficiency_range():
    motor = _motor()
    st = motor.operating_point(0.5, 600.0, 44.4)
    assert 0.0 < st.efficiency < 1.0


def test_dc_current_consistent():
    motor = _motor()
    bus_v = 44.4
    st = motor.operating_point(0.5, 600.0, bus_v)
    assert abs(st.dc_current_A * bus_v - st.dc_power_W) < 1e-6
