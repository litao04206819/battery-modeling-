import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmr.config import load_yaml
from bmr.motor.pmsm import PMSMMotor


def _motor(**overrides):
    cfg = load_yaml("motor.yaml")
    cfg.update(overrides)
    return PMSMMotor(cfg)


def test_no_field_weakening_at_low_speed():
    """Well below no-load speed the machine runs at id = 0 (MTPA)."""
    motor = _motor()
    st = motor.operating_point(0.5, 300.0, 44.4)
    assert not st.field_weakening
    assert st.id_A == 0.0


def test_field_weakening_activates_above_no_load_speed():
    """Above the no-load speed the back-EMF exceeds the bus ceiling -> id < 0."""
    motor = _motor()
    # No-load speed at 44.4 V is ~558 rad/s; 600 rad/s forces field weakening.
    st = motor.operating_point(0.4, 600.0, 44.4)
    assert st.field_weakening
    assert st.id_A < 0.0
    assert st.feasible
    # Phase voltage is held on (not above) the available limit.
    v_max = motor._voltage_limit(44.4)
    assert abs(st.phase_voltage_V - v_max) < 1e-2


def test_overspeed_point_flagged_infeasible():
    """A point far above the achievable envelope is flagged infeasible."""
    motor = _motor()
    st = motor.operating_point(0.5, 4000.0, 30.0)
    assert not st.feasible


def test_field_weakening_can_be_disabled():
    motor = _motor(field_weakening=False)
    st = motor.operating_point(0.4, 600.0, 44.4)
    assert not st.field_weakening
    assert st.id_A == 0.0


def test_current_limit_feasibility_flag():
    motor = _motor(max_phase_current_A=1.0)  # absurdly low limit
    st = motor.operating_point(2.0, 600.0, 44.4)
    assert st.feasible is False
