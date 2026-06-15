import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmr.config import load_yaml
from bmr.rotor.bemt import BEMTRotor


def _rotor():
    return BEMTRotor(load_yaml("rotor.yaml"))


def test_thrust_increases_with_speed():
    rotor = _rotor()
    t_low = rotor.solve(200.0).thrust_N
    t_high = rotor.solve(400.0).thrust_N
    assert t_high > t_low > 0


def test_inverse_solve_recovers_thrust():
    rotor = _rotor()
    target = 30.0  # N
    state = rotor.thrust_to_speed(target)
    assert abs(state.thrust_N - target) < 0.5
    assert state.rotor_speed_rad_s > 0
    assert state.mech_power_W > 0


def test_figure_of_merit_plausible():
    rotor = _rotor()
    state = rotor.thrust_to_speed(30.0)
    # Real rotors have FM ~0.4-0.8; just bound it to a sane physical range.
    assert 0.0 < state.figure_of_merit < 1.0


def test_power_equals_torque_times_speed():
    rotor = _rotor()
    state = rotor.solve(300.0)
    assert abs(state.mech_power_W - state.torque_Nm * state.rotor_speed_rad_s) < 1e-6
