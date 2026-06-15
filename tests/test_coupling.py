import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmr.aircraft.airframe import Airframe
from bmr.config import load_yaml
from bmr.coupling.profile_builder import build_demand_profile
from bmr.mission.profile import Mission
from bmr.motor.pmsm import PMSMMotor
from bmr.rotor.bemt import BEMTRotor


def _system():
    airframe = Airframe.from_dict(load_yaml("aircraft.yaml"))
    rotor = BEMTRotor(load_yaml("rotor.yaml"), air_density_kg_m3=airframe.air_density_kg_m3)
    motor = PMSMMotor(load_yaml("motor.yaml"))
    mission = Mission.from_dict(load_yaml("mission_hover_cruise.yaml"))
    return airframe, rotor, motor, mission


def test_mission_discretisation_length():
    _, _, _, mission = _system()
    # 30+60+300+120+60+30 = 600 s at dt=1 -> 600 samples.
    assert len(mission) == 600
    assert mission.dt == 1.0


def test_demand_profile_positive_and_sized():
    airframe, rotor, motor, mission = _system()
    profile = build_demand_profile(mission, airframe, rotor, motor)
    assert len(profile.pack_power_W) == len(mission)
    assert np.all(profile.pack_power_W > 0)
    # Climb segment (factor 1.25) must demand more power than hover (1.0).
    hover_power = profile.pack_power_W[0]
    climb_power = profile.pack_power_W[40]  # within the climb segment
    assert climb_power > hover_power


def test_energy_magnitude_reasonable():
    airframe, rotor, motor, mission = _system()
    profile = build_demand_profile(mission, airframe, rotor, motor)
    energy_Wh = np.trapezoid(profile.pack_power_W, profile.time_s) / 3600.0
    # A 10 min multirotor mission should consume on the order of 0.05-5 kWh.
    assert 50.0 < energy_Wh < 5000.0
