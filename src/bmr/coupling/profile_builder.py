"""Build the battery power/current demand profile from mission + rotor + motor.

This is the main (one-directional) coupling path:

    mission sample -> per-rotor thrust -> BEMT (rotor speed, torque)
                   -> PMSM (DC power per motor) -> sum over rotors -> pack power

The bus voltage used here is the airframe nominal value; the optional
:mod:`bmr.coupling.feedback_loop` refines it with the battery terminal voltage.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bmr.aircraft.airframe import Airframe
from bmr.mission.profile import Mission
from bmr.motor.pmsm import PMSMMotor
from bmr.rotor.bemt import BEMTRotor


@dataclass
class DemandProfile:
    """Time series of the propulsion demand and intermediate quantities."""

    time_s: np.ndarray
    pack_power_W: np.ndarray
    per_rotor_thrust_N: np.ndarray
    rotor_speed_rad_s: np.ndarray
    rotor_torque_Nm: np.ndarray
    motor_efficiency: np.ndarray
    rotor_figure_of_merit: np.ndarray


def build_demand_profile(
    mission: Mission,
    airframe: Airframe,
    rotor: BEMTRotor,
    motor: PMSMMotor,
    bus_voltage_V: float | None = None,
) -> DemandProfile:
    """Compute the pack power demand over the whole mission."""
    bus_v = bus_voltage_V if bus_voltage_V is not None else airframe.nominal_bus_voltage_V
    n = len(mission)

    pack_power = np.zeros(n)
    thrust = np.zeros(n)
    speed = np.zeros(n)
    torque = np.zeros(n)
    eta = np.zeros(n)
    fom = np.zeros(n)

    for i in range(n):
        T_req = airframe.per_rotor_thrust_N(mission.thrust_factor[i])
        v_axial = mission.axial_velocity_m_s[i]

        rotor_state = rotor.thrust_to_speed(T_req, axial_velocity_m_s=v_axial)
        motor_state = motor.operating_point(
            rotor_state.torque_Nm, rotor_state.rotor_speed_rad_s, bus_v
        )

        thrust[i] = T_req
        speed[i] = rotor_state.rotor_speed_rad_s
        torque[i] = rotor_state.torque_Nm
        fom[i] = rotor_state.figure_of_merit
        eta[i] = motor_state.efficiency
        pack_power[i] = motor_state.dc_power_W * airframe.num_rotors

    return DemandProfile(
        time_s=mission.time_s,
        pack_power_W=pack_power,
        per_rotor_thrust_N=thrust,
        rotor_speed_rad_s=speed,
        rotor_torque_Nm=torque,
        motor_efficiency=eta,
        rotor_figure_of_merit=fom,
    )
