"""Optional bus-voltage feedback co-simulation.

Steps the battery one timestep at a time. At each step the propulsion demand is
recomputed using the battery's *actual* terminal (bus) voltage from the previous
step, so the voltage sag under load feeds back into the PMSM current draw. This
corrects the one-directional approximation in
:mod:`bmr.coupling.profile_builder`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bmr.aircraft.airframe import Airframe
from bmr.battery.drive import BatterySimulator
from bmr.mission.profile import Mission
from bmr.motor.pmsm import PMSMMotor
from bmr.rotor.bemt import BEMTRotor


@dataclass
class CoupledResult:
    time_s: np.ndarray
    pack_power_W: np.ndarray
    pack_voltage_V: np.ndarray
    pack_current_A: np.ndarray
    cell_temperature_K: np.ndarray
    soc: np.ndarray


def run_feedback_simulation(
    mission: Mission,
    airframe: Airframe,
    rotor: BEMTRotor,
    motor: PMSMMotor,
    battery: BatterySimulator,
) -> CoupledResult:
    """Co-simulate the full system with bus-voltage feedback."""
    n = len(mission)
    dt = mission.dt

    bus_v = airframe.nominal_bus_voltage_V  # seed for the first step
    t_out, p_out, v_out, i_out, temp_out, soc_out = ([] for _ in range(6))

    for k in range(n):
        T_req = airframe.per_rotor_thrust_N(mission.thrust_factor[k])
        v_axial = mission.axial_velocity_m_s[k]

        rotor_state = rotor.thrust_to_speed(T_req, axial_velocity_m_s=v_axial)
        motor_state = motor.operating_point(
            rotor_state.torque_Nm, rotor_state.rotor_speed_rad_s, bus_v
        )
        pack_power = motor_state.dc_power_W * airframe.num_rotors

        result = battery.step(dt, pack_power)

        # Feed the new terminal voltage back for the next step.
        bus_v = battery.last_pack_voltage_V

        t_out.append(mission.time_s[k])
        p_out.append(pack_power)
        v_out.append(result.pack_voltage_V[-1])
        i_out.append(result.pack_current_A[-1])
        temp_out.append(result.cell_temperature_K[-1])
        soc_out.append(result.soc[-1])

    return CoupledResult(
        time_s=np.array(t_out),
        pack_power_W=np.array(p_out),
        pack_voltage_V=np.array(v_out),
        pack_current_A=np.array(i_out),
        cell_temperature_K=np.array(temp_out),
        soc=np.array(soc_out),
    )
