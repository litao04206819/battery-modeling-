"""PMSM (permanent-magnet synchronous motor) dq-axis model.

Steady-state dq model with id = 0 (maximum-torque-per-amp for a surface PMSM)
plus a loss breakdown: copper (I^2 R), iron (hysteresis + eddy), and mechanical
(viscous). Maps a mechanical operating point (torque, speed) to the electrical
DC-bus demand (power, current) at a given bus voltage.

Conventions
-----------
* ``iq`` is treated as the RMS q-axis current; copper loss = 3 Rs iq^2.
* Torque constant Kt [N.m/A] is taken equal to the back-EMF constant
  Ke [V.s/rad] derived from KV: ``Ke = 60 / (2 pi KV)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MotorState:
    """Electrical operating point for one motor."""

    dc_power_W: float
    dc_current_A: float
    phase_current_A: float
    phase_voltage_V: float
    efficiency: float
    copper_loss_W: float
    iron_loss_W: float
    mech_loss_W: float


class PMSMMotor:
    def __init__(self, cfg: dict):
        self.p = int(cfg["pole_pairs"])
        self.KV = float(cfg["kv_rpm_per_volt"])
        self.Rs = float(cfg["phase_resistance_ohm"])
        self.Ld = float(cfg["d_axis_inductance_H"])
        self.Lq = float(cfg["q_axis_inductance_H"])
        self.k_h = float(cfg["iron_loss_coeff_hyst"])
        self.k_e = float(cfg["iron_loss_coeff_eddy"])
        self.c_mech = float(cfg["mechanical_loss_coeff"])
        self.inverter_efficiency = float(cfg.get("inverter_efficiency", 0.97))

        # Back-EMF / torque constant (SI).
        self.Ke = 60.0 / (2.0 * np.pi * self.KV)   # V.s/rad
        self.Kt = self.Ke                          # N.m/A (ideal)
        self.lambda_pm = self.Ke                   # flux linkage [Wb]

    def operating_point(self, torque_Nm: float, mech_speed_rad_s: float,
                        bus_voltage_V: float) -> MotorState:
        """Return the DC-bus demand for a given mechanical operating point."""
        omega = float(mech_speed_rad_s)
        omega_e = self.p * omega                 # electrical angular frequency
        f_e = omega_e / (2.0 * np.pi)            # electrical frequency [Hz]

        # q-axis current to produce the torque (id = 0 control).
        iq = torque_Nm / self.Kt
        i_phase = abs(iq)

        # Loss terms.
        p_copper = 3.0 * self.Rs * iq**2
        p_iron = self.k_h * abs(f_e) + self.k_e * f_e**2
        p_mech_loss = self.c_mech * omega**2

        # Mechanical output and electrical (shaft-side) input.
        p_mech_out = torque_Nm * omega
        p_elec = p_mech_out + p_copper + p_iron + p_mech_loss

        # dq voltages -> phase voltage magnitude.
        vd = -omega_e * self.Lq * iq
        vq = self.Rs * iq + omega_e * self.lambda_pm
        v_phase = float(np.hypot(vd, vq))

        # DC bus side (inverter losses).
        p_dc = p_elec / max(self.inverter_efficiency, 1e-6)
        i_dc = p_dc / max(bus_voltage_V, 1e-6)
        efficiency = p_mech_out / p_dc if p_dc > 0 else 0.0

        return MotorState(
            dc_power_W=p_dc,
            dc_current_A=i_dc,
            phase_current_A=i_phase,
            phase_voltage_V=v_phase,
            efficiency=efficiency,
            copper_loss_W=p_copper,
            iron_loss_W=p_iron,
            mech_loss_W=p_mech_loss,
        )
