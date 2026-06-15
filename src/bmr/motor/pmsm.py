"""PMSM (permanent-magnet synchronous motor) dq-axis model.

Steady-state dq model with a loss breakdown -- copper (I^2 R), iron
(hysteresis + eddy) and mechanical (viscous) -- mapping a mechanical operating
point (torque, speed) to the electrical DC-bus demand (power, current).

Control strategy
----------------
* Below base speed the machine runs at ``id = 0`` (this is MTPA for a surface
  PMSM where ``Ld == Lq`` and there is no reluctance torque).
* When the required phase voltage would exceed the inverter ceiling
  (``V_phase_max = bus_voltage / sqrt(3)`` for space-vector PWM), **field
  weakening** injects a negative ``id`` to suppress the flux and keep the
  voltage on the limit ellipse. This is solved analytically from the dq voltage
  equations. Field weakening can be disabled via ``field_weakening: false``.
* The phase current is clamped to ``max_phase_current_A``; if the demand cannot
  be met within the voltage/current limits the returned state is flagged
  ``feasible = False``.

Conventions
-----------
Peak-amplitude dq convention:

* Mechanical back-EMF constant ``Ke = 60 / (2 pi sqrt(3) KV)`` [V_peak.s/rad],
  chosen so the no-load speed at full bus voltage is ``KV * V_dc`` (rpm) given
  the SVPWM phase-voltage ceiling ``V_phase_peak = V_dc / sqrt(3)``.
* Flux linkage ``lambda_pm = Ke / p`` (so ``omega_e * lambda_pm`` equals the
  mechanical back-EMF), torque constant ``Kt = 1.5 * Ke`` [N.m/A_peak].
* ``T = Kt * iq`` (surface PMSM, ``id`` contributes no torque).
* Copper loss with peak amplitudes: ``1.5 * Rs * (id^2 + iq^2)``.
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
    id_A: float = 0.0
    iq_A: float = 0.0
    field_weakening: bool = False
    feasible: bool = True


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
        self.max_phase_current = float(cfg.get("max_phase_current_A", np.inf))
        self.max_line_voltage = float(cfg.get("max_line_voltage_V", np.inf))
        self.use_field_weakening = bool(cfg.get("field_weakening", True))

        # Mechanical back-EMF constant (V_peak per mech rad/s). The sqrt(3)
        # makes the no-load speed at full bus = KV * V_dc given the SVPWM ceiling.
        self.Ke = 60.0 / (2.0 * np.pi * np.sqrt(3.0) * self.KV)
        self.lambda_pm = self.Ke / self.p          # flux linkage [Wb]
        self.Kt = 1.5 * self.Ke                    # N.m/A_peak

    def _voltage_limit(self, bus_voltage_V: float) -> float:
        """Available phase-voltage amplitude (SVPWM), capped by the inverter."""
        v_svpwm = bus_voltage_V / np.sqrt(3.0)
        v_hw = self.max_line_voltage / np.sqrt(3.0)
        return float(min(v_svpwm, v_hw))

    def _phase_voltage(self, id_a: float, iq_a: float, omega_e: float) -> float:
        vd = self.Rs * id_a - omega_e * self.Lq * iq_a
        vq = self.Rs * iq_a + omega_e * (self.lambda_pm + self.Ld * id_a)
        return float(np.hypot(vd, vq))

    def _field_weakening_id(self, iq_a: float, omega_e: float, v_max: float) -> float:
        """Solve for the (negative) id that puts the phase voltage on v_max.

        Voltage ellipse: |V(id)|^2 = v_max^2 expands to a quadratic in id.
        Picks the root closest to zero (least flux weakening). If the limit is
        unreachable, returns the vertex (minimum-voltage) id.
        """
        # vd = Rs*id - Xq*iq ;  vq = Xd*id + (Rs*iq + we*lambda)
        xq = omega_e * self.Lq
        xd = omega_e * self.Ld
        b = -xq * iq_a                       # vd = Rs*id + b
        d = self.Rs * iq_a + omega_e * self.lambda_pm  # vq = Xd*id + d

        qa = self.Rs**2 + xd**2
        qb = 2.0 * (self.Rs * b + xd * d)
        qc = b**2 + d**2 - v_max**2

        disc = qb**2 - 4.0 * qa * qc
        if disc <= 0 or qa <= 0:
            # Cannot reach the limit ellipse: sit at the voltage minimum (vertex).
            return -qb / (2.0 * qa)
        sq = np.sqrt(disc)
        r1 = (-qb + sq) / (2.0 * qa)
        r2 = (-qb - sq) / (2.0 * qa)
        # Field weakening uses id <= 0; choose the root closest to zero.
        candidates = [r for r in (r1, r2) if r <= 1e-9]
        if not candidates:
            return min(r1, r2)
        return max(candidates)

    def operating_point(self, torque_Nm: float, mech_speed_rad_s: float,
                        bus_voltage_V: float) -> MotorState:
        """Return the DC-bus demand for a given mechanical operating point."""
        omega = float(mech_speed_rad_s)
        omega_e = self.p * omega                 # electrical angular frequency
        f_e = omega_e / (2.0 * np.pi)            # electrical frequency [Hz]

        # q-axis current to produce the torque; d-axis starts at 0 (MTPA).
        iq = torque_Nm / self.Kt
        id_a = 0.0
        v_max = self._voltage_limit(bus_voltage_V)
        fw_active = False

        if self.use_field_weakening and self._phase_voltage(0.0, iq, omega_e) > v_max:
            id_a = self._field_weakening_id(iq, omega_e, v_max)
            fw_active = True

        v_phase = self._phase_voltage(id_a, iq, omega_e)

        # Feasibility: within both the phase-current and phase-voltage limits.
        # (If even maximum field weakening cannot hold v_max, the point is
        # over-speed/over-torque for this bus voltage.)
        i_phase = float(np.hypot(id_a, iq))
        feasible = (i_phase <= self.max_phase_current + 1e-9) and (
            v_phase <= v_max + 1e-3
        )

        # Loss terms (copper now includes the d-axis current; peak-amplitude).
        p_copper = 1.5 * self.Rs * (id_a**2 + iq**2)
        p_iron = self.k_h * abs(f_e) + self.k_e * f_e**2
        p_mech_loss = self.c_mech * omega**2

        # Mechanical output and electrical (shaft-side) input.
        p_mech_out = torque_Nm * omega
        p_elec = p_mech_out + p_copper + p_iron + p_mech_loss

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
            id_A=id_a,
            iq_A=iq,
            field_weakening=fw_active,
            feasible=feasible,
        )


def compute_efficiency_map(motor: PMSMMotor, speed_rpm: np.ndarray,
                           torque_Nm: np.ndarray, bus_voltage_V: float):
    """Evaluate motor efficiency over a torque-speed grid at a fixed bus voltage.

    Returns ``(efficiency, field_weakening)`` 2D arrays indexed ``[torque, speed]``.
    Infeasible operating points (voltage/current saturation) are set to NaN in
    the efficiency array.
    """
    speed_rad_s = np.asarray(speed_rpm) * 2.0 * np.pi / 60.0
    eff = np.full((len(torque_Nm), len(speed_rpm)), np.nan)
    fw = np.zeros_like(eff, dtype=bool)

    for i, tq in enumerate(torque_Nm):
        for j, w in enumerate(speed_rad_s):
            st = motor.operating_point(float(tq), float(w), bus_voltage_V)
            fw[i, j] = st.field_weakening
            if st.feasible and st.efficiency > 0:
                eff[i, j] = st.efficiency
    return eff, fw
