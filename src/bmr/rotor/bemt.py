"""Blade-element momentum theory (BEMT) rotor aerodynamics.

Discretises the blade into radial elements and balances blade-element forces
against annulus momentum to solve for the induced inflow, including a Prandtl
tip-loss factor. Works in axial flight (hover / climb / descent); forward
airspeed enters as an additional axial inflow component (a momentum-theory
approximation adequate for a propulsion-energy study).

Solving directions
------------------
* :meth:`solve` - forward: given rotor speed -> thrust, torque, power.
* :meth:`thrust_to_speed` - inverse: given a required thrust -> rotor speed,
  torque, power (Brent root-find on :meth:`solve`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass
class RotorState:
    """Result of a single BEMT operating-point evaluation."""

    thrust_N: float
    torque_Nm: float
    mech_power_W: float
    rotor_speed_rad_s: float
    figure_of_merit: float


class BEMTRotor:
    def __init__(self, cfg: dict, air_density_kg_m3: float = 1.225):
        geo = cfg["geometry"]
        self.R = 0.5 * float(geo["diameter_m"])
        self.B = int(geo["num_blades"])
        self.root = float(geo["root_cutout_frac"])
        self.ne = int(geo["num_elements"])
        self.rho = float(air_density_kg_m3)

        # Radial stations (element centres) and width.
        edges = np.linspace(self.root * self.R, self.R, self.ne + 1)
        self.r = 0.5 * (edges[:-1] + edges[1:])
        self.dr = np.diff(edges)

        # Chord and twist distributions (linear root->tip).
        rR = self.r / self.R
        frac = (rR - self.root) / (1.0 - self.root)
        self.chord = (
            float(cfg["chord"]["root_m"])
            + (float(cfg["chord"]["tip_m"]) - float(cfg["chord"]["root_m"])) * frac
        )
        twist_deg = (
            float(cfg["twist"]["root_deg"])
            + (float(cfg["twist"]["tip_deg"]) - float(cfg["twist"]["root_deg"])) * frac
        )
        self.theta = np.deg2rad(twist_deg)

        af = cfg["aerofoil"]
        self.cl_alpha = float(af["cl_alpha_per_rad"])
        self.alpha_0 = np.deg2rad(float(af["alpha_0_deg"]))
        self.cl_max = float(af["cl_max"])
        self.cd0 = float(af["cd0"])
        self.cd_k = float(af["cd_induced_k"])

        self.disk_area = np.pi * self.R**2

    # ------------------------------------------------------------------
    def _element_inflow(self, omega: float, r: float, c: float, theta: float,
                        dr: float, Vc: float) -> tuple[float, float]:
        """Fixed-point solve for induced velocity vi at one annulus.

        Returns (dT, dQ) for the annulus.
        """
        U_T = omega * r
        if U_T <= 0:
            return 0.0, 0.0

        vi = 1.0  # initial guess [m/s]
        for _ in range(100):
            U_P = Vc + vi
            phi = np.arctan2(U_P, U_T)
            U2 = U_T**2 + U_P**2

            # Prandtl tip-loss factor.
            f = self.B / 2.0 * (self.R - r) / max(r * np.sin(phi), 1e-6)
            F = (2.0 / np.pi) * np.arccos(np.exp(-abs(f)))
            F = max(F, 1e-4)

            alpha = theta - phi
            cl = np.clip(self.cl_alpha * (alpha - self.alpha_0),
                         -self.cl_max, self.cl_max)
            cd = self.cd0 + self.cd_k * cl**2

            dL = 0.5 * self.rho * U2 * c * cl * dr
            dD = 0.5 * self.rho * U2 * c * cd * dr
            dT = self.B * (dL * np.cos(phi) - dD * np.sin(phi))

            # Annulus momentum: dT = 4*pi*rho*(Vc+vi)*vi*r*dr*F
            denom = 4.0 * np.pi * self.rho * (Vc + vi) * r * dr * F
            if denom <= 0:
                break
            vi_new = dT / denom
            if not np.isfinite(vi_new) or vi_new < 0:
                vi_new = max(vi * 0.5, 1e-3)
            if abs(vi_new - vi) < 1e-4:
                vi = vi_new
                break
            vi = 0.5 * vi + 0.5 * vi_new  # under-relax

        # Recompute final forces with converged vi.
        U_P = Vc + vi
        phi = np.arctan2(U_P, U_T)
        U2 = U_T**2 + U_P**2
        alpha = theta - phi
        cl = np.clip(self.cl_alpha * (alpha - self.alpha_0), -self.cl_max, self.cl_max)
        cd = self.cd0 + self.cd_k * cl**2
        dL = 0.5 * self.rho * U2 * c * cl * dr
        dD = 0.5 * self.rho * U2 * c * cd * dr
        dT = self.B * (dL * np.cos(phi) - dD * np.sin(phi))
        dQ = self.B * (dL * np.sin(phi) + dD * np.cos(phi)) * r
        return dT, dQ

    # ------------------------------------------------------------------
    def solve(self, rotor_speed_rad_s: float, axial_velocity_m_s: float = 0.0
              ) -> RotorState:
        """Forward solve: rotor speed -> thrust/torque/power."""
        omega = float(rotor_speed_rad_s)
        T = 0.0
        Q = 0.0
        for i in range(self.ne):
            dT, dQ = self._element_inflow(
                omega, self.r[i], self.chord[i], self.theta[i],
                self.dr[i], axial_velocity_m_s,
            )
            T += dT
            Q += dQ

        P = Q * omega
        # Hover figure of merit (ideal power / actual power).
        if T > 0 and P > 0:
            P_ideal = T**1.5 / np.sqrt(2.0 * self.rho * self.disk_area)
            fom = P_ideal / P
        else:
            fom = 0.0
        return RotorState(
            thrust_N=T, torque_Nm=Q, mech_power_W=P,
            rotor_speed_rad_s=omega, figure_of_merit=fom,
        )

    def thrust_to_speed(self, required_thrust_N: float,
                        axial_velocity_m_s: float = 0.0,
                        omega_max_rad_s: float = 1500.0) -> RotorState:
        """Inverse solve: required thrust -> rotor operating point."""
        if required_thrust_N <= 0:
            return self.solve(0.0, axial_velocity_m_s)

        def residual(omega: float) -> float:
            return self.solve(omega, axial_velocity_m_s).thrust_N - required_thrust_N

        # Bracket: find an upper omega that produces enough thrust.
        lo, hi = 1.0, 50.0
        while residual(hi) < 0 and hi < omega_max_rad_s:
            hi *= 1.5
        omega = brentq(residual, lo, hi, xtol=1e-2)
        return self.solve(omega, axial_velocity_m_s)
