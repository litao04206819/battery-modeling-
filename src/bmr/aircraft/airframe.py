"""Airframe-level trim: convert a flight condition into per-rotor thrust demand."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Airframe:
    mass_kg: float
    num_rotors: int
    gravity_m_s2: float = 9.80665
    air_density_kg_m3: float = 1.225
    nominal_bus_voltage_V: float = 44.4

    @classmethod
    def from_dict(cls, cfg: dict) -> "Airframe":
        return cls(
            mass_kg=float(cfg["mass_kg"]),
            num_rotors=int(cfg["num_rotors"]),
            gravity_m_s2=float(cfg.get("gravity_m_s2", 9.80665)),
            air_density_kg_m3=float(cfg.get("air_density_kg_m3", 1.225)),
            nominal_bus_voltage_V=float(cfg.get("nominal_bus_voltage_V", 44.4)),
        )

    @property
    def weight_N(self) -> float:
        return self.mass_kg * self.gravity_m_s2

    def per_rotor_thrust_N(self, thrust_factor: float) -> float:
        """Total thrust = thrust_factor * weight, shared equally across rotors."""
        return thrust_factor * self.weight_N / self.num_rotors
