"""Parse a mission YAML into a discretised time series of flight conditions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MissionSample:
    """A single time sample's flight condition."""

    time_s: float
    segment_name: str
    seg_type: str
    thrust_factor: float
    axial_velocity_m_s: float   # vertical climb rate (+up) or cruise inflow proxy


@dataclass
class Mission:
    time_s: np.ndarray
    thrust_factor: np.ndarray
    axial_velocity_m_s: np.ndarray
    segment_name: list[str]
    dt: float

    @classmethod
    def from_dict(cls, cfg: dict) -> "Mission":
        dt = float(cfg.get("time_step_s", 1.0))
        times: list[float] = []
        tf: list[float] = []
        vax: list[float] = []
        names: list[str] = []

        t = 0.0
        for seg in cfg["segments"]:
            n = int(round(float(seg["duration_s"]) / dt))
            seg_type = seg["type"]
            factor = float(seg.get("thrust_factor", 1.0))
            # Axial inflow: climb rate for hover/climb/descend; cruise uses
            # forward airspeed as an axial-inflow proxy in the BEMT momentum model.
            if seg_type == "cruise":
                v_axial = float(seg.get("airspeed_m_s", 0.0))
            else:
                v_axial = float(seg.get("climb_rate_m_s", 0.0))
            for _ in range(n):
                times.append(t)
                tf.append(factor)
                vax.append(v_axial)
                names.append(seg["name"])
                t += dt

        return cls(
            time_s=np.array(times),
            thrust_factor=np.array(tf),
            axial_velocity_m_s=np.array(vax),
            segment_name=names,
            dt=dt,
        )

    def __len__(self) -> int:
        return len(self.time_s)
