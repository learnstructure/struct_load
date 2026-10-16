"""
ASCE 7-16 / ASCE 7-22 Directional Wind Force Calculator.
Compliant with ASCE 7 Chapter 26 and Chapter 27 (MWFRS).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class WindParameters:
    """Design wind criteria for a building site."""
    basic_wind_speed: float         # V in m/s (SI) or mph (US)
    exposure_category: str = "B"    # 'B' (suburban/urban), 'C' (open terrain), 'D' (coastal)
    risk_category: str = "II"       # 'I', 'II', 'III', 'IV'
    kd: float = 0.85                # Wind directionality factor (Table 26.6-1)
    kzt: float = 1.0                # Topographic factor (Section 26.8)
    ke: float = 1.0                 # Ground elevation factor (Section 26.9)
    gust_factor: float = 0.85       # Gust-effect factor G (Section 26.11)


@dataclass
class StoryWindResult:
    """Calculated lateral wind forces at each story level."""
    total_base_shear: float
    story_pressures: Dict[int, float]  # story index -> net pressure (kPa or psf)
    story_forces: Dict[int, float]     # story index -> lateral force (kN or kips)


def calculate_kz(z: float, exposure: str = "B", units: Union[str, Any, dict] = "kN, m") -> float:
    """
    Velocity pressure exposure coefficient Kz per ASCE 7 Table 26.10-1.
    """
    from struct_core import parse_units, conversion_factor, Units
    active_units = parse_units(units)
    z_m = z * conversion_factor(active_units, Units(force="kN", length="m"), "length")
    z_eff = max(4.57, z_m)  # Min height 15 ft (~4.57 m)

    exp = exposure.upper().strip()
    if exp == "B":
        alpha, zg = 7.0, 365.76
    elif exp == "C":
        alpha, zg = 9.5, 274.32
    elif exp == "D":
        alpha, zg = 11.5, 213.36
    else:
        raise ValueError(
            f"Invalid exposure category '{exposure}'. ASCE 7 specifies exposure categories 'B', 'C', or 'D'."
        )

    kz = 2.01 * ((z_eff / zg) ** (2.0 / alpha))
    return round(kz, 3)


def calculate_wind_forces(
    story_elevations: List[float],
    tributary_widths: List[float],
    story_heights: List[float],
    params: WindParameters,
    units: Union[str, Any, dict] = "kN, m",
) -> StoryWindResult:
    """
    Calculate story lateral wind forces along the height of a building frame.

    Parameters
    ----------
    story_elevations : List[float]
        Elevation of each level (m or ft).
    tributary_widths : List[float]
        Width of building normal to wind tributary to this frame (m or ft).
    story_heights : List[float]
        Tributary story height for each level (m or ft).
    params : WindParameters
        Wind criteria (speed, exposure, etc.).
    units : str, Units, or dict
        Active engineering units.

    Returns
    -------
    StoryWindResult
        Contains net pressures and story lateral forces.
    """
    from struct_core import parse_units
    active_units = parse_units(units)
    is_si = active_units.length in ("m", "mm", "cm")

    # Velocity pressure qz:
    # SI: qz = 0.613 * Kz * Kzt * Kd * Ke * V^2 (N/m² -> /1000 for kPa)
    # US: qz = 0.00256 * Kz * Kzt * Kd * Ke * V^2 (psf)
    coeff = 0.613 if is_si else 0.00256

    story_pressures: Dict[int, float] = {}
    story_forces: Dict[int, float] = {}
    total_force = 0.0

    # Net pressure coefficient: Windward Cp=0.8, Leeward Cp=-0.5 (net Cp = 1.3)
    net_cp = 1.3
    g = params.gust_factor

    for idx, (z, trib_w, h_story) in enumerate(zip(story_elevations, tributary_widths, story_heights)):
        kz = calculate_kz(z, params.exposure_category, units=active_units)
        qz_raw = coeff * kz * params.kzt * params.kd * params.ke * (params.basic_wind_speed ** 2)

        if is_si:
            qz = qz_raw / 1000.0  # Convert N/m² to kPa
        else:
            qz = qz_raw  # psf

        # Net lateral design pressure p = qz * G * Cp
        p_net = round(qz * g * net_cp, 3)
        story_pressures[idx + 1] = p_net

        # Story lateral force F = p_net * Tributary Area
        area = trib_w * h_story
        f_story = round(p_net * area, 3)
        story_forces[idx + 1] = f_story
        total_force += f_story

    return StoryWindResult(
        total_base_shear=round(total_force, 3),
        story_pressures=story_pressures,
        story_forces=story_forces,
    )
