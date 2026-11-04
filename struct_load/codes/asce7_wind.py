"""
ASCE 7-22 Directional Wind Force Calculator (Chapter 26 and Chapter 27, MWFRS).

Windward wall pressure uses qz at each level (Cp = 0.8); leeward wall pressure uses qh at the mean
roof height with Cp from L/B (Fig. 27.3-1). Internal pressure cancels on the net lateral load.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


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
    l_over_b: float = 1.0           # Plan depth / width ratio for the leeward Cp (Fig. 27.3-1); 1.0 gives -0.5


@dataclass
class StoryWindResult:
    """Calculated lateral wind forces at each story level."""
    total_base_shear: float
    story_pressures: Dict[int, float]  # story index -> net pressure (kPa or psf)
    story_forces: Dict[int, float]     # story index -> lateral force (kN or kips)


# ASCE 7-22 Table 26.11-1 terrain exposure constants: alpha, zg (m)
_EXPOSURE_CONSTANTS = {"B": (7.5, 1000.0), "C": (9.8, 750.0), "D": (11.5, 590.0)}


def calculate_kz(z: float, exposure: str = "B", *, units: Union[str, Any, dict]) -> float:
    """
    Velocity pressure exposure coefficient per ASCE 7-22 Table 26.10-1, Note 1:
    Kz = 2.41 (z / zg)^(2/alpha) for 4.6 m <= z <= zg (z < 4.6 m uses 4.6 m; z > zg gives 2.41),
    with alpha and zg from Table 26.11-1.
    """
    from struct_core import parse_units, conversion_factor, Units
    active_units = parse_units(units)
    z_m = z * conversion_factor(active_units, Units(force="kN", length="m"), "length")

    exp = exposure.upper().strip()
    if exp not in _EXPOSURE_CONSTANTS:
        raise ValueError(
            f"Invalid exposure category '{exposure}'. ASCE 7 specifies exposure categories 'B', 'C', or 'D'."
        )
    alpha, zg = _EXPOSURE_CONSTANTS[exp]
    z_eff = min(max(4.6, z_m), zg)
    return round(2.41 * ((z_eff / zg) ** (2.0 / alpha)), 3)


def leeward_wall_cp(l_over_b: float) -> float:
    """Leeward wall pressure coefficient Cp (ASCE 7-22 Fig. 27.3-1): -0.5 (L/B <= 1), -0.3 (2), -0.2 (>= 4)."""
    if l_over_b <= 1.0:
        return -0.5
    if l_over_b <= 2.0:
        return -0.5 + 0.2 * (l_over_b - 1.0)
    if l_over_b < 4.0:
        return -0.3 + 0.1 * (l_over_b - 2.0) / 2.0
    return -0.2


def calculate_wind_forces(
    story_elevations: List[float],
    tributary_widths: List[float],
    story_heights: List[float],
    params: WindParameters,
    *,
    units: Union[str, Any, dict],
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
    from struct_core import conversion_factor, parse_units
    active_units = parse_units(units)
    is_si = active_units.length in ("m", "mm", "cm")

    # Velocity pressure (Eq. 26.10-1): q = 0.00256 Kz Kzt Ke V^2 (psf, V in mph), 0.613 Kz Kzt Ke V^2
    # (N/m^2, V in m/s). Kd enters the MWFRS pressure p = q Kd G Cp (Eq. 27.3-1).
    coeff = 0.613 if is_si else 0.00256
    # pressures are returned in the model's pressure unit so that p x (model area) is a model force
    to_model = (
        conversion_factor("kN, m", active_units, "pressure") if is_si
        else conversion_factor("lbf, ft", active_units, "pressure")
    )

    def velocity_pressure(z: float) -> float:
        kz = calculate_kz(z, params.exposure_category, units=active_units)
        q = coeff * kz * params.kzt * params.ke * (params.basic_wind_speed ** 2)
        return (q / 1000.0 if is_si else q) * to_model  # kPa or psf -> model pressure unit

    story_pressures: Dict[int, float] = {}
    story_forces: Dict[int, float] = {}
    total_force = 0.0

    g = params.gust_factor
    # Leeward wall: Cp from L/B with qh at the mean roof height (taken as the top level), uniform over height
    qh = velocity_pressure(max(story_elevations))
    p_leeward = qh * params.kd * g * abs(leeward_wall_cp(params.l_over_b))

    for idx, (z, trib_w, h_story) in enumerate(zip(story_elevations, tributary_widths, story_heights)):
        # Windward wall: Cp = 0.8 with qz at the level
        p_windward = velocity_pressure(z) * params.kd * g * 0.8
        p_net = p_windward + p_leeward
        story_pressures[idx + 1] = p_net

        # Story lateral force F = p_net * Tributary Area
        area = trib_w * h_story
        f_story = p_net * area
        story_forces[idx + 1] = f_story
        total_force += f_story

    return StoryWindResult(
        total_base_shear=total_force,
        story_pressures=story_pressures,
        story_forces=story_forces,
    )
