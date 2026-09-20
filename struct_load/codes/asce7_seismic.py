"""
ASCE 7-16 / ASCE 7-22 Equivalent Lateral Force (ELF) Seismic Procedure.
Compliant with ASCE 7 Chapter 11 and Section 12.8.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class SeismicParameters:
    """Design spectral response accelerations and system coefficients."""
    sds: float                     # Design spectral response acceleration at short periods (g)
    sd1: float                     # Design spectral response acceleration at 1-sec period (g)
    r_factor: float = 8.0          # Response modification coefficient (e.g. 8.0 for RC Special Moment Frame)
    importance_factor: float = 1.0 # Seismic importance factor I_e (Risk Cat I/II: 1.0, III: 1.25, IV: 1.5)
    tl: float = 8.0                # Long-period transition period (s)
    system_type: str = "rc_smf"    # 'rc_smf', 'steel_smf', 'rc_imf', 'rc_shear_wall', 'other'


@dataclass
class StoryForceResult:
    """Calculated lateral seismic forces at each story level."""
    base_shear: float
    cs: float
    ta: float
    total_weight: float
    story_forces: Dict[int, float]  # story index -> F_x (kN or kips)


def approximate_fundamental_period(
    total_height: float,
    system_type: str = "rc_smf",
    unit_system: str = "si",
) -> float:
    """
    Calculate approximate fundamental period Ta per ASCE 7 Section 12.8.2.1:
    Ta = Ct * hn^x

    Parameters
    ----------
    total_height : float
        Building height from base to highest level (m for SI, ft for US).
    system_type : str
        Structural system type ('rc_smf', 'steel_smf', 'rc_shear_wall', 'other').
    unit_system : str
        'si' (meters) or 'us' (feet).

    Returns
    -------
    float
        Approximate fundamental period in seconds.
    """
    is_us = unit_system.lower() == "us"
    sys = system_type.lower()

    if "steel_smf" in sys or "steel_moment" in sys:
        ct = 0.028 if is_us else 0.0724
        x = 0.8
    elif "rc_smf" in sys or "concrete_moment" in sys:
        ct = 0.016 if is_us else 0.0466
        x = 0.9
    elif "steel_braced" in sys or "eccentrically" in sys:
        ct = 0.030 if is_us else 0.0731
        x = 0.75
    else:  # All other systems / shear walls
        ct = 0.020 if is_us else 0.0488
        x = 0.75

    ta = ct * (total_height ** x)
    return round(ta, 3)


def calculate_seismic_response_coefficient(
    params: SeismicParameters,
    period: float,
) -> float:
    """
    Calculate seismic response coefficient Cs per ASCE 7 Section 12.8.1.1:
    Cs = SDS / (R / Ie)
    """
    r_ie = params.r_factor / params.importance_factor
    cs_basic = params.sds / r_ie

    # Upper bound (Eq. 12.8-3 & 12.8-4)
    if period <= params.tl:
        cs_max = params.sd1 / (period * r_ie)
    else:
        cs_max = (params.sd1 * params.tl) / ((period ** 2) * r_ie)

    # Lower bound (Eq. 12.8-5)
    cs_min = max(0.044 * params.sds * params.importance_factor, 0.01)

    # S1 >= 0.6g condition (Eq. 12.8-6)
    # (Optional conservatism: if sd1 is high)
    if params.sd1 >= 0.6:
        cs_min = max(cs_min, 0.5 * params.sd1 / r_ie)

    cs = min(cs_basic, cs_max)
    cs = max(cs, cs_min)
    return round(cs, 4)


def calculate_seismic_elf(
    story_elevations: List[float],
    story_weights: List[float],
    params: SeismicParameters,
    unit_system: str = "si",
    custom_period: Optional[float] = None,
) -> StoryForceResult:
    """
    Perform full ASCE 7 Equivalent Lateral Force (ELF) analysis:
    Computes total base shear V and distributes lateral forces Fx along height.

    Parameters
    ----------
    story_elevations : List[float]
        Elevation of each story above base (m or ft), sorted ascending.
        E.g. [3.5, 7.0] for a 2-story building.
    story_weights : List[float]
        Effective seismic weight of each story level (kN or kips).
    params : SeismicParameters
        Seismic design parameters (SDS, SD1, R, etc.).
    unit_system : str
        'si' or 'us'.
    custom_period : float, optional
        User-specified or modal period T. If None, computes Ta.

    Returns
    -------
    StoryForceResult
        Contains base shear and lateral force Fx at each story.
    """
    if len(story_elevations) != len(story_weights):
        raise ValueError("story_elevations and story_weights must have the same length.")

    hn = story_elevations[-1]
    ta = custom_period if custom_period is not None else approximate_fundamental_period(hn, params.system_type, unit_system)
    cs = calculate_seismic_response_coefficient(params, ta)

    w_total = sum(story_weights)
    base_shear = round(cs * w_total, 3)

    # Exponent k related to structural period (ASCE 7 Eq. 12.8-12)
    if ta <= 0.5:
        k = 1.0
    elif ta >= 2.5:
        k = 2.0
    else:
        k = 1.0 + (ta - 0.5) / 2.0  # Linear interpolation between 1 and 2

    # Vertical distribution factor Cvx = (wx * hx^k) / sum(wi * hi^k)
    weighted_heights = [w * (h ** k) for w, h in zip(story_weights, story_elevations)]
    denom = sum(weighted_heights)

    story_forces: Dict[int, float] = {}
    for idx, (wh, elev) in enumerate(zip(weighted_heights, story_elevations)):
        cvx = (wh / denom) if denom > 1e-9 else (1.0 / len(story_elevations))
        fx = round(cvx * base_shear, 3)
        story_forces[idx + 1] = fx

    return StoryForceResult(
        base_shear=base_shear,
        cs=cs,
        ta=ta,
        total_weight=w_total,
        story_forces=story_forces,
    )
