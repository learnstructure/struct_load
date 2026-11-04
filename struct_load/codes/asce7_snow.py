"""
ASCE 7-22 Snow and Roof Live Load Provisions.
Compliant with ASCE 7-22 Chapter 7 (Snow Loads) and Section 4.5 (Roof Live Loads).

ASCE 7-22 ground snow loads are strength-level and risk-category specific, so the snow
importance factor Is of earlier editions no longer appears in the roof snow load.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


# ASCE 7-22 Table 7.3-4: minimum snow load upper limit pm,max (psf, kPa) by Risk Category
_PM_MAX: Dict[str, tuple] = {
    "I": (25.0, 1.20),
    "II": (30.0, 1.44),
    "III": (35.0, 1.68),
    "IV": (40.0, 1.92),
}


@dataclass
class SnowParameters:
    """ASCE 7-22 Chapter 7 Snow Design Parameters."""
    ground_snow_load: float          # pg in kPa (SI) or psf (US), Risk-Category-specific (Sec 7.2)
    exposure_factor_ce: float = 1.0  # Ce per Table 7.3-1 (0.9 to 1.2)
    thermal_factor_ct: float = 1.0   # Ct per Tables 7.3-2 and 7.3-3 (1.0 for heated)
    risk_category: str = "II"        # Risk Category per Table 1.5-1 (sets pm,max, Table 7.3-4)
    roof_slope_deg: float = 0.0      # Roof slope angle in degrees


def calculate_flat_roof_snow_load(
    pg: float,
    *,
    units: Union[str, Any, dict],
    ce: float = 1.0,
    ct: float = 1.0,
    risk_category: str = "II",
) -> float:
    """
    Calculate the uniform design roof snow load per ASCE 7-22 Section 7.3:
    pf = 0.7 * Ce * Ct * pg  (Eq. 7.3-1)

    The minimum low-slope roof snow load pm (Sec 7.3.3) is pm = pg where pg <= pm,max and
    pm = pm,max otherwise (Table 7.3-4). pm is a separate uniform load case, so the governing
    uniform roof snow load is max(pf, pm).

    Parameters
    ----------
    pg : float
        Ground snow load in the active pressure unit (kPa for 'kN, m', psf for 'lbf, ft',
        ksf for 'kip, ft').
    ce : float
        Exposure factor Ce.
    ct : float
        Thermal factor Ct.
    risk_category : str
        Risk Category 'I', 'II', 'III', or 'IV' (Table 7.3-4).
    units : str, Units, or dict
        Active engineering units.

    Returns
    -------
    float
        Governing uniform roof snow load in the active pressure unit.
    """
    from struct_core import conversion_factor, parse_units
    rc = str(risk_category).strip().upper()
    if rc not in _PM_MAX:
        raise ValueError(f"Risk category must be one of {list(_PM_MAX)}, got '{risk_category}'.")
    active_units = parse_units(units)
    pm_psf, pm_kpa = _PM_MAX[rc]
    # Convert the tabulated limit of the matching unit family (psf for US, kPa for SI)
    if active_units.length in ("in", "ft") or active_units.force in ("kip", "lbf"):
        pm_max = pm_psf * conversion_factor("lbf, ft", active_units, "pressure")
    else:
        pm_max = pm_kpa * conversion_factor("kN, m", active_units, "pressure")

    pf_basic = 0.7 * ce * ct * pg
    pm = pg if pg <= pm_max else pm_max
    return max(pf_basic, pm)


def calculate_roof_live_load(
    *,
    units: Union[str, Any, dict],
    tributary_area: float = 20.0,
    slope_rise_over_12: float = 0.0,
) -> float:
    """
    Calculate minimum roof live load Lr per ASCE 7 Section 4.5.3:
    Lr = 20 * R1 * R2 (psf)
    or
    Lr = 0.96 * R1 * R2 (kPa)

    12 psf <= Lr <= 20 psf  (0.58 kPa <= Lr <= 0.96 kPa).

    Parameters
    ----------
    tributary_area : float
        Tributary area of roof member At in active area units.
    slope_rise_over_12 : float
        Roof slope pitch F (inches of rise per foot). Default 0 for flat roof.
    units : str, Units, or dict
        Active engineering units.

    Returns
    -------
    float
        Minimum roof live load pressure Lr in the active pressure unit.
    """
    from struct_core import parse_units, conversion_factor, Units
    active_units = parse_units(units)
    is_us = active_units.length in ("in", "ft") or active_units.force in ("kip", "lbf")

    # Eq. 4.8-1 reduction factors are defined on the tributary area in ft²
    at_sqft = tributary_area * conversion_factor(active_units, Units(force="kN", length="ft"), "area")

    # R1 tributary area reduction factor
    if at_sqft <= 200.0:
        r1 = 1.0
    elif at_sqft < 600.0:
        r1 = 1.2 - 0.001 * at_sqft
    else:
        r1 = 0.6

    # R2 slope reduction factor
    f = max(0.0, slope_rise_over_12)
    if f <= 4.0:
        r2 = 1.0
    elif f < 12.0:
        r2 = 1.2 - 0.05 * f
    else:
        r2 = 0.6

    if is_us:
        lr_psf = max(12.0, min(20.0, 20.0 * r1 * r2))
        return lr_psf * conversion_factor("lbf, ft", active_units, "pressure")
    lr_kpa = max(0.575, min(0.9576, 0.9576 * r1 * r2))  # 12-20 psf in kPa
    return lr_kpa * conversion_factor("kN, m", active_units, "pressure")
