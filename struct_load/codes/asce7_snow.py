"""
ASCE 7-16 / ASCE 7-22 Snow and Roof Live Load Provisions.
Compliant with ASCE 7 Chapter 7 (Snow Loads) and Section 4.5 (Roof Live Loads).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class SnowParameters:
    """ASCE 7 Chapter 7 Snow Design Parameters."""
    ground_snow_load: float          # pg in kPa (SI) or psf (US)
    exposure_factor_ce: float = 1.0  # Ce per Table 7.3-1 (0.9 to 1.2)
    thermal_factor_ct: float = 1.0   # Ct per Table 7.3-2 (1.0 for heated)
    importance_factor_is: float = 1.0# Is per Table 1.5-2 (Cat I: 0.8, II: 1.0, III: 1.1, IV: 1.2)
    roof_slope_deg: float = 0.0      # Roof slope angle in degrees


def calculate_flat_roof_snow_load(
    pg: float,
    ce: float = 1.0,
    ct: float = 1.0,
    is_factor: float = 1.0,
    units: Union[str, Any, dict] = "kN, m",
) -> float:
    """
    Calculate flat roof snow load pf per ASCE 7 Section 7.3:
    pf = 0.7 * Ce * Ct * Is * pg

    Subject to minimum roof snow load pm:
    - If pg <= 20 psf (0.96 kPa): pm = Is * pg
    - If pg > 20 psf (0.96 kPa):  pm = 20 * Is psf (0.96 * Is kPa)

    Parameters
    ----------
    pg : float
        Ground snow load in active pressure units.
    ce : float
        Exposure factor Ce.
    ct : float
        Thermal factor Ct.
    is_factor : float
        Snow importance factor Is.
    units : str, Units, or dict
        Active engineering units.

    Returns
    -------
    float
        Design flat roof snow load.
    """
    from struct_core import parse_units
    active_units = parse_units(units)
    is_us = active_units.length in ("in", "ft")
    threshold = 20.0 if is_us else 0.9576  # 20 psf in kPa = 0.9576 kPa

    pf_basic = 0.7 * ce * ct * is_factor * pg

    if pg <= threshold:
        pm = is_factor * pg
    else:
        pm = threshold * is_factor

    pf = max(pf_basic, pm)
    return round(pf, 3)


def calculate_roof_live_load(
    tributary_area: float = 20.0,
    slope_rise_over_12: float = 0.0,
    units: Union[str, Any, dict] = "kN, m",
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
        Minimum roof live load pressure Lr.
    """
    from struct_core import parse_units, conversion_factor, Units
    active_units = parse_units(units)
    is_us = active_units.length in ("in", "ft") or active_units.force in ("kip", "lb", "lbf")

    # In US building code practice, tributary area is in ft²; in SI it is in m²
    if is_us:
        at_sqft = tributary_area
    else:
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
        lr_base = 20.0 * r1 * r2
        lr = max(12.0, min(20.0, lr_base))
    else:
        lr_base = 0.9576 * r1 * r2  # ~20 psf in kPa
        lr = max(0.575, min(0.9576, lr_base))

    return round(lr, 3)
