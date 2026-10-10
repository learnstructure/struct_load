"""
ASCE 7 / IBC Occupancy Live Load Tables and Reduction Engine.
Compliant with ASCE 7-16/22 Table 4.3-1 and Section 4.7.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class OccupancyLiveLoad:
    """Design live load specification for a building occupancy type."""
    key: str
    name: str
    uniform_psf: float
    uniform_kpa: float
    partition_psf: float = 0.0
    partition_kpa: float = 0.0
    reducible: bool = True
    description: str = ""


# ASCE 7-16 / ASCE 7-22 Table 4.3-1 Minimum Uniformly Distributed Live Loads
_OCCUPANCY_DATABASE: Dict[str, OccupancyLiveLoad] = {
    "office": OccupancyLiveLoad(
        key="office",
        name="Offices",
        uniform_psf=50.0,
        uniform_kpa=2.40,
        partition_psf=15.0,
        partition_kpa=0.72,
        reducible=True,
        description="Office space per ASCE 7 Table 4.3-1. 15 psf partition load added per Section 4.3.2.",
    ),
    "residential": OccupancyLiveLoad(
        key="residential",
        name="Residential (Habitable Rooms)",
        uniform_psf=40.0,
        uniform_kpa=1.92,
        partition_psf=0.0,
        partition_kpa=0.0,
        reducible=True,
        description="Residential dwellings, apartments, hotel rooms.",
    ),
    "classroom": OccupancyLiveLoad(
        key="classroom",
        name="Schools (Classrooms)",
        uniform_psf=40.0,
        uniform_kpa=1.92,
        partition_psf=0.0,
        partition_kpa=0.0,
        reducible=True,
        description="Classrooms in educational facilities.",
    ),
    "corridor_first_floor": OccupancyLiveLoad(
        key="corridor_first_floor",
        name="Corridors (First Floor)",
        uniform_psf=100.0,
        uniform_kpa=4.79,
        partition_psf=0.0,
        partition_kpa=0.0,
        reducible=False,
        description="First-floor corridors in public and commercial buildings.",
    ),
    "corridor_upper": OccupancyLiveLoad(
        key="corridor_upper",
        name="Corridors (Upper Floors)",
        uniform_psf=80.0,
        uniform_kpa=3.83,
        partition_psf=0.0,
        partition_kpa=0.0,
        reducible=True,
        description="Corridors above first floor in commercial buildings.",
    ),
    "retail_first_floor": OccupancyLiveLoad(
        key="retail_first_floor",
        name="Retail Stores (First Floor)",
        uniform_psf=100.0,
        uniform_kpa=4.79,
        partition_psf=0.0,
        partition_kpa=0.0,
        reducible=False,
        description="First-floor retail sales areas.",
    ),
    "retail_upper": OccupancyLiveLoad(
        key="retail_upper",
        name="Retail Stores (Upper Floors)",
        uniform_psf=75.0,
        uniform_kpa=3.59,
        partition_psf=0.0,
        partition_kpa=0.0,
        reducible=True,
        description="Upper-floor retail sales areas.",
    ),
    "storage_light": OccupancyLiveLoad(
        key="storage_light",
        name="Storage (Light)",
        uniform_psf=125.0,
        uniform_kpa=5.99,
        partition_psf=0.0,
        partition_kpa=0.0,
        reducible=False,
        description="Light storage warehouse space. Heavy loads (>100 psf) are not reducible per ASCE 7-22.",
    ),
    "storage_heavy": OccupancyLiveLoad(
        key="storage_heavy",
        name="Storage (Heavy)",
        uniform_psf=250.0,
        uniform_kpa=11.97,
        partition_psf=0.0,
        partition_kpa=0.0,
        reducible=False,
        description="Heavy industrial warehouse storage.",
    ),
    "assembly_fixed": OccupancyLiveLoad(
        key="assembly_fixed",
        name="Assembly (Fixed Seats)",
        uniform_psf=60.0,
        uniform_kpa=2.87,
        partition_psf=0.0,
        partition_kpa=0.0,
        reducible=False,
        description="Theaters, auditoriums with fixed seating.",
    ),
    "assembly_movable": OccupancyLiveLoad(
        key="assembly_movable",
        name="Assembly (Movable Seats / Lobbies)",
        uniform_psf=100.0,
        uniform_kpa=4.79,
        partition_psf=0.0,
        partition_kpa=0.0,
        reducible=False,
        description="Lobbies, ballrooms, assembly halls with movable seating.",
    ),
    "parking": OccupancyLiveLoad(
        key="parking",
        name="Parking Garages (Passenger Cars)",
        uniform_psf=40.0,
        uniform_kpa=1.92,
        partition_psf=0.0,
        partition_kpa=0.0,
        reducible=False,
        description="Passenger vehicle parking structures.",
    ),
    "roof_ordinary": OccupancyLiveLoad(
        key="roof_ordinary",
        name="Roof (Ordinary Flat/Pitched)",
        uniform_psf=20.0,
        uniform_kpa=0.96,
        partition_psf=0.0,
        partition_kpa=0.0,
        reducible=True,
        description="Ordinary flat, pitched, or curved roofs (minimum roof live load Lr).",
    ),
}

# Alias lookup mapping for flexible queries (e.g. "offices", "residence", "school", etc.)
_OCCUPANCY_ALIASES: Dict[str, str] = {
    "offices": "office",
    "commercial": "office",
    "business": "office",
    "residential": "residential",
    "apartment": "residential",
    "dwelling": "residential",
    "house": "residential",
    "hotel": "residential",
    "school": "classroom",
    "education": "classroom",
    "classrooms": "classroom",
    "corridor": "corridor_upper",
    "corridors": "corridor_upper",
    "retail": "retail_first_floor",
    "store": "retail_first_floor",
    "shop": "retail_first_floor",
    "storage": "storage_light",
    "warehouse": "storage_light",
    "assembly": "assembly_movable",
    "theater": "assembly_fixed",
    "parking": "parking",
    "garage": "parking",
    "roof": "roof_ordinary",
}


def lookup_occupancy(query: str) -> OccupancyLiveLoad:
    """
    Search the ASCE 7 occupancy database using a fuzzy/alias match.
    Falls back to 'office' if unrecognized.
    """
    norm = query.strip().lower().replace(" ", "_").replace("-", "_")
    if norm in _OCCUPANCY_DATABASE:
        return _OCCUPANCY_DATABASE[norm]

    target_key = _OCCUPANCY_ALIASES.get(norm)
    if target_key and target_key in _OCCUPANCY_DATABASE:
        return _OCCUPANCY_DATABASE[target_key]

    # Substring search
    for key, val in _OCCUPANCY_DATABASE.items():
        if key in norm or norm in key:
            return val

    raise KeyError(
        f"Occupancy '{query}' not found in ASCE 7 database. "
        f"Valid occupancy keys include: {list(_OCCUPANCY_DATABASE.keys())}."
    )


def get_occupancy_live_load(
    occupancy: str,
    include_partitions: bool = True,
    unit_system: str = "si",
) -> float:
    """
    Retrieve the standard design live load intensity for an occupancy.

    Parameters
    ----------
    occupancy : str
        Occupancy type or name (e.g. 'office', 'residential', 'classroom').
    include_partitions : bool
        Whether to add standard partition allowance (15 psf / 0.72 kPa for offices).
    unit_system : str
        'si' for kPa (kN/m²), 'us' for psf.

    Returns
    -------
    float
        Design uniform live load pressure.
    """
    occ = lookup_occupancy(occupancy)
    if unit_system.lower() == "us":
        load = occ.uniform_psf
        if include_partitions and occ.partition_psf > 0:
            load += occ.partition_psf
    else:
        load = occ.uniform_kpa
        if include_partitions and occ.partition_kpa > 0:
            load += occ.partition_kpa
    return round(load, 3)


def calculate_live_load_reduction(
    L0: float,
    tributary_area: float,
    k_ll: float = 2.0,
    is_multi_story: bool = False,
    unit_system: str = "si",
) -> float:
    """
    Calculate reduced design live load per ASCE 7-16/22 Eq. 4.7-1.

    Parameters
    ----------
    L0 : float
        Unreduced live load (kPa or psf).
    tributary_area : float
        Tributary area A_T (m² for SI, ft² for US).
    k_ll : float
        Live load element factor (4.0 for columns/walls, 2.0 for beams/girders, 1.0 for slabs).
    is_multi_story : bool
        Whether member supports multiple floors (controls minimum reduction factor 0.40 vs 0.50).
    unit_system : str
        'si' or 'us'.

    Returns
    -------
    float
        Reduced design live load L (same unit as L0).
    """
    # Convert area to ft² for the empirical ASCE 7 formula:
    # L = L0 * (0.25 + 15 / sqrt(K_LL * A_T))
    area_ft2 = tributary_area if unit_system.lower() == "us" else tributary_area * 10.7639

    ai = k_ll * area_ft2
    if ai < 400.0:
        return L0  # No reduction if influence area K_LL * A_T < 400 ft²

    reduction_factor = 0.25 + (15.0 / math.sqrt(ai))

    # Minimum limits per ASCE 7-22 Section 4.7.2:
    # L >= 0.50 L0 for members supporting 1 floor
    # L >= 0.40 L0 for members supporting 2 or more floors
    min_factor = 0.40 if is_multi_story else 0.50
    reduction_factor = max(min_factor, min(1.0, reduction_factor))

    return round(L0 * reduction_factor, 3)
