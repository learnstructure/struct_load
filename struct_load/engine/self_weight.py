"""
Self-weight calculation engine for structural elements.
Calculates distributed gravity loads directly from element geometry and material density.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Union
from struct_core import (
    CircularSection,
    DistributedLoad,
    GeneralSection,
    LoadCase,
    RectangularSection,
    StructuralModel,
)


def calculate_section_area(section: object) -> float:
    """
    Calculate cross-sectional area A (m² for SI, in² for US) from a section schema.
    """
    if isinstance(section, RectangularSection):
        return section.b * section.h
    elif isinstance(section, CircularSection):
        return math.pi * ((section.d / 2.0) ** 2)
    elif isinstance(section, GeneralSection):
        return getattr(section, "A", 0.0)
    elif hasattr(section, "A") and getattr(section, "A") > 0:
        return float(getattr(section, "A"))
    elif hasattr(section, "b") and hasattr(section, "h"):
        return float(getattr(section, "b")) * float(getattr(section, "h"))
    return 0.0


def calculate_self_weight(
    model: StructuralModel,
    load_case_id: str = "Dead_SW",
    gravity_accel: float = 9.80665,
    unit_system: str = "si",
    direction: str = "-y",
) -> LoadCase:
    """
    Compute self-weight line loads for all beam/column elements in a StructuralModel.

    Parameters
    ----------
    model : StructuralModel
        The structural model containing materials, sections, and elements.
    load_case_id : str
        ID for the generated LoadCase.
    gravity_accel : float
        Acceleration due to gravity (9.81 m/s² for SI, 32.2 ft/s² for US).
    unit_system : str
        'si' (density in kg/m³, area in m² -> load in kN/m)
        'us' (density in pcf, area in in² -> load in kip/ft).
    direction : str
        Direction of gravity: '-y' (2D plane frames) or '-z' (3D space frames).

    Returns
    -------
    LoadCase
        A struct_core.LoadCase populated with DistributedLoad items.
    """
    materials_map = {m.id: m for m in model.materials}
    sections_map = {s.id: s for s in model.sections}

    element_loads: List[DistributedLoad] = []

    for elem in model.elements:
        # Only beam and truss elements have continuous self-weight
        if getattr(elem, "type", "") not in ("beam", "truss"):
            continue

        mat = materials_map.get(getattr(elem, "material_id", None))
        sec = sections_map.get(getattr(elem, "section_id", None))
        if not mat or not sec:
            continue

        area = calculate_section_area(sec)
        # Material density rho: kg/m³ for SI, or pcf (lb/ft³) for US
        rho = getattr(mat, "rho", 2400.0) or 2400.0

        if unit_system.lower() == "si":
            # mass per length = rho * area (kg/m)
            # weight per length = (rho * area * g) / 1000 (kN/m)
            w = (rho * area * gravity_accel) / 1000.0
        else:
            # US Customary: rho in pcf, area in in²
            # w = (rho * (area / 144.0)) / 1000.0 (kip/ft)
            w = (rho * (area / 144.0)) / 1000.0

        w_mag = round(w, 4)
        if w_mag <= 0:
            continue

        dir_lower = direction.lower()
        if dir_lower == "-y":
            load = DistributedLoad(element_id=elem.id, wx=0.0, wy=-w_mag)
        elif dir_lower == "+y":
            load = DistributedLoad(element_id=elem.id, wx=0.0, wy=w_mag)
        elif dir_lower == "-z":
            # For 3D elements local/global convention
            load = DistributedLoad(element_id=elem.id, wx=0.0, wy=-w_mag)
        else:
            load = DistributedLoad(element_id=elem.id, wx=0.0, wy=-w_mag)

        element_loads.append(load)

    return LoadCase(
        id=load_case_id,
        name="Dead Load (Self-Weight)",
        element_loads=element_loads,
        include_self_weight=True,
    )
