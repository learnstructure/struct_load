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
    Project,
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
        Acceleration due to gravity (9.81 m/s² for SI, 386.088 in/s² or 32.2 ft/s² for US).
    unit_system : str
        'si' (density in kg/m³, area in m² -> load in kN/m)
        'us' (density in pcf/UnitWeight, area in in² -> load in kip/in or kip/ft).
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
        elem_type = str(getattr(elem, "type", "")).lower()
        if elem_type not in ("beam", "truss", "column", "frame"):
            continue

        mat = materials_map.get(getattr(elem, "material_id", None))
        sec = sections_map.get(getattr(elem, "section_id", None))
        if not mat or not sec:
            continue

        area = calculate_section_area(sec)
        if area <= 0.0:
            continue

        # Check explicit unit_weight on material first
        unit_weight = getattr(mat, "unit_weight", None)
        if unit_weight is not None and unit_weight > 0.0:
            w = unit_weight * area
        else:
            rho = getattr(mat, "rho", None)
            if rho is None or rho <= 0.0:
                rho = 2400.0 if unit_system.lower() == "si" else 150.0

            if unit_system.lower() == "si":
                # mass per length = rho * area (kg/m)
                # weight per length = (rho * area * g) / 1000 (kN/m)
                w = (rho * area * gravity_accel) / 1000.0
            else:
                # US Customary
                if rho > 1.0:
                    # rho in pcf (e.g. 150 pcf)
                    if gravity_accel > 100.0:
                        w = (rho / 1728.0) * area / 1000.0  # kip/in
                    else:
                        w = (rho * (area / 144.0)) / 1000.0  # kip/ft
                else:
                    # Mass density in kip*s^2/in^4
                    w = rho * gravity_accel * area

        w_mag = float(w)
        if w_mag <= 0.0:
            continue

        dir_lower = direction.lower()
        if dir_lower in ("-z", "z-", "down_z", "gravity"):
            load = DistributedLoad(element_id=elem.id, wx=0.0, wy=0.0, wz=-w_mag)
        elif dir_lower in ("+z", "z+"):
            load = DistributedLoad(element_id=elem.id, wx=0.0, wy=0.0, wz=w_mag)
        elif dir_lower in ("-y", "y-", "down_y"):
            load = DistributedLoad(element_id=elem.id, wx=0.0, wy=-w_mag, wz=0.0)
        elif dir_lower in ("+y", "y+"):
            load = DistributedLoad(element_id=elem.id, wx=0.0, wy=w_mag, wz=0.0)
        elif dir_lower in ("-x", "x-"):
            load = DistributedLoad(element_id=elem.id, wx=-w_mag, wy=0.0, wz=0.0)
        elif dir_lower in ("+x", "x+"):
            load = DistributedLoad(element_id=elem.id, wx=w_mag, wy=0.0, wz=0.0)
        else:
            load = DistributedLoad(element_id=elem.id, wx=0.0, wy=0.0, wz=-w_mag)

        element_loads.append(load)

    return LoadCase(
        id=load_case_id,
        name="Dead Load (Self-Weight)",
        element_loads=element_loads,
        include_self_weight=True,
    )


def apply_self_weight(
    source: Union[StructuralModel, Project],
    target_case_id: Optional[str] = None,
    gravity_accel: Optional[float] = None,
    unit_system: Optional[str] = None,
    direction: Optional[str] = None,
) -> Union[StructuralModel, Project]:
    """
    Materialize self-weight distributed loads for structural elements.

    Scans load cases in the model. For any LoadCase where
    `include_self_weight=True`, computes element self-weight loads, appends
    them to `case.element_loads`, and sets `case.include_self_weight=False`
    to ensure self-weight is never double-counted.

    If no case has `include_self_weight=True`, applies self-weight to
    `target_case_id` (or the default 'DEAD' load case).

    Parameters
    ----------
    source : StructuralModel or Project
        The structural model or project to update.
    target_case_id : str, optional
        Target load case ID if no case currently has include_self_weight=True.
    gravity_accel : float, optional
        Acceleration due to gravity. Defaults to 9.80665 m/s² (SI) or 386.088 in/s² (US).
    unit_system : str, optional
        'si' or 'us'. If source is a Project, automatically inferred from metadata.units.
    direction : str, optional
        '-z' (3D space frames) or '-y' (2D plane frames). Inferred if None.

    Returns
    -------
    StructuralModel or Project
        The updated source with materialized self-weight loads.
    """
    if isinstance(source, Project):
        if source.model is None:
            raise ValueError("Project contains no structural model.")
        model = source.model
        if getattr(source, "metadata", None) and getattr(source.metadata, "units", None):
            units = source.metadata.units
            if unit_system is None:
                if getattr(units, "is_us_customary", False):
                    unit_system = "us"
                elif getattr(units, "is_metric", False):
                    unit_system = "si"
                else:
                    length_u = str(getattr(units, "length", "m")).lower()
                    force_u = str(getattr(units, "force", "kn")).lower()
                    if length_u in ("in", "ft", "inch", "feet") or force_u in ("kip", "kips", "lb", "lbf", "psi", "ksi"):
                        unit_system = "us"
                    else:
                        unit_system = "si"
            if gravity_accel is None:
                gravity_accel = getattr(units, "gravity_acceleration", None)
    elif isinstance(source, StructuralModel):
        model = source
    else:
        raise TypeError(f"Expected Project or StructuralModel, got {type(source).__name__}")

    if unit_system is None:
        unit_system = "si"

    if direction is None:
        has_z = any(abs(getattr(n, "z", 0.0) or 0.0) > 1e-6 for n in model.nodes)
        direction = "-z" if has_z else "-y"

    g_accel = gravity_accel
    if g_accel is None:
        g_accel = 386.08858 if unit_system.lower() == "us" else 9.80665

    # Identify load cases requesting self-weight
    cases_with_sw = [lc for lc in model.load_cases if getattr(lc, "include_self_weight", False)]

    if cases_with_sw:
        for lc in cases_with_sw:
            sw_case = calculate_self_weight(
                model=model,
                load_case_id=lc.id,
                gravity_accel=g_accel,
                unit_system=unit_system,
                direction=direction,
            )
            lc.element_loads.extend(sw_case.element_loads)
            lc.include_self_weight = False
    else:
        target_id = target_case_id or "DEAD"
        target_case = next((lc for lc in model.load_cases if str(lc.id).upper() == str(target_id).upper()), None)
        if not target_case:
            target_case = next((lc for lc in model.load_cases if "DEAD" in str(lc.id).upper()), None)

        sw_case = calculate_self_weight(
            model=model,
            load_case_id=target_id,
            gravity_accel=g_accel,
            unit_system=unit_system,
            direction=direction,
        )

        if target_case:
            target_case.element_loads.extend(sw_case.element_loads)
            target_case.include_self_weight = False
        else:
            sw_case.include_self_weight = False
            model.load_cases.append(sw_case)

    return source
