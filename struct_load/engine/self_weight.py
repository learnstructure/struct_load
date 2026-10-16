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
    Units,
    parse_units,
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
    gravity_accel: Optional[float] = None,
    units: Optional[Union[str, Units, dict]] = None,
    direction: Optional[str] = None,
) -> LoadCase:
    """
    Compute self-weight line loads for all beam/column elements in a StructuralModel.

    Parameters
    ----------
    model : StructuralModel
        The structural model containing materials, sections, and elements.
    load_case_id : str
        ID for the generated LoadCase.
    gravity_accel : float, optional
        Acceleration due to gravity. If None, derived from active model Units.
    units : str, Units, or dict, optional
        Engineering units (e.g. 'kN, m' or 'kip, in' or Units instance).
        If None, automatically read from model.units or model.metadata.units.
    direction : str, optional
        Direction of gravity: '-z' (3D space frames) or '-y' (2D plane frames).
        If None, automatically inferred from element and node geometry.

    Returns
    -------
    LoadCase
        A struct_core.LoadCase populated with DistributedLoad items.
    """
    # 1. Resolve active Units
    if units is not None:
        active_units = parse_units(units) if isinstance(units, (str, dict)) else units
    elif hasattr(model, "units") and model.units is not None:
        active_units = model.units
    elif getattr(model, "metadata", None) and getattr(model.metadata, "units", None):
        active_units = model.metadata.units
    else:
        active_units = Units(force="kN", length="m")

    # 2. Resolve gravity acceleration
    g_accel = gravity_accel if gravity_accel is not None else active_units.gravity_acceleration

    # 3. Resolve gravity direction
    if direction is None:
        has_z = (
            any(abs(getattr(n, "z", 0.0) or 0.0) > 1e-6 for n in model.nodes)
            or any(
                bool(getattr(s, "uz", False)) or bool(getattr(s, "rx", False)) or bool(getattr(s, "ry", False))
                for s in getattr(model, "supports", [])
            )
            or any(
                abs(getattr(pl, "fz", 0.0) or 0.0) > 1e-6 or abs(getattr(pl, "mx", 0.0) or 0.0) > 1e-6 or abs(getattr(pl, "my", 0.0) or 0.0) > 1e-6
                for lc in getattr(model, "load_cases", [])
                for pl in getattr(lc, "point_loads", [])
            )
            or any(
                abs(getattr(dl, "wz", 0.0) or 0.0) > 1e-6
                for lc in getattr(model, "load_cases", [])
                for dl in getattr(lc, "element_loads", [])
            )
        )
        direction = "-z" if has_z else "-y"

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
        if unit_weight is not None and float(unit_weight) > 0.0:
            w = float(unit_weight) * area
        else:
            rho = getattr(mat, "rho", None)
            if rho is not None and float(rho) > 0.0:
                rho_dyn = float(rho) * active_units.consistent_mass_factor
                w = rho_dyn * g_accel * area
            else:
                from struct_core import conversion_factor
                concrete_uw = 24.0 * conversion_factor("kN, m", active_units, "weight_density")
                w = concrete_uw * area

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
    units: Optional[Union[str, Units, dict]] = None,
    direction: Optional[str] = None,
) -> Union[StructuralModel, Project]:
    """
    Materialize self-weight distributed loads for structural elements.

    Scans load cases in the model. For any LoadCase where
    `include_self_weight=True`, computes element self-weight loads, appends
    them to `case.element_loads`, and sets `case.include_self_weight=False`
    to ensure self-weight is never double-counted.

    If no case has `include_self_weight=True`, applies self-weight to
    `target_case_id` (defaulting to the 'DEAD' load case).

    Parameters
    ----------
    source : StructuralModel or Project
        The structural model or project to update.
    target_case_id : str, optional
        Target load case ID if no case currently has include_self_weight=True. Default: "DEAD".
    gravity_accel : float, optional
        Acceleration due to gravity. Inferred from model units if None.
    units : str, Units, or dict, optional
        Units specification. Automatically inferred from source.units or source.metadata.units if None.
    direction : str, optional
        Direction of gravity: '-z' (3D space frames) or '-y' (2D plane frames).
        Automatically inferred from geometry if None.

    Returns
    -------
    StructuralModel or Project
        The updated source (supports both in-place mutation and assignment).
    """
    # 1. Direct handling for fem3d.Structure or fem2d.Structure
    if hasattr(source, "elements") and hasattr(source, "add_load") and not hasattr(source, "load_cases"):
        active_units = parse_units(units or getattr(source, "units", None) or "kN, m")
        g_accel = gravity_accel if gravity_accel is not None else active_units.gravity_acceleration
        mass_factor = active_units.consistent_mass_factor

        # Detect 2D vs 3D
        is_3d = False
        elem_values = list(source.elements.values())
        for el in elem_values:
            if type(el).__name__ in ("FrameElement", "Frame3D", "Beam3D") or hasattr(el, "w3") or hasattr(el, "wz"):
                is_3d = True
                break
            for n in (getattr(el, "node_i", None), getattr(el, "node_j", None)):
                if n is not None and abs(getattr(n, "z", 0.0) or 0.0) > 1e-6:
                    is_3d = True
                    break

        grav_dir = (direction or ("-z" if is_3d else "-y")).lower()

        # Check existing loads on source to avoid double counting
        already_applied_eids = set()
        for load in getattr(source, "loads", []):
            if getattr(load, "is_self_weight", False):
                elem = getattr(load, "element", None)
                if elem is not None:
                    already_applied_eids.add(getattr(elem, "id", None) or getattr(elem, "eid", None))

        for eid, el in source.elements.items():
            if eid in already_applied_eids or getattr(el, "eid", None) in already_applied_eids:
                continue

            mat = getattr(el, "material", None)
            sec = getattr(el, "section", None)
            if not mat or not sec:
                continue

            area = getattr(sec, "A", None)
            if area is None or float(area) <= 0.0:
                if hasattr(sec, "width") and hasattr(sec, "depth"):
                    area = float(sec.width) * float(sec.depth)
                elif hasattr(sec, "b") and hasattr(sec, "h"):
                    area = float(sec.b) * float(sec.h)
                else:
                    continue

            area = float(area)
            if area <= 0.0:
                continue

            unit_weight = getattr(mat, "unit_weight", None)
            if unit_weight is not None and float(unit_weight) > 0.0:
                w = float(unit_weight) * area
            else:
                rho = getattr(mat, "rho", None)
                if rho is not None and float(rho) > 0.0:
                    rho_dyn = float(rho) * mass_factor
                    w = rho_dyn * g_accel * area
                else:
                    from struct_core import conversion_factor
                    concrete_uw = 24.0 * conversion_factor("kN, m", active_units, "weight_density")
                    w = concrete_uw * area

            if is_3d:
                from fem3d.loads import DistributedLoad as Fem3dDistributedLoad
                if grav_dir in ("-z", "z-", "-3"):
                    sw_load = Fem3dDistributedLoad(el, wz=-w, coord_system="global", is_self_weight=True)
                elif grav_dir in ("-y", "y-", "-2"):
                    sw_load = Fem3dDistributedLoad(el, wy=-w, coord_system="global", is_self_weight=True)
                elif grav_dir in ("-x", "x-", "-1"):
                    sw_load = Fem3dDistributedLoad(el, wx=-w, coord_system="global", is_self_weight=True)
                else:
                    sw_load = Fem3dDistributedLoad(el, wz=-w, coord_system="global", is_self_weight=True)
                source.add_load(sw_load)
            else:
                from fem2d.loads import DistributedLoad as Fem2dDistributedLoad
                c = getattr(el, "cos", 1.0)
                s = getattr(el, "sin", 0.0)
                if grav_dir in ("-y", "y-"):
                    wx_loc = -w * s
                    wy_loc = -w * c
                elif grav_dir in ("-x", "x-"):
                    wx_loc = -w * c
                    wy_loc = w * s
                else:
                    wx_loc = -w * s
                    wy_loc = -w * c
                sw_load = Fem2dDistributedLoad(element=el, wx=wx_loc, wy=wy_loc)
                sw_load.is_self_weight = True
                source.add_load(sw_load)

        return source

    if isinstance(source, Project):
        if source.model is None:
            raise ValueError("Project contains no structural model.")
        model = source.model
        if units is None and getattr(source, "units", None) is not None:
            units = source.units
        elif units is None and getattr(source, "metadata", None) and getattr(source.metadata, "units", None):
            units = source.metadata.units
    elif isinstance(source, StructuralModel):
        model = source
        if units is None and getattr(source, "units", None) is not None:
            units = source.units
    else:
        raise TypeError(f"Expected Project, StructuralModel, or Structure, got {type(source).__name__}")

    # Identify load cases requesting self-weight
    cases_with_sw = [lc for lc in model.load_cases if getattr(lc, "include_self_weight", False)]

    if cases_with_sw:
        for lc in cases_with_sw:
            sw_case = calculate_self_weight(
                model=model,
                load_case_id=lc.id,
                gravity_accel=gravity_accel,
                units=units,
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
            gravity_accel=gravity_accel,
            units=units,
            direction=direction,
        )

        if target_case:
            target_case.element_loads.extend(sw_case.element_loads)
            target_case.include_self_weight = False
        else:
            sw_case.include_self_weight = False
            model.load_cases.append(sw_case)

    return source

