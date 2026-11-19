"""
Self-weight calculation engine for structural elements.
Calculates distributed gravity loads directly from element geometry and material density.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union
from struct_core import (
    CircularSection,
    DistributedLoad,
    GeneralSection,
    LoadCase,
    Project,
    RectangularSection,
    ShellLoad,
    StructuralModel,
    Units,
    parse_units,
)


def material_weight_density(material: object, units: Units) -> float:
    """
    Weight per unit volume of ``material`` in ``units``: its ``unit_weight``, else its mass density ``rho`` times
    gravity. An explicit zero density (``unit_weight=0.0`` with ``rho=0.0`` or ``None``) returns ``0.0`` for
    massless elements with lumped joint masses. A material with no density specified (``unit_weight=None`` and
    ``rho=0.0``/``None``) raises ``ValueError`` (no density is assumed).
    """
    unit_weight = getattr(material, "unit_weight", None)
    if unit_weight is not None and float(unit_weight) > 0.0:
        return float(unit_weight)
    rho = getattr(material, "rho", None)
    if rho is not None and float(rho) > 0.0:
        return float(rho) * units.consistent_mass_factor * units.gravity_acceleration
    if unit_weight is not None and float(unit_weight) == 0.0 and (rho is None or float(rho) == 0.0):
        return 0.0
    name = getattr(material, "id", None) or type(material).__name__
    raise ValueError(f"Material {name!r} has no density (unit_weight or rho); its self-weight cannot be computed.")


def _modifier(element: object, key: str) -> float:
    mods = getattr(element, "stiffness_modifiers", None) or {}
    return float(mods.get(key, 1.0))


def element_weight_per_length(element: object, section: object, material: object, units: Units) -> float:
    """Self-weight per unit length of a frame element: unit weight x section area x the element's ``weight``
    modifier. The ``A`` modifier is not applied: as in SAP2000 and ETABS, it changes the axial stiffness only, and the
    weight and mass have their own modifiers (checked against SAP2000 27.1, struct_link tests)."""
    return material_weight_density(material, units) * calculate_section_area(section) * _modifier(element, "weight")


def element_mass_per_length(element: object, section: object, material: object, units: Units,
                            gravity_accel: float) -> float:
    """Self-mass per unit length of a frame element (consistent mass units): unit weight x section area / g x the
    element's ``mass`` modifier (the ``A`` modifier is not applied; see :func:`element_weight_per_length`)."""
    return (material_weight_density(material, units) * calculate_section_area(section) / gravity_accel
            * _modifier(element, "mass"))


def _solver_weight_density(material: object, gravity_accel: float) -> float:
    """Weight per unit volume of a fem2d / fem3d material: its ``unit_weight``, else its mass density times gravity.
    Solver materials hold ``rho`` in consistent units (e.g. t/m3 for kN, m), so no mass-unit factor applies."""
    unit_weight = getattr(material, "unit_weight", None)
    if unit_weight is not None and float(unit_weight) > 0.0:
        return float(unit_weight)
    rho = getattr(material, "rho", None)
    if rho is not None and float(rho) > 0.0:
        return float(rho) * gravity_accel
    if unit_weight is not None and float(unit_weight) == 0.0 and (rho is None or float(rho) == 0.0):
        return 0.0
    raise ValueError(f"Material {type(material).__name__} has no density (unit_weight or rho); its self-weight cannot "
                     f"be computed.")


def shell_weight_per_area(section: object, material: object, units: Units) -> float:
    """Self-weight per unit area of a shell element: unit weight x membrane thickness th (CSI Analysis Reference
    Manual §10.12)."""
    return material_weight_density(material, units) * float(section.thickness)


def shell_tributary(model: StructuralModel) -> Dict[object, Tuple[float, List[Tuple[object, float]]]]:
    """Area of every shell element and the fraction of it lumped at each node: ``{element_id: (area, [(node_id,
    fraction), ...])}``. The fractions follow the diagonal of the consistent mass matrix, as SAP2000 lumps shell mass
    and uniform loads (CSI Analysis Reference Manual §10.11; ``struct_core.geometry.shell_corner_weights``).
    Elements with missing nodes are refused."""
    from struct_core.geometry import model_shell_coordinates, shell_areas, shell_corner_weights

    out: Dict[object, Tuple[float, List[Tuple[object, float]]]] = {}
    groups = model_shell_coordinates(model)
    n_found = sum(len(ids) for ids, _ in groups.values())
    if n_found != len(model.shell_elements):
        raise ValueError("Shell elements reference missing nodes; validate the model first.")
    by_id = {el.id: el for el in model.shell_elements}
    for ids, coords in groups.values():
        areas = shell_areas(coords)
        weights = shell_corner_weights(coords)
        for i, eid in enumerate(ids):
            out[eid] = (float(areas[i]), list(zip(by_id[eid].node_ids, (float(w) for w in weights[i]))))
    return out


def shell_self_weights(model: StructuralModel, units: Units) -> Dict[object, float]:
    """Self-weight per unit area of every shell element, ``{element_id: weight}`` (CSI §10.12)."""
    materials_map = {m.id: m for m in model.materials}
    sections_map = {s.id: s for s in model.shell_sections}
    out: Dict[object, float] = {}
    for el in model.shell_elements:
        sec = sections_map.get(el.section_id)
        if sec is None:
            raise ValueError(f"Shell element {el.id!r} references missing shell section {el.section_id!r}.")
        mat = materials_map.get(sec.material_id)
        if mat is None:
            raise ValueError(f"Shell section {sec.id!r} on element {el.id!r} references missing material "
                             f"{sec.material_id!r}.")
        out[el.id] = shell_weight_per_area(sec, mat, units)
    return out


def _shell_self_weight_loads(model: StructuralModel, units: Units, direction: str) -> List[ShellLoad]:
    """Self-weight of the shell elements as uniform shell loads, one per distinct weight per area."""
    weights = shell_self_weights(model, units)
    by_weight: Dict[float, List[object]] = {}
    for eid, w in weights.items():
        if w > 0.0:
            by_weight.setdefault(w, []).append(eid)
    d = direction.lower()
    axis = {"-z": "z", "z-": "z", "down_z": "z", "gravity": "z", "+z": "z", "z+": "z",
            "-y": "y", "y-": "y", "down_y": "y", "+y": "y", "y+": "y",
            "-x": "x", "x-": "x", "+x": "x", "x+": "x"}.get(d)
    if axis is None:
        raise ValueError(f"Unknown self-weight direction {direction!r}")
    sign = 1.0 if d.startswith("+") or d.endswith("+") else -1.0
    return [ShellLoad(element_ids=eids, pressure=sign * w, direction=axis, is_self_weight=True)
            for w, eids in by_weight.items()]


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
    Compute self-weight line loads for all beam/column elements in a StructuralModel, and uniform loads (weight per
    unit area, CSI Analysis Reference Manual §10.12) for its shell elements.

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
        A struct_core.LoadCase populated with DistributedLoad items and, for shell elements, ShellLoad items.
    """
    # 1. Resolve active Units
    if units is not None:
        active_units = parse_units(units) if isinstance(units, (str, dict)) else units
    elif getattr(model, "units", None) is not None:
        active_units = model.units
    elif getattr(model, "metadata", None) and getattr(model.metadata, "units", None):
        active_units = model.metadata.units
    else:
        raise ValueError("Units unknown: pass units= or use a Project")

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

        sec_id = getattr(elem, "section_id", None)
        sec = sections_map.get(sec_id)
        if sec is None:
            raise ValueError(f"Element {elem.id!r} references missing section {sec_id!r}.")
        mat_id = getattr(sec, "material_id", None)
        mat = materials_map.get(mat_id)  # the section owns the material
        if mat is None:
            raise ValueError(f"Section {sec.id!r} on element {elem.id!r} references missing material {mat_id!r}.")

        if calculate_section_area(sec) <= 0.0:
            continue

        w_mag = float(element_weight_per_length(elem, sec, mat, active_units))
        if w_mag <= 0.0:
            continue

        dir_lower = direction.lower()
        if dir_lower in ("-z", "z-", "down_z", "gravity"):
            load = DistributedLoad(element_id=elem.id, wx=0.0, wy=0.0, wz=-w_mag, is_self_weight=True)
        elif dir_lower in ("+z", "z+"):
            load = DistributedLoad(element_id=elem.id, wx=0.0, wy=0.0, wz=w_mag, is_self_weight=True)
        elif dir_lower in ("-y", "y-", "down_y"):
            load = DistributedLoad(element_id=elem.id, wx=0.0, wy=-w_mag, wz=0.0, is_self_weight=True)
        elif dir_lower in ("+y", "y+"):
            load = DistributedLoad(element_id=elem.id, wx=0.0, wy=w_mag, wz=0.0, is_self_weight=True)
        elif dir_lower in ("-x", "x-"):
            load = DistributedLoad(element_id=elem.id, wx=-w_mag, wy=0.0, wz=0.0, is_self_weight=True)
        elif dir_lower in ("+x", "x+"):
            load = DistributedLoad(element_id=elem.id, wx=w_mag, wy=0.0, wz=0.0, is_self_weight=True)
        else:
            load = DistributedLoad(element_id=elem.id, wx=0.0, wy=0.0, wz=-w_mag, is_self_weight=True)

        element_loads.append(load)

    shell_loads = _shell_self_weight_loads(model, active_units, direction) if model.shell_elements else []

    return LoadCase(
        id=load_case_id,
        name="Dead Load (Self-Weight)",
        element_loads=element_loads,
        shell_loads=shell_loads,
        include_self_weight=True,
    )


def _new_shell_self_weight(case: LoadCase, sw_case: LoadCase) -> List[ShellLoad]:
    """Self-weight shell loads of ``sw_case`` for the shell elements that do not yet carry self-weight in ``case``."""
    done = {eid for sl in case.shell_loads if sl.is_self_weight for eid in sl.element_ids}
    out = []
    for sl in sw_case.shell_loads:
        eids = [eid for eid in sl.element_ids if eid not in done]
        if eids:
            out.append(sl.model_copy(update={"element_ids": eids}))
    return out


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
        Target load case ID if no case currently has include_self_weight=True. Default: "SelfWeight".
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

        # Detect 2D vs 3D
        is_3d = bool(getattr(source, "shells", None))  # shell elements exist only in fem3d
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

            # Springs have no physical section (fem3d: none; fem2d: a unit placeholder section)
            sec = getattr(el, "section", None)
            if sec is None or type(el).__name__ == "SpringElement":
                continue
            mat = sec.material  # the section owns the material

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

            w = _solver_weight_density(mat, g_accel) * area

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

        # Shell elements of a fem3d Structure: unit weight x membrane thickness per unit area (CSI Analysis Reference
        # Manual §10.12), along the gravity axis
        shells = getattr(source, "shells", None) or {}
        if shells:
            from fem3d.loads import ShellLoad as Fem3dShellLoad

            # the same directions as the frame loads above: gravity acts in the negative sense of the axis
            axis = {"-y": "y", "y-": "y", "-2": "y", "-x": "x", "x-": "x", "-1": "x"}.get(grav_dir, "z")
            done = {load.shell.id for load in getattr(source, "loads", [])
                    if getattr(load, "is_self_weight", False) and getattr(load, "shell", None) is not None}
            for sid, shell in shells.items():
                if sid in done:
                    continue
                w = _solver_weight_density(shell.material, g_accel) * shell.thickness
                source.add_load(Fem3dShellLoad(shell, -w, axis, is_self_weight=True))

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
            already_eids = {dl.element_id for dl in lc.element_loads if getattr(dl, "is_self_weight", False)}
            new_loads = [dl for dl in sw_case.element_loads if dl.element_id not in already_eids]
            lc.element_loads.extend(new_loads)
            lc.shell_loads.extend(_new_shell_self_weight(lc, sw_case))
            lc.include_self_weight = False
    else:
        target_id = target_case_id or "DEAD"
        target_case = next((lc for lc in model.load_cases if str(lc.id).upper() == str(target_id).upper()), None)

        sw_case = calculate_self_weight(
            model=model,
            load_case_id=target_id,
            gravity_accel=gravity_accel,
            units=units,
            direction=direction,
        )

        if target_case:
            already_eids = {dl.element_id for dl in target_case.element_loads if getattr(dl, "is_self_weight", False)}
            new_loads = [dl for dl in sw_case.element_loads if dl.element_id not in already_eids]
            target_case.element_loads.extend(new_loads)
            target_case.shell_loads.extend(_new_shell_self_weight(target_case, sw_case))
            target_case.include_self_weight = False
        else:
            sw_case.include_self_weight = False
            model.load_cases.append(sw_case)

    return source

