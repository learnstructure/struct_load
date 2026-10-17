"""
Mass source calculation engine for structural models.
Evaluates inertial mass distributions, effective seismic weight, and center of mass
from element geometries, material densities, assigned lumped masses, and load patterns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from struct_core import (
    MassSource,
    MassSourceFactor,
    Project,
    StructuralModel,
    Units,
    parse_units,
)

from .geometry import (
    detect_model_dimensionality,
    detect_story_elevations,
)
from .self_weight import calculate_section_area


@dataclass
class MassSourceResult:
    """
    Result of inertial mass source evaluation across a structural model.

    Attributes
    ----------
    nodal_masses : Dict[Any, Tuple[float, float, float]]
        Translational lumped mass (mx, my, mz) assigned to each node ID.
    element_masses : Dict[Any, float]
        Additional distributed non-structural mass per unit length for each element ID.
    total_seismic_weight : float
        Total effective seismic weight W in force units (e.g. kN or kips).
    total_mass : float
        Total inertial mass M in mass units (e.g. tonnes or kip*s^2/in).
    center_of_mass : Tuple[float, float, float]
        Global centroid of inertial mass (X_cm, Y_cm, Z_cm).
    story_masses : Dict[float, float]
        Total lumped mass mapped to each story elevation.
    mass_source : MassSource
        The source configuration used for evaluation.
    """
    nodal_masses: Dict[Any, Tuple[float, float, float]] = field(default_factory=dict)
    element_masses: Dict[Any, float] = field(default_factory=dict)
    total_seismic_weight: float = 0.0
    total_mass: float = 0.0
    center_of_mass: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    story_masses: Dict[float, float] = field(default_factory=dict)
    mass_source: Optional[MassSource] = None
    gravity_accel: float = 9.80665

    @property
    def total_weight(self) -> float:
        """Alias for total_seismic_weight."""
        return self.total_seismic_weight

    @property
    def story_weights(self) -> Dict[float, float]:
        """Effective seismic weights mapped to each story elevation."""
        return {elev: m * self.gravity_accel for elev, m in self.story_masses.items()}


def calculate_mass_source(
    model: Union[StructuralModel, Project],
    mass_source: Optional[Union[MassSource, Dict[str, Any], str]] = None,
    gravity_accel: Optional[float] = None,
    units: Optional[Union[str, Units, dict]] = None,
) -> MassSourceResult:
    """
    Calculate inertial mass distribution and effective seismic weight for a model.

    Parameters
    ----------
    model : StructuralModel or Project
        The structural model or project containing nodes, elements, materials, and loads.
    mass_source : MassSource, dict, str, or None, optional
        Mass source specification. If None, uses `model.mass_source` or a default
        configuration (100% element self-mass and added mass).
        If str, looks up the mass source ID in `model.mass_sources`.
    gravity_accel : float, optional
        Acceleration due to gravity. Defaults to `active_units.gravity_acceleration`.
    units : str, Units, or dict, optional
        Units specification (e.g. 'kN, m' or 'kip, in'). Inferred automatically from model if None.

    Returns
    -------
    MassSourceResult
        Dataclass containing nodal masses, element distributed masses, total weight,
        total mass, center of mass, and story mass distribution.
    """
    if isinstance(model, Project):
        proj = model
        if proj.model is None:
            raise ValueError("Project contains no structural model.")
        if units is None and getattr(proj, "units", None) is not None:
            units = proj.units
        elif units is None and getattr(proj, "metadata", None) and getattr(proj.metadata, "units", None):
            units = proj.metadata.units
        model = proj.model
    elif hasattr(model, "units") and units is None and model.units is not None:
        units = model.units

    active_units = parse_units(units or "kN, m")

    if gravity_accel is None:
        gravity_accel = active_units.gravity_acceleration

    # 1. Resolve MassSource object
    ms: MassSource
    if mass_source is None:
        if getattr(model, "mass_source", None) is not None:
            ms = model.mass_source
        else:
            ms = MassSource(
                id="MS_DEFAULT",
                name="Default Mass Source",
                include_element_mass=True,
                include_added_mass=True,
                factors=[],
            )
    elif isinstance(mass_source, str):
        found = None
        for s in getattr(model, "mass_sources", []):
            if s.id == mass_source:
                found = s
                break
        if found is None and getattr(model, "mass_source", None) and model.mass_source.id == mass_source:
            found = model.mass_source
        if found is None:
            raise ValueError(f"Mass source with ID '{mass_source}' not found in model.")
        ms = found
    elif isinstance(mass_source, dict):
        ms = MassSource(**mass_source)
    elif isinstance(mass_source, MassSource):
        ms = mass_source
    else:
        raise TypeError(f"Unsupported mass_source type: {type(mass_source)}")

    dim = detect_model_dimensionality(model)
    is_2d = dim == "2d"

    # Fast node lookup
    nodes_map = {n.id: n for n in model.nodes}
    elements_map = {e.id: e for e in model.elements}
    materials_map = {m.id: m for m in model.materials}
    sections_map = {s.id: s for s in model.sections}
    load_cases_map = {lc.id: lc for lc in model.load_cases}

    nodal_masses: Dict[Any, List[float]] = {n.id: [0.0, 0.0, 0.0] for n in model.nodes}
    element_masses: Dict[Any, float] = {e.id: 0.0 for e in model.elements}

    # 3. Incorporate user-assigned added mass
    if ms.include_added_mass:
        for n in model.nodes:
            m_val = getattr(n, "mass", 0.0)
            if isinstance(m_val, (int, float)) and m_val > 0.0:
                if ms.lateral_only:
                    nodal_masses[n.id][0] += float(m_val)
                    if not is_2d:
                        nodal_masses[n.id][1] += float(m_val)
                else:
                    nodal_masses[n.id][0] += float(m_val)
                    nodal_masses[n.id][1] += float(m_val)
                    nodal_masses[n.id][2] += float(m_val)
            elif isinstance(m_val, (list, tuple)):
                for idx_c in range(min(3, len(m_val))):
                    if not (ms.lateral_only and idx_c == (1 if is_2d else 2)):
                        nodal_masses[n.id][idx_c] += float(m_val[idx_c])

        for e in model.elements:
            elem_m = getattr(e, "mass", 0.0)
            if elem_m and float(elem_m) > 0.0:
                element_masses[e.id] += float(elem_m)

    # 4. Incorporate applied load patterns (m = c_k * W_k / g)
    for factor_item in ms.factors:
        cid = factor_item.load_case_id
        c_scale = float(factor_item.factor)
        if cid not in load_cases_map or c_scale == 0.0:
            continue
        lc = load_cases_map[cid]

        # 4a. Point loads
        for pl in lc.point_loads:
            nid = pl.node_id
            if nid not in nodal_masses:
                continue

            # Gravity force magnitude
            if is_2d:
                # 2D frame: gravity along Y
                f_grav = abs(pl.fy) if pl.fy != 0.0 else math.sqrt(pl.fx**2 + pl.fy**2)
            else:
                # 3D frame: gravity along Z
                f_grav = abs(pl.fz) if pl.fz != 0.0 else math.sqrt(pl.fx**2 + pl.fy**2 + pl.fz**2)

            if f_grav <= 0.0:
                continue

            # In SI: force is in kN, gravity in m/s^2 -> mass in tonnes (kN / (m/s^2) = 1000 N / 9.81 = tonnes)
            # In US: force is in kips, gravity in in/s^2 -> mass in kip*s^2/in
            m_equiv = (f_grav * c_scale) / gravity_accel

            if ms.lateral_only:
                nodal_masses[nid][0] += m_equiv
                if not is_2d:
                    nodal_masses[nid][1] += m_equiv
            else:
                nodal_masses[nid][0] += m_equiv
                nodal_masses[nid][1] += m_equiv
                nodal_masses[nid][2] += m_equiv

        # 4b. Distributed element loads
        for dl in lc.element_loads:
            eid = dl.element_id
            if eid not in element_masses:
                continue

            if is_2d:
                w_grav = abs(dl.wy) if dl.wy != 0.0 else math.sqrt(dl.wx**2 + dl.wy**2)
            else:
                w_grav = abs(dl.wz) if dl.wz != 0.0 else math.sqrt(dl.wx**2 + dl.wy**2 + dl.wz**2)

            if w_grav <= 0.0:
                continue

            w_equiv = (w_grav * c_scale) / gravity_accel
            element_masses[eid] += w_equiv

        # 4c. Area loads (projected to bounding nodes or elevation)
        for al in getattr(lc, "area_loads", []):
            press = float(getattr(al, "pressure", 0.0) or 0.0)
            if press <= 0.0:
                continue
            b_nodes = getattr(al, "node_ids", [])
            trib_w = getattr(al, "tributary_width", None)

            if b_nodes and len(b_nodes) >= 3:
                # Polygon area
                pts = [nodes_map[nid] for nid in b_nodes if nid in nodes_map]
                if len(pts) >= 3:
                    area_poly = 0.5 * abs(
                        sum(
                            pts[i].x * (pts[(i + 1) % len(pts)].y - pts[i - 1].y)
                            for i in range(len(pts))
                        )
                    )
                    tot_force = press * area_poly
                    m_per_node = (tot_force * c_scale) / (len(pts) * gravity_accel)
                    for pt in pts:
                        if ms.lateral_only:
                            nodal_masses[pt.id][0] += m_per_node
                            if not is_2d:
                                nodal_masses[pt.id][1] += m_per_node
                        else:
                            nodal_masses[pt.id][0] += m_per_node
                            nodal_masses[pt.id][1] += m_per_node
                            nodal_masses[pt.id][2] += m_per_node

    # 5. Compute Element lengths, element self-masses, and Center of Mass
    total_m_x = 0.0
    total_m_y = 0.0
    total_m_z = 0.0
    weighted_x = 0.0
    weighted_y = 0.0
    weighted_z = 0.0

    # Nodal mass contributions
    for nid, m_vec in nodal_masses.items():
        m_trans = m_vec[0] if ms.lateral_only else max(m_vec)
        nd = nodes_map[nid]
        nz = getattr(nd, "z", 0.0)
        total_m_x += m_vec[0]
        total_m_y += m_vec[1]
        total_m_z += m_vec[2]
        weighted_x += m_trans * nd.x
        weighted_y += m_trans * nd.y
        weighted_z += m_trans * nz

    # Element self-mass and distributed mass contributions
    for e in model.elements:
        if e.start_node not in nodes_map or e.end_node not in nodes_map:
            continue
        n_i = nodes_map[e.start_node]
        n_j = nodes_map[e.end_node]
        L = math.sqrt(
            (n_j.x - n_i.x) ** 2
            + (n_j.y - n_i.y) ** 2
            + (getattr(n_j, "z", 0.0) - getattr(n_i, "z", 0.0)) ** 2
        )
        if L <= 1e-12:
            continue

        e_extra_m = element_masses.get(e.id, 0.0)
        e_self_m = 0.0

        if ms.include_element_mass:
            mat = materials_map.get(getattr(e, "material_id", None))
            sec = sections_map.get(getattr(e, "section_id", None))
            if mat and sec:
                area = calculate_section_area(sec)
                unit_weight = getattr(mat, "unit_weight", None)
                if unit_weight is not None and float(unit_weight) > 0.0:
                    # w_self = unit_weight * area -> m_self = w_self / g
                    e_self_m = (float(unit_weight) * area) / gravity_accel
                else:
                    rho = getattr(mat, "rho", None)
                    if rho is not None and float(rho) > 0.0:
                        rho_dyn = float(rho) * active_units.consistent_mass_factor
                        e_self_m = rho_dyn * area
                    else:
                        from struct_core import conversion_factor
                        concrete_uw = 24.0 * conversion_factor("kN, m", active_units, "weight_density")
                        e_self_m = (concrete_uw * area) / gravity_accel

        total_elem_m = (e_self_m + e_extra_m) * L
        mid_x = 0.5 * (n_i.x + n_j.x)
        mid_y = 0.5 * (n_i.y + n_j.y)
        mid_z = 0.5 * (getattr(n_i, "z", 0.0) + getattr(n_j, "z", 0.0))

        total_m_x += total_elem_m
        total_m_y += total_elem_m
        total_m_z += 0.0 if ms.lateral_only else total_elem_m
        weighted_x += total_elem_m * mid_x
        weighted_y += total_elem_m * mid_y
        weighted_z += total_elem_m * mid_z

    total_inertial_mass = total_m_x if ms.lateral_only else (total_m_x + total_m_y + total_m_z) / 3.0
    total_seismic_weight = total_inertial_mass * gravity_accel

    cm_x = weighted_x / total_inertial_mass if total_inertial_mass > 1e-12 else 0.0
    cm_y = weighted_y / total_inertial_mass if total_inertial_mass > 1e-12 else 0.0
    cm_z = weighted_z / total_inertial_mass if total_inertial_mass > 1e-12 else 0.0

    # 6. Story mass aggregation
    story_elevations = getattr(model, "story_elevations", None)
    if not story_elevations:
        story_elevations = detect_story_elevations(model)

    story_masses: Dict[float, float] = {}
    if story_elevations:
        for elev in story_elevations:
            story_masses[elev] = 0.0

        for nid, m_vec in nodal_masses.items():
            nd = nodes_map[nid]
            z_coord = getattr(nd, "z", nd.y if is_2d else 0.0)
            closest_elev = min(story_elevations, key=lambda el: abs(el - z_coord))
            m_eff = m_vec[0] if ms.lateral_only else max(m_vec)
            story_masses[closest_elev] += m_eff

        for e in model.elements:
            if e.start_node not in nodes_map or e.end_node not in nodes_map:
                continue
            n_i = nodes_map[e.start_node]
            n_j = nodes_map[e.end_node]
            z_i = getattr(n_i, "z", n_i.y if is_2d else 0.0)
            z_j = getattr(n_j, "z", n_j.y if is_2d else 0.0)
            L = math.sqrt(
                (n_j.x - n_i.x) ** 2
                + (n_j.y - n_i.y) ** 2
                + (getattr(n_j, "z", 0.0) - getattr(n_i, "z", 0.0)) ** 2
            )
            if L <= 1e-12:
                continue

            e_extra_m = element_masses.get(e.id, 0.0)
            e_self_m = 0.0
            if ms.include_element_mass:
                mat = materials_map.get(getattr(e, "material_id", None))
                sec = sections_map.get(getattr(e, "section_id", None))
                if mat and sec:
                    area = calculate_section_area(sec)
                    unit_weight = getattr(mat, "unit_weight", None)
                    if unit_weight is not None and float(unit_weight) > 0.0:
                        e_self_m = (float(unit_weight) * area) / gravity_accel
                    else:
                        rho = getattr(mat, "rho", None)
                        if rho is not None and float(rho) > 0.0:
                            rho_dyn = float(rho) * active_units.consistent_mass_factor
                            e_self_m = rho_dyn * area
                        else:
                            from struct_core import conversion_factor
                            concrete_uw = 24.0 * conversion_factor("kN, m", active_units, "weight_density")
                            e_self_m = (concrete_uw * area) / gravity_accel

            total_elem_m = (e_self_m + e_extra_m) * L
            if abs(z_i - z_j) < 1e-4:
                # Horizontal member at elevation
                elev = 0.5 * (z_i + z_j)
                closest_elev = min(story_elevations, key=lambda el: abs(el - elev))
                story_masses[closest_elev] += total_elem_m
            else:
                # Vertical or inclined member: split tributary mass between endpoints
                closest_elev_i = min(story_elevations, key=lambda el: abs(el - z_i))
                closest_elev_j = min(story_elevations, key=lambda el: abs(el - z_j))
                story_masses[closest_elev_i] += 0.5 * total_elem_m
                story_masses[closest_elev_j] += 0.5 * total_elem_m

    # Clean dict output for nodal masses (tuple format)
    nodal_tuples = {nid: (vec[0], vec[1], vec[2]) for nid, vec in nodal_masses.items()}

    return MassSourceResult(
        nodal_masses=nodal_tuples,
        element_masses=element_masses,
        total_seismic_weight=total_seismic_weight,
        total_mass=total_inertial_mass,
        center_of_mass=(cm_x, cm_y, cm_z),
        story_masses=story_masses,
        mass_source=ms,
        gravity_accel=gravity_accel,
    )
