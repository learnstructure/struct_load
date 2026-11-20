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
    conversion_factor,
    parse_units,
)

from .geometry import (
    detect_model_dimensionality,
    detect_stories,
    resolve_base_elevation,
)
from .self_weight import element_mass_per_length, shell_self_weights, shell_tributary

# 3-point Gauss-Legendre rule on [-1, 1] (exact for polynomials up to degree 5;
# linear load intensity * cubic Timoshenko end-reaction polynomial has degree <= 4).
_GL3_XI = (-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0))
_GL3_W = (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)


def _beam_point_end_i_fraction(s: float, phi: float, rel_i: bool, rel_j: bool) -> float:
    """Fraction of a transverse unit point load at relative position ``s`` in [0, 1] carried to end ``i``
    by a Timoshenko/Euler-Bernoulli beam with shear parameter ``phi`` and end moment releases."""
    if rel_i and rel_j:
        return 1.0 - s
    om = 1.0 - s
    denom = 1.0 + phi
    v_i = (om * om * (1.0 + 2.0 * s) + om * phi) / denom
    if not rel_i and not rel_j:
        return v_i
    if rel_i:
        m_i = (s * om * om + 0.5 * s * om * phi) / denom
        return v_i - (6.0 / (4.0 + phi)) * m_i
    # rel_j only
    m_j = -(s * s * om + 0.5 * s * om * phi) / denom
    return v_i - (6.0 / (4.0 + phi)) * m_j


def _member_local_axes(
    ni: Any, nj: Any, *, is_2d: bool, roll_angle_deg: float = 0.0
) -> Tuple[float, Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    """Return ``(L, e1, e2, e3)`` matching fem2d/fem3d member local axes."""
    dx = float(nj.x) - float(ni.x)
    dy = float(nj.y) - float(ni.y)
    dz = 0.0 if is_2d else (float(getattr(nj, "z", 0.0)) - float(getattr(ni, "z", 0.0)))
    L = math.sqrt(dx * dx + dy * dy + dz * dz)
    if L <= 1e-12:
        return 0.0, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    e1 = (dx / L, dy / L, dz / L)
    if is_2d:
        e2 = (-dy / L, dx / L, 0.0)
        e3 = (0.0, 0.0, 1.0)
        if abs(roll_angle_deg) > 1e-9:
            c = math.cos(math.radians(roll_angle_deg))
            e2 = (c * e2[0], c * e2[1], 0.0)
        return L, e1, e2, e3

    if dx * dx + dy * dy < 1e-12 * L * L:
        # Vertical member: fem3d default e2 = +Y, e3 = e1 x e2 (-X for +Z)
        v2 = (0.0, 1.0, 0.0)
        v3 = (
            e1[1] * v2[2] - e1[2] * v2[1],
            e1[2] * v2[0] - e1[0] * v2[2],
            e1[0] * v2[1] - e1[1] * v2[0],
        )
    else:
        # Non-vertical: v3 = normalize(e1 x +Z), v2 = v3 x e1
        c3 = (e1[1], -e1[0], 0.0)
        n3 = math.hypot(c3[0], c3[1])
        v3 = (c3[0] / n3, c3[1] / n3, 0.0)
        v2 = (
            v3[1] * e1[2] - v3[2] * e1[1],
            v3[2] * e1[0] - v3[0] * e1[2],
            v3[0] * e1[1] - v3[1] * e1[0],
        )
    if abs(roll_angle_deg) > 1e-9:
        ang = math.radians(roll_angle_deg)
        ca, sa = math.cos(ang), math.sin(ang)
        e2 = (ca * v2[0] + sa * v3[0], ca * v2[1] + sa * v3[1], ca * v2[2] + sa * v3[2])
        e3 = (-sa * v2[0] + ca * v3[0], -sa * v2[1] + ca * v3[1], -sa * v2[2] + ca * v3[2])
    else:
        e2, e3 = v2, v3
    return L, e1, e2, e3


def _member_shear_and_releases(
    elem: Any, sec: Any, mat: Any, L: float
) -> Tuple[float, float, bool, bool, bool, bool]:
    """Return ``(phi_2, phi_3, rel_i2, rel_j2, rel_i3, rel_j3)`` for a frame/truss element."""
    if str(getattr(elem, "type", "")).lower() == "truss" or L <= 1e-12:
        return 0.0, 0.0, True, True, True, True
    ri = getattr(elem, "release_i", None) or [False] * 6
    rj = getattr(elem, "release_j", None) or [False] * 6
    rel_i2 = bool(ri[5]) if len(ri) > 5 else False  # M3 at I (1-2 plane)
    rel_j2 = bool(rj[5]) if len(rj) > 5 else False  # M3 at J (1-2 plane)
    rel_i3 = bool(ri[4]) if len(ri) > 4 else False  # M2 at I (1-3 plane)
    rel_j3 = bool(rj[4]) if len(rj) > 4 else False  # M2 at J (1-3 plane)

    if not getattr(elem, "include_shear_deformation", False) or sec is None or mat is None:
        return 0.0, 0.0, rel_i2, rel_j2, rel_i3, rel_j3

    sec_type = str(getattr(sec, "type", "")).lower()
    if sec_type == "rectangular":
        b, h = float(sec.b), float(sec.h)
        I3 = b * (h ** 3) / 12.0
        I2 = h * (b ** 3) / 12.0
        As2 = (5.0 / 6.0) * b * h
        As3 = As2
    elif sec_type == "circular":
        d = float(sec.d)
        I3 = math.pi * (d ** 4) / 64.0
        I2 = I3
        As2 = 0.9 * math.pi * (d ** 2) / 4.0
        As3 = As2
    else:
        I3 = float(getattr(sec, "I3", 0.0) or 0.0)
        I2 = float(getattr(sec, "I2", 0.0) or 0.0)
        As2 = float(getattr(sec, "As2", 0.0) or 0.0)
        As3 = float(getattr(sec, "As3", 0.0) or 0.0)

    mods = getattr(elem, "stiffness_modifiers", None) or {}
    I3 *= float(mods.get("I3", 1.0))
    I2 *= float(mods.get("I2", 1.0))
    As2 *= float(mods.get("As2", 1.0))
    As3 *= float(mods.get("As3", 1.0))

    nu = float(getattr(mat, "nu", 0.3) or 0.0)
    # 12 E I / (G As L^2) = 24 (1 + nu) I / (As L^2)
    phi_2 = (24.0 * (1.0 + nu) * I3 / (As2 * L * L)) if As2 > 0.0 else 0.0
    phi_3 = (24.0 * (1.0 + nu) * I2 / (As3 * L * L)) if As3 > 0.0 else 0.0
    return phi_2, phi_3, rel_i2, rel_j2, rel_i3, rel_j3


def _consistent_end_i_force_3d(
    w_vec: Tuple[float, float, float],
    s: float,
    e1: Tuple[float, float, float],
    e2: Tuple[float, float, float],
    e3: Tuple[float, float, float],
    phi_2: float,
    phi_3: float,
    rel_i2: bool,
    rel_j2: bool,
    rel_i3: bool,
    rel_j3: bool,
    *,
    is_local: bool,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Consistent end-node force vectors ``(F_i, F_j)`` in global coordinates for a force ``w_vec`` at ``s``."""
    if is_local:
        w1, w2, w3 = w_vec
        w_glob = (
            w1 * e1[0] + w2 * e2[0] + w3 * e3[0],
            w1 * e1[1] + w2 * e2[1] + w3 * e3[1],
            w1 * e1[2] + w2 * e2[2] + w3 * e3[2],
        )
    else:
        w_glob = w_vec
        w1 = w_vec[0] * e1[0] + w_vec[1] * e1[1] + w_vec[2] * e1[2]
        w2 = w_vec[0] * e2[0] + w_vec[1] * e2[1] + w_vec[2] * e2[2]
        w3 = w_vec[0] * e3[0] + w_vec[1] * e3[1] + w_vec[2] * e3[2]

    f1_i = (1.0 - s) * w1
    f2_i = _beam_point_end_i_fraction(s, phi_2, rel_i2, rel_j2) * w2
    f3_i = _beam_point_end_i_fraction(s, phi_3, rel_i3, rel_j3) * w3
    F_i = (
        f1_i * e1[0] + f2_i * e2[0] + f3_i * e3[0],
        f1_i * e1[1] + f2_i * e2[1] + f3_i * e3[1],
        f1_i * e1[2] + f2_i * e2[2] + f3_i * e3[2],
    )
    F_j = (w_glob[0] - F_i[0], w_glob[1] - F_i[1], w_glob[2] - F_i[2])
    return F_i, F_j


def _grav_split_from_end_forces(
    F_i: Tuple[float, float, float],
    F_j: Tuple[float, float, float],
    v_idx: int,
) -> Tuple[float, float]:
    """Extract gravity-axis (or total magnitude when purely non-vertical) end-node weights ``(W_i, W_j)``."""
    tot_v = F_i[v_idx] + F_j[v_idx]
    if abs(tot_v) > 1e-12:
        sign = 1.0 if tot_v >= 0.0 else -1.0
        return max(0.0, sign * F_i[v_idx]), max(0.0, sign * F_j[v_idx])
    # Fallback for purely non-vertical load included in mass source: split total magnitude along end forces
    ni = math.sqrt(F_i[0] ** 2 + F_i[1] ** 2 + F_i[2] ** 2)
    nj = math.sqrt(F_j[0] ** 2 + F_j[1] ** 2 + F_j[2] ** 2)
    return ni, nj


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
    messages: List[str] = field(default_factory=list)

    @property
    def story_weights(self) -> Dict[float, float]:
        """Effective seismic weights mapped to each story elevation."""
        return {elev: m * self.gravity_accel for elev, m in self.story_masses.items()}


def calculate_mass_source(
    model: Union[StructuralModel, Project],
    mass_source: Optional[Union[MassSource, Dict[str, Any], str]] = None,
    gravity_accel: Optional[float] = None,
    units: Optional[Union[str, Units, dict]] = None,
    *,
    base_elevation: Optional[float] = None,
    messages: Optional[List[str]] = None,
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

    if units is None:
        raise ValueError("Units unknown: pass units= or use a Project")

    active_units = parse_units(units)

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

    from struct_core import vertical_axis as get_vertical_axis
    v_axis = get_vertical_axis(model)
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

    # 3. Incorporate user-assigned added mass (struct_core stores Node.mass in active_units.mass;
    # convert to consistent solver mass units F*T^2/L via active_units.consistent_mass_factor)
    mass_factor = active_units.consistent_mass_factor
    if ms.include_added_mass:
        for n in model.nodes:
            m_val = getattr(n, "mass", 0.0)
            if isinstance(m_val, (int, float)) and m_val > 0.0:
                m_cons = float(m_val) * mass_factor
                if ms.lateral_only:
                    if v_axis == "z":
                        nodal_masses[n.id][0] += m_cons
                        nodal_masses[n.id][1] += m_cons
                    else:
                        nodal_masses[n.id][0] += m_cons
                        if not is_2d:
                            nodal_masses[n.id][2] += m_cons
                else:
                    nodal_masses[n.id][0] += m_cons
                    nodal_masses[n.id][1] += m_cons
                    nodal_masses[n.id][2] += m_cons
            elif isinstance(m_val, (list, tuple)):
                vert_idx = 2 if v_axis == "z" else 1
                for idx_c in range(min(3, len(m_val))):
                    if not (ms.lateral_only and idx_c == vert_idx):
                        nodal_masses[n.id][idx_c] += float(m_val[idx_c]) * mass_factor

        for e in model.elements:
            elem_m = getattr(e, "mass", 0.0)
            if elem_m and float(elem_m) > 0.0:
                element_masses[e.id] += float(elem_m) * mass_factor

    # 3b. Shell elements: mass lumped at their nodes in proportion to the diagonal of the consistent mass matrix,
    # on the three translations (CSI Analysis Reference Manual §10.11). Self-mass from the section (CSI §10.12
    # weight / g) when element mass is included; uniform shell loads of the mass-source patterns below.
    def _add_node_mass(nid: Any, m_equiv: float) -> None:
        if ms.lateral_only:
            if v_axis == "z":
                nodal_masses[nid][0] += m_equiv
                nodal_masses[nid][1] += m_equiv
            else:
                nodal_masses[nid][0] += m_equiv
                if not is_2d:
                    nodal_masses[nid][2] += m_equiv
        else:
            nodal_masses[nid][0] += m_equiv
            nodal_masses[nid][1] += m_equiv
            nodal_masses[nid][2] += m_equiv

    shell_trib = shell_tributary(model) if model.shell_elements else {}
    if ms.include_element_mass and shell_trib:
        for eid, w_area in shell_self_weights(model, active_units).items():
            area, shares = shell_trib[eid]
            for nid, frac in shares:
                _add_node_mass(nid, w_area * area * frac / gravity_accel)

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
            if v_axis == "z":
                f_grav = abs(pl.fz) if pl.fz != 0.0 else math.sqrt(pl.fx**2 + pl.fy**2 + pl.fz**2)
            else:
                f_grav = abs(pl.fy) if pl.fy != 0.0 else math.sqrt(pl.fx**2 + pl.fy**2)

            if f_grav <= 0.0:
                continue

            # In SI: force is in kN, gravity in m/s^2 -> mass in tonnes (kN / (m/s^2) = 1000 N / 9.81 = tonnes)
            # In US: force is in kips, gravity in in/s^2 -> mass in kip*s^2/in
            m_equiv = (f_grav * c_scale) / gravity_accel

            if ms.lateral_only:
                if v_axis == "z":
                    nodal_masses[nid][0] += m_equiv
                    nodal_masses[nid][1] += m_equiv
                else:
                    nodal_masses[nid][0] += m_equiv
                    if not is_2d:
                        nodal_masses[nid][2] += m_equiv
            else:
                nodal_masses[nid][0] += m_equiv
                nodal_masses[nid][1] += m_equiv
                nodal_masses[nid][2] += m_equiv

        # 4b. Distributed element loads -> lumped to end nodes using the element's consistent static end
        #     reactions (matching SAP2000's load-to-mass conversion, including Timoshenko shear and end releases).
        v_idx = 2 if v_axis == "z" else 1
        for dl in lc.element_loads:
            eid = dl.element_id
            el_obj = elements_map.get(eid)
            if el_obj is None:
                continue
            if ms.include_element_mass and dl.is_self_weight:
                continue
            if el_obj.start_node not in nodes_map or el_obj.end_node not in nodes_map:
                continue

            a = float(getattr(dl, "start", 0.0))
            b = float(getattr(dl, "end", 1.0))
            span_frac = b - a
            if span_frac <= 0.0:
                continue

            ni, nj = nodes_map[el_obj.start_node], nodes_map[el_obj.end_node]
            L, e1, e2, e3 = _member_local_axes(
                ni, nj, is_2d=is_2d, roll_angle_deg=float(getattr(el_obj, "roll_angle", 0.0) or 0.0)
            )
            if L <= 1e-12:
                continue

            sec = sections_map.get(getattr(el_obj, "section_id", None))
            mat = materials_map.get(getattr(sec, "material_id", None)) if sec is not None else None
            phi_2, phi_3, rel_i2, rel_j2, rel_i3, rel_j3 = _member_shear_and_releases(el_obj, sec, mat, L)

            is_loc = bool(getattr(dl, "is_local", False))
            if is_loc:
                w_start = (float(dl.w1), float(dl.w2), float(dl.w3))
                w_end = (
                    float(dl.w1_end) if dl.w1_end is not None else w_start[0],
                    float(dl.w2_end) if dl.w2_end is not None else w_start[1],
                    float(dl.w3_end) if dl.w3_end is not None else w_start[2],
                )
            else:
                w_start = (float(dl.wx), float(dl.wy), 0.0 if is_2d else float(dl.wz))
                w_end = (
                    float(dl.wx_end) if dl.wx_end is not None else w_start[0],
                    float(dl.wy_end) if dl.wy_end is not None else w_start[1],
                    0.0 if is_2d else (float(dl.wz_end) if dl.wz_end is not None else w_start[2]),
                )

            if all(abs(v) <= 1e-15 for v in w_start + w_end):
                continue

            # Exact 3-point Gauss-Legendre integration over [a, b] (integrand degree <= 4)
            Fi_sum = [0.0, 0.0, 0.0]
            Fj_sum = [0.0, 0.0, 0.0]
            half_len = 0.5 * span_frac * L
            for xi, gw in zip(_GL3_XI, _GL3_W):
                t = 0.5 * (xi + 1.0)
                s = a + t * span_frac
                w_s = (
                    w_start[0] + t * (w_end[0] - w_start[0]),
                    w_start[1] + t * (w_end[1] - w_start[1]),
                    w_start[2] + t * (w_end[2] - w_start[2]),
                )
                fi_s, fj_s = _consistent_end_i_force_3d(
                    w_s, s, e1, e2, e3, phi_2, phi_3, rel_i2, rel_j2, rel_i3, rel_j3, is_local=is_loc
                )
                wt = gw * half_len
                for k in range(3):
                    Fi_sum[k] += wt * fi_s[k]
                    Fj_sum[k] += wt * fj_s[k]

            wi_grav, wj_grav = _grav_split_from_end_forces(
                (Fi_sum[0], Fi_sum[1], Fi_sum[2]),
                (Fj_sum[0], Fj_sum[1], Fj_sum[2]),
                v_idx,
            )
            if wi_grav > 0.0:
                _add_node_mass(el_obj.start_node, (wi_grav * c_scale) / gravity_accel)
            if wj_grav > 0.0:
                _add_node_mass(el_obj.end_node, (wj_grav * c_scale) / gravity_accel)

        # 4b-2. Element point loads -> lumped to end nodes using consistent static end reactions
        for epl in getattr(lc, "element_point_loads", []):
            eid = epl.element_id
            el_obj = elements_map.get(eid)
            if el_obj is None:
                continue
            if el_obj.start_node not in nodes_map or el_obj.end_node not in nodes_map:
                continue
            ni, nj = nodes_map[el_obj.start_node], nodes_map[el_obj.end_node]
            L, e1, e2, e3 = _member_local_axes(
                ni, nj, is_2d=is_2d, roll_angle_deg=float(getattr(el_obj, "roll_angle", 0.0) or 0.0)
            )
            if L <= 1e-12:
                continue

            is_loc = bool(getattr(epl, "is_local", False))
            p_vec = (float(epl.fx), float(epl.fy), 0.0 if (is_2d and not is_loc) else float(epl.fz))
            if all(abs(v) <= 1e-15 for v in p_vec):
                continue

            sec = sections_map.get(getattr(el_obj, "section_id", None))
            mat = materials_map.get(getattr(sec, "material_id", None)) if sec is not None else None
            phi_2, phi_3, rel_i2, rel_j2, rel_i3, rel_j3 = _member_shear_and_releases(el_obj, sec, mat, L)
            s = float(getattr(epl, "position", 0.5))
            fi_p, fj_p = _consistent_end_i_force_3d(
                p_vec, s, e1, e2, e3, phi_2, phi_3, rel_i2, rel_j2, rel_i3, rel_j3, is_local=is_loc
            )
            wi_grav, wj_grav = _grav_split_from_end_forces(fi_p, fj_p, v_idx)
            if wi_grav > 0.0:
                _add_node_mass(el_obj.start_node, (wi_grav * c_scale) / gravity_accel)
            if wj_grav > 0.0:
                _add_node_mass(el_obj.end_node, (wj_grav * c_scale) / gravity_accel)

        # 4b-3. Uniform loads on shell elements (shell self-weight is already counted by include_element_mass)
        for sl in getattr(lc, "shell_loads", []):
            if ms.include_element_mass and sl.is_self_weight:
                continue
            press = abs(float(sl.pressure))
            if press <= 0.0:
                continue
            for eid in sl.element_ids:
                if eid not in shell_trib:
                    raise ValueError(f"Shell load in load case {cid!r} references missing shell element {eid!r}.")
                area, shares = shell_trib[eid]
                for nid, frac in shares:
                    _add_node_mass(nid, press * area * frac * c_scale / gravity_accel)

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
                            if v_axis == "z":
                                nodal_masses[pt.id][0] += m_per_node
                                nodal_masses[pt.id][1] += m_per_node
                            else:
                                nodal_masses[pt.id][0] += m_per_node
                                if not is_2d:
                                    nodal_masses[pt.id][2] += m_per_node
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

        if ms.include_element_mass and str(getattr(e, "type", "")).lower() != "spring":
            sec_id = getattr(e, "section_id", None)
            sec = sections_map.get(sec_id)
            if sec is None:
                raise ValueError(f"Element {e.id!r} references missing section {sec_id!r}.")
            mat_id = getattr(sec, "material_id", None)
            mat = materials_map.get(mat_id)  # the section owns the material
            if mat is None:
                raise ValueError(f"Section {sec.id!r} on element {e.id!r} references missing material {mat_id!r}.")
            e_self_m = element_mass_per_length(e, sec, mat, active_units, gravity_accel)

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

    # 6. Story mass aggregation (mass at the restrained base lands in the base bucket,
    # never snapped up to Story 1; base is taken from base_elevation or lowest supports)
    ms_messages: List[str] = [] if messages is None else messages
    stories_list = list(getattr(model, "stories", None) or [])
    if not stories_list:
        stories_list = detect_stories(model)

    base_elev_val: Optional[float] = None
    if base_elevation is not None:
        base_elev_val = float(base_elevation)
    elif getattr(model, "base_elevation", None) is not None:
        base_elev_val = float(model.base_elevation)
    elif getattr(model, "supports", None):
        base_elev_val = resolve_base_elevation(model, messages=ms_messages)

    elevs: List[float] = []
    if base_elev_val is not None:
        elevs.append(base_elev_val)
    for st in stories_list:
        ev = float(st.elevation)
        if not any(abs(ev - existing) <= 1e-9 for existing in elevs):
            elevs.append(ev)
    elevs.sort()

    story_masses: Dict[float, float] = {}
    if elevs:
        for elev in elevs:
            story_masses[elev] = 0.0

        for nid, m_vec in nodal_masses.items():
            nd = nodes_map[nid]
            z_coord = getattr(nd, "z", 0.0) if v_axis == "z" else nd.y
            closest_elev = min(elevs, key=lambda el: abs(el - z_coord))
            m_eff = m_vec[0] if ms.lateral_only else max(m_vec)
            story_masses[closest_elev] += m_eff

        for e in model.elements:
            if e.start_node not in nodes_map or e.end_node not in nodes_map:
                continue
            n_i = nodes_map[e.start_node]
            n_j = nodes_map[e.end_node]
            z_i = getattr(n_i, "z", 0.0) if v_axis == "z" else n_i.y
            z_j = getattr(n_j, "z", 0.0) if v_axis == "z" else n_j.y
            L = math.sqrt(
                (n_j.x - n_i.x) ** 2
                + (n_j.y - n_i.y) ** 2
                + (getattr(n_j, "z", 0.0) - getattr(n_i, "z", 0.0)) ** 2
            )
            if L <= 1e-12:
                continue

            e_extra_m = element_masses.get(e.id, 0.0)
            e_self_m = 0.0
            if ms.include_element_mass and str(getattr(e, "type", "")).lower() != "spring":
                sec_id = getattr(e, "section_id", None)
                sec = sections_map.get(sec_id)
                if sec is None:
                    raise ValueError(f"Element {e.id!r} references missing section {sec_id!r}.")
                mat_id = getattr(sec, "material_id", None)
                mat = materials_map.get(mat_id)  # the section owns the material
                if mat is None:
                    raise ValueError(f"Section {sec.id!r} on element {e.id!r} references missing material {mat_id!r}.")
                e_self_m = element_mass_per_length(e, sec, mat, active_units, gravity_accel)

            total_elem_m = (e_self_m + e_extra_m) * L
            if abs(z_i - z_j) < 1e-4:
                # Horizontal member at elevation
                elev = 0.5 * (z_i + z_j)
                closest_elev = min(elevs, key=lambda el: abs(el - elev))
                story_masses[closest_elev] += total_elem_m
            else:
                # Vertical or inclined member: split tributary mass between endpoints
                closest_elev_i = min(elevs, key=lambda el: abs(el - z_i))
                closest_elev_j = min(elevs, key=lambda el: abs(el - z_j))
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
        messages=list(ms_messages),
    )


def create_asce7_mass_source(
    model: Optional[Union[StructuralModel, Project]] = None,
    dead_load_case_ids: Optional[List[str]] = None,
    storage_live_case_ids: Optional[List[str]] = None,
    general_live_case_ids: Optional[List[str]] = None,
    include_general_live: bool = False,
    general_live_factor: float = 0.0,
    snow_load_case_ids: Optional[List[str]] = None,
    flat_roof_snow_pressure: float = 0.0,
    partition_load_case_ids: Optional[List[str]] = None,
    equipment_case_ids: Optional[List[str]] = None,
    roof_garden_case_ids: Optional[List[str]] = None,
    include_element_mass: bool = True,
    include_added_mass: bool = True,
    lateral_only: bool = False,
    units: Optional[Union[str, Units, dict]] = None,
    source_id: str = "MS_ASCE7",
    name: str = "ASCE 7-22 Effective Seismic Weight",
) -> MassSource:
    """
    Construct a code-compliant MassSource per ASCE 7-22 Section 12.7.2.

    Parameters & Load Factors per ASCE 7-22 Section 12.7.2:
    ------------------------------------------------------
    1. Dead Load (structural self-weight & superimposed dead):
       - Factor: 1.0 (Section 12.7.2)
    2. Storage Live Load (areas used for storage):
       - Factor: 0.25 minimum (Section 12.7.2 Item 1)
       - Note: Per Section 12.7.2 Exception (b), public garages and open parking structures
         need not be included (factor = 0.0).
    3. General Floor Live Load (offices, residential, retail, classrooms, etc.):
       - Factor: 0.0 (Excluded per Section 12.7.2 unless explicitly requested via
         `include_general_live=True` or `general_live_factor`).
    4. Partitions:
       - Factor: 1.0 (Section 12.7.2 Item 2: actual partition weight or minimum
         10 psf / 0.48 kN/m^2 of floor area).
    5. Permanent Equipment:
       - Factor: 1.0 (Section 12.7.2 Item 3: total operating weight).
    6. Snow Load:
       - Factor: 0.15 if flat roof snow load Pf > 45 psf (2.16 kN/m^2) (Section 12.7.2 Item 4).
       - Factor: 0.0 if Pf <= 45 psf (2.16 kN/m^2).
    7. Roof Gardens & Landscaping:
       - Factor: 1.0 (Section 12.7.2 Item 5 per Section 3.1.4).

    Parameters
    ----------
    model : StructuralModel or Project, optional
        Structural model to auto-detect load cases from if specific case lists are omitted.
    dead_load_case_ids : List[str], optional
        List of load case IDs representing Dead loads (factor = 1.0).
    storage_live_case_ids : List[str], optional
        List of load case IDs representing Storage live loads (factor = 0.25).
    general_live_case_ids : List[str], optional
        List of load case IDs representing General (non-storage) live loads (factor = 0.0).
    include_general_live : bool, default=False
        Whether to include non-storage live loads in seismic weight.
    general_live_factor : float, default=0.0
        Factor for general live load if include_general_live is True.
    snow_load_case_ids : List[str], optional
        List of load case IDs representing Snow loads.
    flat_roof_snow_pressure : float, default=0.0
        Flat roof snow load Pf in active pressure units (e.g. kN/m^2 or psf).
        If Pf > 2.16 kN/m^2 (45 psf), a factor of 0.15 is applied; otherwise 0.0.
    partition_load_case_ids : List[str], optional
        List of load case IDs representing partition loads (factor = 1.0).
    equipment_case_ids : List[str], optional
        List of load case IDs representing permanent equipment (factor = 1.0).
    roof_garden_case_ids : List[str], optional
        List of load case IDs representing roof gardens / landscaping (factor = 1.0).
    include_element_mass : bool, default=True
        Whether to calculate element inertial self-mass from geometry and density.
    include_added_mass : bool, default=True
        Whether to include user-assigned lumped nodal and element masses.
    lateral_only : bool, default=False
        Whether dynamic mass participates only in lateral degrees of freedom.
    units : str, Units, or dict, optional
        Active units specification.
    source_id : str, default="MS_ASCE7"
        Identifier for the MassSource instance.
    name : str, default="ASCE 7-22 Effective Seismic Weight"
        Human-readable description.

    Returns
    -------
    MassSource
        Configured MassSource object.
    """
    raw_model = model.model if isinstance(model, Project) else model
    load_cases = getattr(raw_model, "load_cases", []) if raw_model is not None else []
    all_case_ids = [lc.id for lc in load_cases]

    # Auto-detection if model is provided and explicit lists are omitted
    if raw_model is not None:
        if dead_load_case_ids is None:
            dead_load_case_ids = [
                cid for cid in all_case_ids
                if any(k in str(cid).lower() for k in ("dead", "sw", "sdl"))
            ]
        if storage_live_case_ids is None:
            storage_live_case_ids = [
                cid for cid in all_case_ids
                if any(k in str(cid).lower() for k in ("storage", "stor"))
            ]
        if general_live_case_ids is None:
            storage_set = set(storage_live_case_ids or [])
            general_live_case_ids = [
                cid for cid in all_case_ids
                if any(k in str(cid).lower() for k in ("live", "ll"))
                and cid not in storage_set
                and not any(k in str(cid).lower() for k in ("roof", "lr", "storage", "stor"))
            ]
        if partition_load_case_ids is None:
            partition_load_case_ids = [
                cid for cid in all_case_ids
                if any(k in str(cid).lower() for k in ("partition", "part"))
            ]
        if snow_load_case_ids is None:
            snow_load_case_ids = [
                cid for cid in all_case_ids
                if any(k in str(cid).lower() for k in ("snow",))
            ]
        if equipment_case_ids is None:
            equipment_case_ids = [
                cid for cid in all_case_ids
                if any(k in str(cid).lower() for k in ("equip", "mep"))
            ]
        if roof_garden_case_ids is None:
            roof_garden_case_ids = [
                cid for cid in all_case_ids
                if any(k in str(cid).lower() for k in ("garden", "landscap"))
            ]

    # Resolve unit threshold for snow load (ASCE 7-22 Sec 12.7.2 Item 4: 45 psf / 2.16 kN/m^2)
    active_units = parse_units(units or getattr(getattr(raw_model, "metadata", None), "units", None) or getattr(raw_model, "units", None) or "kN, m")
    threshold_pressure = 2.16 * conversion_factor("kN, m", active_units, "pressure")
    snow_factor = 0.15 if flat_roof_snow_pressure > (threshold_pressure - 1e-6) else 0.0

    factors: List[MassSourceFactor] = []

    # 1. Dead load: 1.0
    for cid in (dead_load_case_ids or []):
        factors.append(MassSourceFactor(load_case_id=cid, factor=1.0))

    # 2. Storage live load: 0.25 (ASCE 7-22 Section 12.7.2 Item 1)
    for cid in (storage_live_case_ids or []):
        factors.append(MassSourceFactor(load_case_id=cid, factor=0.25))

    # 3. Partitions: 1.0 (ASCE 7-22 Section 12.7.2 Item 2)
    for cid in (partition_load_case_ids or []):
        factors.append(MassSourceFactor(load_case_id=cid, factor=1.0))

    # 4. Permanent equipment: 1.0 (ASCE 7-22 Section 12.7.2 Item 3)
    for cid in (equipment_case_ids or []):
        factors.append(MassSourceFactor(load_case_id=cid, factor=1.0))

    # 5. Snow: 0.15 if Pf > 45 psf (2.16 kN/m^2), else 0.0 (ASCE 7-22 Section 12.7.2 Item 4)
    if snow_factor > 0.0:
        for cid in (snow_load_case_ids or []):
            factors.append(MassSourceFactor(load_case_id=cid, factor=snow_factor))

    # 6. Roof garden / landscaping: 1.0 (ASCE 7-22 Section 12.7.2 Item 5)
    for cid in (roof_garden_case_ids or []):
        factors.append(MassSourceFactor(load_case_id=cid, factor=1.0))

    # 7. General live load: 0.0 unless explicitly requested
    gen_factor = general_live_factor if include_general_live else 0.0
    if gen_factor > 0.0:
        for cid in (general_live_case_ids or []):
            factors.append(MassSourceFactor(load_case_id=cid, factor=gen_factor))

    return MassSource(
        id=source_id,
        name=name,
        factors=factors,
        include_element_mass=include_element_mass,
        include_added_mass=include_added_mass,
        lateral_only=lateral_only,
    )

