"""
Tributary load distribution engine.
Translates floor area surface loads (kPa or psf) onto supporting beam elements (kN/m or kip/ft).
Supports both 2D plane frames (gravity along -Y) and 3D space frames (gravity along -Z).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from struct_core import (
    AreaLoad,
    DistributedLoad,
    LoadCase,
    StructuralModel,
)

from .geometry import (
    detect_model_dimensionality,
    detect_story_elevations,
    get_beam_elevation,
    get_beam_length,
    get_beam_span_and_orientation,
    is_horizontal_beam,
)


def distribute_tributary_load(
    pressure: float,
    tributary_width: float,
    direction: str = "-z",
) -> Tuple[float, float, float]:
    """
    Compute 3D distributed line load vector (wx, wy, wz) from surface pressure and tributary width:
    w = pressure * tributary_width

    Parameters
    ----------
    pressure : float
        Surface pressure (force/area, e.g. kPa or psf).
    tributary_width : float
        Tributary width (m or ft).
    direction : str
        Global load direction ('-z', '+z', '-y', '+y', '-x', '+x').

    Returns
    -------
    Tuple[float, float, float]
        (wx, wy, wz)
    """
    mag = float(pressure * tributary_width)
    d = direction.lower().strip()
    if d in ("-z", "z-", "down_z", "gravity"):
        return 0.0, 0.0, -mag
    elif d in ("+z", "z+"):
        return 0.0, 0.0, mag
    elif d in ("-y", "y-", "down_y"):
        return 0.0, -mag, 0.0
    elif d in ("+y", "y+"):
        return 0.0, mag, 0.0
    elif d in ("-x", "x-"):
        return -mag, 0.0, 0.0
    elif d in ("+x", "x+"):
        return mag, 0.0, 0.0
    raise ValueError(
        f"Invalid load direction: '{direction}'. Expected '-z', '+z', '-y', '+y', '-x', or '+x'."
    )


def distribute_1d_tributary(
    pressure: float,
    tributary_width: float,
    direction: str = "-y",
    return_3d: bool = False,
) -> Union[Tuple[float, float], Tuple[float, float, float]]:
    """
    Compute equivalent line load intensity from surface pressure and tributary width.
    Retains backwards-compatibility returning (wx, wy) for 2D, or (wx, wy, wz) if 3D requested.

    Parameters
    ----------
    pressure : float
        Surface pressure intensity.
    tributary_width : float
        Tributary framing width.
    direction : str
        Direction of applied load.
    return_3d : bool
        If True, returns a 3-element tuple (wx, wy, wz).

    Returns
    -------
    Tuple[float, float] or Tuple[float, float, float]
    """
    wx, wy, wz = distribute_tributary_load(pressure, tributary_width, direction=direction)
    if return_3d or "z" in direction.lower():
        return wx, wy, wz
    return wx, wy


def distribute_two_way_slab(
    pressure: float,
    short_span: float,
    long_span: float,
    design_action: str = "moment",
) -> Tuple[float, float, float]:
    """
    Two-way 45-degree tributary distribution (yield-line theory / ACI 318):
    - Short beam receives triangular load:
      peak w_peak = pressure * (short_span / 2)
      Moment equivalent: w_eq = pressure * (short_span / 3)
      Shear equivalent:  w_eq = pressure * (short_span / 4)
    - Long beam receives trapezoidal load:
      peak w_peak = pressure * (short_span / 2)
      Moment equivalent: w_eq = w_peak * (1 - 1 / (3 * m^2)) where m = long_span / short_span.
      Shear equivalent:  w_eq = w_peak * (1 - 1 / (2 * m))

    Parameters
    ----------
    pressure : float
        Surface pressure intensity.
    short_span : float
        Clear span of short side of the panel S.
    long_span : float
        Clear span of long side of the panel L.
    design_action : str
        'moment' (for maximum flexural design) or 'shear' (for maximum support reaction/shear).

    Returns
    -------
    Tuple[w_peak, w_eq_short, w_eq_long]
    """
    s = min(short_span, long_span)
    l = max(short_span, long_span)
    m = max(1.0, l / s) if s > 1e-6 else 1.0

    w_peak = pressure * (s / 2.0)
    action = design_action.lower()

    if "shear" in action:
        w_eq_short = pressure * (s / 4.0)
        w_eq_long = w_peak * (1.0 - (1.0 / (2.0 * m)))
    else:  # "moment" default
        w_eq_short = pressure * (s / 3.0)
        w_eq_long = w_peak * (1.0 - (1.0 / (3.0 * (m ** 2))))

    return round(w_peak, 4), round(w_eq_short, 4), round(w_eq_long, 4)


def apply_floor_load_to_frame(
    model: StructuralModel,
    pressure: float,
    tributary_width: float,
    target_case_id: str,
    case_name: Optional[str] = None,
    story_elevations: Optional[List[float]] = None,
    direction: Optional[str] = None,
    tolerance: float = 0.05,
) -> LoadCase:
    """
    Distribute uniform floor pressure across horizontal beam elements at specified floor levels.
    Automatically senses whether model is 2D or 3D:
    - 2D: Story elevations along Y, horizontal beams in X, gravity load along -Y (wy).
    - 3D: Story elevations along Z, horizontal beams in X-Y, gravity load along -Z (wz).

    Parameters
    ----------
    model : StructuralModel
        The structural frame model.
    pressure : float
        Surface pressure intensity (kN/m² or psf).
    tributary_width : float
        Tributary framing width (m or ft).
    target_case_id : str
        ID for the resulting LoadCase.
    case_name : str, optional
        Human-readable name for the LoadCase.
    story_elevations : List[float], optional
        Elevations (Y coordinates for 2D, Z for 3D) of floor beams.
        If None, automatically detected above ground level.
    direction : str, optional
        Direction of gravity: '-z' for 3D, '-y' for 2D. Inferred if None.
    tolerance : float
        Elevation and coordinate matching tolerance.

    Returns
    -------
    LoadCase
        Configured struct_core.LoadCase with DistributedLoad elements.
    """
    nodes_map = {n.id: n for n in model.nodes}
    element_loads: List[DistributedLoad] = []

    is_3d = detect_model_dimensionality(model, tolerance=tolerance) == "3d"
    if direction is None:
        direction = "-z" if is_3d else "-y"

    wx, wy, wz = distribute_tributary_load(pressure, tributary_width, direction=direction)

    elevs = story_elevations or detect_story_elevations(model, tolerance=tolerance)

    for elem in getattr(model, "elements", []):
        elem_type = str(getattr(elem, "type", "")).lower()
        if elem_type not in ("beam", "frame", ""):
            continue

        n_start = nodes_map.get(elem.start_node)
        n_end = nodes_map.get(elem.end_node)
        if not n_start or not n_end:
            continue

        if not is_horizontal_beam(elem, nodes_map, is_3d=is_3d, tolerance=tolerance):
            continue

        beam_elev = get_beam_elevation(elem, nodes_map, is_3d=is_3d)

        if elevs is not None and len(elevs) > 0:
            matches = any(abs(beam_elev - elev) <= tolerance for elev in elevs)
            if not matches:
                continue
        else:
            if beam_elev <= tolerance:
                continue

        element_loads.append(
            DistributedLoad(
                element_id=elem.id,
                wx=round(wx, 4),
                wy=round(wy, 4),
                wz=round(wz, 4),
            )
        )

    return LoadCase(
        id=target_case_id,
        name=case_name or f"Floor Load ({target_case_id})",
        element_loads=element_loads,
    )


def distribute_area_loads_to_beams(
    model: StructuralModel,
    target_case_id: Optional[str] = None,
    load_type_filter: Optional[str] = None,
    use_equivalent_uniform: bool = True,
    design_action: str = "moment",
    tolerance: float = 0.05,
    clear_area_loads: bool = False,
) -> int:
    """
    Scan model load cases, convert all AreaLoad definitions into DistributedLoad instances
    on perimeter beam elements, and append them to case.element_loads.

    Handles:
    - One-way slabs: distributes tributary strip w = q * (S / 2) to beams perpendicular
      to the span axis; beams parallel to span axis receive 0 load.
    - Two-way slabs: 45° yield-line distribution. Applies triangular equivalent load to
      short beams and trapezoidal equivalent load to long beams.

    Parameters
    ----------
    model : StructuralModel
        Structural model containing nodes, elements, and load cases with area loads.
    target_case_id : str, optional
        If specified, only processes the load case matching this ID.
    load_type_filter : str, optional
        Filter by load nature ('dead', 'live', 'snow', etc.).
    use_equivalent_uniform : bool
        If True, converts triangular/trapezoidal distributions to equivalent uniform line loads.
    design_action : str
        'moment' or 'shear' for equivalent load formulation.
    tolerance : float
        Geometric tolerance for node and element matching.
    clear_area_loads : bool
        If True, clears processed area_loads from the load case.

    Returns
    -------
    int
        Total number of distributed beam loads generated.
    """
    nodes_map = {n.id: n for n in model.nodes}
    is_3d = detect_model_dimensionality(model, tolerance=tolerance) == "3d"

    total_loads_added = 0

    for lc in model.load_cases:
        if target_case_id is not None and str(lc.id).lower() != str(target_case_id).lower():
            continue

        raw_area_loads = getattr(lc, "area_loads", [])
        if not raw_area_loads:
            continue

        loads_to_process: List[AreaLoad] = []
        for al in raw_area_loads:
            if load_type_filter and str(getattr(al, "load_type", "")).lower() != load_type_filter.lower():
                continue
            loads_to_process.append(al)

        if not loads_to_process:
            continue

        new_element_loads: List[DistributedLoad] = []

        for al in loads_to_process:
            pressure = float(al.pressure)
            if abs(pressure) < 1e-9:
                continue

            node_ids = getattr(al, "node_ids", [])
            elev = getattr(al, "story_elevation", None)
            one_way = getattr(al, "one_way", True)
            span_axis = str(getattr(al, "span_direction_axis", "x")).lower()

            if node_ids and len(node_ids) >= 3:
                # Panel defined by explicit boundary polygon nodes
                poly_nodes = [nodes_map[nid] for nid in node_ids if nid in nodes_map]
                if len(poly_nodes) < 3:
                    continue

                x_coords = [n.x for n in poly_nodes]
                y_coords = [n.y for n in poly_nodes]
                lx = max(x_coords) - min(x_coords)
                ly = max(y_coords) - min(y_coords)

                # Find edges of polygon
                edges: Set[Tuple[Any, Any]] = set()
                n_count = len(node_ids)
                for i in range(n_count):
                    n_a = node_ids[i]
                    n_b = node_ids[(i + 1) % n_count]
                    edges.add((min(str(n_a), str(n_b)), max(str(n_a), str(n_b))))

                # Find bounding beam elements on these edges
                bounding_beams = []
                for elem in model.elements:
                    edge_key = (min(str(elem.start_node), str(elem.end_node)),
                                max(str(elem.start_node), str(elem.end_node)))
                    if edge_key in edges:
                        bounding_beams.append(elem)

                if not bounding_beams:
                    continue

                for elem in bounding_beams:
                    span_len, orientation = get_beam_span_and_orientation(elem, nodes_map, is_3d=is_3d)

                    if one_way:
                        # In one-way slab:
                        # If slab spans along X (between Y-beams):
                        # supporting beams are orientation 'y', span is lx, trib width = lx / 2
                        # If slab spans along Y (between X-beams):
                        # supporting beams are orientation 'x', span is ly, trib width = ly / 2
                        if span_axis == "x":
                            if orientation == "y":
                                w_line = pressure * (lx / 2.0)
                            else:
                                w_line = 0.0
                        else:  # span_axis == "y"
                            if orientation == "x":
                                w_line = pressure * (ly / 2.0)
                            else:
                                w_line = 0.0
                    else:
                        # Two-way slab:
                        # Short span S = min(lx, ly), Long span L = max(lx, ly)
                        s = min(lx, ly)
                        l = max(lx, ly)
                        _, w_eq_short, w_eq_long = distribute_two_way_slab(
                            pressure, s, l, design_action=design_action
                        )
                        # Identify whether this beam is on the short span or long span
                        if lx <= ly:
                            # X-beams span lx (short side) -> w_eq_short
                            # Y-beams span ly (long side) -> w_eq_long
                            w_line = w_eq_short if orientation == "x" else w_eq_long
                        else:
                            # Y-beams span ly (short side) -> w_eq_short
                            # X-beams span lx (long side) -> w_eq_long
                            w_line = w_eq_short if orientation == "y" else w_eq_long

                    if abs(w_line) > 1e-6:
                        if is_3d:
                            new_element_loads.append(
                                DistributedLoad(
                                    element_id=elem.id,
                                    wx=0.0,
                                    wy=0.0,
                                    wz=round(-w_line, 4),
                                )
                            )
                        else:
                            new_element_loads.append(
                                DistributedLoad(
                                    element_id=elem.id,
                                    wx=0.0,
                                    wy=round(-w_line, 4),
                                )
                            )

            elif elev is not None and getattr(al, "tributary_width", None) is not None:
                # Area load with story elevation and tributary width
                trib_w = float(al.tributary_width)
                case_proxy = apply_floor_load_to_frame(
                    model=model,
                    pressure=pressure,
                    tributary_width=trib_w,
                    target_case_id=lc.id,
                    story_elevations=[float(elev)],
                    tolerance=tolerance,
                )
                new_element_loads.extend(case_proxy.element_loads)

        if new_element_loads:
            lc.element_loads.extend(new_element_loads)
            total_loads_added += len(new_element_loads)

        if clear_area_loads:
            lc.area_loads = [al for al in lc.area_loads if al not in loads_to_process]

    return total_loads_added
