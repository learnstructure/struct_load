"""
Tributary load distribution engine.
Translates floor area surface loads (kPa or psf) onto supporting beam elements (kN/m or kip/ft).
Supports both 2D plane frames (gravity along -Y) and 3D space frames (gravity along -Z).
"""

from __future__ import annotations

import math
from typing import Any, Literal, Dict, List, Optional, Set, Tuple, Union
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


def _distribute_3d_floor_panels(
    model: StructuralModel,
    pressure: float,
    story_elevations: List[float],
    direction: str = "-z",
    design_action: str = "shear",
    tolerance: float = 0.05,
    allow_partial_floor: bool = False,
) -> Tuple[Dict[Union[int, str], float], Dict[float, float], List[Union[int, str]]]:
    """
    Detect rectangular panels in a 3D building frame at each story elevation,
    and compute equivalent uniform line loads on bounding beams using two-way
    yield-line theory (distribute_two_way_slab with shear equilibrium).
    Returns a dict mapping element_id -> line_load_magnitude.
    """
    nodes_map = {n.id: n for n in model.nodes}
    beam_loads: Dict[Union[int, str], float] = {}
    loaded_areas_by_elev: Dict[float, float] = {}
    unloaded_beams: List[Union[int, str]] = []

    for elev in story_elevations:
        elev_beams = []
        for elem in getattr(model, "elements", []):
            elem_type = str(getattr(elem, "type", "")).lower()
            if elem_type not in ("beam", "frame", ""):
                continue
            if not is_horizontal_beam(elem, nodes_map, is_3d=True, tolerance=tolerance):
                continue
            be = get_beam_elevation(elem, nodes_map, is_3d=True)
            if abs(be - elev) <= tolerance:
                elev_beams.append(elem)

        if not elev_beams:
            continue

        x_beams: Dict[Tuple[float, float, float], Any] = {}
        y_beams: Dict[Tuple[float, float, float], Any] = {}
        all_x = set()
        all_y = set()

        for b in elev_beams:
            n1 = nodes_map.get(b.start_node)
            n2 = nodes_map.get(b.end_node)
            if not n1 or not n2:
                continue
            x1, y1 = round(float(n1.x), 2), round(float(n1.y), 2)
            x2, y2 = round(float(n2.x), 2), round(float(n2.y), 2)
            all_x.update([x1, x2])
            all_y.update([y1, y2])

            if abs(y1 - y2) <= tolerance and abs(x1 - x2) > tolerance:
                y_pos = round((y1 + y2) / 2.0, 2)
                min_x = min(x1, x2)
                max_x = max(x1, x2)
                x_beams[(y_pos, min_x, max_x)] = b
            elif abs(x1 - x2) <= tolerance and abs(y1 - y2) > tolerance:
                x_pos = round((x1 + x2) / 2.0, 2)
                min_y = min(y1, y2)
                max_y = max(y1, y2)
                y_beams[(x_pos, min_y, max_y)] = b

        sorted_x = sorted(list(all_x))
        sorted_y = sorted(list(all_y))

        loaded_area_level = 0.0
        outline_area_level = 0.0
        loaded_beams_at_level = set()

        for ix in range(len(sorted_x) - 1):
            x_left = sorted_x[ix]
            x_right = sorted_x[ix + 1]
            lx = x_right - x_left
            if lx <= tolerance:
                continue

            for iy in range(len(sorted_y) - 1):
                y_bot = sorted_y[iy]
                y_top = sorted_y[iy + 1]
                ly = y_top - y_bot
                if ly <= tolerance:
                    continue

                b_bot = x_beams.get((y_bot, x_left, x_right))
                b_top = x_beams.get((y_top, x_left, x_right))
                b_left = y_beams.get((x_left, y_bot, y_top))
                b_right = y_beams.get((x_right, y_bot, y_top))

                k = sum(1 for b in (b_bot, b_top, b_left, b_right) if b is not None)
                if k >= 2:
                    outline_area_level += lx * ly

                if b_bot and b_top and b_left and b_right:
                    loaded_area_level += lx * ly
                    loaded_beams_at_level.update([b_bot.id, b_top.id, b_left.id, b_right.id])
                    s = min(lx, ly)
                    l = max(lx, ly)
                    _, w_eq_short, w_eq_long = distribute_two_way_slab(
                        pressure, s, l, design_action=design_action
                    )
                    if lx <= ly:
                        w_x = w_eq_short
                        w_y = w_eq_long
                    else:
                        w_x = w_eq_long
                        w_y = w_eq_short

                    for elem in (b_bot, b_top):
                        beam_loads[elem.id] = beam_loads.get(elem.id, 0.0) + w_x
                    for elem in (b_left, b_right):
                        beam_loads[elem.id] = beam_loads.get(elem.id, 0.0) + w_y

        loaded_areas_by_elev[elev] = round(loaded_area_level, 4)
        if outline_area_level > tolerance:
            if loaded_area_level < outline_area_level - 1e-4:
                if not allow_partial_floor:
                    missing_area = outline_area_level - loaded_area_level
                    raise ValueError(
                        f"Floor level z = {elev:.2f}: loaded panel area ({loaded_area_level:.2f}) "
                        f"is less than floor outline area ({outline_area_level:.2f}) by {missing_area:.2f}. "
                        f"Some floor bays are not closed by bounding beams. "
                        f"Close the panels with beams, specify one_way_span ('x' or 'y'), "
                        f"or set allow_partial_floor=True."
                    )

        for b in elev_beams:
            if b.id not in loaded_beams_at_level:
                unloaded_beams.append(b.id)

    return beam_loads, loaded_areas_by_elev, unloaded_beams


def _distribute_3d_one_way(
    model: StructuralModel,
    pressure: float,
    story_elevations: Sequence[float],
    one_way_span: str,
    direction: str = "-z",
    tolerance: float = 0.05,
) -> Dict[Any, float]:
    """
    Distribute one-way slab pressure to horizontal beam lines in a 3D frame.

    When one_way_span is 'y', the slab spans along Y between beam lines running in X.
    Only beams running along X receive slab load; beams along Y receive no load.
    Each beam line receives q * tributary_width, where tributary_width is half the
    distance to the adjacent beam lines on each side (edge beam lines receive half-bay width).

    When one_way_span is 'x', beams running along Y receive load based on bay spacing in X.
    """
    span_dir = one_way_span.lower().strip()
    if span_dir not in ("x", "y"):
        raise ValueError(f"one_way_span must be 'x' or 'y', got {one_way_span!r}")

    nodes_map = {n.id: n for n in model.nodes}
    elevs = list(story_elevations)
    beam_loads: Dict[Any, float] = {}

    for elev in elevs:
        level_beams = []
        for elem in getattr(model, "elements", []):
            if str(getattr(elem, "type", "")).lower() not in ("beam", "frame", ""):
                continue
            ns = nodes_map.get(elem.start_node)
            ne = nodes_map.get(elem.end_node)
            if not ns or not ne:
                continue
            if abs(ns.z - elev) <= tolerance and abs(ne.z - elev) <= tolerance:
                dx = abs(ne.x - ns.x)
                dy = abs(ne.y - ns.y)
                if dx > tolerance or dy > tolerance:
                    level_beams.append((elem, ns, ne, dx, dy))

        if not level_beams:
            continue

        if span_dir == "y":
            # Slab spans along Y, supported by beams running along X (dx > dy)
            supporting = [b for b in level_beams if b[3] > b[4] and b[3] > tolerance]
            coords = sorted({round((b[1].y + b[2].y) / 2.0, 4) for b in supporting})
            if not coords:
                continue
            trib_map = {}
            for i, y_coord in enumerate(coords):
                w_left = (y_coord - coords[i - 1]) / 2.0 if i > 0 else 0.0
                w_right = (coords[i + 1] - y_coord) / 2.0 if i < len(coords) - 1 else 0.0
                trib_map[y_coord] = w_left + w_right

            for elem, ns, ne, dx, dy in supporting:
                y_c = round((ns.y + ne.y) / 2.0, 4)
                w_trib = trib_map.get(y_c, 0.0)
                beam_loads[elem.id] = pressure * w_trib

        elif span_dir == "x":
            # Slab spans along X, supported by beams running along Y (dy > dx)
            supporting = [b for b in level_beams if b[4] > b[3] and b[4] > tolerance]
            coords = sorted({round((b[1].x + b[2].x) / 2.0, 4) for b in supporting})
            if not coords:
                continue
            trib_map = {}
            for i, x_coord in enumerate(coords):
                w_left = (x_coord - coords[i - 1]) / 2.0 if i > 0 else 0.0
                w_right = (coords[i + 1] - x_coord) / 2.0 if i < len(coords) - 1 else 0.0
                trib_map[x_coord] = w_left + w_right

            for elem, ns, ne, dx, dy in supporting:
                x_c = round((ns.x + ne.x) / 2.0, 4)
                w_trib = trib_map.get(x_c, 0.0)
                beam_loads[elem.id] = pressure * w_trib

    return beam_loads


def apply_floor_load_to_frame(
    model: StructuralModel,
    pressure: float,
    tributary_width: Optional[float] = None,
    target_case_id: str = "FloorLoad",
    case_name: Optional[str] = None,
    story_elevations: Optional[List[float]] = None,
    direction: Optional[str] = None,
    two_way: bool = False,
    one_way_span: Optional[Literal["x", "y"]] = None,
    tolerance: float = 0.05,
    load_type: Optional[str] = None,
    allow_partial_floor: bool = False,
) -> LoadCase:
    """
    Distribute uniform floor pressure across horizontal beam elements at specified floor levels.
    Automatically senses whether model is 2D or 3D:
    - 2D: Story elevations along Y, horizontal beams in X, gravity load along -Y (wy).
    - 3D: Story elevations along Z, horizontal beams in X-Y, gravity load along -Z (wz).
      When two_way=True, detects floor panels and distributes via 45-degree yield line.

    Parameters
    ----------
    model : StructuralModel
        The structural frame model.
    pressure : float
        Surface pressure intensity (kN/m² or psf).
    tributary_width : float
        Tributary framing width (m or ft) for 1D beam tributary strip.
    target_case_id : str
        ID for the resulting LoadCase.
    case_name : str, optional
        Human-readable name for the LoadCase.
    story_elevations : List[float], optional
        Elevations (Y coordinates for 2D, Z for 3D) of floor beams.
        If None, automatically detected above ground level.
    direction : str, optional
        Direction of gravity: '-z' for 3D, '-y' for 2D. Inferred if None.
    two_way : bool
        If True in 3D, distributes floor loads across bounding beams using two-way yield lines.
    tolerance : float
        Elevation and coordinate matching tolerance.

    Returns
    -------
    LoadCase
        Configured struct_core.LoadCase with DistributedLoad elements.
    """
    from struct_core import vertical_axis as get_vertical_axis

    v_axis = get_vertical_axis(model)
    nodes_map = {n.id: n for n in model.nodes}
    element_loads: List[DistributedLoad] = []

    is_3d = detect_model_dimensionality(model, tolerance=tolerance) == "3d"
    if direction is None:
        direction = "-z" if (is_3d or v_axis == "z") else "-y"

    elevs = story_elevations or detect_story_elevations(model, tolerance=tolerance)

    if load_type is None:
        cid_lower = str(target_case_id).lower()
        if "dead" in cid_lower or "sdl" in cid_lower:
            load_type = "dead"
        elif "partition" in cid_lower:
            load_type = "partition"
        elif "live" in cid_lower:
            load_type = "live"

    if is_3d:
        if two_way:
            panel_loads, loaded_areas, unloaded = _distribute_3d_floor_panels(
                model=model,
                pressure=pressure,
                story_elevations=elevs,
                direction=direction,
                tolerance=tolerance,
                allow_partial_floor=allow_partial_floor,
            )
            if not panel_loads:
                raise ValueError(
                    f"No closed floor panels found at story levels {elevs}; specify one_way_span ('x' or 'y') in 3D "
                    f"or tributary_width in 2D."
                )
            for eid, w_val in panel_loads.items():
                element_loads.append(
                    DistributedLoad(
                        element_id=eid,
                        wx=0.0,
                        wy=0.0,
                        wz=round(-float(w_val), 4),
                    )
                )
            lc = LoadCase(
                id=target_case_id,
                name=case_name or f"Floor Load ({target_case_id})",
                load_type=load_type,
                element_loads=element_loads,
            )
            lc.loaded_floor_area = loaded_areas
            lc.unloaded_beams = unloaded
            return lc
        else:
            if tributary_width is not None and one_way_span is None:
                raise ValueError(
                    "tributary_width is for 2D frames only; in 3D specify one_way_span ('x' or 'y') or use two_way=True."
                )
            if one_way_span is None:
                raise ValueError("one_way_span ('x' or 'y') is required for 3D one-way floor load distribution.")
            one_way_loads = _distribute_3d_one_way(
                model=model,
                pressure=pressure,
                story_elevations=elevs,
                one_way_span=one_way_span,
                direction=direction,
                tolerance=tolerance,
            )
            for eid, w_val in one_way_loads.items():
                element_loads.append(
                    DistributedLoad(
                        element_id=eid,
                        wx=0.0,
                        wy=0.0,
                        wz=round(-float(w_val), 4),
                    )
                )
            return LoadCase(
                id=target_case_id,
                name=case_name or f"Floor Load ({target_case_id})",
                load_type=load_type,
                element_loads=element_loads,
            )

    if tributary_width is None:
        raise ValueError("tributary_width (model length unit) is required for 2D floor load distribution.")

    wx, wy, wz = distribute_tributary_load(pressure, tributary_width, direction=direction)

    for elem in getattr(model, "elements", []):
        elem_type = str(getattr(elem, "type", "")).lower()
        if elem_type not in ("beam", "frame", ""):
            continue

        n_start = nodes_map.get(elem.start_node)
        n_end = nodes_map.get(elem.end_node)
        if not n_start or not n_end:
            continue

        if not is_horizontal_beam(elem, nodes_map, is_3d=is_3d, tolerance=tolerance, vertical_axis=v_axis):
            continue

        beam_elev = get_beam_elevation(elem, nodes_map, is_3d=is_3d, vertical_axis=v_axis)

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

    if load_type is None:
        cid_lower = str(target_case_id).lower()
        if "dead" in cid_lower or "sdl" in cid_lower:
            load_type = "dead"
        elif "partition" in cid_lower:
            load_type = "partition"
        elif "live" in cid_lower:
            load_type = "live"

    return LoadCase(
        id=target_case_id,
        name=case_name or f"Floor Load ({target_case_id})",
        load_type=load_type,
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
