"""
Geometric analysis and dimension-sensing engine for structural frames.
Detects 2D vs 3D frames, extracts floor elevations, and classifies horizontal beams.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from struct_core import Node, StructuralModel


def detect_model_dimensionality(model: StructuralModel, tolerance: float = 1e-4) -> str:
    """
    Detect whether a structural model is 2D (plane frame in X-Y or X-Z) or 3D (space frame in X-Y-Z).

    Parameters
    ----------
    model : StructuralModel
        The structural frame model.
    tolerance : float
        Coordinate threshold above which spatial extent is treated as non-zero.

    Returns
    -------
    str
        '3d' if the model has non-zero spatial extent along all 3 dimensions, otherwise '2d'.
    """
    dim = getattr(model, "dimension", None)
    if dim and str(dim).upper() in ("2D", "3D"):
        return str(dim).lower()

    if not getattr(model, "nodes", None):
        return "2d"

    xs = [float(n.x) for n in model.nodes]
    ys = [float(n.y) for n in model.nodes]
    zs = [float(getattr(n, "z", 0.0) or 0.0) for n in model.nodes]

    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    span_z = max(zs) - min(zs)

    dims_with_extent = sum(1 for span in (span_x, span_y, span_z) if span > tolerance)
    return "3d" if dims_with_extent >= 3 else "2d"


def resolve_base_elevation(
    model: StructuralModel,
    *,
    messages: Optional[List[str]] = None,
) -> float:
    """
    Return the structural base elevation for ASCE 7-22 §12.8.3 / §26.10 height measurements.

    Uses ``model.base_elevation`` when set. Otherwise infers the elevation of the lowest
    supported node and appends an explanatory note to ``messages`` (if provided). Never
    defaults silently to 0.
    """
    base_elev = getattr(model, "base_elevation", None)
    if base_elev is not None:
        return float(base_elev)

    from struct_core import vertical_axis as get_vertical_axis

    v_axis = get_vertical_axis(model)
    use_z = (v_axis == "z")
    nodes_map = {n.id: n for n in getattr(model, "nodes", [])}
    sup_coords: List[float] = []
    for sup in getattr(model, "supports", []) or []:
        nd = nodes_map.get(sup.node_id)
        if nd is not None:
            sup_coords.append(float(getattr(nd, "z", 0.0) if use_z else nd.y))
    if sup_coords:
        inferred = min(sup_coords)
        if messages is not None:
            messages.append(
                f"Model has no base_elevation; used lowest support elevation ({inferred:g}) as the base datum."
            )
        return inferred

    raise ValueError(
        "Model has no base_elevation and no supports from which to determine the base elevation. "
        "Set model.base_elevation explicitly or define supports."
    )


def detect_stories(
    model: StructuralModel,
    tolerance: float = 0.05,
    *,
    messages: Optional[List[str]] = None,
) -> List[Any]:
    """
    Identify named stories above the base datum.

    Returns ``model.stories`` (sorted ascending by elevation) when defined. Otherwise infers
    unique floor levels from horizontal beams (or nodes) strictly above the base elevation
    and names them ``Story1``, ``Story2``, ...

    Parameters
    ----------
    model : StructuralModel
        The structural model.
    tolerance : float
        Tolerance for grouping similar elevation coordinates.
    messages : List[str], optional
        If provided, records a note when the base elevation is inferred from supports.

    Returns
    -------
    List[Story]
        Sorted list of named stories above base (ascending by elevation).
    """
    from struct_core import Story, vertical_axis as get_vertical_axis

    explicit_stories = getattr(model, "stories", None)
    if explicit_stories:
        return sorted(list(explicit_stories), key=lambda s: float(s.elevation))

    is_3d = detect_model_dimensionality(model, tolerance=tolerance) == "3d"
    v_axis = get_vertical_axis(model)
    use_z = (v_axis == "z")
    nodes_map = {n.id: n for n in model.nodes}

    if getattr(model, "base_elevation", None) is not None:
        base_elev = float(model.base_elevation)
    elif getattr(model, "supports", None):
        base_elev = resolve_base_elevation(model, messages=messages)
    elif model.nodes:
        base_elev = min(float(getattr(n, "z", 0.0) if use_z else n.y) for n in model.nodes)
    else:
        return []

    # Gather elevations from horizontal beams first (most accurate structural indicator)
    beam_elevs: List[float] = []
    for elem in getattr(model, "elements", []):
        elem_type = str(getattr(elem, "type", "")).lower()
        if elem_type not in ("beam", "frame", ""):
            continue
        n_start = nodes_map.get(elem.start_node)
        n_end = nodes_map.get(elem.end_node)
        if not n_start or not n_end:
            continue

        if is_horizontal_beam(elem, nodes_map, is_3d=is_3d, tolerance=tolerance, vertical_axis=v_axis):
            elev = get_beam_elevation(elem, nodes_map, is_3d=is_3d, vertical_axis=v_axis)
            if elev > base_elev + tolerance:
                beam_elevs.append(round(elev, 4))

    raw_elevs = beam_elevs if beam_elevs else [
        round(float(getattr(n, "z", 0.0) if use_z else getattr(n, "y", 0.0)), 4)
        for n in model.nodes
        if float(getattr(n, "z", 0.0) if use_z else getattr(n, "y", 0.0)) > base_elev + tolerance
    ]

    unique_sorted = sorted(list(set(raw_elevs)))
    filtered_elevations: List[float] = []
    for elev in unique_sorted:
        if not filtered_elevations or abs(elev - filtered_elevations[-1]) > tolerance:
            filtered_elevations.append(elev)

    return [Story(name=f"Story{idx}", elevation=ev) for idx, ev in enumerate(filtered_elevations, start=1)]


def is_horizontal_beam(
    elem: Any,
    nodes_map: Dict[Any, Node],
    is_3d: bool = False,
    tolerance: float = 0.05,
    vertical_axis: Optional[str] = None,
) -> bool:
    """
    Check if a beam element is oriented horizontally at a floor elevation.

    - In 3D: start and end nodes have matching Z coordinates (|z1 - z2| <= tol)
      and span length in X-Y plane > tolerance.
    - In 2D:
      - If vertical_axis == 'z': start and end nodes have matching Z coordinates (|z1 - z2| <= tol)
        and span length in X-Y plane > tolerance.
      - If vertical_axis == 'y' (default): start and end nodes have matching Y coordinates (|y1 - y2| <= tol)
        and span length in X > tolerance.
    """
    n_start = nodes_map.get(elem.start_node)
    n_end = nodes_map.get(elem.end_node)
    if not n_start or not n_end:
        return False

    v_axis = (vertical_axis or ("z" if is_3d else "y")).lower()
    if v_axis == "z":
        z_start = getattr(n_start, "z", 0.0) or 0.0
        z_end = getattr(n_end, "z", 0.0) or 0.0
        dz = abs(z_start - z_end)
        xy_dist = math.hypot(n_end.x - n_start.x, n_end.y - n_start.y)
        return dz <= tolerance and xy_dist > tolerance
    else:
        dy = abs(n_start.y - n_end.y)
        dx = abs(n_end.x - n_start.x)
        return dy <= tolerance and dx > tolerance


def get_beam_elevation(
    elem: Any,
    nodes_map: Dict[Any, Node],
    is_3d: bool = False,
    vertical_axis: Optional[str] = None,
) -> float:
    """
    Get the mid-elevation of a beam element (Z for vertical_axis == 'z' or 3D, Y for 2D).
    """
    n_start = nodes_map[elem.start_node]
    n_end = nodes_map[elem.end_node]
    v_axis = (vertical_axis or ("z" if is_3d else "y")).lower()
    if v_axis == "z":
        z1 = getattr(n_start, "z", 0.0) or 0.0
        z2 = getattr(n_end, "z", 0.0) or 0.0
        return (z1 + z2) / 2.0
    return (n_start.y + n_end.y) / 2.0


def get_beam_length(elem: Any, nodes_map: Dict[Any, Node]) -> float:
    """Calculate the 3D Euclidean length of an element."""
    n_start = nodes_map[elem.start_node]
    n_end = nodes_map[elem.end_node]
    dx = n_end.x - n_start.x
    dy = n_end.y - n_start.y
    dz = (getattr(n_end, "z", 0.0) or 0.0) - (getattr(n_start, "z", 0.0) or 0.0)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def get_beam_span_and_orientation(
    elem: Any,
    nodes_map: Dict[Any, Node],
    is_3d: bool,
) -> Tuple[float, str]:
    """
    Get length and orientation axis ('x', 'y', or 'diagonal') of a horizontal beam.
    """
    n_start = nodes_map[elem.start_node]
    n_end = nodes_map[elem.end_node]
    length = get_beam_length(elem, nodes_map)

    if not is_3d:
        return length, "x"

    dx = abs(n_end.x - n_start.x)
    dy = abs(n_end.y - n_start.y)
    if dx > 2.0 * dy:
        return length, "x"
    elif dy > 2.0 * dx:
        return length, "y"
    return length, "diagonal"
