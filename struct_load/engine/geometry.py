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
    Detect whether a structural model is 2D (plane frame in X-Y) or 3D (space frame in X-Y-Z).

    Parameters
    ----------
    model : StructuralModel
        The structural frame model.
    tolerance : float
        Coordinate threshold above which Z coordinates are treated as 3D.

    Returns
    -------
    str
        '3d' if any node has abs(z) > tolerance, otherwise '2d'.
    """
    for node in model.nodes:
        z = getattr(node, "z", 0.0) or 0.0
        if abs(z) > tolerance:
            return "3d"
    return "2d"


def detect_story_elevations(
    model: StructuralModel,
    tolerance: float = 0.05,
) -> List[float]:
    """
    Identify unique floor / story elevations above ground level (> tolerance).

    In a 3D model, story elevations correspond to Z coordinates.
    In a 2D plane frame, story elevations correspond to Y coordinates.

    Parameters
    ----------
    model : StructuralModel
        The structural model.
    tolerance : float
        Tolerance for grouping similar elevation coordinates.

    Returns
    -------
    List[float]
        Sorted list of story elevations (ascending).
    """
    # If the model explicitly specifies story elevations, use them
    explicit_elevs = getattr(model, "story_elevations", None)
    if explicit_elevs:
        filtered = sorted([round(float(e), 4) for e in explicit_elevs if float(e) > tolerance])
        if filtered:
            return filtered

    is_3d = detect_model_dimensionality(model, tolerance=tolerance) == "3d"
    nodes_map = {n.id: n for n in model.nodes}

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

        if is_horizontal_beam(elem, nodes_map, is_3d=is_3d, tolerance=tolerance):
            elev = get_beam_elevation(elem, nodes_map, is_3d=is_3d)
            if elev > tolerance:
                beam_elevs.append(round(elev, 4))

    raw_elevs = beam_elevs if beam_elevs else [
        round(float(getattr(n, "z", 0.0) if is_3d else getattr(n, "y", 0.0)), 4)
        for n in model.nodes
        if float(getattr(n, "z", 0.0) if is_3d else getattr(n, "y", 0.0)) > tolerance
    ]

    unique_sorted = sorted(list(set(raw_elevs)))
    filtered_elevations: List[float] = []
    for elev in unique_sorted:
        if not filtered_elevations or abs(elev - filtered_elevations[-1]) > tolerance:
            filtered_elevations.append(elev)

    return filtered_elevations


def is_horizontal_beam(
    elem: Any,
    nodes_map: Dict[Any, Node],
    is_3d: bool,
    tolerance: float = 0.05,
) -> bool:
    """
    Check if a beam element is oriented horizontally at a floor elevation.

    - In 3D: start and end nodes have matching Z coordinates (|z1 - z2| <= tol)
      and span length in X-Y plane > tolerance.
    - In 2D: start and end nodes have matching Y coordinates (|y1 - y2| <= tol)
      and span length in X > tolerance.
    """
    n_start = nodes_map.get(elem.start_node)
    n_end = nodes_map.get(elem.end_node)
    if not n_start or not n_end:
        return False

    if is_3d:
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
    is_3d: bool,
) -> float:
    """
    Get the mid-elevation of a beam element (Z for 3D, Y for 2D).
    """
    n_start = nodes_map[elem.start_node]
    n_end = nodes_map[elem.end_node]
    if is_3d:
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
