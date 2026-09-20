"""
Tributary load distribution engine.
Translates floor area surface loads (kPa or psf) onto supporting beam elements (kN/m or kip/ft).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
from struct_core import (
    AreaLoad,
    DistributedLoad,
    LoadCase,
    StructuralModel,
)


def distribute_1d_tributary(
    pressure: float,
    tributary_width: float,
    direction: str = "-y",
) -> Tuple[float, float]:
    """
    Compute equivalent line load intensity (wx, wy) from surface pressure and tributary width:
    w = pressure * tributary_width
    """
    mag = pressure * tributary_width
    if direction.lower() == "-y":
        return 0.0, -mag
    elif direction.lower() == "+y":
        return 0.0, mag
    elif direction.lower() == "-x":
        return -mag, 0.0
    elif direction.lower() == "+x":
        return mag, 0.0
    return 0.0, -mag


def distribute_two_way_slab(
    pressure: float,
    short_span: float,
    long_span: float,
) -> Tuple[float, float, float]:
    """
    Two-way 45-degree tributary distribution (yield-line theory):
    - Short beam receives triangular load: peak w_peak = pressure * (short_span / 2)
      Equivalent uniform load for bending: w_eq = pressure * (short_span / 3)
    - Long beam receives trapezoidal load: peak w_peak = pressure * (short_span / 2)
      Equivalent uniform load for bending: w_eq = pressure * (short_span / 2) * (1 - 1/(3 * m^2)) where m = long_span / short_span.

    Returns
    -------
    Tuple[w_peak, w_eq_short, w_eq_long]
    """
    m = max(1.0, long_span / short_span)
    w_peak = pressure * (short_span / 2.0)
    w_eq_short = pressure * (short_span / 3.0)
    w_eq_long = w_peak * (1.0 - (1.0 / (3.0 * (m ** 2))))
    return round(w_peak, 3), round(w_eq_short, 3), round(w_eq_long, 3)


def apply_floor_load_to_frame(
    model: StructuralModel,
    pressure: float,
    tributary_width: float,
    target_case_id: str,
    case_name: Optional[str] = None,
    story_elevations: Optional[List[float]] = None,
    direction: str = "-y",
    tolerance: float = 0.05,
) -> LoadCase:
    """
    Distribute uniform floor pressure across horizontal beam elements at specified floor levels.

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
        Elevations (Y coordinates for 2D frame, Z for 3D frame) of floor beams.
        If None, automatically detects all horizontal beams above foundation (y > 0).
    direction : str
        Direction of gravity: '-y' or '-z'.
    tolerance : float
        Elevation matching tolerance.

    Returns
    -------
    LoadCase
        Configured struct_core.LoadCase with DistributedLoad elements.
    """
    nodes_map = {n.id: n for n in model.nodes}
    element_loads: List[DistributedLoad] = []

    wx, wy = distribute_1d_tributary(pressure, tributary_width, direction=direction)

    for elem in model.elements:
        if getattr(elem, "type", "") != "beam":
            continue

        n_start = nodes_map.get(elem.start_node)
        n_end = nodes_map.get(elem.end_node)
        if not n_start or not n_end:
            continue

        # Check if beam is horizontal (Y coordinates approximately equal in 2D frame)
        is_horizontal = abs(n_start.y - n_end.y) <= tolerance and abs(n_start.x - n_end.x) > tolerance
        if not is_horizontal:
            continue

        beam_elev = (n_start.y + n_end.y) / 2.0

        # If floor elevations specified, only apply to matching elevations
        if story_elevations is not None:
            matches = any(abs(beam_elev - elev) <= tolerance for elev in story_elevations)
            if not matches:
                continue
        else:
            # By default, don't load ground/foundation beams at y <= 0
            if beam_elev <= tolerance:
                continue

        element_loads.append(
            DistributedLoad(
                element_id=elem.id,
                wx=round(wx, 4),
                wy=round(wy, 4),
            )
        )

    return LoadCase(
        id=target_case_id,
        name=case_name or f"Floor Load ({target_case_id})",
        element_loads=element_loads,
    )
