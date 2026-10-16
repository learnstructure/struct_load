"""
Tests for panel and AreaLoad tributary distribution in struct_load.
"""

import pytest
from struct_core import AreaLoad, LoadCase
from struct_load.engine.tributary import (
    apply_floor_load_to_frame,
    distribute_area_loads_to_beams,
    distribute_tributary_load,
    distribute_two_way_slab,
)
from helpers import build_3d_frame_model


def test_distribute_tributary_load_directions():
    # 2.0 kPa * 3.0 m = 6.0 kN/m
    wx, wy, wz = distribute_tributary_load(2.0, 3.0, direction="-z")
    assert (wx, wy, wz) == (0.0, 0.0, -6.0)

    wx, wy, wz = distribute_tributary_load(2.0, 3.0, direction="+z")
    assert (wx, wy, wz) == (0.0, 0.0, 6.0)

    wx, wy, wz = distribute_tributary_load(2.0, 3.0, direction="-y")
    assert (wx, wy, wz) == (0.0, -6.0, 0.0)

    wx, wy, wz = distribute_tributary_load(2.0, 3.0, direction="-x")
    assert (wx, wy, wz) == (-6.0, 0.0, 0.0)


def test_distribute_two_way_slab():
    # q = 5.0 kPa, short span S = 4.0 m, long span L = 6.0 m -> m = 1.5
    # w_peak = 5.0 * (4.0 / 2) = 10.0 kN/m
    # Moment: w_eq_short = 5.0 * (4.0 / 3) = 6.667 kN/m
    # w_eq_long = 10.0 * (1 - 1 / (3 * 2.25)) = 10.0 * (1 - 1/6.75) = 8.519 kN/m
    w_peak, w_eq_s, w_eq_l = distribute_two_way_slab(5.0, 4.0, 6.0, design_action="moment")
    assert w_peak == 10.0
    assert w_eq_s == pytest.approx(6.667, abs=1e-3)
    assert w_eq_l == pytest.approx(8.519, abs=1e-3)

    # Shear: w_eq_short = 5.0 * (4.0 / 4) = 5.0 kN/m
    # w_eq_long = 10.0 * (1 - 1 / (2 * 1.5)) = 10.0 * (1 - 1/3) = 6.667 kN/m
    w_peak_v, w_eq_sv, w_eq_lv = distribute_two_way_slab(5.0, 4.0, 6.0, design_action="shear")
    assert w_eq_sv == 5.0
    assert w_eq_lv == pytest.approx(6.667, abs=1e-3)


def test_apply_floor_load_3d_frame():
    model = build_3d_frame_model()
    # Apply 3.0 kPa over 2.5 m tributary width -> 7.5 kN/m along -Z
    case = apply_floor_load_to_frame(
        model=model,
        pressure=3.0,
        tributary_width=2.5,
        target_case_id="FloorLive",
    )
    assert len(case.element_loads) == 8  # 4 beams on story 1 + 4 beams on story 2
    for el in case.element_loads:
        assert el.wx == 0.0
        assert el.wy == 0.0
        assert el.wz == -7.5


def test_distribute_area_loads_one_way():
    model = build_3d_frame_model()
    # Add an AreaLoad on Story 1 (nodes 5, 6, 7, 8: 6m in X, 5m in Y)
    # Spanning in 'x' direction:
    # Beams perpendicular to X (spanning in Y: B2 and B4) receive load:
    # Tributary width = Lx / 2 = 6.0 / 2 = 3.0 m
    # Line load = 4.0 kPa * 3.0 m = 12.0 kN/m
    # Beams B1 and B3 parallel to X receive 0 load
    al = AreaLoad(
        id="SLAB_1",
        pressure=4.0,
        load_type="live",
        node_ids=[5, 6, 7, 8],
        one_way=True,
        span_direction_axis="x",
    )
    lc = LoadCase(id="Live", area_loads=[al])
    model.load_cases.append(lc)

    count = distribute_area_loads_to_beams(model, target_case_id="Live")
    assert count == 2  # B2 and B4 loaded

    loads_by_elem = {l.element_id: l for l in lc.element_loads}
    assert "B2" in loads_by_elem
    assert "B4" in loads_by_elem
    assert loads_by_elem["B2"].wz == -12.0
    assert loads_by_elem["B4"].wz == -12.0


def test_distribute_area_loads_two_way():
    model = build_3d_frame_model()
    # 4.0 kPa on Story 1 (Lx = 6.0m, Ly = 5.0m)
    # Ly is short span (5.0m), Lx is long span (6.0m)
    # Short side beams (B2, B4) span 5m -> receive triangular equivalent load
    # Long side beams (B1, B3) span 6m -> receive trapezoidal equivalent load
    al = AreaLoad(
        id="SLAB_TWO_WAY",
        pressure=4.0,
        load_type="dead",
        node_ids=[5, 6, 7, 8],
        one_way=False,
    )
    lc = LoadCase(id="Dead", area_loads=[al])
    model.load_cases.append(lc)

    count = distribute_area_loads_to_beams(model, target_case_id="Dead")
    assert count == 4  # All 4 bounding beams loaded

    loads_by_elem = {l.element_id: l for l in lc.element_loads}
    # Short beams: w_eq_short = 4.0 * (5.0 / 3) = 6.667 kN/m
    assert abs(loads_by_elem["B2"].wz) == pytest.approx(6.667, abs=1e-3)
    assert abs(loads_by_elem["B4"].wz) == pytest.approx(6.667, abs=1e-3)
    # Long beams: w_eq_long = 4.0 * 2.5 * (1 - 1 / (3 * 1.2^2)) = 10 * (1 - 1/4.32) = 7.685 kN/m
    assert abs(loads_by_elem["B1"].wz) == pytest.approx(7.685, abs=1e-3)
    assert abs(loads_by_elem["B3"].wz) == pytest.approx(7.685, abs=1e-3)
