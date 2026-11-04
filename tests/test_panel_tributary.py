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
    # F2: 3D one-way distribution requires one_way_span ('x' or 'y')
    # Spanning in Y: beams along X (B1, B3 on each story) carry load:
    # Trib width = 5.0 / 2 = 2.5 m -> wz = -7.5 kN/m. Beams along Y carry 0.
    case = apply_floor_load_to_frame(
        model=model,
        pressure=3.0,
        one_way_span="y",
        target_case_id="FloorLive",
    )
    assert len(case.element_loads) == 4  # 2 beams on story 1 + 2 beams on story 2
    for el in case.element_loads:
        assert el.wx == 0.0
        assert el.wy == 0.0
        assert el.wz == -7.5

    # 3D with tributary_width raises ValueError
    with pytest.raises(ValueError, match="tributary_width is for 2D frames only"):
        apply_floor_load_to_frame(model=model, pressure=3.0, tributary_width=2.5)


def test_3d_floor_load_equilibrium_one_way_and_two_way():
    """F2: One-way in X, one-way in Y, and two-way in 3D all sum to q * floor area within 0.1%."""
    from struct_core.structural_model import Node, BeamElement, RectangularSection, ElasticMaterial, StructuralModel

    # 2x2 bays, 6m x 5m, single story at z = 3.5m (total floor area = 12m x 10m = 120 m^2)
    nodes = []
    nid = 1
    for z in (0.0, 3.5):
        for y in (0.0, 5.0, 10.0):
            for x in (0.0, 6.0, 12.0):
                nodes.append(Node(id=nid, x=x, y=y, z=z))
                nid += 1
    # Beams at z = 3.5 (nodes 10 through 18)
    elements = []
    # Columns
    for i in range(1, 10):
        elements.append(BeamElement(id=f"C{i}", start_node=i, end_node=i+9, section_id="SEC"))
    # Beams along X at y=0, 5, 10
    eid = 1
    nodes_z35 = { (round(n.x, 1), round(n.y, 1)): n.id for n in nodes if n.z == 3.5 }
    for y in (0.0, 5.0, 10.0):
        for x_start, x_end in ((0.0, 6.0), (6.0, 12.0)):
            elements.append(BeamElement(id=f"BX{eid}", start_node=nodes_z35[(x_start, y)],
                                        end_node=nodes_z35[(x_end, y)], section_id="SEC"))
            eid += 1
    # Beams along Y at x=0, 6, 12
    eid = 1
    for x in (0.0, 6.0, 12.0):
        for y_start, y_end in ((0.0, 5.0), (5.0, 10.0)):
            elements.append(BeamElement(id=f"BY{eid}", start_node=nodes_z35[(x, y_start)],
                                        end_node=nodes_z35[(x, y_end)], section_id="SEC"))
            eid += 1

    model = StructuralModel(dimension="3D", nodes=nodes, elements=elements)
    nodes_map = {n.id: n for n in nodes}
    def beam_len(elem):
        ns, ne = nodes_map[elem.start_node], nodes_map[elem.end_node]
        return ((ne.x - ns.x)**2 + (ne.y - ns.y)**2 + (ne.z - ns.z)**2)**0.5

    q = 3.12  # kPa, floor area = 120 m^2 -> total load = 374.4 kN
    total_expected = 120.0 * q

    # 1. One-way span along Y
    case_y = apply_floor_load_to_frame(model, pressure=q, one_way_span="y")
    elem_dict = {e.id: e for e in elements}
    tot_y = sum(abs(l.wz) * beam_len(elem_dict[l.element_id]) for l in case_y.element_loads)
    assert tot_y == pytest.approx(total_expected, rel=1e-3)

    # 2. One-way span along X
    case_x = apply_floor_load_to_frame(model, pressure=q, one_way_span="x")
    tot_x = sum(abs(l.wz) * beam_len(elem_dict[l.element_id]) for l in case_x.element_loads)
    assert tot_x == pytest.approx(total_expected, rel=1e-3)

    # 3. Two-way
    case_2w = apply_floor_load_to_frame(model, pressure=q, two_way=True)
    tot_2w = sum(abs(l.wz) * beam_len(elem_dict[l.element_id]) for l in case_2w.element_loads)
    assert tot_2w == pytest.approx(total_expected, rel=1e-3)


def test_3d_two_way_empty_panels_raises():
    """F3: When two_way is requested in 3D but no closed floor panels exist, raise ValueError."""
    from struct_core.structural_model import Node, BeamElement, StructuralModel

    # Single beam at z = 3.5 (no closed panels)
    nodes = [Node(id=1, x=0.0, y=0.0, z=3.5), Node(id=2, x=6.0, y=0.0, z=3.5)]
    elements = [BeamElement(id="B1", start_node=1, end_node=2, section_id="SEC")]
    model = StructuralModel(dimension="3D", nodes=nodes, elements=elements)

    with pytest.raises(ValueError, match="No closed floor panels found"):
        apply_floor_load_to_frame(model, pressure=3.0, two_way=True)


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


def test_h2_open_floor_panel_detection_and_allow_partial():
    """H2: An open panel at a floor level raises ValueError unless allow_partial_floor=True."""
    from struct_core.structural_model import Node, BeamElement, StructuralModel

    # 2 bays along X (0..6, 6..12), 1 bay along Y (0..5), single story at z = 3.5
    nodes = []
    nid = 1
    for z in (0.0, 3.5):
        for y in (0.0, 5.0):
            for x in (0.0, 6.0, 12.0):
                nodes.append(Node(id=nid, x=x, y=y, z=z))
                nid += 1
    # Nodes at z=3.5 have IDs 7..12:
    # 7: (0,0), 8: (6,0), 9: (12,0)
    # 10: (0,5), 11: (6,5), 12: (12,5)
    elements = []
    # Columns
    for i in range(1, 7):
        elements.append(BeamElement(id=f'C{i}', start_node=i, end_node=i+6, section_id='SEC'))
    # Beams:
    # Bay 1 (x: 0..6, y: 0..5):
    elements.append(BeamElement(id='B_bot1', start_node=7, end_node=8, section_id='SEC'))   # y=0, x: 0..6
    elements.append(BeamElement(id='B_top1', start_node=10, end_node=11, section_id='SEC')) # y=5, x: 0..6
    elements.append(BeamElement(id='B_left', start_node=7, end_node=10, section_id='SEC'))  # x=0, y: 0..5
    elements.append(BeamElement(id='B_mid', start_node=8, end_node=11, section_id='SEC'))   # x=6, y: 0..5
    # Bay 2 (x: 6..12, y: 0..5):
    elements.append(BeamElement(id='B_bot2', start_node=8, end_node=9, section_id='SEC'))   # y=0, x: 6..12
    # B_top2 (y=5, x: 6..12 between nodes 11 and 12) is REMOVED!
    elements.append(BeamElement(id='B_right', start_node=9, end_node=12, section_id='SEC')) # x=12, y: 0..5

    model = StructuralModel(dimension='3D', nodes=nodes, elements=elements)
    q = 3.12  # kPa (office live load)

    # 1. Without allow_partial_floor, it must raise ValueError naming the missing 30 m^2
    with pytest.raises(ValueError, match='loaded panel area \(30.00\) is less than floor outline area \(60.00\) by 30.00'):
        apply_floor_load_to_frame(model, pressure=q, two_way=True, allow_partial_floor=False)

    # 2. With allow_partial_floor=True, it succeeds, loads 93.6 kN on beams, and reports the areas
    case = apply_floor_load_to_frame(model, pressure=q, two_way=True, allow_partial_floor=True)
    nodes_map = {n.id: n for n in nodes}
    def blen(elem):
        ns, ne = nodes_map[elem.start_node], nodes_map[elem.end_node]
        return ((ne.x - ns.x)**2 + (ne.y - ns.y)**2)**0.5
    elem_dict = {e.id: e for e in elements}
    total_load = sum(abs(l.wz) * blen(elem_dict[l.element_id]) for l in case.element_loads)
    assert total_load == pytest.approx(93.6, rel=1e-3)
    assert case.loaded_floor_area == {3.5: 30.0}
    # Beams B_bot2 and B_right border no closed panel
    assert set(case.unloaded_beams) == {'B_bot2', 'B_right'}
