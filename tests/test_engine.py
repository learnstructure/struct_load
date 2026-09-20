"""
Unit tests for struct_load engine subpackage (self-weight and tributary distribution).
"""

import pytest
from struct_core import (
    BeamElement,
    ElasticMaterial,
    Node,
    RectangularSection,
    StructuralModel,
)
from struct_load.engine.self_weight import calculate_self_weight, calculate_section_area
from struct_load.engine.tributary import (
    distribute_1d_tributary,
    distribute_two_way_slab,
    apply_floor_load_to_frame,
)


def _build_portal_frame():
    nodes = [
        Node(id=1, x=0.0, y=0.0),
        Node(id=2, x=0.0, y=3.5),
        Node(id=3, x=6.0, y=3.5),
        Node(id=4, x=6.0, y=0.0),
    ]
    mat = ElasticMaterial(id="Conc", E=2.5e7, rho=2400.0)  # 2400 kg/m3
    sec_beam = RectangularSection(id="B1", b=0.30, h=0.50)  # Area = 0.15 m2
    sec_col = RectangularSection(id="C1", b=0.40, h=0.40)   # Area = 0.16 m2

    elements = [
        BeamElement(id=1, start_node=1, end_node=2, material_id="Conc", section_id="C1"),
        BeamElement(id=2, start_node=2, end_node=3, material_id="Conc", section_id="B1"),
        BeamElement(id=3, start_node=4, end_node=3, material_id="Conc", section_id="C1"),
    ]
    return StructuralModel(nodes=nodes, materials=[mat], sections=[sec_beam, sec_col], elements=elements)


def test_section_area():
    rec = RectangularSection(id="R1", b=0.30, h=0.50)
    assert calculate_section_area(rec) == pytest.approx(0.15)


def test_self_weight():
    model = _build_portal_frame()
    sw_case = calculate_self_weight(model, load_case_id="SW", unit_system="si")
    assert sw_case.id == "SW"
    assert len(sw_case.element_loads) == 3

    # For Beam 2 (b=0.30, h=0.50, rho=2400):
    # w = (2400 * 0.15 * 9.80665) / 1000 = ~3.53 kN/m
    beam_load = next(l for l in sw_case.element_loads if l.element_id == 2)
    assert beam_load.wy == pytest.approx(-3.53, rel=1e-2)


def test_tributary_1d_and_two_way():
    wx, wy = distribute_1d_tributary(pressure=2.5, tributary_width=4.0)
    assert wx == 0.0
    assert wy == -10.0  # 2.5 * 4 = 10 kN/m downward

    w_peak, w_short, w_long = distribute_two_way_slab(pressure=3.0, short_span=4.0, long_span=6.0)
    assert w_peak == 6.0  # 3.0 * (4 / 2) = 6.0
    assert w_short == 4.0 # 3.0 * (4 / 3) = 4.0
    assert w_long > w_short


def test_apply_floor_load_to_frame():
    model = _build_portal_frame()
    # Apply 2.0 kPa floor load with 5 m tributary width -> 10 kN/m on beam 2
    lc = apply_floor_load_to_frame(model, pressure=2.0, tributary_width=5.0, target_case_id="LL")
    assert lc.id == "LL"
    assert len(lc.element_loads) == 1  # Only the horizontal beam 2
    assert lc.element_loads[0].element_id == 2
    assert lc.element_loads[0].wy == -10.0
