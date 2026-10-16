"""
Unit tests for LoadGenerator end-to-end load workflow.
"""

import pytest
from struct_core import (
    BeamElement,
    ElasticMaterial,
    Node,
    Project,
    RectangularSection,
    StructuralModel,
    Support,
)
from struct_load.generator import LoadGenerator


def _build_test_frame():
    nodes = [
        Node(id=1, x=0.0, y=0.0),
        Node(id=2, x=0.0, y=3.5),
        Node(id=3, x=6.0, y=3.5),
        Node(id=4, x=6.0, y=0.0),
    ]
    mat = ElasticMaterial(id="Concrete_4ksi", E=2.5e7, rho=2400.0)
    sec_beam = RectangularSection(id="Beam", b=0.30, h=0.50)
    sec_col = RectangularSection(id="Col", b=0.40, h=0.40)
    elements = [
        BeamElement(id=1, start_node=1, end_node=2, material_id="Concrete_4ksi", section_id="Col"),
        BeamElement(id=2, start_node=2, end_node=3, material_id="Concrete_4ksi", section_id="Beam"),
        BeamElement(id=3, start_node=4, end_node=3, material_id="Concrete_4ksi", section_id="Col"),
    ]
    supports = [Support.fixed(node_id=1), Support.fixed(node_id=4)]
    model = StructuralModel(nodes=nodes, materials=[mat], sections=[sec_beam, sec_col], elements=elements, supports=supports)
    return Project(model=model)


def test_load_generator_gravity_and_combinations():
    project = _build_test_frame()
    gen = LoadGenerator(units="kN, m")

    # 1. Apply gravity loads (Office occupancy + 5m tributary + self weight)
    dead_case, live_case = gen.apply_gravity_loads(
        project,
        occupancy="office",
        tributary_width=5.0,
        superimposed_dead=1.5,
        include_self_weight=True,
    )
    assert dead_case.id == "Dead"
    assert live_case.id == "Live"
    assert len(project.model.load_cases) == 2

    # Verify beam 2 has combined dead load (self weight + SDL)
    b2_dead = next(l for l in dead_case.element_loads if l.element_id == 2)
    # SDL = 1.5 kPa * 5m = 7.5 kN/m; SW = 2400 * 0.15 * 9.81 / 1000 = ~3.53 kN/m -> total ~ 11.03 kN/m
    assert b2_dead.wy == pytest.approx(-11.03, rel=0.05)

    # Verify beam 2 has office live load (2.40 + 0.72 = 3.12 kPa * 5m = 15.6 kN/m)
    b2_live = next(l for l in live_case.element_loads if l.element_id == 2)
    assert b2_live.wy == pytest.approx(-15.6, rel=0.05)

    # 2. Apply lateral seismic load
    eq_case, elf_res = gen.apply_lateral_seismic(
        project,
        sds=0.75,
        sd1=0.40,
        r_factor=8.0,
    )
    assert eq_case.id == "EQ_X"
    assert len(eq_case.point_loads) == 1
    assert eq_case.point_loads[0].fx > 0.0

    # 3. Generate LRFD load combinations
    combs = gen.generate_combinations(project, design_method="LRFD")
    assert len(combs) >= 3
    comb_ids = [c.id for c in combs]
    assert "COMB_1.4D" in comb_ids
    assert "COMB_1.2D+1.6L" in comb_ids
    assert "COMB_1.2D+1.0L+1.0E" in comb_ids
    assert len(project.model.load_combinations) == len(combs)
