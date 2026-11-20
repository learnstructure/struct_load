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
from struct_load.codes.asce7_seismic import SeismicParameters
from struct_load.generator import LoadGenerator


def _build_test_frame():
    nodes = [
        Node(id=1, x=0.0, y=0.0),
        Node(id=2, x=0.0, y=3.5),
        Node(id=3, x=6.0, y=3.5),
        Node(id=4, x=6.0, y=0.0),
    ]
    mat = ElasticMaterial(id="Concrete_4ksi", E=2.5e7, rho=2400.0)
    sec_beam = RectangularSection(id="Beam", material_id="Concrete_4ksi", b=0.30, h=0.50)
    sec_col = RectangularSection(id="Col", material_id="Concrete_4ksi", b=0.40, h=0.40)
    elements = [
        BeamElement(id=1, start_node=1, end_node=2, section_id="Col"),
        BeamElement(id=2, start_node=2, end_node=3, section_id="Beam"),
        BeamElement(id=3, start_node=4, end_node=3, section_id="Col"),
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
    # Office floors carry a partition allowance: the partition weight case feeds the seismic weight
    assert [lc.id for lc in project.model.load_cases] == ["Dead", "Live", "PartitionWeight"]

    # Beam 2 carries SDL and self-weight as separate entries; only the self-weight entry is flagged
    b2_dead = [l for l in dead_case.element_loads if l.element_id == 2]
    # SDL = 1.5 kPa * 5m = 7.5 kN/m; SW = 2400 * 0.15 * 9.81 / 1000 = ~3.53 kN/m -> total ~ 11.03 kN/m
    assert sum(l.wy for l in b2_dead) == pytest.approx(-11.03, rel=0.05)
    assert [l.is_self_weight for l in b2_dead].count(True) == 1
    assert sum(l.wy for l in b2_dead if l.is_self_weight) == pytest.approx(-3.53, rel=0.05)

    # Verify beam 2 has office live load (2.40 + 0.72 = 3.12 kPa * 5m = 15.6 kN/m)
    b2_live = next(l for l in live_case.element_loads if l.element_id == 2)
    assert b2_live.wy == pytest.approx(-15.6, rel=0.05)

    # 2. Apply lateral seismic load
    params = SeismicParameters(sds=0.75, sd1=0.40, site_class="D", risk_category="II", r_factor=8.0)
    eq_case, elf_res = gen.apply_lateral_seismic(
        project,
        params=params,
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


def test_generate_combinations_factors_every_dead_case():
    """Models with separate self-weight and superimposed dead cases get the dead factor on both."""
    from struct_core import LoadCase

    project = _build_test_frame()
    project.model.load_cases = [
        LoadCase(id="DEAD", include_self_weight=True),
        LoadCase(id="SuperDead"),
        LoadCase(id="LIVE"),
    ]
    combs = LoadGenerator(units="kN, m").generate_combinations(project)
    by_id = {c.id: {f.load_case_id: f.factor for f in c.factors} for c in combs}
    assert by_id["COMB_1.4D"] == {"DEAD": 1.4, "SuperDead": 1.4}
    assert by_id["COMB_1.2D+1.6L"] == {"DEAD": 1.2, "SuperDead": 1.2, "LIVE": 1.6}


def test_generate_combinations_reports_omitted_load_cases():
    """Load cases left out of generated combinations (other, temperature, soil, fluid, partition, or duplicates) are
    reported in LoadGenerator.messages with their ID, type, and reason."""
    from struct_core import LoadCase

    project = _build_test_frame()
    project.model.load_cases = [
        LoadCase(id="DEAD", load_type="dead", include_self_weight=True),
        LoadCase(id="LIVE", load_type="live"),
        LoadCase(id="PAT1", load_type="other"),
        LoadCase(id="TEMP", load_type="temperature"),
        LoadCase(id="SOIL", load_type="soil"),
        LoadCase(id="FLUID", load_type="fluid"),
        LoadCase(id="PartitionWeight", load_type="partition"),
        LoadCase(id="WX1", load_type="wind", direction="X"),
        LoadCase(id="WX2", load_type="wind", direction="X"),
    ]
    gen = LoadGenerator(units="kN, m")
    combs = gen.generate_combinations(project)
    assert len(combs) > 0
    assert len(gen.messages) == 6
    text = "\n".join(gen.messages)
    for case_id, kind in (
        ("PAT1", "other"),
        ("TEMP", "temperature"),
        ("SOIL", "soil"),
        ("FLUID", "fluid"),
        ("PartitionWeight", "partition"),
        ("WX2", "wind"),
    ):
        assert any(case_id in m and kind in m for m in gen.messages), f"Missing message for {case_id} ({kind}): {text}"


def test_calculate_model_seismic_weight_includes_beam_self_weight_with_superdead():
    """Superimposed dead loads (is_self_weight=False) on beams must not suppress beam self-weight."""
    from struct_core import DistributedLoad, LoadCase, Story
    from struct_load import apply_self_weight, calculate_mass_source, create_asce7_mass_source

    model = StructuralModel(
        dimension="3D",
        vertical_axis="Z",
        units="kN, m",
        nodes=[
            Node(id="1", x=0.0, y=0.0, z=0.0),
            Node(id="2", x=6.0, y=0.0, z=0.0),
            Node(id="3", x=0.0, y=0.0, z=3.5),
            Node(id="4", x=6.0, y=0.0, z=3.5),
        ],
        supports=[Support.fixed_3d("1"), Support.fixed_3d("2")],
        materials=[ElasticMaterial(id="C28", E=25e6, nu=0.2, unit_weight=24.0)],
        sections=[
            RectangularSection(id="COL", material_id="C28", b=0.4, h=0.4),
            RectangularSection(id="BM", material_id="C28", b=0.3, h=0.5),
        ],
        elements=[
            BeamElement(id="C1", start_node="1", end_node="3", section_id="COL"),
            BeamElement(id="C2", start_node="2", end_node="4", section_id="COL"),
            BeamElement(id="B1", start_node="3", end_node="4", section_id="BM"),
        ],
        base_elevation=0.0,
        stories=[Story(name="Roof", elevation=3.5)],
        load_cases=[
            LoadCase(id="DEAD", load_type="dead", include_self_weight=True),
            LoadCase(
                id="SuperDead",
                load_type="super_dead",
                include_self_weight=False,
                element_loads=[DistributedLoad(element_id="B1", wz=-10.0, is_self_weight=False)],
            ),
        ],
    )
    gen = LoadGenerator(units="kN, m")
    w_stories = gen.calculate_model_seismic_weight(model, [3.5], base_elevation=0.0)
    # 2 columns above mid-height: 2 * 24.0 * 0.16 * (3.5 / 2) = 13.44 kN
    # 1 beam self-weight: 24.0 * 0.15 * 6.0 = 21.60 kN
    # SuperDead on beam: 10.0 * 6.0 = 60.00 kN -> total = 95.04 kN
    expected = 13.44 + 21.60 + 60.00
    assert w_stories == [pytest.approx(expected, rel=1e-6)]

    ms = create_asce7_mass_source(model, include_element_mass=True)
    ms_res = calculate_mass_source(model, mass_source=ms, units="kN, m", base_elevation=0.0)
    assert ms_res.story_weights[3.5] == pytest.approx(expected, rel=1e-6)
    assert ms_res.total_seismic_weight == pytest.approx(expected + 13.44, rel=1e-6)

    # Also verify materialized self-weight (is_self_weight=True) is not double-counted
    model_mat = model.model_copy(deep=True)
    apply_self_weight(model_mat, target_case_id="DEAD", units="kN, m")
    w_mat = gen.calculate_model_seismic_weight(model_mat, [3.5], base_elevation=0.0)
    assert w_mat == [pytest.approx(expected, rel=1e-6)]

