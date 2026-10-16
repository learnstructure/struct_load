"""
Tests for Mass Source calculation engine in struct_load.
"""

import pytest
from struct_core import (
    BeamElement,
    DistributedLoad,
    ElasticMaterial,
    LoadCase,
    MassSource,
    MassSourceFactor,
    Node,
    PointLoad,
    RectangularSection,
    StructuralModel,
)
from struct_load import calculate_mass_source


@pytest.fixture
def sample_2d_portal_model():
    """Create a 2D portal frame with Dead and Live load cases."""
    n1 = Node(id="N1", x=0.0, y=0.0)
    n2 = Node(id="N2", x=6.0, y=0.0)
    n3 = Node(id="N3", x=0.0, y=4.0)
    n4 = Node(id="N4", x=6.0, y=4.0)

    mat = ElasticMaterial(id="CONC", E=25e6, rho=2400.0)
    sec = RectangularSection(id="COL", b=0.4, h=0.4)

    c1 = BeamElement(id="C1", start_node="N1", end_node="N3", material_id="CONC", section_id="COL")
    c2 = BeamElement(id="C2", start_node="N2", end_node="N4", material_id="CONC", section_id="COL")
    beam = BeamElement(id="B1", start_node="N3", end_node="N4", material_id="CONC", section_id="COL")

    # Dead load case: 20 kN/m along beam
    lc_dead = LoadCase(
        id="DEAD",
        name="Dead Load",
        element_loads=[DistributedLoad(element_id="B1", wy=-20.0)],
        point_loads=[PointLoad(node_id="N3", fy=-10.0), PointLoad(node_id="N4", fy=-10.0)],
    )

    # Live load case: 10 kN/m along beam
    lc_live = LoadCase(
        id="LIVE",
        name="Live Load",
        element_loads=[DistributedLoad(element_id="B1", wy=-10.0)],
        point_loads=[PointLoad(node_id="N3", fy=-5.0), PointLoad(node_id="N4", fy=-5.0)],
    )

    model = StructuralModel(
        nodes=[n1, n2, n3, n4],
        elements=[c1, c2, beam],
        materials=[mat],
        sections=[sec],
        load_cases=[lc_dead, lc_live],
        story_elevations=[0.0, 4.0],
    )
    return model


def test_mass_source_load_combinations(sample_2d_portal_model):
    """
    Test mass source evaluating 1.0 DEAD + 0.25 LIVE.
    Beam length = 6 m.
    Dead load:
        - Beam distributed: 20 kN/m * 6 m = 120 kN -> mass = 120 / 9.80665 = 12.2366 tonne
        - Point loads: 10 kN + 10 kN = 20 kN -> mass = 20 / 9.80665 = 2.0394 tonne
    Live load (scaled by 0.25):
        - Beam distributed: 0.25 * 10 kN/m * 6 m = 15 kN -> mass = 15 / 9.80665 = 1.5296 tonne
        - Point loads: 0.25 * (5 + 5) = 2.5 kN -> mass = 2.5 / 9.80665 = 0.2549 tonne
    Total applied load weight = 120 + 20 + 15 + 2.5 = 157.5 kN
    """
    model = sample_2d_portal_model
    ms = MassSource(
        id="MS_ASCE7",
        include_element_mass=False,  # exclude self-mass to test load conversion directly
        include_added_mass=True,
        factors=[
            MassSourceFactor(load_case_id="DEAD", factor=1.0),
            MassSourceFactor(load_case_id="LIVE", factor=0.25),
        ],
    )

    res = calculate_mass_source(model, mass_source=ms)

    assert res.total_seismic_weight == pytest.approx(157.5, rel=1e-4)
    expected_mass = 157.5 / 9.80665
    assert res.total_mass == pytest.approx(expected_mass, rel=1e-4)

    # Check beam distributed extra mass
    # w_dead = 20 kN/m, w_live_scaled = 2.5 kN/m -> w_tot = 22.5 kN/m
    # mass per length = 22.5 / 9.80665 = 2.2943 tonne/m
    assert res.element_masses["B1"] == pytest.approx(22.5 / 9.80665, rel=1e-4)

    # Check nodal masses at N3 and N4
    # P_tot per node = 10 (dead) + 0.25 * 5 (live) = 11.25 kN
    # m_node = 11.25 / 9.80665 = 1.1471 tonne
    assert res.nodal_masses["N3"][0] == pytest.approx(11.25 / 9.80665, rel=1e-4)
    assert res.nodal_masses["N4"][0] == pytest.approx(11.25 / 9.80665, rel=1e-4)


def test_mass_source_with_element_self_mass(sample_2d_portal_model):
    """
    Test mass source including element self-mass from material density.
    Concrete density = 2400 kg/m^3. Section A = 0.4 * 0.4 = 0.16 m^2.
    Column height = 4 m (two columns = 8 m). Beam span = 6 m. Total length = 14 m.
    Volume = 14 * 0.16 = 2.24 m^3.
    Self mass = 2.24 * 2400 = 5376 kg = 5.376 tonnes.
    Self weight = 5.376 * 9.80665 = 52.72 kN.
    """
    model = sample_2d_portal_model
    ms = MassSource(
        id="MS_SELF",
        include_element_mass=True,
        include_added_mass=True,
        factors=[],  # No loads, pure self-mass
    )

    res = calculate_mass_source(model, mass_source=ms)

    assert res.total_mass == pytest.approx(5.376, rel=1e-3)
    assert res.total_seismic_weight == pytest.approx(5.376 * 9.80665, rel=1e-3)
    # Center of mass X should be 3.0 m (symmetric frame)
    assert res.center_of_mass[0] == pytest.approx(3.0, abs=1e-4)


def test_mass_source_lateral_only(sample_2d_portal_model):
    """Test lateral_only flag suppresses vertical mass."""
    model = sample_2d_portal_model
    ms = MassSource(
        id="MS_LAT",
        include_element_mass=False,
        include_added_mass=True,
        lateral_only=True,
        factors=[MassSourceFactor(load_case_id="DEAD", factor=1.0)],
    )

    res = calculate_mass_source(model, mass_source=ms)

    # In 2D with lateral_only=True, vertical (index 1 / Y) mass should be 0.0
    assert res.nodal_masses["N3"][0] > 0.0
    assert res.nodal_masses["N3"][1] == 0.0
