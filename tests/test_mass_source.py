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
    sec = RectangularSection(id="COL", material_id="CONC", b=0.4, h=0.4)

    c1 = BeamElement(id="C1", start_node="N1", end_node="N3", section_id="COL")
    c2 = BeamElement(id="C2", start_node="N2", end_node="N4", section_id="COL")
    beam = BeamElement(id="B1", start_node="N3", end_node="N4", section_id="COL")

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
        units="kN, m",
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


def test_asce7_mass_source_storage_vs_general_live():
    """
    Test ASCE 7-22 Section 12.7.2 requirements:
    - Dead load: 100% (factor = 1.0)
    - Storage live load: 25% (factor = 0.25, Item 1)
    - General live load: 0% (factor = 0.0, excluded)
    - Partitions: 100% (factor = 1.0, Item 2)
    - Permanent equipment: 100% (factor = 1.0, Item 3)
    """
    from struct_load import create_asce7_mass_source

    n1 = Node(id="N1", x=0.0, y=0.0)
    n2 = Node(id="N2", x=6.0, y=0.0)
    mat = ElasticMaterial(id="MAT", E=25e6, rho=2400.0)
    sec = RectangularSection(id="SEC", material_id="MAT", b=0.3, h=0.5)
    b1 = BeamElement(id="B1", start_node="N1", end_node="N2", section_id="SEC")

    lc_dead = LoadCase(id="DEAD", element_loads=[DistributedLoad(element_id="B1", wy=-20.0)])
    lc_office = LoadCase(id="LIVE_OFFICE", element_loads=[DistributedLoad(element_id="B1", wy=-10.0)])
    lc_storage = LoadCase(id="STORAGE_LIVE", element_loads=[DistributedLoad(element_id="B1", wy=-30.0)])
    lc_part = LoadCase(id="PARTITION", element_loads=[DistributedLoad(element_id="B1", wy=-5.0)])
    lc_equip = LoadCase(id="EQUIPMENT", element_loads=[DistributedLoad(element_id="B1", wy=-8.0)])

    model = StructuralModel(
        units="kN, m",
        nodes=[n1, n2],
        elements=[b1],
        materials=[mat],
        sections=[sec],
        load_cases=[lc_dead, lc_office, lc_storage, lc_part, lc_equip],
    )

    # Auto-detect mass source without snow
    ms = create_asce7_mass_source(model, include_element_mass=False)

    factors_dict = {f.load_case_id: f.factor for f in ms.factors}

    # Verify factors per ASCE 7-22 Section 12.7.2
    assert factors_dict["DEAD"] == 1.0
    assert factors_dict["STORAGE_LIVE"] == 0.25
    assert factors_dict["PARTITION"] == 1.0
    assert factors_dict["EQUIPMENT"] == 1.0
    # General office live load must NOT be included in seismic weight
    assert "LIVE_OFFICE" not in factors_dict

    # Calculate seismic weight:
    # Dead: 20 * 6 = 120 kN
    # Storage (25%): 0.25 * 30 * 6 = 45 kN
    # Partitions: 5 * 6 = 30 kN
    # Equipment: 8 * 6 = 48 kN
    # Total W = 120 + 45 + 30 + 48 = 243 kN
    res = calculate_mass_source(model, mass_source=ms)
    assert res.total_seismic_weight == pytest.approx(243.0, rel=1e-4)


def test_asce7_mass_source_snow_threshold():
    """
    Test ASCE 7-22 Section 12.7.2 Item 4:
    - Snow load Pf <= 45 psf (2.16 kN/m^2) -> 0% included.
    - Snow load Pf > 45 psf (2.16 kN/m^2) -> 15% of uniform snow load included.
    """
    from struct_load import create_asce7_mass_source

    n1 = Node(id="N1", x=0.0, y=0.0)
    n2 = Node(id="N2", x=6.0, y=0.0)
    mat = ElasticMaterial(id="MAT", E=25e6, rho=2400.0)
    sec = RectangularSection(id="SEC", material_id="MAT", b=0.3, h=0.5)
    b1 = BeamElement(id="B1", start_node="N1", end_node="N2", section_id="SEC")

    lc_dead = LoadCase(id="DEAD", element_loads=[DistributedLoad(element_id="B1", wy=-10.0)])
    lc_snow = LoadCase(id="SNOW", element_loads=[DistributedLoad(element_id="B1", wy=-12.0)])

    model = StructuralModel(
        units="kN, m",nodes=[n1, n2], elements=[b1], materials=[mat], sections=[sec], load_cases=[lc_dead, lc_snow])

    # Case A: Low snow load Pf = 1.5 kPa (<= 2.16 kPa threshold) -> 0% snow
    ms_low_snow = create_asce7_mass_source(
        model,
        flat_roof_snow_pressure=1.5,
        include_element_mass=False,
    )
    factors_low = {f.load_case_id: f.factor for f in ms_low_snow.factors}
    assert "SNOW" not in factors_low

    res_low = calculate_mass_source(model, mass_source=ms_low_snow)
    assert res_low.total_seismic_weight == pytest.approx(60.0, rel=1e-4)  # 10 kN/m * 6 m = 60 kN

    # Case B: Heavy snow load Pf = 2.5 kPa (> 2.16 kPa threshold) -> 15% snow
    ms_heavy_snow = create_asce7_mass_source(
        model,
        flat_roof_snow_pressure=2.5,
        include_element_mass=False,
    )
    factors_heavy = {f.load_case_id: f.factor for f in ms_heavy_snow.factors}
    assert factors_heavy["SNOW"] == 0.15

    # Total W = Dead (60 kN) + 0.15 * Snow (12 * 6 = 72 kN) = 60 + 10.8 = 70.8 kN
    res_heavy = calculate_mass_source(model, mass_source=ms_heavy_snow)
    assert res_heavy.total_seismic_weight == pytest.approx(70.8, rel=1e-4)



def test_mass_source_skips_self_weight_loads_when_element_mass_included():
    """Self-weight loads materialized into a dead case must not be counted a second time as mass."""
    from struct_core import Project, Support
    from struct_load import LoadGenerator

    model = StructuralModel(
        units="kN, m",
        nodes=[Node(id=1, x=0.0, y=0.0), Node(id=2, x=6.0, y=0.0), Node(id=3, x=0.0, y=4.0), Node(id=4, x=6.0, y=4.0)],
        materials=[ElasticMaterial(id="CONC", E=25e6, rho=2400.0)],
        sections=[RectangularSection(id="SEC", material_id="CONC", b=0.4, h=0.4)],
        elements=[
            BeamElement(id="C1", start_node=1, end_node=3, section_id="SEC"),
            BeamElement(id="C2", start_node=2, end_node=4, section_id="SEC"),
            BeamElement(id="B1", start_node=3, end_node=4, section_id="SEC"),
        ],
        supports=[Support.fixed(node_id=1), Support.fixed(node_id=2)],
    )
    LoadGenerator(units="kN, m").apply_gravity_loads(
        Project(model=model), occupancy="office", tributary_width=5.0, superimposed_dead=1.5
    )
    ms = MassSource(id="MS", include_element_mass=True, factors=[MassSourceFactor(load_case_id="Dead", factor=1.0)])
    res = calculate_mass_source(model, mass_source=ms, units="kN, m")

    g = 9.80665
    self_weight = 2400.0 * g / 1000.0 * 0.16 * (4.0 + 4.0 + 6.0)
    sdl = 1.5 * 5.0 * 6.0
    assert res.total_seismic_weight == pytest.approx(self_weight + sdl, rel=1e-6)
