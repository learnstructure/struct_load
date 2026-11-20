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
    Story,
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
        base_elevation=0.0,
        stories=[Story(name="Story1", elevation=4.0)],
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

    # Member distributed loads are lumped to end nodes via consistent static end reactions
    # (for symmetric uniform loads: 50% to N3, 50% to N4).
    # w_tot = 22.5 kN/m * 6 m = 135 kN -> 67.5 kN at N3 and 67.5 kN at N4.
    # Point loads add 10 + 0.25 * 5 = 11.25 kN at each node -> 78.75 kN / g per node.
    assert res.element_masses["B1"] == pytest.approx(0.0, abs=1e-12)
    assert res.nodal_masses["N3"][0] == pytest.approx(78.75 / 9.80665, rel=1e-4)
    assert res.nodal_masses["N4"][0] == pytest.approx(78.75 / 9.80665, rel=1e-4)


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


def test_element_mass_and_weight_modifiers_act_as_in_sap2000():
    """The element's mass modifier scales its self-mass, the weight modifier its self-weight, and the A modifier
    neither (SAP2000 27.1, measured in the struct_link tests). Regression: all three were ignored."""
    from struct_core import Project, Support
    from struct_load import calculate_self_weight, element_mass_per_length, element_weight_per_length

    def model(mods):
        return StructuralModel(
            nodes=[Node(id="A", x=0.0, y=0.0, z=0.0), Node(id="B", x=4.0, y=0.0, z=0.0)],
            materials=[ElasticMaterial(id="C", E=25e6, unit_weight=24.0)],
            sections=[RectangularSection(id="R", material_id="C", b=0.3, h=0.5)],
            elements=[BeamElement(id="E", start_node="A", end_node="B", section_id="R", stiffness_modifiers=mods)],
            supports=[Support.fixed_3d("A")],
            units="kN, m",
        )

    def weight(mods):
        return -calculate_self_weight(model(mods), units="kN, m").element_loads[0].wz

    def mass(mods):
        return calculate_mass_source(model(mods), units="kN, m").total_mass

    assert weight(None) == pytest.approx(24.0 * 0.15)
    assert weight({"weight": 0.5}) == pytest.approx(0.5 * weight(None))
    assert weight({"A": 0.5, "mass": 0.5}) == pytest.approx(weight(None))
    assert mass({"mass": 0.5}) == pytest.approx(0.5 * mass(None))
    assert mass({"A": 0.5, "weight": 0.5}) == pytest.approx(mass(None))
    m = model({"mass": 0.25, "weight": 2.0})
    elem, sec, mat = m.elements[0], m.sections[0], m.materials[0]
    from struct_core import parse_units
    units = parse_units("kN, m")
    assert element_weight_per_length(elem, sec, mat, units) == pytest.approx(2.0 * 3.6)
    assert element_mass_per_length(elem, sec, mat, units, 9.80665) == pytest.approx(0.25 * 3.6 / 9.80665)


def test_nodal_added_mass_converted_to_consistent_units():
    """Node.mass is stored in units.mass (e.g. kg in 'kN, m'); calculate_mass_source outputs consistent mass (tonnes)."""
    from struct_core import Support

    model = StructuralModel(
        units="kN, m",
        nodes=[Node(id="1", x=0.0, y=0.0, z=0.0), Node(id="2", x=4.0, y=0.0, z=0.0, mass=10_000.0)],
        materials=[ElasticMaterial(id="M", E=30e6, rho=0.0)],
        sections=[RectangularSection(id="S", material_id="M", b=0.3, h=0.3)],
        elements=[BeamElement(id="E1", start_node="1", end_node="2", section_id="S")],
        supports=[Support.fixed_3d("1")],
    )
    ms = MassSource(id="MS", include_element_mass=False, include_added_mass=True, factors=[])
    res = calculate_mass_source(model, mass_source=ms, units="kN, m")
    assert res.nodal_masses["2"] == pytest.approx((10.0, 10.0, 10.0), rel=1e-9)
    assert res.total_mass == pytest.approx(10.0, rel=1e-9)
    assert res.total_seismic_weight == pytest.approx(10.0 * 9.80665, rel=1e-9)


def test_asymmetric_member_loads_lumped_by_consistent_end_reactions():
    """Member loads in a MassSource lump to end nodes using consistent static end reactions (matching SAP2000):
    fixed-fixed Euler-Bernoulli (13/16 : 3/16), fixed-fixed Timoshenko, pinned-pinned lever rule (3/4 : 1/4),
    and off-center element point loads."""
    from struct_core import ConcreteMaterial, ElementPointLoad, Support

    g = 9.80665

    def _run(shear: bool, rel_i=None, rel_j=None, dl=None, epl=None):
        model = StructuralModel(
            dimension="3D",
            vertical_axis="Z",
            units="kN, m",
            nodes=[Node(id="1", x=0.0, y=0.0, z=3.5), Node(id="2", x=6.0, y=0.0, z=3.5)],
            supports=[Support.fixed_3d("1"), Support.fixed_3d("2")],
            materials=[ConcreteMaterial(id="C30", E=25.0e6, nu=0.2, fc=30.0e3, unit_weight=24.0)],
            sections=[RectangularSection(id="S1", material_id="C30", b=0.4, h=0.4)],
            elements=[
                BeamElement(
                    id="B1",
                    start_node="1",
                    end_node="2",
                    section_id="S1",
                    include_shear_deformation=shear,
                    release_i=rel_i or [False] * 6,
                    release_j=rel_j or [False] * 6,
                )
            ],
            load_cases=[LoadCase(id="SDL", element_loads=dl or [], element_point_loads=epl or [])],
        )
        ms = MassSource(
            id="MS",
            include_element_mass=False,
            include_added_mass=False,
            factors=[MassSourceFactor(load_case_id="SDL", factor=1.0)],
        )
        return calculate_mass_source(model, mass_source=ms, units="kN, m")

    # 1. Fixed-fixed Euler-Bernoulli partial load on [0, 0.5]: 13/16 at node 1, 3/16 at node 2
    r_eb = _run(False, dl=[DistributedLoad(element_id="B1", wz=-30.0, start=0.0, end=0.5)])
    m_tot = 90.0 / g
    assert r_eb.nodal_masses["1"][0] == pytest.approx((13.0 / 16.0) * m_tot, rel=1e-12)
    assert r_eb.nodal_masses["2"][0] == pytest.approx((3.0 / 16.0) * m_tot, rel=1e-12)

    # 2. Fixed-fixed Timoshenko (phi = 0.0128) partial load on [0, 0.5]
    phi = 0.0128
    r_tim = _run(True, dl=[DistributedLoad(element_id="B1", wz=-30.0, start=0.0, end=0.5)])
    f1_tim = (13.0 + 12.0 * phi) / (16.0 * (1.0 + phi))
    assert r_tim.nodal_masses["1"][0] == pytest.approx(f1_tim * m_tot, rel=1e-12)
    assert r_tim.nodal_masses["2"][0] == pytest.approx((1.0 - f1_tim) * m_tot, rel=1e-12)

    # 3. Pinned-pinned (M3 released at both ends): exact simple-beam lever rule 0.75 / 0.25
    rel_pin = [False, False, False, False, True, True]
    r_pin = _run(True, rel_i=rel_pin, rel_j=rel_pin, dl=[DistributedLoad(element_id="B1", wz=-30.0, start=0.0, end=0.5)])
    assert r_pin.nodal_masses["1"][0] == pytest.approx(0.75 * m_tot, rel=1e-12)
    assert r_pin.nodal_masses["2"][0] == pytest.approx(0.25 * m_tot, rel=1e-12)

    # 4. Off-center ElementPointLoad at s=0.25 (Timoshenko fixed-fixed)
    r_pt = _run(True, epl=[ElementPointLoad(element_id="B1", position=0.25, fz=-90.0)])
    f1_pt = (0.75 ** 2 * 1.5 + 0.75 * phi) / (1.0 + phi)
    assert r_pt.nodal_masses["1"][0] == pytest.approx(f1_pt * m_tot, rel=1e-12)
    assert r_pt.nodal_masses["2"][0] == pytest.approx((1.0 - f1_pt) * m_tot, rel=1e-12)


def test_explicit_zero_density_material_allowed_while_missing_density_refused():
    """Materials with explicit unit_weight=0.0 and rho=0.0 represent massless elements (with lumped joint masses)
    and return zero self-mass/self-weight; materials with unit_weight=None and rho=0.0 are still refused."""
    from struct_core import Support
    from struct_load.engine.self_weight import calculate_self_weight, material_weight_density
    from struct_core import parse_units

    units = parse_units("kN, m")
    mat_zero = ElasticMaterial(id="MASSLESS", E=25e6, unit_weight=0.0, rho=0.0)
    assert material_weight_density(mat_zero, units) == 0.0

    mat_missing = ElasticMaterial(id="UNSPEC", E=25e6)  # unit_weight=None, rho=0.0 default
    with pytest.raises(ValueError, match="has no density"):
        material_weight_density(mat_missing, units)

    model = StructuralModel(
        units="kN, m",
        nodes=[Node(id="1", x=0.0, y=0.0, z=0.0), Node(id="2", x=4.0, y=0.0, z=0.0, mass=5000.0)],
        materials=[mat_zero],
        sections=[RectangularSection(id="S", material_id="MASSLESS", b=0.3, h=0.3)],
        elements=[BeamElement(id="E1", start_node="1", end_node="2", section_id="S")],
        supports=[Support.fixed_3d("1")],
    )
    sw = calculate_self_weight(model, units="kN, m")
    assert sw.element_loads == []
    ms = MassSource(id="MS", include_element_mass=True, include_added_mass=True, factors=[])
    res = calculate_mass_source(model, mass_source=ms, units="kN, m")
    assert res.total_mass == pytest.approx(5.0, rel=1e-12)


def test_base_mass_excluded_consistently_with_or_without_explicit_base_elevation():
    """Mass at the restrained base is excluded identically whether base_elevation is explicit or inferred
    from lowest supports, and identically across calculate_mass_source and calculate_model_seismic_weight."""
    from struct_core import Support
    from struct_load import LoadGenerator, SeismicParameters

    def _make_portal(base_elev):
        return StructuralModel(
            units="kN, m",
            nodes=[
                Node(id="N1", x=0.0, y=0.0),
                Node(id="N2", x=6.0, y=0.0),
                Node(id="N3", x=0.0, y=4.0),
                Node(id="N4", x=6.0, y=4.0),
            ],
            supports=[Support.fixed("N1"), Support.fixed("N2")],
            materials=[ElasticMaterial(id="CONC", E=25e6, unit_weight=24.0)],
            sections=[RectangularSection(id="S", material_id="CONC", b=0.4, h=0.4)],
            elements=[
                BeamElement(id="C1", start_node="N1", end_node="N3", section_id="S"),
                BeamElement(id="C2", start_node="N2", end_node="N4", section_id="S"),
                BeamElement(id="B1", start_node="N3", end_node="N4", section_id="S"),
            ],
            load_cases=[
                LoadCase(id="PART", load_type="partition", element_loads=[DistributedLoad(element_id="B1", wy=-20.0)]),
            ],
            base_elevation=base_elev,
            stories=[Story(name="Story1", elevation=4.0)],
        )

    # Beam weight = 6*0.16*24 = 23.04; each column = 4*0.16*24 = 15.36; beam partition = 120.0.
    # Above base (Story1): partition (120) + beam self-weight (23.04) + top half of 2 columns (15.36) = 158.40 kN.
    # Base (excluded): bottom half of 2 columns = 15.36 kN.
    expected_story1_w = 120.0 + 6.0 * 0.16 * 24.0 + 4.0 * 0.16 * 24.0  # 158.40 kN
    ms = MassSource(
        id="MS1", include_element_mass=True, include_added_mass=False,
        factors=[MassSourceFactor(load_case_id="PART", factor=1.0)],
    )
    m_explicit = _make_portal(0.0)
    m_inferred = _make_portal(None)

    res_exp = calculate_mass_source(m_explicit, mass_source=ms)
    res_inf = calculate_mass_source(m_inferred, mass_source=ms)
    assert res_exp.story_weights[4.0] == pytest.approx(expected_story1_w, rel=1e-6)
    assert res_inf.story_weights[4.0] == pytest.approx(expected_story1_w, rel=1e-6)
    assert res_inf.story_weights[0.0] == pytest.approx(15.36, rel=1e-6)
    assert any("lowest support" in msg for msg in res_inf.messages)

    params = SeismicParameters(sds=1.0, sd1=0.4, site_class="D", risk_category="II", r_factor=8.0)
    gen_exp_ms = LoadGenerator(units="kN, m")
    _, elf_exp_ms = gen_exp_ms.apply_lateral_seismic(m_explicit, params=params, mass_source=ms)
    gen_inf_ms = LoadGenerator(units="kN, m")
    _, elf_inf_ms = gen_inf_ms.apply_lateral_seismic(m_inferred, params=params, mass_source=ms)
    gen_exp_w = LoadGenerator(units="kN, m")
    _, elf_exp_w = gen_exp_w.apply_lateral_seismic(_make_portal(0.0), params=params)
    gen_inf_w = LoadGenerator(units="kN, m")
    _, elf_inf_w = gen_inf_w.apply_lateral_seismic(_make_portal(None), params=params)

    for elf in (elf_exp_ms, elf_inf_ms, elf_exp_w, elf_inf_w):
        assert elf.base_shear == pytest.approx(0.125 * expected_story1_w, rel=1e-4)
    assert any("lowest support" in msg for msg in gen_inf_ms.messages)
    assert any("lowest support" in msg for msg in gen_inf_w.messages)



