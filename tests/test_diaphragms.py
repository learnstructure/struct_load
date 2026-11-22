"""Verification tests for Milestone 3: struct_load mass source, seismic weight, and ELF with rigid diaphragms."""

import numpy as np
import pytest

from struct_core import (
    BeamElement,
    ConcreteMaterial,
    Diaphragm,
    DistributedLoad,
    LoadCase,
    MassSource,
    MassSourceFactor,
    Node,
    PointLoad,
    Project,
    RectangularSection,
    Story,
    StructuralModel,
    Support,
)
from struct_load import (
    DiaphragmMassSummary,
    LoadGenerator,
    SeismicParameters,
    calculate_mass_source,
)
from fem3d.adapters import model_from_core


def _asymmetric_2story_project(*, with_diaphragms: bool = True, use_dead_case_for_mass: bool = True) -> Project:
    """
    2-story 3D frame on an 8 m x 6 m plan with strongly asymmetric mass distribution
    so each floor's center of mass is shifted well away from its geometric centroid (4.0, 3.0).
    """
    Lx, Ly = 8.0, 6.0
    zs = [0.0, 3.5, 7.0]
    nodes = []
    for k, z in enumerate(zs):
        for j, y in enumerate((0.0, Ly)):
            for i, x in enumerate((0.0, Lx)):
                nid = 100 * k + 10 * j + i + 1
                # Heavy added nodal mass on corner (0, 0), medium on (Lx, 0), light elsewhere
                mass = 0.0
                inertia = 0.0
                if k > 0:
                    if i == 0 and j == 0:
                        mass, inertia = 25000.0, 42000.0
                    elif i == 1 and j == 0:
                        mass, inertia = 9000.0, 12000.0
                    elif i == 0 and j == 1:
                        mass, inertia = 4000.0, 6000.0
                    else:
                        mass, inertia = 2000.0, 3000.0
                nodes.append(Node(id=nid, x=x, y=y, z=z, mass=mass, inertia=inertia))

    supports = [Support.fixed_3d(1), Support.fixed_3d(2), Support.fixed_3d(11), Support.fixed_3d(12)]
    elems = []
    for k in (0, 1):
        for offset in (1, 2, 11, 12):
            elems.append(BeamElement(
                id=f"C_{k}_{offset}",
                start_node=100 * k + offset,
                end_node=100 * (k + 1) + offset,
                section_id="COL",
            ))
    for k in (1, 2):
        b = 100 * k
        elems.extend([
            BeamElement(id=f"BX1_{k}", start_node=b + 1, end_node=b + 2, section_id="BM"),
            BeamElement(id=f"BX2_{k}", start_node=b + 11, end_node=b + 12, section_id="BM"),
            BeamElement(id=f"BY1_{k}", start_node=b + 1, end_node=b + 11, section_id="BM"),
            BeamElement(id=f"BY2_{k}", start_node=b + 2, end_node=b + 12, section_id="BM"),
        ])

    dead_point_loads = []
    dead_elem_loads = []
    if use_dead_case_for_mass:
        # Extra gravity dead load concentrated near corner (0, 0) and along the south beam BX1
        dead_point_loads = [
            PointLoad(node_id=101, fz=-180.0),
            PointLoad(node_id=201, fz=-140.0),
            PointLoad(node_id=202, fz=-30.0),
        ]
        dead_elem_loads = [
            DistributedLoad(element_id="BX1_1", wz=-24.0),
            DistributedLoad(element_id="BX1_2", wz=-20.0),
        ]

    diaphragms = (
        [
            Diaphragm(id="D1", name="Floor_1", node_ids=[101, 102, 111, 112], axis="Z"),
            Diaphragm(id="D2", name="Roof", node_ids=[201, 202, 211, 212], axis="Z"),
        ]
        if with_diaphragms
        else []
    )
    model = StructuralModel(
        dimension="3D",
        vertical_axis="Z",
        units="kN, m",
        nodes=nodes,
        supports=supports,
        materials=[ConcreteMaterial(id="C30", E=25.0e6, nu=0.2, fc=30.0e3, unit_weight=24.0)],
        sections=[
            RectangularSection(id="COL", material_id="C30", b=0.45, h=0.45),
            RectangularSection(id="BM", material_id="C30", b=0.30, h=0.55),
        ],
        elements=elems,
        base_elevation=0.0,
        stories=[Story(name="Floor_1", elevation=3.5), Story(name="Roof", elevation=7.0)],
        diaphragms=diaphragms,
        load_cases=[
            LoadCase(
                id="DEAD",
                load_type="dead",
                include_self_weight=False,
                point_loads=dead_point_loads,
                element_loads=dead_elem_loads,
            )
        ],
    )
    return Project(model=model, units="kN, m")


def test_mass_source_diaphragm_summary_matches_fem3d_master_and_preserves_nodal_values():
    proj_d = _asymmetric_2story_project(with_diaphragms=True)
    proj_nd = _asymmetric_2story_project(with_diaphragms=False)
    ms = MassSource(
        id="MS1",
        include_element_mass=True,
        include_added_mass=True,
        factors=[MassSourceFactor(load_case_id="DEAD", factor=1.0)],
    )
    proj_d.model.mass_source = ms
    proj_nd.model.mass_source = ms

    res_d = calculate_mass_source(proj_d, mass_source=ms)
    res_nd = calculate_mass_source(proj_nd, mass_source=ms)

    # Mass source and seismic weight values are unchanged by diaphragms
    assert res_nd.diaphragm_masses == {}
    assert res_d.total_mass == res_nd.total_mass
    assert res_d.total_seismic_weight == res_nd.total_seismic_weight
    assert res_d.center_of_mass == res_nd.center_of_mass
    assert res_d.story_masses == res_nd.story_masses
    assert res_d.nodal_masses == res_nd.nodal_masses

    # Check read-only DiaphragmMassSummary against fem3d's T^T M T master block
    s = model_from_core(proj_d, mass_source=ms, include_self_weight=False)
    _, M_r = s.get_reduced_matrices(lumped=True, include_rotational_inertia=False)

    assert set(res_d.diaphragm_masses.keys()) == {"D1", "D2"}
    for did in ("D1", "D2"):
        summary = res_d.diaphragm_masses[did]
        assert isinstance(summary, DiaphragmMassSummary)
        meta = s.diaphragm_masters[did]
        c1, c2, cr = meta["master_dofs"]
        L_d = meta["scale_radius"]

        assert summary.mass_1 == pytest.approx(float(M_r[c1, c1]), rel=1e-12)
        assert summary.mass_2 == pytest.approx(float(M_r[c2, c2]), rel=1e-12)
        assert summary.mass == pytest.approx(0.5 * (float(M_r[c1, c1]) + float(M_r[c2, c2])), rel=1e-12)
        assert summary.weight == pytest.approx(summary.mass * res_d.gravity_accel, rel=1e-12)
        assert summary.rotational_inertia == pytest.approx(float(M_r[cr, cr]) * (L_d ** 2), rel=1e-12)
        np.testing.assert_allclose(summary.center_of_mass, meta["master_coords"], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("axis", ["X", "Y"])
@pytest.mark.parametrize("lateral_only", [False, True])
def test_diaphragm_mass_summary_x_y_axes_and_lateral_only(axis: str, lateral_only: bool):
    # Create an X- or Y-normal single-plane frame with added nodal mass + inertia and compare against fem3d T^T M T
    coords = [(1, 0, 0, 0), (2, 5, 0, 0), (3, 5, 4, 0), (4, 0, 4, 0),
              (5, 0, 0, 3), (6, 5, 0, 3), (7, 5, 4, 3), (8, 0, 4, 3)]
    nodes = [Node(id=i, x=x, y=y, z=z, mass=1000.0 * i if i >= 5 else 0.0, inertia=250.0 * i if i >= 5 else 0.0)
             for i, x, y, z in coords]
    if axis == "X":
        # Nodes 2, 3, 6, 7 lie at x = 5.0
        d_nids = [2, 3, 6, 7]
        sup = [Support.fixed_3d(1), Support.fixed_3d(4)]
        nodes[1].mass, nodes[2].mass = 5000.0, 2000.0
    else:  # "Y": nodes 3, 4, 7, 8 lie at y = 4.0
        d_nids = [3, 4, 7, 8]
        sup = [Support.fixed_3d(1), Support.fixed_3d(2)]
        nodes[3].mass = 6000.0

    model = StructuralModel(
        dimension="3D",
        vertical_axis="Z",
        units="kN, m",
        nodes=nodes,
        supports=sup,
        materials=[ConcreteMaterial(id="C30", E=25.0e6, fc=30.0e3, unit_weight=24.0)],
        sections=[RectangularSection(id="S1", material_id="C30", b=0.4, h=0.4)],
        elements=[
            BeamElement(id="C1", start_node=1, end_node=5, section_id="S1"),
            BeamElement(id="C2", start_node=2, end_node=6, section_id="S1"),
            BeamElement(id="C3", start_node=3, end_node=7, section_id="S1"),
            BeamElement(id="C4", start_node=4, end_node=8, section_id="S1"),
            BeamElement(id="B1", start_node=5, end_node=6, section_id="S1"),
            BeamElement(id="B2", start_node=6, end_node=7, section_id="S1"),
            BeamElement(id="B3", start_node=7, end_node=8, section_id="S1"),
            BeamElement(id="B4", start_node=8, end_node=5, section_id="S1"),
        ],
        diaphragms=[Diaphragm(id="D1", node_ids=d_nids, axis=axis)],
    )
    proj = Project(model=model, units="kN, m")
    ms = MassSource(id="MS1", include_element_mass=True, include_added_mass=True, lateral_only=lateral_only, factors=[])
    ms_res = calculate_mass_source(proj, mass_source=ms)
    s = model_from_core(proj, mass_source=ms, include_self_weight=False)
    _, M_r = s.get_reduced_matrices(lumped=True, include_rotational_inertia=False)

    summary = ms_res.diaphragm_masses["D1"]
    meta = s.diaphragm_masters["D1"]
    c1, c2, cr = meta["master_dofs"]
    L_d = meta["scale_radius"]
    assert summary.mass_1 == pytest.approx(float(M_r[c1, c1]), rel=1e-12)
    assert summary.mass_2 == pytest.approx(float(M_r[c2, c2]), rel=1e-12)
    assert summary.rotational_inertia == pytest.approx(float(M_r[cr, cr]) * (L_d ** 2), rel=1e-12)
    np.testing.assert_allclose(summary.center_of_mass, meta["master_coords"], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("explicit_ms", [True, False])
def test_elf_on_asymmetric_diaphragm_floor_acts_at_center_of_mass_and_matches_master_load(explicit_ms: bool):
    proj = _asymmetric_2story_project(with_diaphragms=True)
    ms = MassSource(
        id="MS1",
        include_element_mass=True,
        include_added_mass=True,
        factors=[MassSourceFactor(load_case_id="DEAD", factor=1.0)],
    )
    if explicit_ms:
        proj.model.mass_source = ms

    gen = LoadGenerator(units="kN, m")
    params = SeismicParameters(sds=0.80, sd1=0.45, site_class="D", risk_category="II", system_type="rc_smf")
    eq_cases, elf_res = gen.apply_lateral_seismic(proj, params=params, direction="both")
    case_x, case_y = eq_cases
    assert any("12.8.4.1" in msg for msg in gen.messages)

    ms_res = calculate_mass_source(proj, mass_source=ms)
    nodes_by_id = {n.id: n for n in proj.model.nodes}
    elev_by_story = {1: 3.5, 2: 7.0}
    diaph_by_story = {1: "D1", 2: "D2"}

    # Verify resultant equals story force and line of action equals exact COM (not geometric centroid (4.0, 3.0))
    for k, elev in elev_by_story.items():
        sum_d = ms_res.diaphragm_masses[diaph_by_story[k]]
        cm_x, cm_y, _ = sum_d.center_of_mass

        pls_x = [pl for pl in case_x.point_loads if abs(nodes_by_id[pl.node_id].z - elev) < 1e-6]
        tot_fx = sum(pl.fx for pl in pls_x)
        y_res = sum(nodes_by_id[pl.node_id].y * pl.fx for pl in pls_x) / tot_fx
        assert tot_fx == pytest.approx(elf_res.story_forces[k], rel=1e-12)
        assert y_res == pytest.approx(cm_y, rel=1e-12)
        assert abs(y_res - 3.0) > 0.5  # geometric centroid is 3.0 m; COM is much further south

        pls_y = [pl for pl in case_y.point_loads if abs(nodes_by_id[pl.node_id].z - elev) < 1e-6]
        tot_fy = sum(pl.fy for pl in pls_y)
        x_res = sum(nodes_by_id[pl.node_id].x * pl.fy for pl in pls_y) / tot_fy
        assert tot_fy == pytest.approx(elf_res.story_forces[k], rel=1e-12)
        assert x_res == pytest.approx(cm_x, rel=1e-12)
        assert abs(x_res - 4.0) > 0.5  # geometric centroid is 4.0 m; COM is much further west

    # Compare fem3d static results against applying the exact story force directly at a master node at COM
    for dir_label, case_obj in (("X", case_x), ("Y", case_y)):
        s_dist = model_from_core(proj, load_case_id=case_obj.id, mass_source=ms, include_self_weight=False)
        disp_dist, reac_dist = s_dist.solve()

        # Static solve places masters at the geometric centroid (4.0, 3.0), so reduced F_r[cr]
        # captures the exact inherent torsional moment Mt = -Fx*(y_cm - 3) or +Fy*(x_cm - 4).
        F_r_geom = s_dist.reduce_vector(s_dist.F)
        for k in (1, 2):
            did = diaph_by_story[k]
            _c1, _c2, cr = s_dist.diaphragm_masters[did]["master_dofs"]
            L_d_geom = s_dist.diaphragm_masters[did]["scale_radius"]
            cm_x, cm_y, _ = ms_res.diaphragm_masses[did].center_of_mass
            expected_f = elf_res.story_forces[k]
            expected_mt = (-expected_f * (cm_y - 3.0)) if dir_label == "X" else (expected_f * (cm_x - 4.0))
            assert F_r_geom[cr] * L_d_geom == pytest.approx(expected_mt, rel=1e-12)

        # With mass-centered master coordinates (T^T F at COM), master torsional moment is zero.
        s_dist.get_reduced_matrices(lumped=True)
        F_r_com = s_dist.reduce_vector(s_dist.F)
        for k in (1, 2):
            c1, c2, cr = s_dist.diaphragm_masters[diaph_by_story[k]]["master_dofs"]
            expected_f = elf_res.story_forces[k]
            assert F_r_com[c1] == pytest.approx(expected_f if dir_label == "X" else 0.0, abs=1e-11)
            assert F_r_com[c2] == pytest.approx(expected_f if dir_label == "Y" else 0.0, abs=1e-11)
            assert abs(F_r_com[cr]) < 1e-11

        # Build an identical structural model adding standalone COM nodes loaded directly at the master point
        proj_master = proj.model_copy(deep=True)
        proj_master.model.load_cases = [lc for lc in proj_master.model.load_cases if lc.id == "DEAD"]
        cm_loads = []
        for k in (1, 2):
            did = diaph_by_story[k]
            cx, cy, cz = ms_res.diaphragm_masses[did].center_of_mass
            cm_nid = f"CM_{k}"
            proj_master.model.nodes.append(Node(id=cm_nid, x=cx, y=cy, z=cz))
            proj_master.model.supports.append(Support(node_id=cm_nid, uz=True, rx=True, ry=True))
            proj_master.model.get_diaphragm(did).node_ids.append(cm_nid)
            f_val = elf_res.story_forces[k]
            cm_loads.append(PointLoad(node_id=cm_nid, fx=f_val if dir_label == "X" else 0.0, fy=f_val if dir_label == "Y" else 0.0))
        proj_master.model.load_cases.append(LoadCase(id="EQ_MASTER", load_type="seismic", point_loads=cm_loads))

        s_master = model_from_core(proj_master, load_case_id="EQ_MASTER", mass_source=ms, include_self_weight=False)
        disp_master, reac_master = s_master.solve()

        for nid in s_dist.nodes:
            np.testing.assert_allclose(
                disp_dist[s_dist.nodes[nid].dofs],
                disp_master[s_master.nodes[nid].dofs],
                rtol=1e-12,
                atol=1e-13,
            )
            np.testing.assert_allclose(
                reac_dist[s_dist.nodes[nid].dofs],
                reac_master[s_master.nodes[nid].dofs],
                rtol=1e-11,
                atol=1e-11,
            )
        for eid in s_dist.elements:
            np.testing.assert_allclose(
                s_dist.elements[eid].get_local_forces(),
                s_master.elements[eid].get_local_forces(),
                rtol=1e-11,
                atol=1e-11,
            )


@pytest.mark.parametrize("dist", ["face", "uniform"])
def test_no_diaphragm_identity_and_mixed_floor_elf(dist: str):
    # Pure no-diaphragm model retains exact legacy behavior
    proj_a = _asymmetric_2story_project(with_diaphragms=False)
    proj_b = _asymmetric_2story_project(with_diaphragms=False)
    params = SeismicParameters(sds=0.80, sd1=0.45, site_class="D", risk_category="II", system_type="rc_smf")

    gen_a = LoadGenerator(units="kN, m")
    cases_a, elf_a = gen_a.apply_lateral_seismic(proj_a, params=params, direction="both", distribution=dist)
    assert not any("12.8.4.1" in m for m in gen_a.messages)

    # Check face/uniform equal-share rounded values on Roof (elevation 7.0)
    roof_pts_x = [pl for pl in cases_a[0].point_loads if pl.node_id in (201, 202, 211, 212)]
    expected_n = 2 if dist == "face" else 4
    assert len(roof_pts_x) == expected_n
    assert all(pl.fx == round(elf_a.story_forces[2] / expected_n, 4) for pl in roof_pts_x)

    # Mixed-floor model: diaphragm on Floor_1 only, NO diaphragm on Roof
    proj_mixed = _asymmetric_2story_project(with_diaphragms=True)
    proj_mixed.model.diaphragms = [proj_mixed.model.diaphragms[0]]  # keep D1 only
    gen_m = LoadGenerator(units="kN, m")
    cases_m, elf_m = gen_m.apply_lateral_seismic(proj_mixed, params=params, direction="both", distribution=dist)
    assert elf_m.story_forces == elf_a.story_forces

    # Floor 1 (with D1) acts at COM; Roof (without diaphragm) matches no-diaphragm distribution identically
    roof_pts_mixed_x = [(pl.node_id, pl.fx) for pl in cases_m[0].point_loads if pl.node_id in (201, 202, 211, 212)]
    roof_pts_orig_x = [(pl.node_id, pl.fx) for pl in roof_pts_x]
    assert roof_pts_mixed_x == roof_pts_orig_x

    ms_res = calculate_mass_source(
        proj_mixed,
        mass_source=MassSource(
            id="MS", include_element_mass=True, include_added_mass=True,
            factors=[MassSourceFactor(load_case_id="DEAD", factor=1.0)],
        ),
    )
    nodes_by_id = {n.id: n for n in proj_mixed.model.nodes}
    f1_pts_x = [pl for pl in cases_m[0].point_loads if pl.node_id in (101, 102, 111, 112)]
    y_res_f1 = sum(nodes_by_id[pl.node_id].y * pl.fx for pl in f1_pts_x) / sum(pl.fx for pl in f1_pts_x)
    assert y_res_f1 == pytest.approx(ms_res.diaphragm_masses["D1"].center_of_mass[1], rel=1e-12)


def test_elf_without_mass_source_matches_asce7_mass_source_with_nodal_masses():
    """
    Regression check: calculate_model_seismic_weight and apply_lateral_seismic without an explicit
    mass source must count Node.mass identically to MS_ASCE7, so ELF total weight equals the
    MS_ASCE7 story weight and the diaphragm resultant acts at the true center of mass.
    """
    from struct_load import create_asce7_mass_source

    proj = _asymmetric_2story_project(with_diaphragms=True)
    model = proj.model
    assert model.mass_source is None

    # Use single-story subset or full 2-story asymmetric model with heavy Node.mass + element self-weight + DEAD loads
    gen = LoadGenerator(units="kN, m")
    elevations = [3.5, 7.0]
    w_elf = gen.calculate_model_seismic_weight(model, elevations, base_elevation=0.0)

    asce7_ms = create_asce7_mass_source(model, units="kN, m")
    ms_res = calculate_mass_source(model, mass_source=asce7_ms, units="kN, m", base_elevation=0.0)
    for k, elev in enumerate(elevations):
        assert w_elf[k] == pytest.approx(round(ms_res.story_weights[elev], 2), abs=1e-9)

    params = SeismicParameters(sds=0.80, sd1=0.45, site_class="D", risk_category="II", system_type="rc_smf")
    (case_x, case_y), elf_res = gen.apply_lateral_seismic(proj, params=params, direction="both")
    assert elf_res.total_weight == pytest.approx(sum(round(ms_res.story_weights[e], 2) for e in elevations), abs=1e-9)

    nodes_by_id = {n.id: n for n in model.nodes}
    for k, (elev, did) in enumerate(((3.5, "D1"), (7.0, "D2")), start=1):
        cm_x, cm_y, _ = ms_res.diaphragm_masses[did].center_of_mass
        pls_x = [pl for pl in case_x.point_loads if abs(nodes_by_id[pl.node_id].z - elev) < 1e-6]
        pls_y = [pl for pl in case_y.point_loads if abs(nodes_by_id[pl.node_id].z - elev) < 1e-6]
        fx_sum = sum(pl.fx for pl in pls_x)
        fy_sum = sum(pl.fy for pl in pls_y)
        assert fx_sum == pytest.approx(elf_res.story_forces[k], rel=1e-12)
        assert fy_sum == pytest.approx(elf_res.story_forces[k], rel=1e-12)
        y_res = sum(nodes_by_id[pl.node_id].y * pl.fx for pl in pls_x) / fx_sum
        x_res = sum(nodes_by_id[pl.node_id].x * pl.fy for pl in pls_y) / fy_sum
        assert y_res == pytest.approx(cm_y, rel=1e-12)
        assert x_res == pytest.approx(cm_x, rel=1e-12)

    # Also verify a model whose seismic weight comes solely from Node.mass (massless frame, no dead loads)
    proj_added_only = _asymmetric_2story_project(with_diaphragms=True, use_dead_case_for_mass=False)
    for mat in proj_added_only.model.materials:
        mat.unit_weight = 0.0
    proj_added_only.model.load_cases = []
    (cx_only, _), elf_only = LoadGenerator(units="kN, m").apply_lateral_seismic(
        proj_added_only, params=params, direction="both"
    )
    ms_only = calculate_mass_source(
        proj_added_only, mass_source=create_asce7_mass_source(proj_added_only.model, units="kN, m"),
        units="kN, m", base_elevation=0.0,
    )
    assert elf_only.total_weight == pytest.approx(
        round(ms_only.story_weights[3.5], 2) + round(ms_only.story_weights[7.0], 2),
        abs=1e-9,
    )
    assert len(cx_only.point_loads) == 8
