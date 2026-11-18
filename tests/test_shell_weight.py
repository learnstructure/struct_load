"""
Shell elements in self-weight, mass source, and story seismic weight. Without them the ELF seismic weight of a model
with slabs or walls would be understated (shells must never be dropped silently).
"""

import math

import pytest

from struct_core import (
    BeamElement,
    ConcreteMaterial,
    LoadCase,
    MassSource,
    MassSourceFactor,
    Node,
    Project,
    RectangularSection,
    ShellElement,
    ShellLoad,
    ShellSection,
    StructuralModel,
    Support,
)
from struct_load import LoadGenerator, calculate_mass_source
from struct_load.engine.self_weight import apply_self_weight, calculate_self_weight

GAMMA = 24.0   # kN/m3
T_SLAB = 0.2   # m
T_WALL = 0.25  # m


def building() -> Project:
    """A 4 m x 3 m slab at z = 3 (2 x 1 elements) on a 3 m high wall along y = 0 (2 elements) and two columns."""
    nodes = []
    nid = 1
    ids = {}
    for z in (0.0, 3.0):
        for y in (0.0, 3.0):
            for x in (0.0, 2.0, 4.0):
                ids[(x, y, z)] = nid
                nodes.append(Node(id=nid, x=x, y=y, z=z))
                nid += 1
    n = ids
    model = StructuralModel(
        nodes=nodes,
        materials=[ConcreteMaterial(id="C", E=25e6, fc=30e3, unit_weight=GAMMA)],
        sections=[RectangularSection(id="COL", material_id="C", b=0.4, h=0.4)],
        shell_sections=[ShellSection(id="SLAB", material_id="C", thickness=T_SLAB),
                        ShellSection(id="WALL", material_id="C", thickness=T_WALL)],
        elements=[BeamElement(id="C1", start_node=n[(0, 3, 0)], end_node=n[(0, 3, 3)], section_id="COL"),
                  BeamElement(id="C2", start_node=n[(4, 3, 0)], end_node=n[(4, 3, 3)], section_id="COL")],
        shell_elements=[
            ShellElement(id="S1", node_ids=[n[(0, 0, 3)], n[(2, 0, 3)], n[(2, 3, 3)], n[(0, 3, 3)]], section_id="SLAB"),
            ShellElement(id="S2", node_ids=[n[(2, 0, 3)], n[(4, 0, 3)], n[(4, 3, 3)], n[(2, 3, 3)]], section_id="SLAB"),
            ShellElement(id="W1", node_ids=[n[(0, 0, 0)], n[(2, 0, 0)], n[(2, 0, 3)], n[(0, 0, 3)]], section_id="WALL"),
            ShellElement(id="W2", node_ids=[n[(2, 0, 0)], n[(4, 0, 0)], n[(4, 0, 3)], n[(2, 0, 3)]], section_id="WALL"),
        ],
        supports=[Support(node_id=n[(x, y, 0.0)], ux=True, uy=True, uz=True, rx=True, ry=True, rz=True)
                  for x in (0.0, 2.0, 4.0) for y in (0.0, 3.0)],
        load_cases=[LoadCase(id="DEAD", load_type="dead", include_self_weight=True),
                    LoadCase(id="SDL", load_type="super_dead",
                             shell_loads=[ShellLoad(element_ids=["S1", "S2"], pressure=-1.5, direction="z")])],
    )
    return Project(model=model, units="kN, m")


SLAB_W = GAMMA * T_SLAB * 4.0 * 3.0     # 57.6 kN
WALL_W = GAMMA * T_WALL * 4.0 * 3.0     # 72.0 kN
COL_W = 2 * GAMMA * 0.4 * 0.4 * 3.0     # 23.04 kN


def test_calculate_self_weight_gives_shell_loads_per_unit_area():
    case = calculate_self_weight(building().model, units="kN, m", direction="-z")
    loads = {sl.pressure: sorted(sl.element_ids) for sl in case.shell_loads}
    assert loads == {-GAMMA * T_SLAB: ["S1", "S2"], -GAMMA * T_WALL: ["W1", "W2"]}
    assert all(sl.direction == "z" and sl.is_self_weight for sl in case.shell_loads)
    assert sorted(dl.element_id for dl in case.element_loads) == ["C1", "C2"]


def test_apply_self_weight_adds_shell_loads_once():
    project = building()
    apply_self_weight(project)
    dead = project.model.load_cases[0]
    total = sum(-sl.pressure * 12.0 for sl in dead.shell_loads)  # each pressure covers 12 m2 (two 2 x 3 elements)
    assert math.isclose(total, SLAB_W + WALL_W)
    dead.include_self_weight = True
    apply_self_weight(project)  # a second pass must not add the shell self-weight again
    assert sum(len(sl.element_ids) for sl in dead.shell_loads) == 4


def test_mass_source_includes_shell_self_mass():
    project = building()
    res = calculate_mass_source(project, MassSource(include_element_mass=True, factors=[]))
    assert math.isclose(res.total_seismic_weight, SLAB_W + WALL_W + COL_W, rel_tol=1e-12)


def test_mass_source_includes_shell_loads_of_its_patterns_without_double_counting_self_weight():
    project = building()
    apply_self_weight(project)  # DEAD now carries self-weight shell loads
    ms = MassSource(include_element_mass=True,
                    factors=[MassSourceFactor(load_case_id="DEAD", factor=1.0),
                             MassSourceFactor(load_case_id="SDL", factor=1.0)])
    res = calculate_mass_source(project, ms)
    assert math.isclose(res.total_seismic_weight, SLAB_W + WALL_W + COL_W + 1.5 * 12.0, rel_tol=1e-12)


def test_mass_source_lumps_shell_mass_at_nodes_and_stories():
    project = building()
    project.model.story_elevations = [0.0, 3.0]
    res = calculate_mass_source(project, MassSource(include_element_mass=True, factors=[]))
    g = res.gravity_accel
    # The wall's mass is shared by its bottom and top nodes; the roof story gets the slab, half the wall, and half
    # the columns
    story_w = res.story_weights
    roof = max(story_w)
    assert math.isclose(story_w[roof], SLAB_W + WALL_W / 2 + COL_W / 2, rel_tol=1e-9)
    corner = project.model.shell_elements[0].node_ids[0]  # slab corner (0, 0, 3): a quarter of S1, a quarter of W1
    assert math.isclose(res.nodal_masses[corner][0] * g, SLAB_W / 8 + WALL_W / 8, rel_tol=1e-9)


def test_lateral_only_mass_source_puts_shell_mass_on_horizontal_dofs():
    project = building()
    res = calculate_mass_source(project, MassSource(include_element_mass=True, lateral_only=True, factors=[]))
    corner = project.model.shell_elements[0].node_ids[0]
    mx, my, mz = res.nodal_masses[corner]
    assert mx > 0 and math.isclose(mx, my) and mz == 0.0
    assert math.isclose(res.total_seismic_weight, SLAB_W + WALL_W + COL_W, rel_tol=1e-12)


def test_story_seismic_weight_includes_shells():
    project = building()
    weights = LoadGenerator().calculate_model_seismic_weight(project.model, elevations=[0.0, 3.0])
    sdl = 1.5 * 12.0  # the SDL case counts at 100 % (ASCE 7-22 §12.7.2)
    assert math.isclose(weights[1], SLAB_W + WALL_W / 2 + COL_W / 2 + sdl, abs_tol=0.011)
    assert math.isclose(sum(weights), SLAB_W + WALL_W + COL_W + sdl, abs_tol=0.02)


def test_story_seismic_weight_counts_shell_self_weight_once():
    project = building()
    apply_self_weight(project)
    project.model.load_cases[0].include_self_weight = False
    weights = LoadGenerator().calculate_model_seismic_weight(project.model, elevations=[0.0, 3.0])
    assert math.isclose(sum(weights), SLAB_W + WALL_W + COL_W + 1.5 * 12.0, abs_tol=0.02)


def test_shell_load_on_missing_element_is_refused():
    project = building()
    project.model.load_cases[1].shell_loads[0].element_ids.append("NOPE")
    ms = MassSource(include_element_mass=True, factors=[MassSourceFactor(load_case_id="SDL", factor=1.0)])
    with pytest.raises(ValueError, match="missing shell element 'NOPE'"):
        calculate_mass_source(project, ms)
