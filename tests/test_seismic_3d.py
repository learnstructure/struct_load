"""
Tests for 3D model-derived seismic weights and multi-axis lateral loads in struct_load.
"""

import pytest
from struct_core import Metadata, Project
from struct_load import LoadGenerator, SeismicParameters
from helpers import build_3d_frame_model


def test_model_derived_seismic_weight():
    model = build_3d_frame_model()
    project = Project(model=model, metadata=Metadata(title="Test Building"))
    gen = LoadGenerator(units="kN, m")

    # Apply gravity loads first so model has Dead load cases
    dead_case, _ = gen.apply_gravity_loads(
        project,
        occupancy="office",
        tributary_width=5.0,
        superimposed_dead=1.5,
        include_self_weight=True,
    )
    assert dead_case.id == "Dead"

    elevations = [3.5, 7.0]
    story_weights = gen.calculate_model_seismic_weight(model, elevations)
    assert len(story_weights) == 2
    assert story_weights[0] > 0.0
    assert story_weights[1] > 0.0

    # Total weight should equal sum of story weights
    total_w = sum(story_weights)
    assert total_w > 100.0  # reasonable for 2-story building frame


def test_apply_lateral_seismic_3d_both_directions():
    model = build_3d_frame_model()
    project = Project(model=model, metadata=Metadata(title="Test Building"))
    gen = LoadGenerator(units="kN, m")

    # Apply gravity loads first
    gen.apply_gravity_loads(project, include_self_weight=True)

    # Apply seismic loads in both directions
    params = SeismicParameters(sds=0.75, sd1=0.40, r_factor=8.0)
    cases, elf_res = gen.apply_lateral_seismic(
        project,
        params=params,
        direction="both",
        distribution="face",
    )

    assert isinstance(cases, list)
    assert len(cases) == 2
    case_ids = {c.id for c in cases}
    assert "EQ_X" in case_ids
    assert "EQ_Y" in case_ids

    case_x = next(c for c in cases if c.id == "EQ_X")
    case_y = next(c for c in cases if c.id == "EQ_Y")

    assert len(case_x.point_loads) == 4  # 2 face nodes per story * 2 stories
    assert len(case_y.point_loads) == 4

    # Check X loads have fx > 0 and fy == 0
    for pl in case_x.point_loads:
        assert pl.fx > 0.0
        assert pl.fy == 0.0

    # Check Y loads have fy > 0 and fx == 0
    for pl in case_y.point_loads:
        assert pl.fy > 0.0
        assert pl.fx == 0.0


def test_generate_combinations_orthogonal_seismic():
    model = build_3d_frame_model()
    project = Project(model=model, metadata=Metadata(title="Test Building"))
    gen = LoadGenerator(units="kN, m")

    gen.apply_gravity_loads(project)
    gen.apply_lateral_seismic(project, direction="both")

    # Generate LRFD combinations with orthogonal combinations enabled
    combs = gen.generate_combinations(project, design_method="LRFD", orthogonal_seismic=True)
    comb_ids = [c.id for c in combs]

    # Should contain standard gravity combinations
    assert "COMB_1.4D" in comb_ids
    assert "COMB_1.2D+1.6L" in comb_ids

    # Should contain orthogonal seismic combinations (+1.0Ex +0.3Ey, etc.)
    has_ortho = any("EQ_X" in c.name and "EQ_Y" in c.name for c in combs)
    assert has_ortho
