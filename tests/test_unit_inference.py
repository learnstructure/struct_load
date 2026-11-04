"""
Tests for unit system auto-inference in struct_load.
"""

import pytest
from struct_core import Metadata, Project, Units
from struct_load import LoadGenerator
from helpers import build_3d_frame_model


def test_units_auto_inference_us():
    model = build_3d_frame_model()
    # US Customary project
    units_us = Units(length="ft", force="kip", time="s", temperature="F")
    meta = Metadata(title="US Building", units=units_us)
    project = Project(model=model, metadata=meta)

    # Do not specify units in LoadGenerator constructor
    gen = LoadGenerator()
    u = gen.get_units(project)
    assert u.length == "ft"
    assert u.force == "kip"

    dead_case, live_case = gen.apply_gravity_loads(
        project,
        occupancy="office",
        one_way_span="y",
        superimposed_dead=0.020,  # 20 psf, in the model's ksf
        include_self_weight=False,
    )

    # Office live load = 50 psf + 15 psf partition = 65 psf = 0.065 ksf
    # Bay Y = 5.0 ft -> trib width = 2.5 ft -> 0.065 * 2.5 = 0.1625 kip/ft
    assert len(live_case.element_loads) > 0
    first_live = live_case.element_loads[0]
    assert abs(first_live.wz) == pytest.approx(0.1625, rel=0.01)


def test_units_auto_inference_si():
    model = build_3d_frame_model()
    units_si = Units(length="m", force="kN", time="s", temperature="C")
    meta = Metadata(title="SI Building", units=units_si)
    project = Project(model=model, metadata=meta)

    gen = LoadGenerator()
    u = gen.get_units(project)
    assert u.length == "m"
    assert u.force == "kN"

    dead_case, live_case = gen.apply_gravity_loads(
        project,
        occupancy="classroom",
        one_way_span="y",
        superimposed_dead=1.0,  # 1 kPa
        include_self_weight=False,
    )
    # classroom: 1.92 kPa * 2.5 m (trib width for 5m bay) = 4.80 kN/m
    assert len(live_case.element_loads) > 0
    first_live = live_case.element_loads[0]
    assert abs(first_live.wz) == pytest.approx(4.80, rel=0.01)


def test_units_canonical_string_inference():
    model = build_3d_frame_model()
    # Direct canonical string initialization
    project = Project(model=model, units="kip, in")
    gen = LoadGenerator()
    u = gen.get_units(project)
    assert u.force == "kip"
    assert u.length == "in"

    project_si = Project(model=build_3d_frame_model(), units="kN, m")
    u_si = gen.get_units(project_si)
    assert u_si.force == "kN"
    assert u_si.length == "m"


def test_apply_gravity_loads_no_double_counting():
    """Verify materialized self-weight sets include_self_weight=False to prevent double counting."""
    from struct_load import apply_self_weight

    project = Project(model=build_3d_frame_model(), units="kN, m")
    gen = LoadGenerator()

    dead_case, live_case = gen.apply_gravity_loads(
        project,
        occupancy="office",
        superimposed_dead=1.5,
        include_self_weight=True,
    )

    # include_self_weight must be False because self-weight is already in element_loads
    assert dead_case.include_self_weight is False
    loads_before = len(dead_case.element_loads)
    assert loads_before > 0

    # Running apply_self_weight should NOT add loads again since include_self_weight is False
    apply_self_weight(project.model, target_case_id=dead_case.id)
    # The dead case should maintain the exact same loads
    assert len(dead_case.element_loads) == loads_before
