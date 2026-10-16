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
        tributary_width=15.0,  # 15 ft
        superimposed_dead=20.0,  # 20 psf
        include_self_weight=False,
    )

    # In US units: office live load = 50 psf + 15 psf partition = 65 psf
    # 65 psf * 15 ft = 975 plf = 0.975 kip/ft (or 975 lb/ft)
    assert len(live_case.element_loads) > 0
    first_live = live_case.element_loads[0]
    # Magnitude: 65 * 15 = 975
    assert abs(first_live.wz) == pytest.approx(975.0, rel=0.01)


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
        tributary_width=4.0,  # 4 m
        superimposed_dead=1.0,  # 1 kPa
        include_self_weight=False,
    )
    # classroom: 1.92 kPa * 4 m = 7.68 kN/m
    assert len(live_case.element_loads) > 0
    first_live = live_case.element_loads[0]
    assert abs(first_live.wz) == pytest.approx(7.68, rel=0.01)
