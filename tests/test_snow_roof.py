"""
Tests for ASCE 7 Snow and Roof Live load provisions in struct_load.
"""

import pytest
from struct_core import Metadata, Project
from struct_load import (
    LoadGenerator,
    calculate_flat_roof_snow_load,
    calculate_roof_live_load,
)
from conftest import build_3d_frame_model


def test_flat_roof_snow_load_basic_and_minimum():
    # SI: pg = 1.5 kPa (> 0.96 kPa). Ce=1.0, Ct=1.0, Is=1.0
    # pf_basic = 0.7 * 1.5 = 1.05 kPa
    # pm = 0.958 kPa
    # pf = max(1.05, 0.958) = 1.05 kPa
    pf = calculate_flat_roof_snow_load(pg=1.5, unit_system="si")
    assert pf == 1.05

    # Low snow zone: pg = 0.5 kPa (<= 0.96 kPa)
    # pf_basic = 0.7 * 0.5 = 0.35 kPa
    # pm = 1.0 * 0.5 = 0.5 kPa (governs!)
    pf_low = calculate_flat_roof_snow_load(pg=0.5, unit_system="si")
    assert pf_low == 0.5

    # US Customary: pg = 30 psf (> 20 psf)
    # pf_basic = 0.7 * 30 = 21 psf. pm = 20 psf. pf = 21 psf
    pf_us = calculate_flat_roof_snow_load(pg=30.0, unit_system="us")
    assert pf_us == 21.0


def test_roof_live_load_bounds():
    # Large area (flat roof): At = 800 sqft (> 600 sqft), pitch = 0
    # R1 = 0.6, R2 = 1.0 -> 20 * 0.6 = 12 psf (code minimum bound)
    lr_us = calculate_roof_live_load(tributary_area=800.0, slope_rise_over_12=0.0, unit_system="us")
    assert lr_us == 12.0

    # Small area: At = 100 sqft (<= 200 sqft)
    # R1 = 1.0, R2 = 1.0 -> 20 psf (code maximum bound)
    lr_small = calculate_roof_live_load(tributary_area=100.0, slope_rise_over_12=0.0, unit_system="us")
    assert lr_small == 20.0


def test_apply_roof_loads_to_3d_frame():
    model = build_3d_frame_model()
    project = Project(model=model, metadata=Metadata(title="Test Building"))
    gen = LoadGenerator(unit_system="si")

    created = gen.apply_roof_loads(
        project,
        ground_snow_load=1.2,
        tributary_width=4.0,
    )
    assert "snow" in created
    assert "roof_live" in created

    snow_case = created["snow"]
    # Only 4 roof beams on story 2 (Z=7.0: B5, B6, B7, B8) should receive load!
    assert len(snow_case.element_loads) == 4
    roof_beam_ids = {"B5", "B6", "B7", "B8"}
    for el in snow_case.element_loads:
        assert el.element_id in roof_beam_ids
        assert el.wz < 0.0  # downward gravity
