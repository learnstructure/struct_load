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
from helpers import build_3d_frame_model


def test_flat_roof_snow_load_asce7_22():
    # ASCE 7-22 Eq. 7.3-1 has no importance factor: pf = 0.7 * Ce * Ct * pg
    # SI, Risk Category II (pm,max = 1.44 kPa): pg = 3.0 kPa -> pf = 2.1 kPa, pm = 1.44 kPa
    assert calculate_flat_roof_snow_load(pg=3.0, units="kN, m") == pytest.approx(2.1)

    # pg <= pm,max: pm = pg governs (0.7 * 0.5 = 0.35 < 0.5)
    assert calculate_flat_roof_snow_load(pg=0.5, units="kN, m") == pytest.approx(0.5)

    # pg = 1.8 kPa > pm,max: pf = 1.26 kPa < pm = 1.44 kPa (Table 7.3-4, RC II)
    assert calculate_flat_roof_snow_load(pg=1.8, units="kN, m") == pytest.approx(1.44)
    # Risk Category IV raises the minimum to 1.92 kPa -> pm = pg = 1.8 kPa
    assert calculate_flat_roof_snow_load(pg=1.8, risk_category="IV", units="kN, m") == pytest.approx(1.8)


def test_flat_roof_snow_load_returns_active_pressure_unit():
    # 40 psf, RC II: pf = 28 psf < pm,max = 30 psf -> 30 psf
    assert calculate_flat_roof_snow_load(pg=40.0, units="lbf, ft") == pytest.approx(30.0)
    # Same load in a kip-ft model is expressed in ksf
    assert calculate_flat_roof_snow_load(pg=0.040, units="kip, ft") == pytest.approx(0.030)


def test_flat_roof_snow_load_rejects_unknown_risk_category():
    with pytest.raises(ValueError):
        calculate_flat_roof_snow_load(pg=1.0, risk_category="V", units="kN, m")


def test_roof_live_load_bounds():
    # Large area (flat roof): At = 800 sqft (> 600 sqft), pitch = 0
    # R1 = 0.6, R2 = 1.0 -> 20 * 0.6 = 12 psf (code minimum bound)
    lr_us = calculate_roof_live_load(tributary_area=800.0, slope_rise_over_12=0.0, units="lbf, ft")
    assert lr_us == pytest.approx(12.0)

    # Small area: At = 100 sqft (<= 200 sqft)
    # R1 = 1.0, R2 = 1.0 -> 20 psf (code maximum bound)
    lr_small = calculate_roof_live_load(tributary_area=100.0, slope_rise_over_12=0.0, units="lbf, ft")
    assert lr_small == pytest.approx(20.0)

    # kip-ft models receive ksf
    lr_ksf = calculate_roof_live_load(tributary_area=100.0, slope_rise_over_12=0.0, units="kip, ft")
    assert lr_ksf == pytest.approx(0.020)


def test_apply_roof_loads_to_3d_frame():
    model = build_3d_frame_model()
    project = Project(model=model, metadata=Metadata(title="Test Building"))
    gen = LoadGenerator(units="kN, m")

    created = gen.apply_roof_loads(
        project,
        ground_snow_load=1.2,
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
