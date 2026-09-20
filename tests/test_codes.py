"""
Unit tests for struct_load codes subpackage (ASCE 7 tables, combinations, seismic, wind).
"""

import pytest
from struct_load.codes.asce7_tables import (
    lookup_occupancy,
    get_occupancy_live_load,
    calculate_live_load_reduction,
)
from struct_load.codes.asce7_combinations import generate_asce7_combinations
from struct_load.codes.asce7_seismic import (
    SeismicParameters,
    calculate_seismic_elf,
    approximate_fundamental_period,
    calculate_seismic_response_coefficient,
)
from struct_load.codes.asce7_wind import (
    WindParameters,
    calculate_kz,
    calculate_wind_forces,
)


def test_asce7_occupancy_lookups():
    occ_office = lookup_occupancy("office")
    assert occ_office.key == "office"
    assert occ_office.uniform_psf == 50.0
    assert occ_office.uniform_kpa == 2.40
    assert occ_office.partition_psf == 15.0

    # Test alias
    occ_apartment = lookup_occupancy("apartment")
    assert occ_apartment.key == "residential"

    # Test pressure values
    ll_si = get_occupancy_live_load("office", include_partitions=True, unit_system="si")
    assert ll_si == pytest.approx(2.40 + 0.72, rel=1e-2)

    ll_us = get_occupancy_live_load("office", include_partitions=True, unit_system="us")
    assert ll_us == 65.0  # 50 + 15 psf


def test_live_load_reduction():
    # Small area (<400 ft2): no reduction
    l_small = calculate_live_load_reduction(L0=50.0, tributary_area=100.0, k_ll=2.0, unit_system="us")
    assert l_small == 50.0

    # Large area (e.g. 500 ft2, influence area = 1000 ft2)
    l_red = calculate_live_load_reduction(L0=50.0, tributary_area=500.0, k_ll=2.0, unit_system="us")
    assert l_red < 50.0
    assert l_red >= 50.0 * 0.50  # Single story min factor 0.50


def test_asce7_combinations_lrfd():
    combs = generate_asce7_combinations(
        dead_case_id="D",
        live_case_id="L",
        wind_case_id="W",
        seismic_case_id="E",
        design_method="LRFD",
    )
    comb_ids = [c.id for c in combs]
    assert "COMB_1.4D" in comb_ids
    assert "COMB_1.2D+1.6L" in comb_ids
    assert "COMB_1.2D+1.0L+1.0W" in comb_ids
    assert "COMB_1.2D+1.0L+1.0E" in comb_ids
    assert "COMB_0.9D+1.0E" in comb_ids
    assert "COMB_0.9D-1.0E" in comb_ids

    # Check factors on 1.2D + 1.6L
    comb_12_16 = next(c for c in combs if c.id == "COMB_1.2D+1.6L")
    factors_map = {f.load_case_id: f.factor for f in comb_12_16.factors}
    assert factors_map["D"] == 1.2
    assert factors_map["L"] == 1.6


def test_asce7_seismic_elf():
    params = SeismicParameters(
        sds=0.75,
        sd1=0.40,
        r_factor=8.0,
        importance_factor=1.0,
        system_type="rc_smf",
    )
    story_elevations = [3.5, 7.0]  # 2 stories, 3.5m each
    story_weights = [150.0, 150.0]  # 150 kN per floor

    res = calculate_seismic_elf(story_elevations, story_weights, params, unit_system="si")
    assert res.base_shear > 0.0
    assert len(res.story_forces) == 2
    # Upper floor force should be greater than or equal to lower floor force
    assert res.story_forces[2] >= res.story_forces[1]
    assert pytest.approx(sum(res.story_forces.values()), rel=1e-2) == res.base_shear


def test_asce7_wind():
    params = WindParameters(basic_wind_speed=40.0, exposure_category="B")
    story_elevations = [3.5, 7.0]
    trib_widths = [5.0, 5.0]
    story_heights = [3.5, 3.5]

    res = calculate_wind_forces(story_elevations, trib_widths, story_heights, params, unit_system="si")
    assert res.total_base_shear > 0.0
    assert len(res.story_forces) == 2
    # Upper story pressure should be higher due to Kz increasing with height
    assert res.story_pressures[2] >= res.story_pressures[1]
