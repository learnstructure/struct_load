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
    ll_si = get_occupancy_live_load("office", include_partitions=True, units="kN, m")
    assert ll_si == pytest.approx(2.40 + 0.72, rel=1e-2)

    ll_us = get_occupancy_live_load("office", include_partitions=True, units="kip, in")
    assert ll_us == 65.0  # 50 + 15 psf


def test_live_load_reduction():
    # Small area (<400 ft2): no reduction
    l_small = calculate_live_load_reduction(L0=50.0, tributary_area=100.0, k_ll=2.0, units="kip, in")
    assert l_small == 50.0

    # Large area (e.g. 500 ft2, influence area = 1000 ft2)
    l_red = calculate_live_load_reduction(L0=50.0, tributary_area=500.0, k_ll=2.0, units="kip, in")
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
        method=2,
    )
    story_elevations = [3.5, 7.0]  # 2 stories, 3.5m each
    story_weights = [150.0, 150.0]  # 150 kN per floor

    res = calculate_seismic_elf(story_elevations, story_weights, params, units="kN, m")
    assert res.base_shear > 0.0
    assert len(res.story_forces) == 2
    # Upper floor force should be greater than or equal to lower floor force
    assert res.story_forces[2] >= res.story_forces[1]
    assert pytest.approx(sum(res.story_forces.values()), rel=1e-2) == res.base_shear

    # Check story shears Vx = sum_{i=x}^n Fi (ASCE 7-22 Eq. 12.8-14)
    assert res.story_shears is not None
    assert pytest.approx(res.story_shears[1], rel=1e-2) == res.base_shear
    assert pytest.approx(res.story_shears[2], rel=1e-2) == res.story_forces[2]


def test_asce7_22_elf_method1_and_method2():
    from struct_load.codes.asce7_seismic import calculate_seismic_response_coefficient

    # Method 2: Short period T = 0.2s <= Ts (0.4 / 1.0 = 0.4s) -> Cs = SDS / (R/Ie) = 1.0 / 8.0 = 0.125
    params_m2 = SeismicParameters(sds=1.00, sd1=0.40, r_factor=8.0, importance_factor=1.0, method=2)
    cs_m2_short = calculate_seismic_response_coefficient(params_m2, 0.2)
    assert cs_m2_short == 0.125

    # Method 2: Moderate period T = 0.8s (Ts < T <= TL) -> Cs = SD1 / (T * (R/Ie)) = 0.40 / (0.8 * 8.0) = 0.0625
    cs_m2_mid = calculate_seismic_response_coefficient(params_m2, 0.8)
    assert cs_m2_mid == 0.0625

    # Method 2: Long period T = 10.0s > TL (8.0s) -> Cs = SD1 * TL / (T^2 * (R/Ie)) = (0.40 * 8.0) / (100 * 8.0) = 0.004
    # But minimum Cs per Eq. (12.8-6) = max(0.044 * 1.0 * 1.0, 0.01) = 0.044
    cs_m2_long = calculate_seismic_response_coefficient(params_m2, 10.0)
    assert cs_m2_long == 0.044

    # Method 1 (Eq. 12.8-2): T < Ts uses maximum Sa = SDS -> Cs = 0.125
    params_m1 = SeismicParameters(sds=1.00, sd1=0.40, r_factor=8.0, importance_factor=1.0, method=1)
    cs_m1_short = calculate_seismic_response_coefficient(params_m1, 0.2)
    assert cs_m1_short == 0.125

    # Method 1 with custom Sa spectrum dictionary
    custom_spec = {0.2: 1.20, 0.5: 1.20, 1.0: 0.60, 2.0: 0.30}
    params_m1_spec = SeismicParameters(
        sds=1.00, sd1=0.40, r_factor=8.0, importance_factor=1.0, method=1, sa_spectrum=custom_spec
    )
    # At T=1.0s, Sa=0.60 -> Cs = 0.60 / 8.0 = 0.075
    cs_m1_spec = calculate_seismic_response_coefficient(params_m1_spec, 1.0)
    assert cs_m1_spec == 0.075


def test_asce7_22_s1_minimum_and_sds_cap():
    from struct_load.codes.asce7_seismic import (
        calculate_seismic_response_coefficient,
        check_sds_cap_eligibility,
        calculate_cw,
        approximate_fundamental_period,
    )

    # Eq. (12.8-7): When S1 >= 0.6g, Cs >= 0.5 * S1 / (R/Ie)
    # S1 = 0.70g, R = 8.0, Ie = 1.0 -> Cs_min = 0.5 * 0.70 / 8.0 = 0.04375 -> rounded 0.0438
    params_s1 = SeismicParameters(sds=1.20, sd1=0.50, s1=0.70, r_factor=8.0, importance_factor=1.0)
    cs_s1 = calculate_seismic_response_coefficient(params_s1, 2.0)
    # Cs basic = 0.50 / (2.0 * 8.0) = 0.03125 < 0.04375
    # Eq. 12.8-6 gives 0.044 * 1.2 * 1 = 0.0528
    assert cs_s1 >= 0.0528

    # High S1 scenario where Eq. 12.8-7 governs
    params_s1_gov = SeismicParameters(sds=0.80, sd1=0.40, s1=0.80, r_factor=5.0, importance_factor=1.0)
    # Eq. 12.8-6: 0.044 * 0.8 * 1.0 = 0.0352
    # Eq. 12.8-7: 0.5 * 0.8 / 5.0 = 0.080
    cs_gov = calculate_seismic_response_coefficient(params_s1_gov, 2.0)
    assert cs_gov == 0.080

    # Section 12.8.1.3 SDS cap test:
    # 3-story building, T = 0.30s <= 0.5s, Site Class C, Risk Cat II, R = 8.0 >= 3.0, rho = 1.0, no irregularities
    params_cap = SeismicParameters(
        sds=1.80,
        sd1=0.40,
        r_factor=8.0,
        site_class="C",
        risk_category="II",
        apply_sds_cap=True,
        has_irregularities=False,
        redundancy_factor=1.0,
        num_stories=3,
    )
    eligible, reason = check_sds_cap_eligibility(params_cap, period=0.30, num_stories=3)
    assert eligible is True

    # Capped SDS = max(1.0, 0.70 * 1.80) = 1.26
    # Cs = 1.26 / 8.0 = 0.1575
    cs_capped = calculate_seismic_response_coefficient(params_cap, period=0.30, num_stories=3)
    assert cs_capped == 0.1575

    # If Site Class is E, cap is ineligible per Criterion 5
    params_site_e = SeismicParameters(
        sds=1.80, sd1=0.40, r_factor=8.0, site_class="E", apply_sds_cap=True, num_stories=3
    )
    eligible_e, _ = check_sds_cap_eligibility(params_site_e, period=0.30, num_stories=3)
    assert eligible_e is False


def test_asce7_22_period_equations():
    from struct_load.codes.asce7_seismic import (
        approximate_fundamental_period,
        calculate_cw,
    )

    # General Eq. (12.8-8): RC frame of 7.0 m
    # Ta = 0.0466 * 7.0^0.9 = 0.269 s
    ta_rc = approximate_fundamental_period(7.0, system_type="rc_smf", units="kN, m")
    assert round(ta_rc, 3) == 0.269

    # Alternative Eq. (12.8-9): Ta = 0.1 * N for 4-story frame with average story height >= 3.0m
    ta_n = approximate_fundamental_period(
        14.0, system_type="rc_smf", units="kN, m", num_stories=4, use_story_count_formula=True
    )
    assert ta_n == 0.40

    # Shear wall Eq. (12.8-10) and Cw Eq. (12.8-11)
    ab = 200.0  # m^2
    walls = [(5.0, 6.0), (5.0, 6.0)]  # Ai = 5 m^2, Di = 6 m
    cw = calculate_cw(ab, walls, hn=20.0)
    assert cw > 0.0
    ta_wall = approximate_fundamental_period(
        20.0, system_type="rc_shear_wall", units="kN, m", cw=cw
    )
    assert ta_wall > 0.0


def test_asce7_wind():
    params = WindParameters(basic_wind_speed=40.0, exposure_category="B")
    story_elevations = [3.5, 7.0]
    trib_widths = [5.0, 5.0]
    story_heights = [3.5, 3.5]

    res = calculate_wind_forces(story_elevations, trib_widths, story_heights, params, units="kN, m")
    assert res.total_base_shear > 0.0
    assert len(res.story_forces) == 2
    # Upper story pressure should be higher due to Kz increasing with height
    assert res.story_pressures[2] >= res.story_pressures[1]
