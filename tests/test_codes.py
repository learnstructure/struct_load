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

    # 50 + 15 psf, returned in the active pressure unit
    assert get_occupancy_live_load("office", include_partitions=True, units="lbf, ft") == pytest.approx(65.0)
    assert get_occupancy_live_load("office", include_partitions=True, units="kip, ft") == pytest.approx(0.065)
    assert get_occupancy_live_load("office", include_partitions=True, units="kip, in") == pytest.approx(65.0 / 144000.0)


def test_live_load_reduction():
    # Small area (<400 ft2): no reduction
    l_small = calculate_live_load_reduction(L0=50.0, tributary_area=100.0, k_ll=2.0, units="kip, ft")
    assert l_small == 50.0

    # Large area (e.g. 500 ft2, influence area = 1000 ft2) in ft and in^2
    l_red = calculate_live_load_reduction(L0=50.0, tributary_area=500.0, k_ll=2.0, units="kip, ft")
    assert l_red < 50.0
    assert l_red >= 50.0 * 0.50  # Single story min factor 0.50
    l_red_in2 = calculate_live_load_reduction(L0=50.0, tributary_area=500.0 * 144.0, k_ll=2.0, units="kip, in")
    assert l_red_in2 == pytest.approx(l_red)


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
        site_class="D",
        risk_category="II",
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
    params_m2 = SeismicParameters(sds=1.00, sd1=0.40, site_class="D", risk_category="II", r_factor=8.0, importance_factor=1.0, tl=8.0, method=2)
    cs_m2_short = calculate_seismic_response_coefficient(params_m2, 0.2)
    assert cs_m2_short == 0.125

    # Method 2: Moderate period T = 0.8s (Ts < T <= TL) -> Cs = SD1 / (T * (R/Ie)) = 0.40 / (0.8 * 8.0) = 0.0625
    cs_m2_mid = calculate_seismic_response_coefficient(params_m2, 0.8)
    assert cs_m2_mid == 0.0625

    # Method 2: Long period T = 10.0s > TL (8.0s) -> Cs = SD1 * TL / (T^2 * (R/Ie)) = (0.40 * 8.0) / (100 * 8.0) = 0.004
    # But minimum Cs per Eq. (12.8-6) = max(0.044 * 1.0 * 1.0, 0.01) = 0.044
    cs_m2_long = calculate_seismic_response_coefficient(params_m2, 10.0)
    assert cs_m2_long == 0.044
    # TL is a mapped site value: a period above the smallest mapped TL (4 s) needs it
    with pytest.raises(ValueError, match="long-period transition period TL"):
        calculate_seismic_response_coefficient(SeismicParameters(sds=1.00, sd1=0.40, site_class="D", risk_category="II", system_type="rc_smf", method=2), 10.0)

    # Method 1 (Eq. 12.8-2) needs the multi-period spectrum of Sec 11.4.5.1
    params_m1 = SeismicParameters(sds=1.00, sd1=0.40, site_class="D", risk_category="II", r_factor=8.0, importance_factor=1.0, method=1)
    with pytest.raises(ValueError, match="multi-period"):
        calculate_seismic_response_coefficient(params_m1, 0.2)

    # Method 1 with custom Sa spectrum dictionary
    custom_spec = {0.0: 0.40, 0.2: 1.20, 0.5: 1.20, 1.0: 0.60, 2.0: 0.30}
    # Sec 12.8.1.1 exception: T = 0.1 s is below the period of maximum Sa -> Sa = 1.20, Cs = 0.15
    params_m1_short = SeismicParameters(
        sds=1.00, sd1=0.40, site_class="D", risk_category="II", r_factor=8.0, importance_factor=1.0, method=1, sa_spectrum=custom_spec
    )
    assert calculate_seismic_response_coefficient(params_m1_short, 0.1) == pytest.approx(0.15)
    params_m1_spec = SeismicParameters(
        sds=1.00, sd1=0.40, site_class="D", risk_category="II", r_factor=8.0, importance_factor=1.0, method=1, sa_spectrum=custom_spec
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
    params_s1 = SeismicParameters(sds=1.20, sd1=0.50, site_class="D", risk_category="II", s1=0.70, r_factor=8.0, importance_factor=1.0)
    cs_s1 = calculate_seismic_response_coefficient(params_s1, 2.0)
    # Cs basic = 0.50 / (2.0 * 8.0) = 0.03125 < 0.04375
    # Eq. 12.8-6 gives 0.044 * 1.2 * 1 = 0.0528
    assert cs_s1 >= 0.0528

    # High S1 scenario where Eq. 12.8-7 governs
    params_s1_gov = SeismicParameters(sds=0.80, sd1=0.40, site_class="D", risk_category="II", s1=0.80, r_factor=5.0, importance_factor=1.0)
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
        sds=1.80, sd1=0.40, site_class="E", risk_category="II", r_factor=8.0, apply_sds_cap=True, num_stories=3
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
    # Intermediate and ordinary RC moment frames share the concrete moment-frame coefficients
    assert approximate_fundamental_period(7.0, system_type="rc_imf", units="kN, m") == ta_rc
    assert approximate_fundamental_period(7.0, system_type="rc_omf", units="kN, m") == ta_rc
    # All steel moment frames use Ct = 0.0724, x = 0.8 (SI)
    ta_steel = round(0.0724 * 7.0 ** 0.8, 3)
    for sys_key in ("steel_smf", "steel_imf", "steel_omf"):
        assert approximate_fundamental_period(7.0, system_type=sys_key, units="kN, m") == ta_steel

    # Alternative Eq. (12.8-9): Ta = 0.1 * N for 4-story frame with average story height >= 3.0m
    ta_n = approximate_fundamental_period(
        14.0, system_type="rc_smf", units="kN, m", num_stories=4, use_story_count_formula=True
    )
    assert ta_n == 0.40
    # Eq. 12.8-9 is limited to moment frames with N <= 12 and average story height >= 3 m
    with pytest.raises(ValueError, match="12.8-9"):
        approximate_fundamental_period(14.0, system_type="rc_shear_wall", units="kN, m", num_stories=4,
                                       use_story_count_formula=True)
    with pytest.raises(ValueError, match="12.8-9"):
        approximate_fundamental_period(10.0, system_type="rc_smf", units="kN, m", num_stories=4,
                                       use_story_count_formula=True)

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


def test_asce7_seismic_design_category():
    from struct_load import determine_seismic_design_category

    # 1. S1 >= 0.75 rule
    assert determine_seismic_design_category(sds=0.5, sd1=0.2, risk_category="II", s1=0.80) == "E"
    assert determine_seismic_design_category(sds=0.5, sd1=0.2, risk_category="IV", s1=0.80) == "F"

    # 2. Category A
    assert determine_seismic_design_category(sds=0.10, sd1=0.04, risk_category="II") == "A"
    assert determine_seismic_design_category(sds=0.10, sd1=0.04, risk_category="IV") == "A"

    # 3. Category B / C
    assert determine_seismic_design_category(sds=0.25, sd1=0.10, risk_category="II") == "B"
    assert determine_seismic_design_category(sds=0.25, sd1=0.10, risk_category="IV") == "C"

    # 4. Category C / D
    assert determine_seismic_design_category(sds=0.40, sd1=0.15, risk_category="II") == "C"
    assert determine_seismic_design_category(sds=0.40, sd1=0.15, risk_category="IV") == "D"

    # 5. Category D (high SDS / SD1)
    assert determine_seismic_design_category(sds=0.75, sd1=0.40, risk_category="II") == "D"
    assert determine_seismic_design_category(sds=0.75, sd1=0.40, risk_category="IV") == "D"

    # 6. Governed by SD1 when SDS is lower
    # SDS = 0.25 (B), SD1 = 0.25 (D) -> governing is D
    assert determine_seismic_design_category(sds=0.25, sd1=0.25, risk_category="II") == "D"


def test_asce7_combinations_ev_and_overstrength():
    from struct_load import generate_asce7_combinations

    # Standard LRFD with SDS = 1.0 -> Ev = 0.2 * 1.0 * D = 0.2 D
    # 1.2D + Ev = 1.4D; 0.9D - Ev = 0.7D
    combs = generate_asce7_combinations(
        dead_case_id="D",
        live_case_id="L",
        seismic_case_id="E",
        design_method="LRFD",
        sds=1.0,
    )
    factors_map = {c.id: {f.load_case_id: f.factor for f in c.factors} for c in combs}

    # Find the seismic combination with 1.4D
    comb_add = next(c for c in combs if "1.4D" in c.id and "E" in c.id)
    factors_add = {f.load_case_id: f.factor for f in comb_add.factors}
    assert factors_add["D"] == 1.4
    assert factors_add["E"] == 1.0
    assert factors_add["L"] == 1.0

    # Counteracting seismic combination: 0.9D - Ev = 0.7D
    comb_sub = next(c for c in combs if "0.7D" in c.id)
    factors_sub = {f.load_case_id: f.factor for f in comb_sub.factors}
    assert factors_sub["D"] == 0.7
    assert abs(factors_sub["E"]) == 1.0

    # SDC B exemption: Ev = 0 even if SDS is high (Section 12.4.2.2 Exception 2a)
    combs_sdc_b = generate_asce7_combinations(
        dead_case_id="D",
        live_case_id="L",
        seismic_case_id="E",
        design_method="LRFD",
        sds=1.0,
        seismic_design_category="B",
    )
    comb_b = next(c for c in combs_sdc_b if "1.2D" in c.id and "E" in c.id)
    factors_b = {f.load_case_id: f.factor for f in comb_b.factors}
    assert factors_b["D"] == 1.2  # Ev omitted in SDC B

    # Overstrength Omega0 = 3.0 (Section 12.4.3)
    combs_omega = generate_asce7_combinations(
        dead_case_id="D",
        live_case_id="L",
        seismic_case_id="E",
        design_method="LRFD",
        sds=1.0,
        include_overstrength=True,
        omega_0=3.0,
    )
    comb_om = next(c for c in combs_omega if "Em" in c.id or "3" in c.id)
    factors_om = {f.load_case_id: f.factor for f in comb_om.factors}
    assert factors_om["E"] == 3.0
    assert factors_om["D"] == 1.4

    # ASD combinations with Ev: (1.0 + 0.14 SDS) D and (0.6 - 0.14 SDS) D
    combs_asd = generate_asce7_combinations(
        dead_case_id="D",
        seismic_case_id="E",
        design_method="ASD",
        sds=1.0,
    )
    comb_asd_add = next(c for c in combs_asd if "1.14D" in c.id)
    factors_asd_add = {f.load_case_id: f.factor for f in comb_asd_add.factors}
    assert factors_asd_add["D"] == 1.14
    assert factors_asd_add["E"] == 0.7


def test_asce7_accidental_torsion_and_amplification():
    from struct_load import calculate_accidental_torsion, calculate_torsional_amplification

    # Base force Fx = 100 kN, building width perpendicular B = 20 m
    # Accidental eccentricity = 0.05 * 20 = 1.0 m
    # Mta = 1.0 * 1.0 * 100 = 100 kN*m
    mta = calculate_accidental_torsion(fx=100.0, perpendicular_dimension=20.0, ax=1.0)
    assert mta == 100.0

    # With amplification factor Ax = 1.5 -> Mta = 150 kN*m
    mta_amp = calculate_accidental_torsion(fx=100.0, perpendicular_dimension=20.0, ax=1.5)
    assert mta_amp == 150.0

    # Torsional amplification Ax (Eq. 12.8-15):
    # delta_max = 15 mm, delta_avg = 10 mm
    # Ax = (15 / (1.2 * 10))^2 = (15 / 12)^2 = 1.25^2 = 1.5625 -> rounded 1.562
    ax = calculate_torsional_amplification(delta_max=15.0, delta_avg=10.0)
    assert ax == pytest.approx(1.562, abs=0.01)

    # Lower bound: if delta_max / (1.2 * delta_avg) <= 1.0, Ax = 1.0
    ax_min = calculate_torsional_amplification(delta_max=10.0, delta_avg=10.0)
    assert ax_min == 1.0

    # Upper bound: Ax capped at 3.0
    ax_max = calculate_torsional_amplification(delta_max=50.0, delta_avg=10.0)
    assert ax_max == 3.0


def test_asce7_story_drift_limits():
    from struct_load import calculate_design_story_drift, check_story_drift_limit

    # Design drift Delta = Cd * (delta_top - delta_bot) / Ie (Eq. 12.8-16)
    # delta_top = 25 mm, delta_bot = 15 mm, Cd = 5.5, Ie = 1.0 -> Delta = 5.5 * 10 = 55 mm
    drift = calculate_design_story_drift(delta_elastic_top=25.0, delta_elastic_bot=15.0, cd=5.5, importance_factor=1.0)
    assert drift == 55.0

    # Story height = 3500 mm, Risk Category II (allowable factor = 0.020)
    # Delta_a = 0.020 * 3500 = 70.0 mm
    is_ok, d_val, d_allow = check_story_drift_limit(story_drift=55.0, story_height=3500.0, risk_category="II")
    assert is_ok is True
    assert d_allow == 70.0

    # Risk Category IV (allowable factor = 0.010)
    # Delta_a = 0.010 * 3500 = 35.0 mm -> 55.0 > 35.0 -> fails
    is_ok_iv, _, d_allow_iv = check_story_drift_limit(story_drift=55.0, story_height=3500.0, risk_category="IV")
    assert is_ok_iv is False
    assert d_allow_iv == 35.0



def _factor_sets(combs):
    return [{f.load_case_id: f.factor for f in c.factors} for c in combs]


def test_asce7_22_seismic_combinations_include_both_senses_of_e():
    # Section 2.3.6 combination 6 must be checked for +E and -E (non-symmetric frames)
    combs = generate_asce7_combinations(
        dead_case_id="D", live_case_id="L", seismic_case_id="E", design_method="LRFD", sds=1.0
    )
    sets = _factor_sets(combs)
    assert {"D": 1.4, "L": 1.0, "E": 1.0} in sets
    assert {"D": 1.4, "L": 1.0, "E": -1.0} in sets
    assert {"D": 0.7, "E": 1.0} in sets
    assert {"D": 0.7, "E": -1.0} in sets


def test_asce7_22_snow_load_factors():
    combs = generate_asce7_combinations(
        dead_case_id="D", live_case_id="L", snow_case_id="S", wind_case_id="W",
        seismic_case_id="E", design_method="LRFD",
    )
    sets = _factor_sets(combs)
    assert {"D": 1.2, "L": 1.6, "S": 0.3} in sets          # 2a: 0.3S
    assert {"D": 1.2, "S": 1.0, "L": 1.0} in sets          # 3a: 1.0S + L
    assert {"D": 1.2, "S": 1.0, "W": 0.5} in sets          # 3a: 1.0S + 0.5W
    assert {"D": 1.2, "W": -1.0, "L": 1.0, "S": 0.3} in sets  # 4a with reversed wind
    assert {"D": 1.2, "E": -1.0, "L": 1.0, "S": 0.15} in sets  # 6: 0.15S
    # No ASCE 7-16 snow factors remain
    assert not any(s.get("S") in (0.5, 1.6, 0.2) for s in sets)


def test_asce7_22_asd_seismic_and_snow_combinations():
    combs = generate_asce7_combinations(
        dead_case_id="D", live_case_id="L", snow_case_id="S", seismic_case_id="E",
        design_method="ASD", sds=1.0,
    )
    sets = _factor_sets(combs)
    assert {"D": 1.0, "S": 0.7} in sets                                  # 3a
    assert {"D": 1.0, "L": 0.75, "S": 0.525} in sets                     # 4a
    assert {"D": 1.14, "E": -0.7} in sets                                 # 8
    assert {"D": 1.105, "E": 0.525, "L": 0.75, "S": 0.1} in sets          # 9
    assert {"D": 0.46, "E": -0.7} in sets                                 # 10


def test_asce7_overstrength_requires_omega_0():
    with pytest.raises(ValueError):
        generate_asce7_combinations(dead_case_id="D", seismic_case_id="E", include_overstrength=True)


def test_asce7_22_table_12_2_1_and_drift_stability_limits():
    from struct_load import (allowable_story_drift, max_stability_coefficient, seismic_system_coefficients,
                             stability_coefficient)

    assert seismic_system_coefficients("rc_smf") == (8.0, 3.0, 5.5)
    assert seismic_system_coefficients("rc_imf") == (5.0, 3.0, 4.5)
    assert seismic_system_coefficients("rc_omf") == (3.0, 3.0, 2.5)
    with pytest.raises(ValueError):
        seismic_system_coefficients("rc_shear_wall_unknown")

    # Table 12.12-1 "all other structures", Risk Category II: 0.020 hsx; SMF in SDC D: / rho (12.12.1.1)
    assert allowable_story_drift(3500.0, "II", "rc_smf", "D", 1.3) == pytest.approx(0.020 * 3500.0 / 1.3)
    assert allowable_story_drift(3500.0, "II", "rc_smf", "C", 1.3) == pytest.approx(0.020 * 3500.0)
    assert allowable_story_drift(3500.0, "IV", "rc_imf", "D", 1.0) == pytest.approx(0.010 * 3500.0)

    # Eq. 12.8-18 and 12.8-19
    assert stability_coefficient(px=1000.0, delta_xe=10.0, vx=100.0, story_height=3500.0) == pytest.approx(10000.0 / 350000.0)
    assert max_stability_coefficient(4.5) == pytest.approx(0.5 / 4.5)
    assert max_stability_coefficient(1.5) == pytest.approx(0.25)
    assert max_stability_coefficient(5.5) == pytest.approx(0.10)  # 0.091 < 0.10: need not be taken less than 0.10


def test_asce7_22_kz_matches_table_26_10_1():
    # Table 26.10-1 values (Exposure B/C/D at 15, 30, 100 ft)
    for z_ft, kz_b, kz_c, kz_d in ((15.0, 0.57, 0.85, 1.03), (30.0, 0.70, 0.98, 1.16), (100.0, 0.95, 1.26, 1.43)):
        assert calculate_kz(z_ft, "B", units="lbf, ft") == pytest.approx(kz_b, abs=0.015)
        assert calculate_kz(z_ft, "C", units="lbf, ft") == pytest.approx(kz_c, abs=0.015)
        assert calculate_kz(z_ft, "D", units="lbf, ft") == pytest.approx(kz_d, abs=0.015)


def test_wind_leeward_uses_qh_and_l_over_b():
    from struct_load.codes.asce7_wind import leeward_wall_cp

    assert leeward_wall_cp(1.0) == -0.5 and leeward_wall_cp(2.0) == pytest.approx(-0.3)
    assert leeward_wall_cp(3.0) == pytest.approx(-0.25) and leeward_wall_cp(5.0) == -0.2

    # Two levels: the leeward pressure is the same at both (qh), so the level difference is windward only
    params = WindParameters(basic_wind_speed=115.0, exposure_category="C", l_over_b=2.0)
    res = calculate_wind_forces([12.0, 24.0], [20.0, 20.0], [12.0, 12.0], params, units="lbf, ft")
    q = lambda z: 0.00256 * calculate_kz(z, "C", units="lbf, ft") * 115.0 ** 2
    leeward = q(24.0) * 0.85 * 0.85 * 0.3
    assert res.story_pressures[1] == pytest.approx(q(12.0) * 0.85 * 0.85 * 0.8 + leeward, abs=1e-3)
    assert res.story_pressures[2] == pytest.approx(q(24.0) * 0.85 * 0.85 * 0.8 + leeward, abs=1e-3)

    # kip-ft models receive ksf pressures and kip forces
    res_kip = calculate_wind_forces([12.0, 24.0], [20.0, 20.0], [12.0, 12.0], params, units="kip, ft")
    assert res_kip.total_base_shear == pytest.approx(res.total_base_shear / 1000.0, rel=1e-3)


def test_partition_weight_case_for_seismic_weight():
    from helpers import build_3d_frame_model
    from struct_core import Project, Metadata
    from struct_load import LoadGenerator, calculate_mass_source

    project = Project(model=build_3d_frame_model(), metadata=Metadata(title="t"))
    gen = LoadGenerator(units="kN, m")
    gen.apply_gravity_loads(project, occupancy="office", superimposed_dead=1.5)
    case = next(lc for lc in project.model.load_cases if lc.id == "PartitionWeight")
    assert case.element_loads and all(dl.wz < 0.0 for dl in case.element_loads)
    from struct_load import create_asce7_mass_source

    ms = create_asce7_mass_source(project)
    assert any(f.load_case_id == "PartitionWeight" and f.factor == 1.0 for f in ms.factors)
    # Not part of any load combination
    combs = gen.generate_combinations(project)
    assert not any(f.load_case_id == "PartitionWeight" for c in combs for f in c.factors)
    # Residential: no partition allowance, no partition weight case
    project2 = Project(model=build_3d_frame_model(), metadata=Metadata(title="t"))
    gen.apply_gravity_loads(project2, occupancy="residential", superimposed_dead=1.5)
    assert all(lc.id != "PartitionWeight" for lc in project2.model.load_cases)


def test_seismic_coefficients_follow_the_system_and_risk_category():
    """R from Table 12.2-1 and Ie from Table 1.5-2 unless given."""
    assert SeismicParameters(sds=1.0, sd1=0.5, site_class="D", risk_category="II", system_type="rc_imf").r_factor == 5.0
    assert SeismicParameters(sds=1.0, sd1=0.5, site_class="D", system_type="rc_smf", risk_category="IV").importance_factor == 1.5
    given = SeismicParameters(sds=1.0, sd1=0.5, site_class="D", risk_category="II", system_type="rc_imf", r_factor=4.0, importance_factor=1.25)
    assert (given.r_factor, given.importance_factor) == (4.0, 1.25)
    with pytest.raises(ValueError):
        SeismicParameters(sds=1.0, sd1=0.5, site_class="D", risk_category="II", system_type="all_other")  # no tabulated R: give r_factor
    with pytest.raises(ValueError):
        SeismicParameters(sds=1.0, sd1=0.5, site_class="D", risk_category="II")  # no system and no R: raises


def test_m2_required_units_and_seismic_parameters():
    from struct_load.codes.asce7_wind import calculate_kz, calculate_wind_forces
    from struct_load.codes.asce7_seismic import approximate_fundamental_period, calculate_seismic_elf, SeismicParameters
    from struct_load.codes.asce7_snow import calculate_flat_roof_snow_load, calculate_roof_live_load
    from struct_load.codes.asce7_tables import get_occupancy_live_load, calculate_live_load_reduction

    # Missing units must raise TypeError on all 8 public functions
    with pytest.raises(TypeError):
        calculate_kz(z=10.0)
    with pytest.raises(TypeError):
        calculate_wind_forces(height=20.0, width=10.0, length=15.0, v=115.0)
    with pytest.raises(TypeError):
        approximate_fundamental_period(total_height=15.0)
    with pytest.raises(TypeError):
        calculate_seismic_elf(story_elevations=[3.0, 6.0], story_weights=[100.0, 100.0], params=SeismicParameters(sds=1.0, sd1=0.4, site_class='D', risk_category='II', r_factor=8.0))
    with pytest.raises(TypeError):
        calculate_flat_roof_snow_load(pg=1.5)
    with pytest.raises(TypeError):
        calculate_roof_live_load(roof_slope=5.0)
    with pytest.raises(TypeError):
        get_occupancy_live_load('office')
    with pytest.raises(TypeError):
        calculate_live_load_reduction(lo=50.0, ai=500.0)

    # Missing site_class or risk_category raises TypeError
    with pytest.raises(TypeError):
        SeismicParameters(sds=1.0, sd1=0.6)

    # Missing system and R raises ValueError
    with pytest.raises(ValueError, match='system_type is required'):
        SeismicParameters(sds=1.0, sd1=0.6, site_class='D', risk_category='II')
