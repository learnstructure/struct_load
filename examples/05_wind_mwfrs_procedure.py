"""
Example 05: ASCE 7-16 / 7-22 Directional Wind Procedure (MWFRS)
==============================================================
This example demonstrates how `struct_load` calculates Main Wind-Force
Resisting System (MWFRS) directional wind loads:
1. Computation of velocity pressure exposure coefficient Kz(z) for Exposures B, C, and D.
2. Calculation of velocity pressure qz and net design pressure across building height.
3. Multi-story lateral wind force calculation and cumulative wind shear.
4. Exposure category sensitivity analysis (comparing urban Exposure B with open Exposure C).
5. Application of lateral wind loads to a 3D structural model using `LoadGenerator`.
"""

from pathlib import Path
import sys

# Ensure root repository and packages are in import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from struct_core import (
    BeamElement,
    ElasticMaterial,
    Node,
    RectangularSection,
    StructuralModel,
)
from struct_load.codes.asce7_wind import (
    WindParameters,
    calculate_kz,
    calculate_wind_forces,
)
from struct_load.generator import LoadGenerator


def demo_kz_and_exposure_profiles():
    print("=" * 70)
    print("1. ASCE 7 Velocity Pressure Exposure Coefficient Kz(z) Profiles")
    print("=" * 70)

    heights_m = [4.5, 9.0, 15.0, 24.0, 36.0, 50.0]

    print(f"{'Height z (m)':<14} | {'Exp B (Urban)':<16} | {'Exp C (Open)':<16} | {'Exp D (Coastal)':<16}")
    print("-" * 68)
    for z in heights_m:
        kz_b = calculate_kz(z, exposure="B", units="kN, m")
        kz_c = calculate_kz(z, exposure="C", units="kN, m")
        kz_d = calculate_kz(z, exposure="D", units="kN, m")
        print(f"{z:<14.1f} | {kz_b:<16.3f} | {kz_c:<16.3f} | {kz_d:<16.3f}")

    print("\nNote: Exposure B models urban/suburban terrain with surface roughness B.")
    print("      Exposure C represents open terrain with scattered obstructions.")
    print("      Exposure D represents flat, unobstructed coastal water/mud flats.")


def demo_mwfrs_story_forces():
    print("\n" + "=" * 70)
    print("2. Multi-Story MWFRS Story Force Calculation (ASCE 7 Ch 26 & 27)")
    print("=" * 70)

    # 5-story commercial building:
    # 5 levels at 3.6 m increments (Total height = 18.0 m)
    # Building width perpendicular to wind tributary to this frame: 15.0 m
    story_elevs = [3.6, 7.2, 10.8, 14.4, 18.0]
    story_heights = [3.6, 3.6, 3.6, 3.6, 3.6]
    trib_widths = [15.0] * 5

    # Design Wind Criteria:
    # Basic wind speed V = 45 m/s (~100 mph), Exposure Category C, Risk Category II
    # Kd = 0.85 (directionality), Kzt = 1.0 (flat ground), Ke = 1.0 (sea level), G = 0.85
    params_c = WindParameters(
        basic_wind_speed=45.0,  # m/s
        exposure_category="C",
        risk_category="II",
        kd=0.85,
        kzt=1.0,
        ke=1.0,
        gust_factor=0.85,
    )

    res_c = calculate_wind_forces(
        story_elevations=story_elevs,
        tributary_widths=trib_widths,
        story_heights=story_heights,
        params=params_c,
        units="kN, m",
    )

    print(f"Basic Wind Speed V:              {params_c.basic_wind_speed:.1f} m/s")
    print(f"Exposure Category:               {params_c.exposure_category}")
    print(f"Wind Directionality Factor Kd:    {params_c.kd:.2f}")
    print(f"Gust-Effect Factor G:            {params_c.gust_factor:.2f}")
    print(f"Total Lateral Wind Base Shear:   {res_c.total_base_shear:.2f} kN\n")

    print(f"{'Story':<7} | {'Elevation (m)':<14} | {'Pressure p (kPa)':<18} | {'Force Fw (kN)':<15} | {'Cumul Shear (kN)':<16}")
    print("-" * 76)

    # Cumulative shears from roof down to base
    n_stories = len(story_elevs)
    cum_shears = {}
    running_shear = 0.0
    for idx in range(n_stories, 0, -1):
        running_shear += res_c.story_forces[idx]
        cum_shears[idx] = running_shear

    for idx in range(n_stories, 0, -1):
        z = story_elevs[idx - 1]
        p_net = res_c.story_pressures[idx]
        fw = res_c.story_forces[idx]
        v_cum = cum_shears[idx]
        print(f"Level {idx:<2} | {z:<14.2f} | {p_net:<18.3f} | {fw:<15.2f} | {v_cum:<16.2f}")

    print("-" * 76)
    print(f"Total Base Shear: {res_c.total_base_shear:.2f} kN")


def demo_exposure_comparison():
    print("\n" + "=" * 70)
    print("3. Sensitivity Comparison: Exposure B vs Exposure C vs Exposure D")
    print("=" * 70)

    story_elevs = [3.5, 7.0, 10.5, 14.0, 17.5]
    story_heights = [3.5] * 5
    trib_widths = [12.0] * 5
    v_wind = 42.0  # m/s

    print(f"{'Exposure Category':<22} | {'Base Shear (kN)':<16} | {'Relative Increase':<18}")
    print("-" * 60)

    res_b = calculate_wind_forces(
        story_elevs, trib_widths, story_heights,
        WindParameters(basic_wind_speed=v_wind, exposure_category="B"),
        units="kN, m"
    )
    res_c = calculate_wind_forces(
        story_elevs, trib_widths, story_heights,
        WindParameters(basic_wind_speed=v_wind, exposure_category="C"),
        units="kN, m"
    )
    res_d = calculate_wind_forces(
        story_elevs, trib_widths, story_heights,
        WindParameters(basic_wind_speed=v_wind, exposure_category="D"),
        units="kN, m"
    )

    base_b = res_b.total_base_shear
    print(f"{'Exposure B (Urban)':<22} | {res_b.total_base_shear:<16.2f} | {'Baseline (1.00x)':<18}")
    print(f"{'Exposure C (Open)':<22} | {res_c.total_base_shear:<16.2f} | {f'{res_c.total_base_shear / base_b:.2f}x':<18}")
    print(f"{'Exposure D (Coastal)':<22} | {res_d.total_base_shear:<16.2f} | {f'{res_d.total_base_shear / base_b:.2f}x':<18}")


def demo_apply_wind_to_model():
    print("\n" + "=" * 70)
    print("4. Applying Wind Loads to Structural Model via LoadGenerator")
    print("=" * 70)

    # 2-story building frame
    model = StructuralModel(units="kN, m")
    for z, z_val in enumerate([0.0, 4.0, 8.0]):
        base_id = z * 2
        model.nodes.extend([
            Node(id=base_id + 1, x=0.0, y=0.0, z=z_val),
            Node(id=base_id + 2, x=8.0, y=0.0, z=z_val),
        ])

    mat = ElasticMaterial(id="Concrete", E=30e6, nu=0.2, rho=2400.0)
    sec = RectangularSection(material_id=mat.id, id="Rect_400x500", b=0.40, h=0.50)
    model.materials.append(mat)
    model.sections.append(sec)

    model.elements.extend([
        BeamElement(id=1, type="beam", start_node=1, end_node=3, section_id=sec.id),
        BeamElement(id=2, type="beam", start_node=2, end_node=4, section_id=sec.id),
        BeamElement(id=3, type="beam", start_node=3, end_node=4, section_id=sec.id),
        BeamElement(id=4, type="beam", start_node=3, end_node=5, section_id=sec.id),
        BeamElement(id=5, type="beam", start_node=4, end_node=6, section_id=sec.id),
        BeamElement(id=6, type="beam", start_node=5, end_node=6, section_id=sec.id),
    ])

    generator = LoadGenerator(units="kN, m")
    wind_case, wind_res = generator.apply_lateral_wind(
        model_or_project=model,
        basic_wind_speed=40.0,  # m/s
        exposure_category="B",
        tributary_width=6.0,
        direction="X",
    )

    print(f"Generated Wind Load Case ID: {wind_case.id} ('{wind_case.name}')")
    print(f"Total Base Shear V_wind = {wind_res.total_base_shear:.2f} kN")
    print(f"Applied Point Loads ({len(wind_case.point_loads)} nodes):")
    for pl in wind_case.point_loads:
        node = next(n for n in model.nodes if n.id == pl.node_id)
        print(f"  Diaphragm Node #{pl.node_id} (Z = {node.z:.1f} m): Fx = {pl.fx:.2f} kN")


def main():
    demo_kz_and_exposure_profiles()
    demo_mwfrs_story_forces()
    demo_exposure_comparison()
    demo_apply_wind_to_model()
    print("\n" + "=" * 70)
    print("Example 05 completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
