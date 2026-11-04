"""
Example 04: ASCE 7-16 / 7-22 Equivalent Lateral Force (ELF) Seismic Procedure
=============================================================================
This example demonstrates how `struct_load` implements the ASCE 7 Section 12.8
Equivalent Lateral Force (ELF) seismic procedure:
1. Computation of approximate fundamental period Ta for different lateral systems.
2. Calculation of seismic response coefficient Cs with code upper and lower bounds.
3. Vertical lateral force distribution exponent k and story factors Cvx.
4. Model-derived seismic weights W_i directly from structural elements and dead loads.
5. Tabular comparison of story forces and cumulative story shears.
"""

from pathlib import Path
import sys

# Ensure root repository and packages are in import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from struct_core import (
    BeamElement,
    DistributedLoad,
    ElasticMaterial,
    LoadCase,
    Node,
    RectangularSection,
    StructuralModel,
)
from struct_load.codes.asce7_seismic import (
    SeismicParameters,
    approximate_fundamental_period,
    calculate_seismic_elf,
    calculate_seismic_response_coefficient,
)
from struct_load.generator import LoadGenerator


def demo_period_and_cs():
    print("=" * 70)
    print("1. ASCE 7-16/22 Period (Ta) and Seismic Coefficient (Cs) Evaluation")
    print("=" * 70)

    building_height_m = 24.0  # 6-story building, 4 m per story

    systems = [
        ("rc_smf", "RC Special Moment Frame"),
        ("steel_smf", "Steel Special Moment Frame"),
        ("rc_shear_wall", "RC Shear Wall / Dual System"),
    ]

    print(f"Building Total Height: {building_height_m:.1f} m\n")
    print(f"{'Structural System':<35} | {'Ta (sec)':<10}")
    print("-" * 50)
    for sys_code, sys_name in systems:
        ta = approximate_fundamental_period(
            total_height=building_height_m, system_type=sys_code, units="kN, m"
        )
        print(f"{sys_name:<35} | {ta:<10.3f}")

    # Seismic Parameters (e.g., High-seismic site in California / Nevada)
    # SDS = 1.00 g, SD1 = 0.60 g, R = 8.0 (RC-SMF), Ie = 1.0 (Risk Cat II), TL = 8.0 s
    params = SeismicParameters(
        sds=1.00,
        sd1=0.60,
        site_class="D",
        risk_category="II",
        r_factor=8.0,
        importance_factor=1.0,
        tl=8.0,
        system_type="rc_smf",
    )

    ta_rc = approximate_fundamental_period(building_height_m, system_type="rc_smf", units="kN, m")
    cs = calculate_seismic_response_coefficient(params=params, period=ta_rc)

    print("\n--- Seismic Design Parameters ---")
    print(f"Design Spectral Acceleration SDS:     {params.sds:.2f} g")
    print(f"Design Spectral Acceleration SD1:     {params.sd1:.2f} g")
    print(f"Response Modification Coeff (R):      {params.r_factor:.1f}")
    print(f"Seismic Importance Factor (Ie):       {params.importance_factor:.2f}")
    print(f"Calculated Fundamental Period Ta:     {ta_rc:.3f} s")
    print(f"Governing Seismic Coeff Cs:          {cs:.4f} (Base Shear V = {cs:.4f} * W)")


def demo_multi_story_elf_distribution():
    print("\n" + "=" * 70)
    print("2. ELF Story Force Distribution across Height (ASCE 7 Sec 12.8.3)")
    print("=" * 70)

    # 4-Story Building Data:
    # Elevations (m): Story 1 = 3.5, Story 2 = 7.0, Story 3 = 10.5, Story 4 (Roof) = 14.0
    # Story Seismic Weights Wi (kN): Floors = 2500 kN, Roof = 1800 kN
    story_elevs = [3.5, 7.0, 10.5, 14.0]
    story_weights = [2500.0, 2500.0, 2500.0, 1800.0]  # kN
    total_w = sum(story_weights)

    params = SeismicParameters(
        sds=0.85,
        sd1=0.45,
        site_class="D",
        risk_category="III",
        r_factor=6.0,          # e.g., Concrete Intermediate Moment Frame
        importance_factor=1.25, # Risk Category III (e.g. assembly/school)
        tl=8.0,
        system_type="rc_smf",
    )

    elf_result = calculate_seismic_elf(
        story_elevations=story_elevs,
        story_weights=story_weights,
        params=params,
        units="kN, m",
    )

    print(f"Total Effective Seismic Weight W:     {elf_result.total_weight:.1f} kN")
    print(f"Fundamental Period Ta:                {elf_result.ta:.3f} s")
    print(f"Seismic Response Coefficient Cs:      {elf_result.cs:.4f}")
    print(f"Total Design Seismic Base Shear V:    {elf_result.base_shear:.2f} kN")

    # Tabulate story forces Fx and story shear Vx
    print("\n" + "-" * 75)
    print(f"{'Story':<7} | {'Elevation (m)':<14} | {'Weight (kN)':<12} | {'Force Fx (kN)':<15} | {'Shear Vx (kN)':<14}")
    print("-" * 75)

    n_stories = len(story_elevs)
    # Calculate cumulative story shear from top to bottom
    story_shears = {}
    cum_shear = 0.0
    for idx in range(n_stories, 0, -1):
        fx = elf_result.story_forces[idx]
        cum_shear += fx
        story_shears[idx] = cum_shear

    for idx in range(n_stories, 0, -1):
        z = story_elevs[idx - 1]
        w_i = story_weights[idx - 1]
        fx = elf_result.story_forces[idx]
        vx = story_shears[idx]
        print(f"Level {idx:<2} | {z:<14.2f} | {w_i:<12.1f} | {fx:<15.2f} | {vx:<14.2f}")

    print("-" * 75)
    print(f"Sum of Story Forces Fx = {sum(elf_result.story_forces.values()):.2f} kN (Base Shear = {elf_result.base_shear:.2f} kN)")


def demo_model_derived_seismic_weight():
    print("\n" + "=" * 70)
    print("3. Model-Derived Seismic Weights via Structural Elements and Dead Loads")
    print("=" * 70)

    # Build a 2-story 3D structural model
    model = StructuralModel(units="kN, m")

    # Ground (Z=0), Floor 1 (Z=3.5), Roof (Z=7.0)
    for z, z_val in enumerate([0.0, 3.5, 7.0]):
        base_id = z * 4
        model.nodes.extend([
            Node(id=base_id + 1, x=0.0, y=0.0, z=z_val),
            Node(id=base_id + 2, x=6.0, y=0.0, z=z_val),
            Node(id=base_id + 3, x=6.0, y=6.0, z=z_val),
            Node(id=base_id + 4, x=0.0, y=6.0, z=z_val),
        ])

    mat = ElasticMaterial(id="Concrete", E=30e6, nu=0.2, rho=2400.0)
    sec_col = RectangularSection(material_id=mat.id, id="Col_500x500", b=0.50, h=0.50)
    sec_bm = RectangularSection(material_id=mat.id, id="Bm_300x500", b=0.30, h=0.50)
    model.materials.append(mat)
    model.sections.extend([sec_col, sec_bm])

    # Columns (Story 1: 1-4 to 5-8, Story 2: 5-8 to 9-12)
    elem_id = 1
    for b_offset, t_offset in [(0, 4), (4, 8)]:
        for i in range(1, 5):
            model.elements.append(
                BeamElement(
                    id=elem_id, type="beam",
                    start_node=b_offset + i, end_node=t_offset + i,
                    section_id=sec_col.id
                )
            )
            elem_id += 1

    # Perimeter floor beams at Floor 1 and Roof
    for offset in [4, 8]:
        ring = [(offset + 1, offset + 2), (offset + 2, offset + 3),
                (offset + 3, offset + 4), (offset + 4, offset + 1)]
        for n1, n2 in ring:
            model.elements.append(
                BeamElement(
                    id=elem_id, type="beam",
                    start_node=n1, end_node=n2,
                    section_id=sec_bm.id
                )
            )
            elem_id += 1

    # Add Superimposed Dead Load (e.g. 20 kN/m line load on floor beams)
    dead_loads = []
    # Floor 1 beams (elem_ids 9, 10, 11, 12)
    for eid in [9, 10, 11, 12]:
        dead_loads.append(DistributedLoad(element_id=eid, wz=-20.0))
    # Roof beams (elem_ids 13, 14, 15, 16)
    for eid in [13, 14, 15, 16]:
        dead_loads.append(DistributedLoad(element_id=eid, wz=-12.0))

    model.load_cases.append(
        LoadCase(id="Dead", name="Superimposed Dead Load", element_loads=dead_loads)
    )

    # Use LoadGenerator to derive exact model seismic weight
    generator = LoadGenerator(units="kN, m")
    elevations = [3.5, 7.0]
    weights = generator.calculate_model_seismic_weight(model, elevations)

    print(f"Calculated story seismic weights directly from model elements and dead load cases:")
    print(f"  Floor 1 (Z = 3.5 m): W_1 = {weights[0]:.2f} kN")
    print(f"  Roof    (Z = 7.0 m): W_2 = {weights[1]:.2f} kN")
    print(f"  Total Seismic Weight W = {sum(weights):.2f} kN")

    # Apply lateral seismic using the generator
    params = SeismicParameters(sds=1.0, sd1=0.5, site_class="D", risk_category="II", r_factor=8.0)
    seismic_cases, elf_res = generator.apply_lateral_seismic(
        model_or_project=model,
        params=params,
        direction="X",
    )
    print(f"\nGenerated Seismic Load Case: {seismic_cases.id} ({seismic_cases.name})")
    print(f"Point loads applied to floor diaphragm nodes: {len(seismic_cases.point_loads)} nodal loads")
    for pl in seismic_cases.point_loads[:4]:
        node = next(n for n in model.nodes if n.id == pl.node_id)
        print(f"  Node #{pl.node_id} (Z={node.z}m): Fx = {pl.fx:.2f} kN")


def main():
    demo_period_and_cs()
    demo_multi_story_elf_distribution()
    demo_model_derived_seismic_weight()
    print("\n" + "=" * 70)
    print("Example 04 completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
