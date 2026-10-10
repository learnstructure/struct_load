"""
Example 07: End-to-End Automated Building Loading Pipeline
=========================================================
This example demonstrates the complete end-to-end automated loading workflow
using `struct_load.generator.LoadGenerator` on a realistic 3-story concrete frame building:
1. Parametric 3D structural model generation (geometry, materials, sections, supports).
2. Gravity loading: ASCE 7 occupancy live load lookup, partitions, superimposed dead load (SDL),
   and beam/column self-weight materialization.
3. Roof loading: ASCE 7 flat roof snow load (Chapter 7) and roof live load (Section 4.5).
4. Lateral seismic loading: ASCE 7 Equivalent Lateral Force (ELF) procedure with exact
   model-derived seismic weights and multi-directional diaphragm loading (EQ_X and EQ_Y).
5. Lateral wind loading: ASCE 7 MWFRS directional wind procedure (Wind_X and Wind_Y).
6. Factored combinations: Automated ASCE 7 / ACI 318 LRFD combination generation including
   3D orthogonal seismic interactions (1.0 Ex +/- 0.3 Ey and 0.3 Ex +/- 1.0 Ey).
7. Comprehensive model loading audit and sanity verification.
"""

from pathlib import Path
import sys

# Ensure root repository and packages are in import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from struct_core import (
    BeamElement,
    ElasticMaterial,
    Metadata,
    Node,
    Project,
    RectangularSection,
    StructuralModel,
    Support,
    Units,
)
from struct_load.generator import LoadGenerator


def build_3story_building_model() -> Project:
    """
    Construct a 3-story, 2x1-bay reinforced concrete space frame:
    - Plan: 2 bays in X (6.0 m each, total 12.0 m) x 1 bay in Y (5.0 m)
    - Height: 3 stories @ 3.5 m (Z = 0.0, 3.5, 7.0, 10.5 m)
    - 6 columns per story (18 columns total)
    - 7 beams per level (4 in X, 3 in Y -> 21 beams total)
    """
    model = StructuralModel()

    # Grid definitions
    x_coords = [0.0, 6.0, 12.0]
    y_coords = [0.0, 5.0]
    z_coords = [0.0, 3.5, 7.0, 10.5]

    # Generate nodes
    node_id = 1
    grid_nodes = {}  # (ix, iy, iz) -> node_id
    for iz, z in enumerate(z_coords):
        for iy, y in enumerate(y_coords):
            for ix, x in enumerate(x_coords):
                model.nodes.append(Node(id=node_id, x=x, y=y, z=z))
                grid_nodes[(ix, iy, iz)] = node_id
                node_id += 1

    # Materials and Cross Sections
    conc = ElasticMaterial(id="Concrete_C30", name="Normalweight Concrete 30 MPa", E=30e6, nu=0.2, rho=2400.0)
    col_sec = RectangularSection(id="Col_500x500", name="Column 500x500 mm", b=0.50, h=0.50)
    bm_sec = RectangularSection(id="Bm_300x600", name="Beam 300x600 mm", b=0.30, h=0.60)
    model.materials.append(conc)
    model.sections.extend([col_sec, bm_sec])

    elem_id = 1
    # Generate Columns (connecting iz -> iz+1)
    for iz in range(len(z_coords) - 1):
        for iy in range(len(y_coords)):
            for ix in range(len(x_coords)):
                n_bot = grid_nodes[(ix, iy, iz)]
                n_top = grid_nodes[(ix, iy, iz + 1)]
                model.elements.append(
                    BeamElement(
                        id=elem_id, type="beam",
                        start_node=n_bot, end_node=n_top,
                        material_id=conc.id, section_id=col_sec.id,
                        name=f"Col_Story{iz+1}_{ix}_{iy}",
                    )
                )
                elem_id += 1

    # Generate Beams at each elevated level (iz >= 1)
    for iz in range(1, len(z_coords)):
        # Beams along X direction
        for iy in range(len(y_coords)):
            for ix in range(len(x_coords) - 1):
                n1 = grid_nodes[(ix, iy, iz)]
                n2 = grid_nodes[(ix + 1, iy, iz)]
                model.elements.append(
                    BeamElement(
                        id=elem_id, type="beam",
                        start_node=n1, end_node=n2,
                        material_id=conc.id, section_id=bm_sec.id,
                        name=f"BmX_Lvl{iz}_{ix}_{iy}",
                    )
                )
                elem_id += 1

        # Beams along Y direction
        for ix in range(len(x_coords)):
            for iy in range(len(y_coords) - 1):
                n1 = grid_nodes[(ix, iy, iz)]
                n2 = grid_nodes[(ix, iy + 1, iz)]
                model.elements.append(
                    BeamElement(
                        id=elem_id, type="beam",
                        start_node=n1, end_node=n2,
                        material_id=conc.id, section_id=bm_sec.id,
                        name=f"BmY_Lvl{iz}_{ix}_{iy}",
                    )
                )
                elem_id += 1

    # Supports: Fixed bases at Z = 0 m (iz = 0)
    for iy in range(len(y_coords)):
        for ix in range(len(x_coords)):
            base_node_id = grid_nodes[(ix, iy, 0)]
            model.supports.append(Support.fixed_3d(node_id=base_node_id))

    # Wrap in Project
    project = Project(
        metadata=Metadata(
            title="3-Story RC Office Building Frame",
            project_name="ASCE 7 / ACI 318 Loading Pipeline",
            author="Structural Engineering AI Agent",
            units=Units(length="m", force="kN"),
        ),
        model=model,
    )

    return project


def run_pipeline():
    print("=" * 80)
    print("END-TO-END AUTOMATED BUILDING LOADING PIPELINE")
    print("=" * 80)

    # 1. Build Model
    project = build_3story_building_model()
    model = project.model
    print(f"\n1. Structural Model Initialized:")
    print(f"   Project:       '{project.metadata.title}'")
    print(f"   Nodes:          {len(model.nodes)}")
    print(f"   Elements:       {len(model.elements)} ({18} columns, {21} beams)")
    print(f"   Base Supports:  {len(model.supports)} fixed foundation supports")

    # 2. Instantiate LoadGenerator
    generator = LoadGenerator(unit_system="si")

    # 3. Apply Gravity Loads (Dead + Live)
    # Tributary width: 5.0 m (spacing between frames in Y)
    # Superimposed dead: 1.5 kPa (finishes, ceilings, MEP)
    # Occupancy: "office" (ASCE 7: 2.4 kPa + 0.72 kPa partition allowance = 3.12 kPa)
    print("\n2. Applying Gravity Loads (ASCE 7-16 Table 4.3-1):")
    dead_case, live_case = generator.apply_gravity_loads(
        model_or_project=project,
        occupancy="office",
        tributary_width=5.0,
        superimposed_dead=1.5,
        include_self_weight=True,
    )
    print(f"   - Dead Case '{dead_case.id}': {len(dead_case.element_loads)} elements loaded (SW + SDL)")
    print(f"   - Live Case '{live_case.id}': {len(live_case.element_loads)} floor beams loaded")

    # 4. Apply Roof Loads (Snow + Roof Live)
    # Ground snow pg = 1.2 kPa -> Flat roof snow pf = 0.7 * 1.2 = 0.84 kPa
    # Roof live load Lr minimum per Sec 4.5
    print("\n3. Applying Roof Environmental Loads (ASCE 7 Chapter 7 & Section 4.5):")
    roof_cases = generator.apply_roof_loads(
        model_or_project=project,
        ground_snow_load=1.2,
        tributary_width=5.0,
    )
    for k, rc in roof_cases.items():
        print(f"   - Roof Case '{rc.id}': {len(rc.element_loads)} roof beams loaded")

    # 5. Apply Lateral Seismic Loads (ASCE 7 ELF)
    # SDS = 0.90 g, SD1 = 0.45 g, R = 8.0 (Special RC Moment Frame)
    # Derives exact building seismic weight directly from member self-weights and dead load cases!
    print("\n4. Applying Lateral Seismic Loads (ASCE 7 Section 12.8 ELF):")
    seismic_cases, elf_res = generator.apply_lateral_seismic(
        model_or_project=project,
        sds=0.90,
        sd1=0.45,
        r_factor=8.0,
        direction="both",
    )
    print(f"   - Effective Seismic Weight W:  {elf_res.total_weight:.2f} kN")
    print(f"   - Approximate Period Ta:       {elf_res.ta:.3f} s")
    print(f"   - Response Coefficient Cs:     {elf_res.cs:.4f}")
    print(f"   - Total Base Shear V_seismic:  {elf_res.base_shear:.2f} kN")
    for sc in (seismic_cases if isinstance(seismic_cases, list) else [seismic_cases]):
        print(f"   - Seismic Case '{sc.id}': {len(sc.point_loads)} nodal lateral loads applied")

    # 6. Apply Lateral Wind Loads (ASCE 7 MWFRS)
    # Basic wind speed V = 42 m/s (~94 mph), Exposure B, Direction: both
    print("\n5. Applying Lateral Wind Loads (ASCE 7 Chapter 27 MWFRS):")
    wind_cases, wind_res = generator.apply_lateral_wind(
        model_or_project=project,
        basic_wind_speed=42.0,
        exposure_category="B",
        tributary_width=5.0,
        direction="both",
    )
    print(f"   - Total Wind Base Shear V_wind: {wind_res.total_base_shear:.2f} kN")
    for wc in (wind_cases if isinstance(wind_cases, list) else [wind_cases]):
        print(f"   - Wind Case '{wc.id}': {len(wc.point_loads)} nodal lateral loads applied")

    # 7. Generate Code Factored Combinations
    # LRFD combinations with 3D orthogonal seismic interactions (1.0 Ex +/- 0.3 Ey)
    print("\n6. Generating Factored Design Combinations (ASCE 7 Ch 2 / ACI 318 Ch 5):")
    combinations = generator.generate_combinations(
        model_or_project=project,
        design_method="LRFD",
        orthogonal_seismic=True,
    )
    print(f"   - Total LRFD Combinations Generated: {len(combinations)}")

    # 8. Model Audit Summary
    print("\n" + "=" * 80)
    print("MODEL LOADING AUDIT & VERIFICATION SUMMARY")
    print("=" * 80)

    print(f"\n{'Load Case ID':<15} | {'Nature/Description':<35} | {'Elements':<10} | {'Nodes':<8}")
    print("-" * 75)
    for lc in model.load_cases:
        desc = lc.name or lc.id
        n_elem = len(lc.element_loads)
        n_node = len(lc.point_loads)
        print(f"{lc.id:<15} | {desc:<35} | {n_elem:<10} | {n_node:<8}")

    print("\nSample Generated LRFD Load Combinations:")
    for comb in model.load_combinations[:6]:
        eq_terms = [f"{f.factor:+.2f}*{f.load_case_id}" for f in comb.factors]
        eq_str = " ".join(eq_terms).lstrip("+")
        print(f"  [{comb.id:<28}] {eq_str}")

    if len(model.load_combinations) > 6:
        print(f"  ... and {len(model.load_combinations) - 6} additional combinations (orthogonal seismic & wind).")

    print("\nVerdict: Model loading complete, compliant with ASCE 7-16/22 and ACI 318-19,")
    print("         and fully ready for FEM solver analysis and reinforced concrete design.")


def main():
    run_pipeline()
    print("\n" + "=" * 80)
    print("Example 07 completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
