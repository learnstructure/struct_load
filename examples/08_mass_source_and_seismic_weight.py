"""
Example 08: Mass Source Definition & Effective Seismic Weight (ASCE 7-22)
========================================================================

Demonstrates:
1. Creating a multi-story building model with Dead, Live, and Partition load cases.
2. Configuring an ASCE 7-22 Section 12.7.2 Mass Source:
   - 100% Dead Load (structural + superimposed)
   - 25% Storage / Sustained Live Load
   - Element self-mass from material density
3. Evaluating effective seismic weight W and inertial mass using `calculate_mass_source()`.
4. Inspecting the global Center of Mass (CM) and story-by-story mass distributions.
"""

from struct_core import (
    BeamElement,
    DistributedLoad,
    ElasticMaterial,
    GeneralSection,
    LoadCase,
    MassSource,
    MassSourceFactor,
    Node,
    PointLoad,
    RectangularSection,
    StructuralModel,
)
from struct_load import calculate_mass_source


def build_3story_building():
    """Build a 2-bay, 3-story steel-concrete commercial frame in SI units."""
    nodes = []
    # Grid: X = [0.0, 6.0, 12.0], Z = [0.0, 3.5, 7.0, 10.5]
    for floor_idx, z in enumerate([0.0, 3.5, 7.0, 10.5]):
        for x_idx, x in enumerate([0.0, 6.0, 12.0]):
            nid = f"N_{floor_idx}_{x_idx}"
            nodes.append(Node(id=nid, x=x, y=0.0, z=z))

    mat_conc = ElasticMaterial(id="CONC", E=30e6, rho=2400.0)  # 2400 kg/m^3
    sec_col = RectangularSection(id="COL_500", b=0.5, h=0.5)   # 500x500 mm
    sec_beam = RectangularSection(id="BEAM_300x600", b=0.3, h=0.6)  # 300x600 mm

    elements = []
    elem_id = 1
    # Columns
    for floor_idx in range(3):
        for x_idx in range(3):
            n_bot = f"N_{floor_idx}_{x_idx}"
            n_top = f"N_{floor_idx + 1}_{x_idx}"
            elements.append(
                BeamElement(
                    id=f"COL_{elem_id}",
                    start_node=n_bot,
                    end_node=n_top,
                    material_id="CONC",
                    section_id="COL_500",
                )
            )
            elem_id += 1

    # Beams
    beam_ids = []
    for floor_idx in range(1, 4):
        for bay in range(2):
            n_start = f"N_{floor_idx}_{bay}"
            n_end = f"N_{floor_idx}_{bay + 1}"
            bid = f"BEAM_{elem_id}"
            beam_ids.append(bid)
            elements.append(
                BeamElement(
                    id=bid,
                    start_node=n_start,
                    end_node=n_end,
                    material_id="CONC",
                    section_id="BEAM_300x600",
                )
            )
            elem_id += 1

    # Superimposed Dead Load (SDL: finishes, MEP, partitions = 15 kN/m per beam)
    sdl_loads = [DistributedLoad(element_id=bid, wz=-15.0) for bid in beam_ids]
    lc_dead = LoadCase(id="DEAD", name="Superimposed Dead Load", element_loads=sdl_loads)

    # Live Load (LL = 12 kN/m per beam)
    ll_loads = [DistributedLoad(element_id=bid, wz=-12.0) for bid in beam_ids]
    lc_live = LoadCase(id="LIVE", name="Live Load", element_loads=ll_loads)

    model = StructuralModel(
        nodes=nodes,
        elements=elements,
        materials=[mat_conc],
        sections=[sec_col, sec_beam],
        load_cases=[lc_dead, lc_live],
        story_elevations=[3.5, 7.0, 10.5],
        units="kN, m",
    )
    return model


def main():
    print("=" * 70)
    print("Example 08: ASCE 7-22 Mass Source & Seismic Weight Engine")
    print("=" * 70)

    model = build_3story_building()
    print(f"Building created with {len(model.nodes)} nodes and {len(model.elements)} elements.")

    # Define ASCE 7-22 Mass Source: 1.0 DEAD + 0.25 LIVE + Element Self-Mass
    ms = MassSource(
        id="MS_ASCE7",
        name="ASCE 7-22 Seismic Weight (1.0 D + 0.25 L + Self)",
        include_element_mass=True,
        include_added_mass=True,
        lateral_only=False,
        factors=[
            MassSourceFactor(load_case_id="DEAD", factor=1.0),
            MassSourceFactor(load_case_id="LIVE", factor=0.25),
        ],
    )

    result = calculate_mass_source(
        model=model,
        mass_source=ms,
    )

    print("\n[Mass Source Evaluation Summary]")
    print(f"  Configuration : {ms.name}")
    print(f"  Total Seismic Weight W : {result.total_seismic_weight:12.2f} kN")
    print(f"  Total Inertial Mass M  : {result.total_mass:12.2f} tonnes")
    print(f"  Center of Mass (CM)    : ({result.center_of_mass[0]:.2f} m, "
          f"{result.center_of_mass[1]:.2f} m, {result.center_of_mass[2]:.2f} m)")

    print("\n[Story Mass Distribution]")
    print(f"  {'Elevation (m)':<15} {'Story Mass (tonnes)':<20} {'Story Weight (kN)':<20}")
    print("  " + "-" * 55)
    for elev, m_story in sorted(result.story_masses.items()):
        w_story = m_story * model.units.gravity_acceleration
        print(f"  {elev:<15.2f} {m_story:<20.2f} {w_story:<20.2f}")

    print("\n" + "=" * 70)
    print("Mass Source calculation completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
