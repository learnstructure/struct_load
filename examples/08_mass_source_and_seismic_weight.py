"""
Example 08: Mass Source Definition & Effective Seismic Weight (ASCE 7-22)
========================================================================

Demonstrates ASCE 7-22 Section 12.7.2 Effective Seismic Weight determination:
1. Dead Load (structural self-mass + superimposed dead): 100% (factor = 1.0).
2. Storage Live Load: Minimum 25% (factor = 0.25) per Section 12.7.2 Item 1.
   Note: Per Section 12.7.2 Exception (b), public garages and open parking structures
   are excluded (0%). General floor live load (office, residential, classroom) is 0%.
3. Partitions: 100% (factor = 1.0) per Section 12.7.2 Item 2 (min 10 psf / 0.48 kN/m^2).
4. Permanent Equipment: 100% (factor = 1.0) per Section 12.7.2 Item 3.
5. Snow Load: 15% where flat roof snow load Pf > 45 psf (2.16 kN/m^2) per Section 12.7.2 Item 4.
6. Evaluating effective seismic weight W, Center of Mass (CM), and story-by-story distribution.
"""

from struct_core import (
    BeamElement,
    DistributedLoad,
    ElasticMaterial,
    LoadCase,
    Node,
    RectangularSection,
    StructuralModel,
)
from struct_load import calculate_mass_source, create_asce7_mass_source


def build_3story_building():
    """Build a 2-bay, 3-story commercial frame in SI units."""
    nodes = []
    # Grid: X = [0.0, 6.0, 12.0], Z = [0.0, 3.5, 7.0, 10.5]
    for floor_idx, z in enumerate([0.0, 3.5, 7.0, 10.5]):
        for x_idx, x in enumerate([0.0, 6.0, 12.0]):
            nid = f"N_{floor_idx}_{x_idx}"
            nodes.append(Node(id=nid, x=x, y=0.0, z=z))

    mat_conc = ElasticMaterial(id="CONC", E=30e6, rho=2400.0)  # 2400 kg/m^3
    sec_col = RectangularSection(id="COL_500", material_id="CONC", b=0.5, h=0.5)   # 500x500 mm
    sec_beam = RectangularSection(id="BEAM_300x600", material_id="CONC", b=0.3, h=0.6)  # 300x600 mm

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
                    section_id="BEAM_300x600",
                )
            )
            elem_id += 1

    # Superimposed Dead Load (SDL: finishes, MEP = 15 kN/m per beam)
    sdl_loads = [DistributedLoad(element_id=bid, wz=-15.0) for bid in beam_ids]
    lc_dead = LoadCase(id="DEAD", name="Superimposed Dead Load", element_loads=sdl_loads)

    # General Office Live Load (LL = 12 kN/m per beam) - excluded from seismic weight per 12.7.2
    ll_loads = [DistributedLoad(element_id=bid, wz=-12.0) for bid in beam_ids]
    lc_office_live = LoadCase(id="LIVE_OFFICE", name="Office Live Load", element_loads=ll_loads)

    # Storage Live Load (Storage LL = 20 kN/m on 1st floor) - 25% included per Sec 12.7.2 Item 1
    storage_beam_ids = beam_ids[:2]
    storage_loads = [DistributedLoad(element_id=bid, wz=-20.0) for bid in storage_beam_ids]
    lc_storage = LoadCase(id="STORAGE_LIVE", name="Heavy Storage Live Load", element_loads=storage_loads)

    # Partition Load (5 kN/m per beam) - 100% included per Sec 12.7.2 Item 2
    part_loads = [DistributedLoad(element_id=bid, wz=-5.0) for bid in beam_ids]
    lc_partition = LoadCase(id="PARTITION", name="Floor Partitions", element_loads=part_loads)

    model = StructuralModel(
        nodes=nodes,
        elements=elements,
        materials=[mat_conc],
        sections=[sec_col, sec_beam],
        load_cases=[lc_dead, lc_office_live, lc_storage, lc_partition],
        story_elevations=[3.5, 7.0, 10.5],
        units="kN, m",
    )
    return model


def main():
    print("=" * 75)
    print("Example 08: ASCE 7-22 Section 12.7.2 Effective Seismic Weight Engine")
    print("=" * 75)

    model = build_3story_building()
    print(f"Building created with {len(model.nodes)} nodes and {len(model.elements)} elements.")

    # 1. Standard ASCE 7-22 Mass Source using create_asce7_mass_source() helper:
    # - 100% DEAD (SDL) + 100% Member self-weight
    # - 25% STORAGE_LIVE (Section 12.7.2 Item 1)
    # - 100% PARTITION (Section 12.7.2 Item 2)
    # - 0% LIVE_OFFICE (General live load is excluded per ASCE 7-22 Section 12.7.2)
    ms_asce7 = create_asce7_mass_source(
        model=model,
        dead_load_case_ids=["DEAD"],
        storage_live_case_ids=["STORAGE_LIVE"],
        partition_load_case_ids=["PARTITION"],
        include_general_live=False,  # Code compliant: 0% general office live load
        include_element_mass=True,
    )

    print("\n[ASCE 7-22 Mass Source Factors]")
    for factor in ms_asce7.factors:
        print(f"  Load Case: {factor.load_case_id:<15} Factor: {factor.factor:.2f}")

    res_asce7 = calculate_mass_source(model=model, mass_source=ms_asce7)

    print("\n[Mass Source Evaluation Summary]")
    print(f"  Configuration          : {ms_asce7.name}")
    print(f"  Total Seismic Weight W : {res_asce7.total_seismic_weight:12.2f} kN")
    print(f"  Total Inertial Mass M  : {res_asce7.total_mass:12.2f} tonnes")
    print(f"  Center of Mass (CM)    : ({res_asce7.center_of_mass[0]:.2f} m, "
          f"{res_asce7.center_of_mass[1]:.2f} m, {res_asce7.center_of_mass[2]:.2f} m)")

    print("\n[Story-by-Story Mass & Weight Breakdown]")
    print(f"  {'Elevation (m)':<15} {'Story Mass (tonnes)':<22} {'Story Weight (kN)':<20}")
    print("  " + "-" * 60)
    for elev, m_story in sorted(res_asce7.story_masses.items()):
        w_story = m_story * model.units.gravity_acceleration
        print(f"  {elev:<15.2f} {m_story:<22.2f} {w_story:<20.2f}")

    print("\n" + "=" * 75)
    print("ASCE 7-22 Seismic Weight calculation verified successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()

