"""
Example 02: Tributary and Area Load Distribution Engine
======================================================
This example demonstrates how `struct_load` translates surface area pressures
into line loads on supporting structural framing members:
1. 1D tributary strip load calculation (2D and 3D).
2. Two-way slab 45-degree yield-line distribution (triangular and trapezoidal loads
   converted into equivalent uniform line loads for flexure and shear).
3. Automated AreaLoad distribution on a 3D structural model panel using
   `distribute_area_loads_to_beams`.
"""

from pathlib import Path
import sys

# Ensure root repository and packages are in import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from struct_core import (
    AreaLoad,
    BeamElement,
    ElasticMaterial,
    LoadCase,
    Node,
    RectangularSection,
    StructuralModel,
)
from struct_load.engine.tributary import (
    apply_floor_load_to_frame,
    distribute_1d_tributary,
    distribute_area_loads_to_beams,
    distribute_tributary_load,
    distribute_two_way_slab,
)


def demo_1d_tributary():
    print("=" * 70)
    print("1. 1D Tributary Strip Distribution")
    print("=" * 70)

    pressure_kpa = 3.0  # 3.0 kPa (e.g. office live load + partitions)
    trib_width_m = 4.5  # 4.5 m bay spacing

    # 2D Plane Frame (gravity down along -Y)
    wx_2d, wy_2d = distribute_1d_tributary(
        pressure=pressure_kpa, tributary_width=trib_width_m, direction="-y"
    )
    print(f"Surface Pressure:       {pressure_kpa:.2f} kPa")
    print(f"Tributary Width:        {trib_width_m:.2f} m")
    print(f"2D Plane Frame Vector:  wx = {wx_2d:.2f} kN/m, wy = {wy_2d:.2f} kN/m (Gravity along -Y)")

    # 3D Space Frame (gravity down along -Z)
    wx_3d, wy_3d, wz_3d = distribute_tributary_load(
        pressure=pressure_kpa, tributary_width=trib_width_m, direction="-z"
    )
    print(
        f"3D Space Frame Vector:  wx = {wx_3d:.2f} kN/m, wy = {wy_3d:.2f} kN/m, wz = {wz_3d:.2f} kN/m (Gravity along -Z)"
    )


def demo_two_way_yield_line():
    print("\n" + "=" * 70)
    print("2. Two-Way Slab Yield-Line Tributary Distribution (45 deg)")
    print("=" * 70)

    # Rectangular floor slab panel: 4.0 m x 6.0 m
    short_span = 4.0  # m
    long_span = 6.0   # m
    pressure = 5.0    # 5.0 kPa total uniform pressure

    aspect_ratio = long_span / short_span
    print(f"Panel Geometry: Short span S = {short_span:.2f} m, Long span L = {long_span:.2f} m")
    print(f"Aspect Ratio (m = L/S): {aspect_ratio:.2f}")
    print(f"Uniform Surface Pressure: {pressure:.2f} kPa\n")

    # Flexural design equivalent loads (same midspan bending moment)
    w_peak, w_eq_short_m, w_eq_long_m = distribute_two_way_slab(
        pressure=pressure, short_span=short_span, long_span=long_span, design_action="moment"
    )

    print("--- Equivalent Uniform Loads for Bending Moment ---")
    print(f"Peak Load Intensity (Apex of Triangle/Trapezoid): {w_peak:.2f} kN/m")
    print(f"Short Beam (Triangular tributary area):         w_eq = {w_eq_short_m:.2f} kN/m")
    print(f"Long Beam  (Trapezoidal tributary area):        w_eq = {w_eq_long_m:.2f} kN/m")

    # Shear design equivalent loads (same total end reaction / shear)
    w_peak_v, w_eq_short_v, w_eq_long_v = distribute_two_way_slab(
        pressure=pressure, short_span=short_span, long_span=long_span, design_action="shear"
    )

    print("\n--- Equivalent Uniform Loads for Shear / Support Reactions ---")
    print(f"Short Beam (Triangular tributary area):         w_eq = {w_eq_short_v:.2f} kN/m")
    print(f"Long Beam  (Trapezoidal tributary area):        w_eq = {w_eq_long_v:.2f} kN/m")


def demo_area_load_distribution_on_model():
    print("\n" + "=" * 70)
    print("3. Automated AreaLoad Materialization on 3D Structural Model")
    print("=" * 70)

    # Build a single-story 3D floor bay:
    # 4 corner columns and 4 perimeter floor beams at elevation Z = 3.5 m
    # Bay dimensions: X in [0, 6], Y in [0, 4], Z in [0, 3.5]
    model = StructuralModel()

    # Base nodes (Z = 0)
    model.nodes.extend([
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=6.0, y=0.0, z=0.0),
        Node(id=3, x=6.0, y=4.0, z=0.0),
        Node(id=4, x=0.0, y=4.0, z=0.0),
    ])

    # Floor nodes (Z = 3.5 m)
    model.nodes.extend([
        Node(id=5, x=0.0, y=0.0, z=3.5),
        Node(id=6, x=6.0, y=0.0, z=3.5),
        Node(id=7, x=6.0, y=4.0, z=3.5),
        Node(id=8, x=0.0, y=4.0, z=3.5),
    ])

    # Materials and sections
    conc = ElasticMaterial(id="Concrete_C30", E=30e6, nu=0.2, rho=2400.0)
    sec_col = RectangularSection(id="Col_40x40", b=0.40, h=0.40)
    sec_bm = RectangularSection(id="Beam_30x50", b=0.30, h=0.50)
    model.materials.append(conc)
    model.sections.extend([sec_col, sec_bm])

    # Columns
    model.elements.extend([
        BeamElement(id=1, type="beam", start_node=1, end_node=5, material_id=conc.id, section_id=sec_col.id),
        BeamElement(id=2, type="beam", start_node=2, end_node=6, material_id=conc.id, section_id=sec_col.id),
        BeamElement(id=3, type="beam", start_node=3, end_node=7, material_id=conc.id, section_id=sec_col.id),
        BeamElement(id=4, type="beam", start_node=4, end_node=8, material_id=conc.id, section_id=sec_col.id),
    ])

    # Perimeter floor beams:
    # E5: (5 -> 6) along X, span = 6m (long beam)
    # E6: (6 -> 7) along Y, span = 4m (short beam)
    # E7: (7 -> 8) along X, span = 6m (long beam)
    # E8: (8 -> 5) along Y, span = 4m (short beam)
    model.elements.extend([
        BeamElement(id=5, type="beam", start_node=5, end_node=6, material_id=conc.id, section_id=sec_bm.id),
        BeamElement(id=6, type="beam", start_node=6, end_node=7, material_id=conc.id, section_id=sec_bm.id),
        BeamElement(id=7, type="beam", start_node=7, end_node=8, material_id=conc.id, section_id=sec_bm.id),
        BeamElement(id=8, type="beam", start_node=8, end_node=5, material_id=conc.id, section_id=sec_bm.id),
    ])

    # Define a 2-way AreaLoad over nodes [5, 6, 7, 8] with pressure = 4.0 kPa
    area_load = AreaLoad(
        id="Floor_Live_Bay1",
        node_ids=[5, 6, 7, 8],
        pressure=4.0,  # 4.0 kPa
        load_type="live",
        one_way=False,
    )
    live_case = LoadCase(
        id="Live",
        name="Occupancy Live Load",
        area_loads=[area_load],
    )
    model.load_cases.append(live_case)

    print(f"Model created with {len(model.elements)} elements and 1 AreaLoad (4.0 kPa, two-way).")
    print(f"Elements before area load distribution: {len(live_case.element_loads)} distributed loads.")

    # Distribute area loads into distributed line loads on perimeter beams
    count = distribute_area_loads_to_beams(
        model=model,
        target_case_id="Live",
        design_action="moment",
        clear_area_loads=False,
    )

    print(f"\nSuccessfully distributed {count} AreaLoad(s) into {len(live_case.element_loads)} beam line loads:")
    for dist_load in live_case.element_loads:
        elem = next(e for e in model.elements if e.id == dist_load.element_id)
        print(
            f"  Beam #{elem.id:02d} (Nodes {elem.start_node}->{elem.end_node}): "
            f"wz = {dist_load.wz:7.2f} kN/m  (wx={dist_load.wx:.1f}, wy={dist_load.wy:.1f})"
        )


def main():
    demo_1d_tributary()
    demo_two_way_yield_line()
    demo_area_load_distribution_on_model()
    print("\n" + "=" * 70)
    print("Example 02 completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
