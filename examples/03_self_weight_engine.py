"""
Example 03: Element Self-Weight Calculation and Materialization Engine
=====================================================================
This example demonstrates how `struct_load` determines and materializes
element self-weight distributed loads from cross-section geometry and material density:
1. Section area and unit self-weight computation in SI (kN/m) and US (kip/ft) units.
2. Self-weight materialization on a 2D plane frame (gravity along -Y).
3. Self-weight materialization on a 3D space frame (gravity along -Z).
4. Safe load materialization workflow ensuring `include_self_weight` is set to False
   after generation to guarantee self-weight is never double-counted.
"""

from pathlib import Path
import sys

# Ensure root repository and packages are in import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from struct_core import (
    BeamElement,
    CircularSection,
    ElasticMaterial,
    GeneralSection,
    LoadCase,
    Metadata,
    Node,
    Project,
    RectangularSection,
    StructuralModel,
    Units,
)
from struct_load.engine.self_weight import (
    apply_self_weight,
    calculate_section_area,
    calculate_self_weight,
)


def demo_unit_self_weight():
    print("=" * 70)
    print("1. Section Area and Linear Self-Weight Calculations")
    print("=" * 70)

    # SI Sections (Concrete: rho = 2400 kg/m3, Steel: rho = 7850 kg/m3)
    conc_rc = RectangularSection(id="RC_300x500", b=0.30, h=0.50)
    conc_circ = CircularSection(id="RC_D400", d=0.40)
    steel_w = GeneralSection(id="W14x90_equiv", A=0.0171, I3=4.16e-4, I2=1.51e-4)

    area_rc = calculate_section_area(conc_rc)
    area_circ = calculate_section_area(conc_circ)
    area_steel = calculate_section_area(steel_w)

    rho_conc = 2400.0  # kg/m3
    rho_steel = 7850.0  # kg/m3
    units_si = Units(force="kN", length="m")
    g = units_si.gravity_acceleration

    w_rc = (rho_conc * area_rc * g) / 1000.0
    w_circ = (rho_conc * area_circ * g) / 1000.0
    w_steel = (rho_steel * area_steel * g) / 1000.0

    print("SI System (kN/m):")
    print(f"  Rectangular 300x500 mm: Area = {area_rc:.4f} m^2, w = {w_rc:.3f} kN/m")
    print(f"  Circular D=400 mm:     Area = {area_circ:.4f} m^2, w = {w_circ:.3f} kN/m")
    print(f"  Steel Section:         Area = {area_steel:.4f} m^2, w = {w_steel:.3f} kN/m")

    # US Customary Section (Concrete: 150 pcf)
    conc_us = RectangularSection(id="RC_12x24_in", b=12.0, h=24.0)
    area_us = calculate_section_area(conc_us)  # 288 in^2
    # w (kip/ft) = 150 lb/ft3 * (288 / 144) ft2 / 1000 = 0.300 kip/ft
    w_us = 150.0 * (area_us / 144.0) / 1000.0
    print("\nUS Customary System:")
    print(f"  Rectangular 12x24 in:   Area = {area_us:.1f} in^2, w = {w_us:.3f} kip/ft")


def demo_2d_plane_frame_self_weight():
    print("\n" + "=" * 70)
    print("2. 2D Plane Frame Self-Weight Materialization (Gravity along -Y)")
    print("=" * 70)

    # Portal Frame: Span = 6.0 m, Story Height = 3.5 m
    model = StructuralModel(units="kN, m")

    # Nodes (X, Y)
    model.nodes.extend([
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=6.0, y=0.0, z=0.0),
        Node(id=3, x=0.0, y=3.5, z=0.0),
        Node(id=4, x=6.0, y=3.5, z=0.0),
    ])

    mat = ElasticMaterial(id="Conc", E=25e6, nu=0.2, rho=2400.0)
    col_sec = RectangularSection(id="Col_400x400", b=0.40, h=0.40)  # Area = 0.16 m2 -> w = 3.766 kN/m
    bm_sec = RectangularSection(id="Bm_300x500", b=0.30, h=0.50)   # Area = 0.15 m2 -> w = 3.530 kN/m
    model.materials.append(mat)
    model.sections.extend([col_sec, bm_sec])

    # Elements
    model.elements.extend([
        BeamElement(id=1, type="beam", start_node=1, end_node=3, material_id=mat.id, section_id=col_sec.id),
        BeamElement(id=2, type="beam", start_node=2, end_node=4, material_id=mat.id, section_id=col_sec.id),
        BeamElement(id=3, type="beam", start_node=3, end_node=4, material_id=mat.id, section_id=bm_sec.id),
    ])

    # Initial Dead load case flagged with include_self_weight=True
    dead_case = LoadCase(
        id="Dead",
        name="Dead Load Case",
        include_self_weight=True,
    )
    model.load_cases.append(dead_case)

    print(f"Before apply_self_weight: include_self_weight = {dead_case.include_self_weight}")
    print(f"Distributed loads in Dead case: {len(dead_case.element_loads)}")

    # Apply self-weight directly to the model
    apply_self_weight(model, units="kN, m", direction="-y")

    updated_case = next(lc for lc in model.load_cases if lc.id == "Dead")
    print(f"\nAfter apply_self_weight:")
    print(f"  include_self_weight flag set to: {updated_case.include_self_weight} (preventing double counting)")
    print(f"  Generated distributed loads: {len(updated_case.element_loads)}")

    for dl in updated_case.element_loads:
        elem = next(e for e in model.elements if e.id == dl.element_id)
        sec = next(s for s in model.sections if s.id == elem.section_id)
        print(
            f"  Element #{elem.id} ({sec.id}): wy = {dl.wy:6.3f} kN/m, "
            f"wx = {dl.wx:.1f}, wz = {dl.wz:.1f}"
        )


def demo_3d_space_frame_self_weight():
    print("\n" + "=" * 70)
    print("3. 3D Space Frame Self-Weight Materialization on Project (Gravity along -Z)")
    print("=" * 70)

    # 3D Bay: 6m x 5m in plan, 4m height
    model = StructuralModel(units="kN, m")

    # Ground nodes
    model.nodes.extend([
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=6.0, y=0.0, z=0.0),
        Node(id=3, x=6.0, y=5.0, z=0.0),
        Node(id=4, x=0.0, y=5.0, z=0.0),
    ])
    # Roof nodes
    model.nodes.extend([
        Node(id=5, x=0.0, y=0.0, z=4.0),
        Node(id=6, x=6.0, y=0.0, z=4.0),
        Node(id=7, x=6.0, y=5.0, z=4.0),
        Node(id=8, x=0.0, y=5.0, z=4.0),
    ])

    mat = ElasticMaterial(id="Conc_C35", E=32e6, nu=0.2, rho=2450.0)
    sec_col = RectangularSection(id="Col_500x500", b=0.50, h=0.50)
    sec_bm = RectangularSection(id="Bm_300x600", b=0.30, h=0.60)
    model.materials.append(mat)
    model.sections.extend([sec_col, sec_bm])

    # 4 columns
    for i, (b_id, t_id) in enumerate([(1, 5), (2, 6), (3, 7), (4, 8)], start=1):
        model.elements.append(
            BeamElement(id=i, type="beam", start_node=b_id, end_node=t_id, material_id=mat.id, section_id=sec_col.id)
        )

    # 4 roof beams
    for i, (n1, n2) in enumerate([(5, 6), (6, 7), (7, 8), (8, 5)], start=5):
        model.elements.append(
            BeamElement(id=i, type="beam", start_node=n1, end_node=n2, material_id=mat.id, section_id=sec_bm.id)
        )

    # Wrap inside struct_core.Project with canonical units
    project = Project(
        model=model,
        units="kN, m",
        metadata=Metadata(title="3D Portal Building Frame"),
    )

    # Materialize self weight via Project wrapper
    apply_self_weight(project)

    dead_case = next(lc for lc in project.model.load_cases if any(k in str(lc.id).lower() for k in ("selfweight", "dead")))
    print(f"Project Units detected automatically: SI")
    print(f"Generated {len(dead_case.element_loads)} 3D self-weight line loads (Case: {dead_case.id}):")

    total_weight = 0.0
    nodes_map = {n.id: n for n in model.nodes}
    for dl in dead_case.element_loads:
        elem = next(e for e in model.elements if e.id == dl.element_id)
        n_a, n_b = nodes_map[elem.start_node], nodes_map[elem.end_node]
        L = ((n_b.x - n_a.x)**2 + (n_b.y - n_a.y)**2 + (n_b.z - n_a.z)**2) ** 0.5
        w_mag = abs(dl.wz)
        total_elem_w = w_mag * L
        total_weight += total_elem_w
        print(f"  Element #{elem.id:02d} ({elem.section_id}): L = {L:4.1f}m, wz = {dl.wz:6.3f} kN/m, total = {total_elem_w:6.2f} kN")

    g_accel = project.units.gravity_acceleration
    print(f"\nTotal Structure Self-Weight = {total_weight:.2f} kN ({total_weight / g_accel:.2f} tonnes)")


def main():
    demo_unit_self_weight()
    demo_2d_plane_frame_self_weight()
    demo_3d_space_frame_self_weight()
    print("\n" + "=" * 70)
    print("Example 03 completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
