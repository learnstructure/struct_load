"""
Example 06: ASCE 7 / ACI 318 Factored Load Combinations Generator
=================================================================
This example demonstrates how `struct_load` generates code-compliant
factored load combinations for strength (LRFD) and serviceability (ASD) design:
1. LRFD Strength design load combinations per ASCE 7 Chapter 2 and ACI 318 Chapter 5.
2. ASD Allowable Stress / Serviceability load combinations.
3. 3D Orthogonal Seismic combinations (1.0 Ex +/- 0.3 Ey and 0.3 Ex +/- 1.0 Ey)
   per ASCE 7 Section 12.5.
4. Programmatic inspection and mathematical equation formatting of generated combinations.
"""

from pathlib import Path
import sys

# Ensure root repository and packages are in import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from struct_core import LoadCombination
from struct_load.codes.asce7_combinations import generate_asce7_combinations


def format_combination_equation(comb: LoadCombination) -> str:
    """Format a LoadCombination object into an equation string."""
    terms = []
    for f in comb.factors:
        sign = "+" if f.factor >= 0 else "-"
        mag = abs(f.factor)
        terms.append(f"{f.factor:+.2f}*{f.load_case_id}")
    eq = " ".join(terms)
    if eq.startswith("+"):
        eq = eq[1:].strip()
    return eq


def demo_lrfd_combinations():
    print("=" * 70)
    print("1. LRFD (Strength Design) Combinations (ASCE 7 Ch 2 / ACI 318 Ch 5)")
    print("=" * 70)

    lrfd_combs = generate_asce7_combinations(
        dead_case_id="Dead",
        live_case_id="Live",
        wind_case_id="Wind_X",
        seismic_case_id="EQ_X",
        roof_live_case_id="RoofLive",
        snow_case_id="Snow",
        design_method="LRFD",
    )

    print(f"Generated {len(lrfd_combs)} LRFD combinations:\n")
    print(f"{'ID':<25} | {'Equation':<45}")
    print("-" * 75)
    for c in lrfd_combs:
        eq_str = format_combination_equation(c)
        print(f"{c.id:<25} | {eq_str:<45}")


def demo_asd_combinations():
    print("\n" + "=" * 70)
    print("2. ASD (Allowable Stress / Serviceability) Combinations")
    print("=" * 70)

    asd_combs = generate_asce7_combinations(
        dead_case_id="Dead",
        live_case_id="Live",
        wind_case_id="Wind_X",
        seismic_case_id="EQ_X",
        snow_case_id="Snow",
        design_method="ASD",
    )

    print(f"Generated {len(asd_combs)} ASD combinations:\n")
    print(f"{'ID':<25} | {'Equation':<45}")
    print("-" * 75)
    for c in asd_combs:
        eq_str = format_combination_equation(c)
        print(f"{c.id:<25} | {eq_str:<45}")


def demo_orthogonal_3d_seismic():
    print("\n" + "=" * 70)
    print("3. 3D Orthogonal Seismic Combinations (ASCE 7 Sec 12.5: 100% + 30%)")
    print("=" * 70)

    # When a 3D building undergoes earthquake excitation, lateral forces occur simultaneously
    # in orthogonal directions. ASCE 7 permits either 3D dynamic analysis or combination
    # of 100% force in one direction + 30% force in the perpendicular direction:
    # (1.0 Ex +/- 0.3 Ey) and (0.3 Ex +/- 1.0 Ey)
    ortho_combs = generate_asce7_combinations(
        dead_case_id="Dead",
        live_case_id="Live",
        seismic_case_id="EQ_X",
        seismic_y_case_id="EQ_Y",
        orthogonal_combinations=True,
        design_method="LRFD",
    )

    seismic_ortho = [c for c in ortho_combs if "EQ_X" in str(c.id) and "EQ_Y" in str(c.id)]
    print(f"Generated {len(seismic_ortho)} 3D orthogonal seismic combination cases:\n")
    print(f"{'ID':<30} | {'Equation':<45}")
    print("-" * 78)
    for c in seismic_ortho:
        eq_str = format_combination_equation(c)
        print(f"{c.id:<30} | {eq_str:<45}")


def demo_summary():
    print("\n" + "=" * 70)
    print("4. Integration Summary")
    print("=" * 70)
    print("The generated `struct_core.LoadCombination` objects can be assigned directly")
    print("to `StructuralModel.load_combinations` and consumed by structural analysis solvers")
    print("(such as `fem2d`, `fem3d`, or OpenSees) and design checkers (such as `aci318`).")


def main():
    demo_lrfd_combinations()
    demo_asd_combinations()
    demo_orthogonal_3d_seismic()
    demo_summary()
    print("\n" + "=" * 70)
    print("Example 06 completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
