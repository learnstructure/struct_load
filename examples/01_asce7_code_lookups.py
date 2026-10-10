"""Example 1 — ASCE 7 & IBC Building Code Lookups and Live Load Reduction.

Demonstrates:
  1. Querying ASCE 7-16/22 & IBC Chapter 16 minimum uniform live loads across occupancies.
  2. Automatic unit handling (US Customary psf vs SI Metric kPa).
  3. Office partition load additions per ASCE 7 Section 4.3.2.
  4. Live load reduction calculations per ASCE 7 Section 4.7 (Eq. 4.7-1).
  5. Flat roof snow load determination per ASCE 7 Chapter 7.
  6. Minimum roof live load calculations per ASCE 7 Section 4.5.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure struct_load and ecosystem packages are discoverable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from struct_load import (
    calculate_flat_roof_snow_load,
    calculate_live_load_reduction,
    calculate_roof_live_load,
    get_occupancy_live_load,
    lookup_occupancy,
)


def run_occupancy_lookups() -> None:
    print("=" * 72)
    print(" 1. ASCE 7-16/22 & IBC MINIMUM UNIFORM OCCUPANCY LIVE LOADS")
    print("=" * 72)
    occupancies = [
        "office",
        "residential",
        "classroom",
        "corridor_first_floor",
        "retail",
        "storage_light",
        "storage_heavy",
        "assembly",
        "roof",
    ]

    header = f"{'Occupancy Key':<22} | {'Uniform (psf)':<14} | {'Uniform (kPa)':<14} | {'Partition':<10}"
    print(header)
    print("-" * len(header))

    for occ_key in occupancies:
        occ = lookup_occupancy(occ_key)
        part_str = f"{occ.partition_psf:.0f} psf" if occ.partition_psf > 0 else "None"
        print(f"{occ.key:<22} | {occ.uniform_psf:>8.1f} psf  | {occ.uniform_kpa:>8.2f} kPa  | {part_str:<10}")

    # Office with partition comparison
    off_base_us = get_occupancy_live_load("office", include_partitions=False, unit_system="us")
    off_full_us = get_occupancy_live_load("office", include_partitions=True, unit_system="us")
    off_base_si = get_occupancy_live_load("office", include_partitions=False, unit_system="si")
    off_full_si = get_occupancy_live_load("office", include_partitions=True, unit_system="si")

    print("\nOffice Partition Allowance Comparison (ASCE 7 Sec 4.3.2):")
    print(f"  US: Base = {off_base_us:.1f} psf -> With Partitions (+15 psf) = {off_full_us:.1f} psf")
    print(f"  SI: Base = {off_base_si:.2f} kPa -> With Partitions (+0.72 kPa) = {off_full_si:.2f} kPa")


def run_live_load_reduction() -> None:
    print("\n" + "=" * 72)
    print(" 2. ASCE 7 SECTION 4.7 LIVE LOAD REDUCTION (Eq. 4.7-1)")
    print("    L = L0 * (0.25 + 15 / sqrt(K_LL * A_T))")
    print("=" * 72)

    l0_psf = 50.0  # Unreduced office live load (psf)
    tributary_areas = [150.0, 300.0, 400.0, 600.0, 1000.0, 1500.0]  # ft^2

    members = [
        ("Interior Column (1-story)", 4.0, False),
        ("Interior Column (multi-story)", 4.0, True),
        ("Beam / Girder", 2.0, False),
        ("Slab Panel", 1.0, False),
    ]

    for member_name, k_ll, is_multi in members:
        print(f"\nMember Type: {member_name} (K_LL = {k_ll:.1f})")
        print(f"  {'Tributary Area':<16} | {'Influence K_LL*At':<18} | {'Reduced L':<12} | {'Reduction %':<12}")
        print("  " + "-" * 64)
        for at in tributary_areas:
            ai = k_ll * at
            l_red = calculate_live_load_reduction(
                L0=l0_psf,
                tributary_area=at,
                k_ll=k_ll,
                is_multi_story=is_multi,
                unit_system="us",
            )
            pct = ((l0_psf - l_red) / l0_psf) * 100.0
            print(f"  {at:>8.0f} sq ft   | {ai:>10.0f} sq ft     | {l_red:>7.1f} psf  | {pct:>7.1f}%")


def run_roof_snow_and_live() -> None:
    print("\n" + "=" * 72)
    print(" 3. ASCE 7 CHAPTER 7 SNOW & SECTION 4.5 ROOF LIVE LOADS")
    print("=" * 72)

    # 1. Flat Roof Snow Load
    # pf = 0.7 * Ce * Ct * Is * pg, subject to code minimum pm
    pg_values_us = [15.0, 25.0, 40.0, 60.0]  # Ground snow load in psf
    print("Flat Roof Snow Load (pf) across ground snow loads (Ce=1.0, Ct=1.0, Is=1.0):")
    print(f"  {'Ground Snow pg':<16} | {'Flat Roof pf':<14} | {'Code Min pm':<14}")
    print("  " + "-" * 48)
    for pg in pg_values_us:
        pf = calculate_flat_roof_snow_load(pg=pg, ce=1.0, ct=1.0, is_factor=1.0, unit_system="us")
        pm = pg if pg <= 20.0 else 20.0
        print(f"  {pg:>8.1f} psf     | {pf:>8.1f} psf    | {pm:>8.1f} psf")

    # 2. Roof Live Load
    # Lr = 20 * R1 * R2 (between 12 psf and 20 psf)
    print("\nMinimum Roof Live Load (Lr) across tributary areas:")
    areas = [100.0, 250.0, 500.0, 800.0]  # ft^2
    for a in areas:
        lr_flat = calculate_roof_live_load(tributary_area=a, slope_rise_over_12=0.0, unit_system="us")
        lr_sloped = calculate_roof_live_load(tributary_area=a, slope_rise_over_12=6.0, unit_system="us")
        print(f"  Area = {a:>5.0f} ft^2 -> Flat Roof (slope=0) = {lr_flat:.1f} psf | Sloped (6:12) = {lr_sloped:.1f} psf")
    print("=" * 72)


def main() -> None:
    run_occupancy_lookups()
    run_live_load_reduction()
    run_roof_snow_and_live()


if __name__ == "__main__":
    main()
