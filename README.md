# struct_load

A building code load determination and management package for the [Structural Engineering Ecosystem](../README.md).

`struct_load` bridges architectural criteria (occupancy, story heights, bay spacing, geographic location) and deterministic structural engineering loads. It provides code table lookups (ASCE 7-16/22 & IBC Chapter 16), automated element self-weight calculation, tributary area distribution, seismic ELF & wind pressure calculators, and standard LRFD/ASD factored load combination generation.

> **AI Philosophy:** The AI decides *what* building to load (e.g. occupancy = "office", location = "Reno, NV"); `struct_load` deterministically computes *how* the loads are calculated per building code equations.

---

## Features

- **ASCE 7 & IBC Occupancy Live Loads**:
  - Full database of standard occupancies (Offices, Residential, Classrooms, Corridors, Retail, Storage, Assembly, Parking, Roof).
  - Built-in partition load allowances (15 psf / 0.72 kPa for offices).
  - ASCE 7 Section 4.7 live load reduction calculator ($L = L_0 (0.25 + 15/\sqrt{K_{LL} A_T})$).
- **Self-Weight Engine**:
  - Scans `struct_core.StructuralModel.elements` and calculates gravity line loads ($w = -\rho A g$) directly from cross-section dimensions and material density.
- **Tributary Distribution Engine**:
  - 2D frames: maps floor surface pressures ($q \text{ in kPa or psf}$) onto horizontal beam elements via bay tributary width ($w = q \times B_{\text{trib}}$).
  - Two-way slabs: 45° yield-line distribution (triangular and trapezoidal load profiles).
- **ASCE 7 Seismic ELF Procedure**:
  - Calculates approximate fundamental period $T_a$, seismic response coefficient $C_s = S_{DS} / (R / I_e)$, base shear $V = C_s W$, and vertical force distribution $F_x$ along building height.
- **ASCE 7 Wind MWFRS Procedure**:
  - Directional procedure computing velocity pressure $q_z$, exposure coefficient $K_z$ (Exposure B, C, D), and story lateral forces.
- **Standard Factored Load Combinations**:
  - Generates ACI 318 / ASCE 7 Strength (LRFD) and Allowable Stress (ASD) combinations ($1.4D$, $1.2D + 1.6L$, $1.2D + 1.0L + 1.0W$, $1.2D + 1.0L + 1.0E$, $0.9D \pm 1.0W$, $0.9D \pm 1.0E$) as `struct_core.LoadCombination` objects.

---

## Installation

```bash
conda activate structeng
pip install -e struct_load/
```

---

## Quick Start

### 1. High-Level Loading via `LoadGenerator`

```python
from struct_core import Project, StructuralModel
from struct_load import LoadGenerator

# LoadGenerator automates dead, live, and combination creation
generator = LoadGenerator(unit_system="si")

# Apply Office gravity loads with 5 m tributary width and self-weight
dead_case, live_case = generator.apply_gravity_loads(
    project,
    occupancy="office",
    tributary_width=5.0,
    superimposed_dead=1.5,
    include_self_weight=True,
)

# Apply ASCE 7 seismic ELF lateral loads
seismic_case, elf_res = generator.apply_lateral_seismic(
    project,
    sds=0.75,
    sd1=0.40,
    r_factor=8.0,
    tributary_width=5.0,
)

# Generate factored ACI 318 / ASCE 7 LRFD combinations
combs = generator.generate_combinations(project, design_method="LRFD")
print(f"Generated {len(combs)} load combinations.")
```

### 2. Direct Code Lookup & Calculators

```python
from struct_load import get_occupancy_live_load, calculate_live_load_reduction

# Look up uniform live load for office in SI (kPa)
q_office = get_occupancy_live_load("office", include_partitions=True, unit_system="si")
print(f"Office live load: {q_office} kPa")  # 3.12 kPa (2.40 + 0.72)

# Check live load reduction for 500 ft² tributary area
q_reduced = calculate_live_load_reduction(
    L0=50.0,
    tributary_area=500.0,
    k_ll=2.0,
    unit_system="us",
)
print(f"Reduced live load: {q_reduced} psf")
```

---

## Running Tests

```bash
conda activate structeng
pytest struct_load/tests -v
```

---

## Architecture Compliance

- **Inputs**: Reads `struct_core.StructuralModel` (nodes, elements, sections, materials).
- **Outputs**: Produces standard `struct_core.LoadCase` and `struct_core.LoadCombination` objects.
- **No Solver Logic**: Does not solve stiffness matrices or check rebar; strictly handles load determination.
