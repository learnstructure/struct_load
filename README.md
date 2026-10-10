# struct_load

A building code load determination and management package for the [Structural Engineering Ecosystem](../README.md).

`struct_load` bridges architectural criteria (occupancy, story heights, bay spacing, geographic location) and deterministic structural engineering loads. It provides code table lookups (ASCE 7-16/22 & IBC Chapter 16), automated element self-weight calculation, tributary area distribution, seismic ELF & wind pressure calculators, roof snow & live load determination, and standard LRFD/ASD factored load combination generation.

> **AI Philosophy:** The AI decides *what* building to load (e.g. occupancy = "office", location = "Reno, NV"); `struct_load` deterministically computes *how* the loads are calculated per building code equations.

---

## Features

- **2D & 3D Dimension & Coordinate Awareness**:
  - Automatically senses model dimensionality (`2d` plane frame vs `3d` space frame).
  - In 2D: Story elevations along $Y$, gravity along $-Y$ ($w_y$), lateral loads along $X$ ($f_x$).
  - In 3D: Story elevations along $Z$, gravity along $-Z$ ($w_z$), lateral loads along $X$ ($f_x$) and $Y$ ($f_y$).
  - Automatic floor elevation detection from beam levels above foundation.
- **Unit System Auto-Inference**:
  - Automatically detects US Customary (`ft`, `in`, `kip`, `psf`) vs SI (`m`, `kN`, `kPa`) directly from `project.metadata.units`.
- **Panel & `AreaLoad` Tributary Distribution**:
  - Maps floor surface pressures directly from `struct_core.AreaLoad` onto perimeter supporting beams.
  - **One-way slabs**: tributary strips ($w = q \cdot S/2$) applied to beams perpendicular to span axis; parallel beams receive 0 load.
  - **Two-way slabs**: 45° yield-line distribution. Computes triangular equivalent uniform loads on short beams and trapezoidal equivalent uniform loads on long beams.
- **ASCE 7 & IBC Occupancy Live Loads**:
  - Full database of standard occupancies (Offices, Residential, Classrooms, Corridors, Retail, Storage, Assembly, Parking, Roof).
  - Built-in partition load allowances (15 psf / 0.72 kPa for offices).
  - ASCE 7 Section 4.7 live load reduction calculator ($L = L_0 (0.25 + 15/\sqrt{K_{LL} A_T})$).
- **Self-Weight Engine**:
  - Scans `struct_core.StructuralModel.elements` and calculates gravity line loads ($w = -\rho A g$) directly from cross-section dimensions and material density.
- **Exact Model-Derived Seismic Weights & 3D ELF**:
  - Calculates exact story seismic weights $W_i$ directly from model Dead loads and member self-weights.
  - Directional seismic loads (`EQ_X`, `EQ_Y`, or `direction="both"`).
  - Calculates approximate fundamental period $T_a$, seismic response coefficient $C_s$, base shear $V = C_s W$, and vertical force distribution $F_x, F_y$ along building height.
- **ASCE 7 Snow & Roof Live Loads**:
  - ASCE 7 Chapter 7 flat roof snow load ($p_f = 0.7 C_e C_t I_s p_g$, bounded by code minimum $p_m$).
  - ASCE 7 Section 4.5 minimum roof live load ($L_r = 20 R_1 R_2$).
  - Applied specifically to roof level framing beams.
- **ASCE 7 Wind MWFRS Procedure**:
  - Directional procedure computing velocity pressure $q_z$, exposure coefficient $K_z$ (Exposure B, C, D), and story lateral forces along $X$ and $Y$.
- **Standard Factored Load Combinations**:
  - Generates ACI 318 / ASCE 7 Strength (LRFD) and Allowable Stress (ASD) combinations ($1.4D$, $1.2D + 1.6L + 0.5(S \text{ or } L_r)$, $1.2D + 1.0L + 1.0W$, $1.2D + 1.0L + 1.0E$, $0.9D \pm 1.0W$, $0.9D \pm 1.0E$).
  - Includes ASCE 7 Section 12.5 orthogonal seismic combinations ($1.0 E_X \pm 0.3 E_Y$ and $0.3 E_X \pm 1.0 E_Y$).

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

# LoadGenerator auto-infers units from project.metadata.units
generator = LoadGenerator()

# Apply Office gravity loads with 5 m tributary width and self-weight
dead_case, live_case = generator.apply_gravity_loads(
    project,
    occupancy="office",
    tributary_width=5.0,
    superimposed_dead=1.5,
    include_self_weight=True,
)

# Apply roof snow and live loads to top floor framing
roof_cases = generator.apply_roof_loads(
    project,
    ground_snow_load=1.5,  # 1.5 kPa (or psf)
    tributary_width=5.0,
)

# Apply ASCE 7 seismic ELF lateral loads in both directions (X and Y)
seismic_cases, elf_res = generator.apply_lateral_seismic(
    project,
    sds=0.75,
    sd1=0.40,
    r_factor=8.0,
    direction="both",
)

# Generate factored ACI 318 / ASCE 7 LRFD combinations (with 3D orthogonal seismic)
combs = generator.generate_combinations(project, design_method="LRFD", orthogonal_seismic=True)
print(f"Generated {len(combs)} load combinations.")
```

### 2. Panel `AreaLoad` Tributary Distribution

```python
from struct_core import AreaLoad, LoadCase
from struct_load import distribute_area_loads_to_beams

# Define two-way slab pressure over panel nodes [5, 6, 7, 8]
al = AreaLoad(
    id="SLAB_FLOOR1",
    pressure=3.5,  # kPa
    load_type="live",
    node_ids=[5, 6, 7, 8],
    one_way=False,
)
project.model.load_cases.append(LoadCase(id="Live", area_loads=[al]))

# Distribute area loads into equivalent beam line loads
n_created = distribute_area_loads_to_beams(project.model, target_case_id="Live")
print(f"Distributed into {n_created} beam line loads.")
```

### 3. Materializing Element Self-Weight (`apply_self_weight`)

```python
from struct_load import apply_self_weight

# Automatically infers units and 2D/3D direction (-z for 3D, -y for 2D)
apply_self_weight(project)
# Element loads are computed from material density and section area,
# appended to case.element_loads, and case.include_self_weight is set to False.
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
