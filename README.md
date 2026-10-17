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
- **Canonical Unit System Auto-Inference**:
  - Automatically reads `Units` directly from `source.units` (`fem3d.Structure`, `fem2d.Structure`, `StructuralModel`) or `project.metadata.units`.
  - Automatically derives exact acceleration due to gravity $g$ from active length units ($9.80665\text{ m/s}^2$, $386.0886\text{ in/s}^2$, $32.174\text{ ft/s}^2$), eliminating manual $g$ specification.
- **Panel & `AreaLoad` Tributary Distribution**:
  - Maps floor surface pressures directly from `struct_core.AreaLoad` onto perimeter supporting beams.
  - **One-way slabs**: tributary strips ($w = q \cdot S/2$) applied to beams perpendicular to span axis; parallel beams receive 0 load.
  - **Two-way slabs**: 45° yield-line distribution. Computes triangular equivalent uniform loads on short beams and trapezoidal equivalent uniform loads on long beams.
- **ASCE 7 & IBC Occupancy Live Loads**:
  - Full database of standard occupancies (Offices, Residential, Classrooms, Corridors, Retail, Storage, Assembly, Parking, Roof).
  - Built-in partition load allowances (15 psf / 0.72 kPa for offices).
  - ASCE 7 Section 4.7 live load reduction calculator ($L = L_0 (0.25 + 15/\sqrt{K_{LL} A_T})$).
- **Universal Self-Weight Engine**:
  - Computes continuous gravity line loads directly from section geometry and material weight density (`unit_weight` or $\rho \cdot g$).
  - Works polymorphically on `fem3d.Structure`, `fem2d.Structure`, `struct_core.StructuralModel`, and `Project`.
  - Automatically detects 3D ($-z$) vs 2D ($-y$) direction and applies `DistributedLoad(..., is_self_weight=True)`.
- **Exact Model-Derived Seismic Weights & 3D ELF**:
  - Calculates exact story seismic weights $W_i$ directly from model Dead loads and member self-weights.
  - Directional seismic loads (`EQ_X`, `EQ_Y`, or `direction="both"`).
  - Calculates approximate fundamental period $T_a$, seismic response coefficient $C_s$, base shear $V = C_s W$, and vertical force distribution $F_x, F_y$ along building height.
- **Mass Source & Seismic Weight Calculation Engine**:
  - Commercial-grade mass source evaluation converting load cases into inertial mass ($m = \sum c_i W_i / g$).
  - Supports structural element self-masses ($\rho \cdot A \cdot L$), assigned nodal lumped masses, element masses, and load case forces (point loads, distributed loads, area loads).
  - Calculates total seismic weight $W$, total inertial mass $M$, global Center of Mass $(X_{cm}, Y_{cm}, Z_{cm})$, and story-by-story mass distribution.
  - Directly consumed by `fem2d` and `fem3d` adapters for modal and response spectrum dynamic analysis.
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
params = SeismicParameters(sds=0.75, sd1=0.40, r_factor=8.0)
seismic_cases, elf_res = generator.apply_lateral_seismic(
    project,
    params=params,
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

`apply_self_weight` can be called directly on analysis solver structures (`fem3d.Structure`, `fem2d.Structure`) or schema containers (`StructuralModel`, `Project`):

```python
from struct_load import apply_self_weight

# Option A: Direct solver application (no schema conversion required)
struct.apply_self_weight()      # via method on fem3d/fem2d Structure
# or: apply_self_weight(struct)

# Option B: Schema model application (infers units and direction automatically)
apply_self_weight(project)  # supports in-place and return: project = apply_self_weight(project)
# Element loads are computed from material unit_weight (or rho * g) and section area,
# appended to DEAD case.element_loads (with is_self_weight=True), and case.include_self_weight is set to False.
```

---

## Running Tests

```bash
conda activate structeng
pytest struct_load/tests -v
```

---

## Examples Suite

The [`examples/`](examples/) directory contains standalone, runnable scripts demonstrating every capability of `struct_load`:

| Script | Description |
|---|---|
| [`01_asce7_code_lookups.py`](examples/01_asce7_code_lookups.py) | ASCE 7 occupancy live load lookups, partition allowances, live load reduction, flat roof snow, and minimum roof live loads. |
| [`02_tributary_and_area_loads.py`](examples/02_tributary_and_area_loads.py) | 1D tributary strips, 2-way slab 45° yield-line distributions (moment and shear equivalents), and automated `AreaLoad` to beam distribution on a 3D bay. |
| [`03_self_weight_engine.py`](examples/03_self_weight_engine.py) | Section area and linear self-weight calculations in SI and US units, 2D plane frame (-Y) and 3D space frame (-Z) self-weight materialization. |
| [`04_seismic_elf_procedure.py`](examples/04_seismic_elf_procedure.py) | ASCE 7 Section 12.8 Equivalent Lateral Force (ELF) procedure, $T_a$, $C_s$, exact model-derived story seismic weights $W_i$, and story shears. |
| [`05_wind_mwfrs_procedure.py`](examples/05_wind_mwfrs_procedure.py) | ASCE 7 Chapter 27 MWFRS directional wind procedure, velocity pressure $q_z$, exposure coefficient $K_z$ profiles (B, C, D), and story wind forces. |
| [`06_factored_load_combinations.py`](examples/06_factored_load_combinations.py) | ASCE 7 / ACI 318 LRFD and ASD combinations generator, and 3D orthogonal seismic combinations ($1.0E_X \pm 0.3E_Y$ and $0.3E_X \pm 1.0E_Y$). |
| [`07_end_to_end_building_loading.py`](examples/07_end_to_end_building_loading.py) | Complete automated loading pipeline on a 3-story concrete building project using `LoadGenerator`. |
| [`08_mass_source_and_seismic_weight.py`](examples/08_mass_source_and_seismic_weight.py) | Commercial-grade Mass Source evaluation converting $1.0D + 0.25L$ into dynamic mass, Center of Mass $(X_{cm}, Y_{cm}, Z_{cm})$, and story mass distribution. |
| [`run_all.py`](examples/run_all.py) | Master runner executing all examples sequentially. |

To run all examples:
```bash
conda activate structeng
python struct_load/examples/run_all.py
```

---

## Architecture Compliance

- **Inputs**: Reads `struct_core.StructuralModel` (nodes, elements, sections, materials).
- **Outputs**: Produces standard `struct_core.LoadCase` and `struct_core.LoadCombination` objects.
- **No Solver Logic**: Does not solve stiffness matrices or check rebar; strictly handles load determination.
