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
  - `apply_gravity_loads(partition_weight=...)` adds a `PartitionWeight` case with the §12.7.2 item 2 partition weight (default 10 psf / 0.48 kPa for occupancies with a partition allowance, when no `custom_live_load` is given). The case adds seismic weight only; load combinations ignore it, because the live load already carries the partition allowance.
  - Pressures are returned in the model's pressure unit (kPa for `kN, m`, ksf for `kip, ft`, psf for `lbf, ft`); `apply_gravity_loads` expects `superimposed_dead` and `custom_live_load` in the same unit.
  - Project inputs are never assumed: `apply_gravity_loads` requires `occupancy` and `superimposed_dead` (0 for none) and, for one-way distribution (every 2D frame), `tributary_width`; `apply_roof_loads` requires `tributary_width` and reduces the minimum roof live load for the members' `tributary_area` only when it is given (Eq. 4.8-1; R1 = 1 otherwise); `apply_lateral_wind` requires the site's `basic_wind_speed` and the frame's `tributary_width`.
  - ASCE 7 Section 4.7 live load reduction calculator ($L = L_0 (0.25 + 15/\sqrt{K_{LL} A_T})$).
- **Universal Self-Weight Engine**:
  - Computes continuous gravity line loads directly from section geometry and the weight density of each section's material (`unit_weight` or $\rho \cdot g$); a material without either raises `ValueError` (no density is assumed).
  - Works polymorphically on `fem3d.Structure`, `fem2d.Structure`, `struct_core.StructuralModel`, and `Project`.
  - Automatically detects 3D ($-z$) vs 2D ($-y$) direction and applies `DistributedLoad(..., is_self_weight=True)`.
- **Exact Model-Derived Seismic Weights & 3D ELF**:
  - Calculates exact story seismic weights $W_i$ directly from model Dead loads and member self-weights.
  - Directional seismic loads (`EQ_X`, `EQ_Y`, or `direction="both"`).
  - Calculates approximate fundamental period $T_a$, seismic response coefficient $C_s$, base shear $V = C_s W$, and vertical force distribution $F_x, F_y$ along building height.
  - $C_s$ Method 1 (Eq. 12.8-2) requires the multi-period spectrum (`sa_spectrum`); Method 2 (Eqs. 12.8-3 to 12.8-5) is the default. Eq. 12.8-7 is evaluated only when $S_1$ is supplied (`StoryForceResult.s1_bound_evaluated`). $T_a = 0.1N$ (Eq. 12.8-9) is accepted only for moment frames with $N \le 12$ and an average story height ≥ 10 ft (3 m).
- **Mass Source & Effective Seismic Weight Engine (ASCE 7-22 Section 12.7.2)**:
  - Full adherence to Section 12.7.2 specifications:
    - Dead load and self-weight: 100% ($1.0D$).
    - Storage live load: minimum 25% ($0.25 L_{\text{storage}}$, Section 12.7.2 Item 1). General office/residential live load is 0% (Item 1 exception b).
    - Partition load: 100% of actual or minimum code partition allowance ($10\text{ psf}$ / $0.48\text{ kPa}$, Item 2).
    - Permanent operating equipment: 100% ($1.0$, Item 3).
    - Snow load: 15% included only when flat roof snow load $P_f > 45\text{ psf}$ ($2.16\text{ kN/m}^2$, Item 4).
    - Roof gardens: 100% ($1.0$, Item 5).
  - Commercial-grade mass source evaluation converting load cases into inertial mass ($m = \sum c_i W_i / g$). With `include_element_mass=True`, loads flagged `is_self_weight` are skipped so member self-weight is counted once (`apply_gravity_loads` stores self-weight and superimposed dead as separate entries of the dead case).
  - `generate_combinations` applies the dead-load factor to every dead-type case (e.g. `DEAD` and `SuperDead`).
  - Supports structural element self-masses ($\rho \cdot A \cdot L$), assigned nodal lumped masses, element masses, and load case forces (point loads, distributed loads, area loads).
  - Calculates total seismic weight $W$, total inertial mass $M$, global Center of Mass $(X_{cm}, Y_{cm}, Z_{cm})$, and story-by-story mass distribution.
  - `create_asce7_mass_source(model, ...)` factory automatically configures code-compliant mass source from model load cases.
  - Directly consumed by `fem2d` and `fem3d` adapters for modal and response spectrum dynamic analysis.
- **Seismic Design Category & Lateral Procedures (ASCE 7-22 Chapters 11 & 12)**:
  - **SDC Determination (Section 11.6)**: Evaluates Risk Category (I–IV), $S_{DS}$, $S_{D1}$, and $S_1 \ge 0.75$ criteria per Tables 11.6-1 and 11.6-2 (Categories A through F).
  - **Accidental Torsion & Amplification (Section 12.8.4)**: Accidental torsional moment $M_{ta} = A_x (0.05 B) F_x$ and torsional amplification factor $A_x = (\delta_{\max} / 1.2 \delta_{\text{avg}})^2$ bounded by $[1.0, 3.0]$.
  - **Design Story Drift (Section 12.8.6 & 12.12)**: Computes design story drift $\Delta = C_d (\delta_{\text{top}} - \delta_{\text{bot}}) / I_e$ (Eq. 12.8-16) and verifies compliance against Table 12.12-1 allowable drift limits ($\Delta_a$).
  - **System keys**: `SEISMIC_SYSTEM_TYPES` lists the accepted keys. `normalize_system_type` lower-cases a key and raises ValueError for anything else, and the agent tools reject unknown systems.
  - **System coefficients and stability**: `seismic_system_coefficients` ($R$, $\Omega_0$, $C_d$ from Table 12.2-1 for RC special, intermediate and ordinary moment frames); `allowable_story_drift` (Table 12.12-1, divided by $\rho$ for moment frames in SDC D–F per §12.12.1.1); `stability_coefficient` ($\theta = P_x \Delta_{xe} / (V_x h_{sx})$, Eq. 12.8-18) and `max_stability_coefficient` ($\theta_{max} = 0.5/(\beta C_d) \le 0.25$, Eq. 12.8-19, not less than 0.10).
- **ASCE 7-22 Snow & Roof Live Loads**:
  - Chapter 7 roof snow load $p_f = 0.7 C_e C_t p_g$ (Eq. 7.3-1; ASCE 7-22 snow loads are strength-level, so there is no $I_s$), with the low-slope minimum $p_m = \min(p_g, p_{m,\max})$ by Risk Category (Table 7.3-4: 25/30/35/40 psf).
  - Section 4.5 minimum roof live load ($L_r = 20 R_1 R_2$, 12–20 psf).
  - Inputs and results are pressures in the model's pressure unit (kPa, psf, ksf, ...).
  - Applied specifically to roof level framing beams.
- **ASCE 7 Wind MWFRS Procedure**:
  - Directional procedure computing velocity pressure $q_z$, exposure coefficient $K_z = 2.41 (z/z_g)^{2/\alpha}$ (ASCE 7-22 Table 26.11-1 constants for Exposure B, C, D), and story lateral forces along $X$ and $Y$.
  - Windward walls use $q_z$ ($C_p = 0.8$); leeward walls use $q_h$ with $C_p$ from $L/B$ (−0.5, −0.3, −0.2; `WindParameters.l_over_b`).
- **ASCE 7-22 Factored Load Combinations (Sections 2.3, 2.4, and 12.4)**:
  - Strength (LRFD, Section 2.3.1/2.3.6): $1.4D$; $1.2D + 1.6L + (0.5L_r \text{ or } 0.3S)$; $1.2D + (1.6L_r \text{ or } 1.0S) + (L \text{ or } 0.5W)$; $1.2D + 1.0W + L + (0.5L_r \text{ or } 0.3S)$; $0.9D + 1.0W$; $(1.2 + 0.2 S_{DS}) D + \rho Q_E + L + 0.15S$; $(0.9 - 0.2 S_{DS}) D + \rho Q_E$.
  - Allowable stress (ASD, Section 2.4.1/2.4.5): $D$; $D + L$; $D + (L_r \text{ or } 0.7S)$; $D + 0.75L + 0.75(L_r \text{ or } 0.7S)$; $D + 0.6W$; $D + 0.75L + 0.45W + 0.75(L_r \text{ or } 0.7S)$; $0.6D + 0.6W$; $(1.0 + 0.14 S_{DS})D + 0.7\rho Q_E$; $(1.0 + 0.105 S_{DS})D + 0.525\rho Q_E + 0.75L + 0.1S$; $(0.6 - 0.14 S_{DS})D + 0.7\rho Q_E$.
  - Wind and seismic loads are reversible: every combination containing them is generated for both senses ($\pm W$, $\pm Q_E$), which matters for frames that are not symmetric.
  - Supports seismic overstrength $E_{mh} = \Omega_0 Q_E$ (Section 12.4.3); `omega_0` (Table 12.2-1) is required with `include_overstrength=True`.
  - Enforces SDC B vertical seismic effect exemption ($E_v = 0$ per Section 12.4.2.2 Exception 2(a)).
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
    ground_snow_load=1.5,  # pg in the model pressure unit (here kPa)
    risk_category="II",    # sets the minimum low-slope roof snow load (Table 7.3-4)
    tributary_width=5.0,
)

# Apply ASCE 7 seismic ELF lateral loads (using exact Section 12.7.2 seismic weight).
# R comes from the system (Table 12.2-1) and Ie from the Risk Category (Table 1.5-2) unless given; TL (the site's
# mapped value) is needed only for periods above 4 s, the smallest mapped TL.
params = SeismicParameters(sds=0.75, sd1=0.40, system_type="rc_smf", risk_category="II")
seismic_cases, elf_res = generator.apply_lateral_seismic(
    project,
    params=params,
    direction="both",
    storage_live_cases=["LIVE_STORAGE"],  # 25% per Section 12.7.2 Item 1
)

# Generate factored ASCE 7-22 LRFD combinations (with vertical seismic Ev = 0.2*Sds*D)
combs = generator.generate_combinations(
    project,
    design_method="LRFD",
    sds=0.75,
    rho=1.3,
    orthogonal_seismic=True,
)
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

### 4. Code-Compliant Mass Source & Seismic Diagnostics (ASCE 7-22)

```python
from struct_load import (
    create_asce7_mass_source,
    determine_seismic_design_category,
    calculate_accidental_torsion,
    calculate_torsional_amplification,
    check_story_drift_limit,
)

# Determine Seismic Design Category (Section 11.6)
sdc = determine_seismic_design_category(risk_category="II", sds=0.75, sd1=0.40)  # returns "D"

# Create ASCE 7-22 Section 12.7.2 mass source automatically
ms = create_asce7_mass_source(
    project.model,
    storage_live_factor=0.25,   # Section 12.7.2 Item 1 (Storage Live)
    general_live_factor=0.0,    # General Live is 0%
    flat_roof_snow_pressure=1.2, # < 2.16 kPa (45 psf) -> 0% per Item 4
)
res = ms.evaluate(project.model)
print(f"Total Seismic Weight: {res.total_weight:.2f} kN, Center of Mass: {res.center_of_mass}")

# Accidental Torsion & Amplification (Section 12.8.4)
M_ta = calculate_accidental_torsion(fx=120.0, building_width=30.0, ax=1.0)  # 180 kN*m
ax = calculate_torsional_amplification(delta_max=0.025, delta_avg=0.018)    # Ax = 1.34

# Story Drift Verification (Section 12.12 & Table 12.12-1)
drift_ok, delta, delta_a = check_story_drift_limit(
    delta_top=0.045, delta_bot=0.020, cd=5.5, ie=1.0, story_height=3.5, occupancy="office"
)
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
