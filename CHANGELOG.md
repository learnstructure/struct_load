# Changelog

All notable changes to `struct_load` are recorded in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/). The code of the paper is the git tag `v2.1-paper`.

## [Unreleased]

### Fixed
- `LoadGenerator.calculate_model_seismic_weight`: checks `getattr(dl, "is_self_weight", False)` before marking an
  element as already carrying materialized self-weight, matching `calculate_mass_source`. Previously, a superimposed
  dead load (`is_self_weight=False`) on a beam caused `calculate_model_seismic_weight` to skip the beam's unmaterialized
  self-weight when `include_self_weight=True`, understating the effective seismic weight (by 17.5% on benchmark `C02`'s
  model when computed without a mass source). Test: `tests/test_generator.py`.

### Changed
- **Named stories and explicit base elevation (ROADMAP §1a):** replaced `detect_story_elevations` with `detect_stories`
  (returning `List[Story]` for levels above the base only) and added `resolve_base_elevation` (using
  `model.base_elevation` when set, or inferring from the lowest supported nodes with a message). Renamed
  `story_elevations` to `stories` everywhere, including `calculate_seismic_elf`, `calculate_wind_forces`,
  `LoadGenerator.apply_lateral_seismic`, `LoadGenerator.apply_lateral_wind`, and floor/mass helpers. Both
  `calculate_seismic_elf` and `calculate_wind_forces` now take `stories: List[Story]` and a required keyword-only
  `base_elevation: float` (no default of 0), measuring heights `h_x = story.elevation - base_elevation` per ASCE 7-22
  §12.8.3. Unified restrained-base mass exclusion across `calculate_mass_source` and
  `LoadGenerator.calculate_model_seismic_weight` (whether `base_elevation` is explicit or inferred from lowest
  supports), matching ETABS 23 and SAP2000 27 effective seismic weight. Tests: `tests/test_codes.py`,
  `tests/test_geometry_3d.py`, `tests/test_mass_source.py`.

### Added
- `LoadGenerator.messages`: `generate_combinations` now records every load case left out of the generated ASCE 7-22
  combinations (`other`, `temperature`, `soil`, `fluid`, `partition`, and extra cases beyond one per direction for
  wind, seismic, snow, or roof live), with its ID, type, and reason. Test: `tests/test_generator.py`.
- `apply_self_weight` on a fem3d `Structure` adds the self-weight of its shell elements (unit weight x membrane
  thickness, CSI §10.12) as fem3d `ShellLoad`s along the gravity axis, once (fem3d/SHELL_PLAN.md, M5). A structure
  with shells is treated as 3D. Test: `fem3d/tests/test_shell_structure.py`.
- Shell elements in self-weight and mass (fem3d/SHELL_PLAN.md, M1); without them the ELF seismic weight of a model
  with slabs or walls was understated:
  - `calculate_self_weight` and `apply_self_weight` add `ShellLoad` self-weight (unit weight x membrane thickness,
    CSI Analysis Reference Manual §10.12), once per element.
  - `calculate_mass_source` lumps shell self-mass and the shell loads of its load patterns at the element nodes, in
    the shares of CSI §10.11, so nodal masses, total weight, center of mass, and story masses include shells.
  - `LoadGenerator.calculate_model_seismic_weight` adds shell loads and the shell self-weight not already in a dead
    case to the stories of the element nodes.
  - Helpers in `struct_load.engine.self_weight`: `shell_weight_per_area`, `shell_self_weights`, and `shell_tributary`.
  - Tests: `tests/test_shell_weight.py`.
- `element_weight_per_length` and `element_mass_per_length`: the self-weight and self-mass per unit length of a
  frame element, with its `weight` and `mass` modifiers (`BeamElement.stiffness_modifiers`).

### Fixed
- `calculate_mass_source` now lumps member distributed and point loads (`DistributedLoad`, `ElementPointLoad`) to
  end nodes using the element's consistent static end reactions (including Timoshenko shear deformation and end
  releases, reducing to the simple-beam lever rule when moment-released), matching SAP2000's load-to-mass conversion
  instead of splitting asymmetric loads 50/50. Test: `tests/test_mass_source.py`.
- `material_weight_density` and `_solver_weight_density` now allow explicit zero density (`unit_weight=0.0` and
  `rho=0.0`, returning `0.0` for massless elements with lumped joint masses) while still refusing materials with
  unspecified density (`unit_weight=None` and `rho=0.0`). Test: `tests/test_mass_source.py`.
- `calculate_mass_source` now converts user-assigned `Node.mass` and element `mass` from `units.mass` to consistent
  solver mass units (`units.consistent_mass_factor`), matching `fem2d`, `fem3d`, and `self_weight`. Test:
  `tests/test_mass_source.py`.
- The element `weight` and `mass` modifiers were ignored. Self-weight (`calculate_self_weight`) now follows the
  weight modifier, and the element self-mass of the mass source and the story seismic weights follow the mass
  modifier (the seismic weight is the mass times g, as in the CSI mass source). The `A` modifier changes neither:
  as SAP2000 27.1 shows (struct_link tests), it scales the axial stiffness only.
- `examples/07_end_to_end_building_loading.py` now passes `two_way=True` instead of `tributary_width` to
  `apply_gravity_loads` and `apply_roof_loads` on its 3D frame.
