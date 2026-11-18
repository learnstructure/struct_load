# Changelog

All notable changes to `struct_load` are recorded in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/). The code of the paper is the git tag `v2.1-paper`.

## [Unreleased]

### Added
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
  - New helpers `shell_weight_per_area`, `shell_self_weights`, and `shell_tributary`.
- `element_weight_per_length` and `element_mass_per_length`: the self-weight and self-mass per unit length of a
  frame element, with its `weight` and `mass` modifiers (`BeamElement.stiffness_modifiers`).

### Fixed
- The element `weight` and `mass` modifiers were ignored. Self-weight (`calculate_self_weight`) now follows the
  weight modifier, and the element self-mass of the mass source and the story seismic weights follow the mass
  modifier (the seismic weight is the mass times g, as in the CSI mass source). The `A` modifier changes neither:
  as SAP2000 27.1 shows (struct_link tests), it scales the axial stiffness only.
