# Changelog

All notable changes to `struct_load` are recorded in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/). The code of the paper is the git tag `v2.1-paper`.

## [Unreleased]

### Added
- `element_weight_per_length` and `element_mass_per_length`: the self-weight and self-mass per unit length of a
  frame element, with its `weight` and `mass` modifiers (`BeamElement.stiffness_modifiers`).

### Fixed
- The element `weight` and `mass` modifiers were ignored. Self-weight (`calculate_self_weight`) now follows the
  weight modifier, and the element self-mass of the mass source and the story seismic weights follow the mass
  modifier (the seismic weight is the mass times g, as in the CSI mass source). The `A` modifier changes neither:
  as SAP2000 27.1 shows (struct_link tests), it scales the axial stiffness only.
