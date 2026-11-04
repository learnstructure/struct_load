"""
struct_load: Automated structural load determination, building code lookups,
and combination generation for the Structural Engineering Ecosystem.
"""

from .codes.asce7_combinations import generate_asce7_combinations
from .codes.asce7_seismic import (
    SeismicParameters,
    StoryForceResult,
    approximate_fundamental_period,
    calculate_accidental_torsion,
    calculate_cw,
    calculate_design_response_spectrum,
    calculate_design_story_drift,
    calculate_seismic_elf,
    calculate_seismic_response_coefficient,
    calculate_torsional_amplification,
    check_sds_cap_eligibility,
    check_story_drift_limit,
    determine_seismic_design_category,
    evaluate_design_spectral_acceleration,
    get_cu_coefficient,
    SEISMIC_SYSTEM_COEFFICIENTS,
    SEISMIC_SYSTEM_TYPES,
    allowable_story_drift,
    max_stability_coefficient,
    normalize_system_type,
    seismic_importance_factor,
    seismic_system_coefficients,
    stability_coefficient,
)
from .codes.asce7_snow import (
    SnowParameters,
    calculate_flat_roof_snow_load,
    calculate_roof_live_load,
)
from .codes.asce7_tables import (
    OccupancyLiveLoad,
    calculate_live_load_reduction,
    get_occupancy_live_load,
    lookup_occupancy,
)
from .codes.asce7_wind import (
    StoryWindResult,
    WindParameters,
    calculate_kz,
    calculate_wind_forces,
)
from .engine.geometry import (
    detect_model_dimensionality,
    detect_story_elevations,
    get_beam_elevation,
    get_beam_length,
    is_horizontal_beam,
)
from .engine.mass_source import (
    MassSourceResult,
    calculate_mass_source,
    create_asce7_mass_source,
)
from .engine.self_weight import (
    apply_self_weight,
    calculate_section_area,
    calculate_self_weight,
)
from .engine.tributary import (
    apply_floor_load_to_frame,
    distribute_1d_tributary,
    distribute_area_loads_to_beams,
    distribute_tributary_load,
    distribute_two_way_slab,
)
from .generator import LoadGenerator

__version__ = "0.2.0"

__all__ = [
    # Top-level Orchestrator
    "LoadGenerator",
    # ASCE 7 Tables & Occupancy
    "OccupancyLiveLoad",
    "lookup_occupancy",
    "get_occupancy_live_load",
    "calculate_live_load_reduction",
    # Combinations
    "generate_asce7_combinations",
    # Seismic ELF & Spectrum
    "SeismicParameters",
    "StoryForceResult",
    "approximate_fundamental_period",
    "calculate_seismic_response_coefficient",
    "calculate_seismic_elf",
    "calculate_design_response_spectrum",
    "evaluate_design_spectral_acceleration",
    "check_sds_cap_eligibility",
    "get_cu_coefficient",
    "calculate_cw",
    "determine_seismic_design_category",
    "calculate_accidental_torsion",
    "calculate_torsional_amplification",
    "calculate_design_story_drift",
    "check_story_drift_limit",
    "SEISMIC_SYSTEM_COEFFICIENTS",
    "SEISMIC_SYSTEM_TYPES",
    "normalize_system_type",
    "seismic_importance_factor",
    "seismic_system_coefficients",
    "allowable_story_drift",
    "stability_coefficient",
    "max_stability_coefficient",
    # Wind MWFRS
    "WindParameters",
    "StoryWindResult",
    "calculate_kz",
    "calculate_wind_forces",
    # Snow & Roof
    "SnowParameters",
    "calculate_flat_roof_snow_load",
    "calculate_roof_live_load",
    # Geometry & Engine
    "detect_model_dimensionality",
    "detect_story_elevations",
    "is_horizontal_beam",
    "get_beam_elevation",
    "get_beam_length",
    "apply_self_weight",
    "calculate_section_area",
    "calculate_self_weight",
    "distribute_1d_tributary",
    "distribute_tributary_load",
    "distribute_two_way_slab",
    "distribute_area_loads_to_beams",
    "apply_floor_load_to_frame",
    "calculate_mass_source",
    "create_asce7_mass_source",
    "MassSourceResult",
]
