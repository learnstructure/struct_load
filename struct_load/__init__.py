"""
struct_load: Automated structural load determination, building code lookups,
and combination generation for the Structural Engineering Ecosystem.
"""

from .codes.asce7_combinations import generate_asce7_combinations
from .codes.asce7_seismic import (
    SeismicParameters,
    StoryForceResult,
    approximate_fundamental_period,
    calculate_seismic_elf,
    calculate_seismic_response_coefficient,
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
from .engine.self_weight import calculate_section_area, calculate_self_weight
from .engine.tributary import (
    apply_floor_load_to_frame,
    distribute_1d_tributary,
    distribute_two_way_slab,
)
from .generator import LoadGenerator

__version__ = "0.1.0"

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
    # Seismic ELF
    "SeismicParameters",
    "StoryForceResult",
    "approximate_fundamental_period",
    "calculate_seismic_response_coefficient",
    "calculate_seismic_elf",
    # Wind MWFRS
    "WindParameters",
    "StoryWindResult",
    "calculate_kz",
    "calculate_wind_forces",
    # Engine
    "calculate_section_area",
    "calculate_self_weight",
    "distribute_1d_tributary",
    "distribute_two_way_slab",
    "apply_floor_load_to_frame",
]
