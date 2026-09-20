"""
High-level LoadGenerator orchestrating automated load determination on structural models.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
from struct_core import (
    DistributedLoad,
    LoadCase,
    LoadCombination,
    Node,
    PointLoad,
    Project,
    StructuralModel,
)

from .codes.asce7_combinations import generate_asce7_combinations
from .codes.asce7_seismic import (
    SeismicParameters,
    StoryForceResult,
    calculate_seismic_elf,
)
from .codes.asce7_tables import get_occupancy_live_load
from .codes.asce7_wind import (
    StoryWindResult,
    WindParameters,
    calculate_wind_forces,
)
from .engine.self_weight import calculate_self_weight
from .engine.tributary import apply_floor_load_to_frame


class LoadGenerator:
    """
    Automated building load determination engine.
    Translates architectural and environmental criteria into deterministic, code-compliant
    struct_core.LoadCase and struct_core.LoadCombination instances.
    """

    def __init__(self, unit_system: str = "si"):
        self.unit_system = unit_system.lower()

    @staticmethod
    def _unwrap_model(model_or_project: Union[StructuralModel, Project]) -> StructuralModel:
        if isinstance(model_or_project, Project):
            return model_or_project.model
        return model_or_project

    def apply_gravity_loads(
        self,
        model_or_project: Union[StructuralModel, Project],
        occupancy: str = "office",
        tributary_width: float = 5.0,
        superimposed_dead: float = 1.5,
        include_self_weight: bool = True,
        custom_live_load: Optional[float] = None,
        dead_case_id: str = "Dead",
        live_case_id: str = "Live",
    ) -> Tuple[LoadCase, LoadCase]:
        """
        Calculate and assign realistic Dead and Live load cases to the structural model.

        Parameters
        ----------
        model_or_project : StructuralModel or Project
            Target structural model.
        occupancy : str
            Building occupancy type (e.g. 'office', 'residential', 'classroom').
        tributary_width : float
            Tributary width of floor framing (m for SI, ft for US).
        superimposed_dead : float
            Superimposed dead load pressure (kPa for SI, psf for US) for finishes/MEP.
        include_self_weight : bool
            Whether to calculate member self-weight directly from sections.
        custom_live_load : float, optional
            Override live load intensity. If None, looked up from ASCE 7 Table 4.3-1.
        dead_case_id : str
            ID for Dead load case.
        live_case_id : str
            ID for Live load case.

        Returns
        -------
        Tuple[LoadCase, LoadCase]
            (Dead_LoadCase, Live_LoadCase)
        """
        model = self._unwrap_model(model_or_project)

        # 1. Live Load from ASCE 7 table
        if custom_live_load is not None:
            ll_pressure = custom_live_load
        else:
            ll_pressure = get_occupancy_live_load(
                occupancy, include_partitions=True, unit_system=self.unit_system
            )

        live_case = apply_floor_load_to_frame(
            model=model,
            pressure=ll_pressure,
            tributary_width=tributary_width,
            target_case_id=live_case_id,
            case_name=f"Live Load ({occupancy.title()})",
        )

        # 2. Dead Load: Superimposed + Self-Weight
        sdl_case = apply_floor_load_to_frame(
            model=model,
            pressure=superimposed_dead,
            tributary_width=tributary_width,
            target_case_id=dead_case_id,
            case_name="Dead Load",
        )

        combined_dead_loads: Dict[Union[int, str], DistributedLoad] = {}
        for dl in sdl_case.element_loads:
            combined_dead_loads[dl.element_id] = dl

        if include_self_weight:
            sw_case = calculate_self_weight(
                model=model,
                load_case_id="SW",
                unit_system=self.unit_system,
            )
            for sw_load in sw_case.element_loads:
                eid = sw_load.element_id
                if eid in combined_dead_loads:
                    existing = combined_dead_loads[eid]
                    combined_dead_loads[eid] = DistributedLoad(
                        element_id=eid,
                        wx=round(existing.wx + sw_load.wx, 4),
                        wy=round(existing.wy + sw_load.wy, 4),
                    )
                else:
                    combined_dead_loads[eid] = sw_load

        final_dead_case = LoadCase(
            id=dead_case_id,
            name="Dead Load (Self-Weight + SDL)",
            element_loads=list(combined_dead_loads.values()),
            include_self_weight=include_self_weight,
        )

        # Attach / replace in model
        model.load_cases = [
            lc for lc in model.load_cases if lc.id not in (dead_case_id, live_case_id)
        ]
        model.load_cases.extend([final_dead_case, live_case])

        return final_dead_case, live_case

    def apply_lateral_seismic(
        self,
        model_or_project: Union[StructuralModel, Project],
        sds: float = 0.75,
        sd1: float = 0.40,
        r_factor: float = 8.0,
        system_type: str = "rc_smf",
        importance_factor: float = 1.0,
        tributary_width: float = 5.0,
        seismic_case_id: str = "EQ_X",
        story_elevations: Optional[List[float]] = None,
    ) -> Tuple[LoadCase, StoryForceResult]:
        """
        Apply ASCE 7 Equivalent Lateral Force (ELF) seismic loads as nodal forces to floor joints.

        Parameters
        ----------
        model_or_project : StructuralModel or Project
            Target structural model.
        sds : float
            Design spectral acceleration at short periods (g).
        sd1 : float
            Design spectral acceleration at 1-sec period (g).
        r_factor : float
            Response modification factor R.
        system_type : str
            Structural system type ('rc_smf', 'steel_smf', etc.).
        tributary_width : float
            Tributary width for frame gravity weight (m or ft).
        seismic_case_id : str
            ID for generated Seismic LoadCase.
        story_elevations : List[float], optional
            Story elevations. If None, automatically detected from beam elevations.

        Returns
        -------
        Tuple[LoadCase, StoryForceResult]
            (Seismic_LoadCase, StoryForceResult)
        """
        model = self._unwrap_model(model_or_project)

        # Detect story levels from model nodes (y > 0)
        elevations = story_elevations or self.detect_story_elevations(model)
        if not elevations:
            elevations = [3.5]  # fallback single story

        # Estimate story seismic weights from tributary area and average floor load (~5 kPa dead)
        # Or calculate from dead load case if present
        story_weights = []
        # Total span estimate
        x_coords = [n.x for n in model.nodes]
        span_length = max(x_coords) - min(x_coords) if x_coords else 6.0

        for _ in elevations:
            # Floor tributary area = span_length * tributary_width
            # Approx dead weight = 5.0 kPa * area (kN)
            w_story = 5.0 * span_length * tributary_width
            story_weights.append(round(w_story, 2))

        params = SeismicParameters(
            sds=sds,
            sd1=sd1,
            r_factor=r_factor,
            importance_factor=importance_factor,
            system_type=system_type,
        )

        elf_result = calculate_seismic_elf(
            story_elevations=elevations,
            story_weights=story_weights,
            params=params,
            unit_system=self.unit_system,
        )

        # Apply story lateral forces Fx to left-most or all nodes at each floor elevation
        point_loads: List[PointLoad] = []
        for idx, elev in enumerate(elevations):
            fx = elf_result.story_forces[idx + 1]
            floor_nodes = [n for n in model.nodes if abs(n.y - elev) <= 0.1]
            if not floor_nodes:
                continue

            # Apply to first node (e.g. left column joint)
            target_node = min(floor_nodes, key=lambda n: n.x)
            point_loads.append(PointLoad(node_id=target_node.id, fx=round(fx, 3), fy=0.0))

        seismic_case = LoadCase(
            id=seismic_case_id,
            name="Seismic Load (ASCE 7 ELF)",
            point_loads=point_loads,
        )

        model.load_cases = [lc for lc in model.load_cases if lc.id != seismic_case_id]
        model.load_cases.append(seismic_case)

        return seismic_case, elf_result

    def apply_lateral_wind(
        self,
        model_or_project: Union[StructuralModel, Project],
        basic_wind_speed: float = 40.0,
        exposure_category: str = "B",
        tributary_width: float = 5.0,
        wind_case_id: str = "Wind_X",
        story_elevations: Optional[List[float]] = None,
    ) -> Tuple[LoadCase, StoryWindResult]:
        """
        Apply ASCE 7 wind lateral forces to the building frame.
        """
        model = self._unwrap_model(model_or_project)
        elevations = story_elevations or self.detect_story_elevations(model)
        if not elevations:
            elevations = [3.5]

        # Calculate story heights
        story_heights = []
        prev_h = 0.0
        for elev in elevations:
            story_heights.append(elev - prev_h)
            prev_h = elev

        wind_params = WindParameters(
            basic_wind_speed=basic_wind_speed,
            exposure_category=exposure_category,
        )

        wind_result = calculate_wind_forces(
            story_elevations=elevations,
            tributary_widths=[tributary_width] * len(elevations),
            story_heights=story_heights,
            params=wind_params,
            unit_system=self.unit_system,
        )

        point_loads: List[PointLoad] = []
        for idx, elev in enumerate(elevations):
            fw = wind_result.story_forces[idx + 1]
            floor_nodes = [n for n in model.nodes if abs(n.y - elev) <= 0.1]
            if not floor_nodes:
                continue

            target_node = min(floor_nodes, key=lambda n: n.x)
            point_loads.append(PointLoad(node_id=target_node.id, fx=round(fw, 3), fy=0.0))

        wind_case = LoadCase(
            id=wind_case_id,
            name="Wind Load (ASCE 7 MWFRS)",
            point_loads=point_loads,
        )

        model.load_cases = [lc for lc in model.load_cases if lc.id != wind_case_id]
        model.load_cases.append(wind_case)

        return wind_case, wind_result

    def generate_combinations(
        self,
        model_or_project: Union[StructuralModel, Project],
        design_method: str = "LRFD",
    ) -> List[LoadCombination]:
        """
        Detect present load cases on the model and generate standard ASCE 7 / ACI 318 combinations.
        """
        model = self._unwrap_model(model_or_project)
        case_ids = {lc.id for lc in model.load_cases}

        dead_id = "Dead" if "Dead" in case_ids else (next((cid for cid in case_ids if "dead" in cid.lower()), None))
        live_id = "Live" if "Live" in case_ids else (next((cid for cid in case_ids if "live" in cid.lower()), None))
        wind_id = next((cid for cid in case_ids if "wind" in cid.lower()), None)
        seismic_id = next((cid for cid in case_ids if "eq" in cid.lower() or "seis" in cid.lower()), None)

        if not dead_id and model.load_cases:
            dead_id = model.load_cases[0].id

        combs = generate_asce7_combinations(
            dead_case_id=dead_id or "Dead",
            live_case_id=live_id,
            wind_case_id=wind_id,
            seismic_case_id=seismic_id,
            design_method=design_method,
        )

        model.load_combinations = combs
        return combs

    @staticmethod
    def detect_story_elevations(model: StructuralModel, tolerance: float = 0.1) -> List[float]:
        """
        Inspect model nodes and horizontal beams to identify unique floor elevations above ground (y > 0).
        """
        y_vals = sorted(list({round(n.y, 2) for n in model.nodes if n.y > tolerance}))
        filtered = []
        for y in y_vals:
            if not filtered or abs(y - filtered[-1]) > tolerance:
                filtered.append(y)
        return filtered
