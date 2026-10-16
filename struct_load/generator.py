"""
High-level LoadGenerator orchestrating automated load determination on structural models.
Supports 2D and 3D frames, unit auto-inference, exact model-derived seismic weights,
multi-directional lateral loads, roof snow and live loads, and area load distribution.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from struct_core import (
    DistributedLoad,
    LoadCase,
    LoadCombination,
    Node,
    PointLoad,
    Project,
    StructuralModel,
    Units,
    parse_units,
)

from .codes.asce7_combinations import generate_asce7_combinations
from .codes.asce7_seismic import (
    SeismicParameters,
    StoryForceResult,
    calculate_seismic_elf,
)
from .codes.asce7_snow import (
    SnowParameters,
    calculate_flat_roof_snow_load,
    calculate_roof_live_load,
)
from .codes.asce7_tables import get_occupancy_live_load
from .codes.asce7_wind import (
    StoryWindResult,
    WindParameters,
    calculate_wind_forces,
)
from .engine.geometry import (
    detect_model_dimensionality,
    detect_story_elevations,
    get_beam_elevation,
    get_beam_length,
    is_horizontal_beam,
)
from .engine.self_weight import calculate_section_area, calculate_self_weight
from .engine.tributary import (
    apply_floor_load_to_frame,
    distribute_area_loads_to_beams,
)


class LoadGenerator:
    """
    Automated building load determination engine.
    Translates architectural and environmental criteria into deterministic, code-compliant
    struct_core.LoadCase and struct_core.LoadCombination instances.
    """

    def __init__(self, units: Optional[Union[str, Units, dict]] = None):
        """
        Initialize LoadGenerator.

        Parameters
        ----------
        units : str, Units, or dict, optional
            Units specification (e.g. 'kN, m' or 'kip, in'). If None, automatically inferred
            from model or project units, or defaults to 'kN, m'.
        """
        self._units = parse_units(units) if units is not None else None

    @property
    def units(self) -> Optional[Units]:
        """Active or default Units."""
        return self._units

    def get_units(self, model_or_project: Union[StructuralModel, Project]) -> Units:
        """Resolve effective Units for a model or project."""
        if self._units:
            return self._units

        if isinstance(model_or_project, Project):
            if getattr(model_or_project, "units", None) is not None:
                return parse_units(model_or_project.units)
            meta = getattr(model_or_project, "metadata", None)
            u = getattr(meta, "units", None) if meta else None
            if u:
                return parse_units(u)
            if model_or_project.model and getattr(model_or_project.model, "units", None) is not None:
                return parse_units(model_or_project.model.units)
        elif hasattr(model_or_project, "units") and model_or_project.units is not None:
            return parse_units(model_or_project.units)

        return Units(force="kN", length="m")

    @staticmethod
    def _unwrap_model(model_or_project: Union[StructuralModel, Project]) -> StructuralModel:
        if isinstance(model_or_project, Project):
            if model_or_project.model is None:
                raise ValueError("Project contains no structural model.")
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
        Automatically senses 2D (wy along -Y) vs 3D (wz along -Z) geometry.
        """
        model = self._unwrap_model(model_or_project)
        active_units = self.get_units(model_or_project)
        is_3d = detect_model_dimensionality(model) == "3d"
        direction = "-z" if is_3d else "-y"

        # 1. Live Load from ASCE 7 table
        if custom_live_load is not None:
            ll_pressure = custom_live_load
        else:
            ll_pressure = get_occupancy_live_load(
                occupancy, include_partitions=True, units=active_units
            )

        live_case = apply_floor_load_to_frame(
            model=model,
            pressure=ll_pressure,
            tributary_width=tributary_width,
            target_case_id=live_case_id,
            case_name=f"Live Load ({occupancy.title()})",
            direction=direction,
        )

        # 2. Dead Load: Superimposed + Self-Weight
        sdl_case = apply_floor_load_to_frame(
            model=model,
            pressure=superimposed_dead,
            tributary_width=tributary_width,
            target_case_id=dead_case_id,
            case_name="Dead Load",
            direction=direction,
        )

        combined_dead_loads: Dict[Union[int, str], DistributedLoad] = {}
        for dl in sdl_case.element_loads:
            combined_dead_loads[dl.element_id] = dl

        if include_self_weight:
            sw_case = calculate_self_weight(
                model=model,
                load_case_id="SW",
                units=active_units,
                direction=direction,
            )
            for sw_load in sw_case.element_loads:
                eid = sw_load.element_id
                if eid in combined_dead_loads:
                    existing = combined_dead_loads[eid]
                    combined_dead_loads[eid] = DistributedLoad(
                        element_id=eid,
                        wx=round(existing.wx + sw_load.wx, 4),
                        wy=round(existing.wy + sw_load.wy, 4),
                        wz=round(existing.wz + sw_load.wz, 4),
                    )
                else:
                    combined_dead_loads[eid] = sw_load

        final_dead_case = LoadCase(
            id=dead_case_id,
            name="Dead Load (Self-Weight + SDL)",
            element_loads=list(combined_dead_loads.values()),
            include_self_weight=include_self_weight,
        )

        # Attach / replace in model (case-insensitive ID match)
        target_ids = {dead_case_id.lower(), live_case_id.lower()}
        model.load_cases = [
            lc for lc in model.load_cases if str(lc.id).lower() not in target_ids
        ]
        model.load_cases.extend([final_dead_case, live_case])

        return final_dead_case, live_case

    def apply_roof_loads(
        self,
        model_or_project: Union[StructuralModel, Project],
        ground_snow_load: Optional[float] = None,
        roof_live_load: Optional[float] = None,
        tributary_width: float = 5.0,
        exposure_factor_ce: float = 1.0,
        thermal_factor_ct: float = 1.0,
        importance_factor: float = 1.0,
        snow_case_id: str = "Snow",
        roof_live_case_id: str = "RoofLive",
    ) -> Dict[str, LoadCase]:
        """
        Calculate and assign ASCE 7 Snow and/or Roof Live load cases to roof level beams.
        """
        model = self._unwrap_model(model_or_project)
        active_units = self.get_units(model_or_project)
        is_3d = detect_model_dimensionality(model) == "3d"
        direction = "-z" if is_3d else "-y"

        elevations = self.detect_story_elevations(model)
        if not elevations:
            raise ValueError(
                "No story elevations detected in model. Ensure horizontal beams or floor nodes exist."
            )
        roof_elevation = elevations[-1]

        created_cases: Dict[str, LoadCase] = {}

        # 1. Snow Load
        if ground_snow_load is not None and ground_snow_load > 0.0:
            pf = calculate_flat_roof_snow_load(
                pg=ground_snow_load,
                ce=exposure_factor_ce,
                ct=thermal_factor_ct,
                is_factor=importance_factor,
                units=active_units,
            )
            snow_case = apply_floor_load_to_frame(
                model=model,
                pressure=pf,
                tributary_width=tributary_width,
                target_case_id=snow_case_id,
                case_name="Snow Load (ASCE 7 Ch 7)",
                story_elevations=[roof_elevation],
                direction=direction,
            )
            model.load_cases = [lc for lc in model.load_cases if lc.id != snow_case_id]
            model.load_cases.append(snow_case)
            created_cases["snow"] = snow_case

        # 2. Roof Live Load
        if roof_live_load is not None and roof_live_load > 0.0:
            lr_pressure = roof_live_load
        else:
            # Code minimum
            lr_pressure = calculate_roof_live_load(
                tributary_area=tributary_width * 6.0,
                slope_rise_over_12=0.0,
                units=active_units,
            )

        roof_live_case = apply_floor_load_to_frame(
            model=model,
            pressure=lr_pressure,
            tributary_width=tributary_width,
            target_case_id=roof_live_case_id,
            case_name="Roof Live Load (ASCE 7 Sec 4.5)",
            story_elevations=[roof_elevation],
            direction=direction,
        )
        model.load_cases = [lc for lc in model.load_cases if lc.id != roof_live_case_id]
        model.load_cases.append(roof_live_case)
        created_cases["roof_live"] = roof_live_case

        return created_cases

    def distribute_area_loads(
        self,
        model_or_project: Union[StructuralModel, Project],
        target_case_id: Optional[str] = None,
        load_type_filter: Optional[str] = None,
        use_equivalent_uniform: bool = True,
        design_action: str = "moment",
        tolerance: float = 0.05,
    ) -> int:
        """
        Distribute surface pressure from AreaLoads onto bounding beam elements.
        """
        model = self._unwrap_model(model_or_project)
        return distribute_area_loads_to_beams(
            model=model,
            target_case_id=target_case_id,
            load_type_filter=load_type_filter,
            use_equivalent_uniform=use_equivalent_uniform,
            design_action=design_action,
            tolerance=tolerance,
        )

    def calculate_model_seismic_weight(
        self,
        model: StructuralModel,
        elevations: List[float],
        tolerance: float = 0.1,
    ) -> List[float]:
        """
        Compute deterministic story seismic weights W_i directly from dead loads
        and member self-weights defined in the model.
        """
        is_3d = detect_model_dimensionality(model, tolerance=tolerance) == "3d"
        nodes_map = {n.id: n for n in model.nodes}

        n_stories = len(elevations)
        story_weights = [0.0] * n_stories

        # Helper to find closest story index
        def _closest_story_idx(z_val: float) -> int:
            diffs = [abs(z_val - elev) for elev in elevations]
            return int(diffs.index(min(diffs)))

        # 1. Sum up all Dead load cases present in the model
        dead_cases = [
            lc for lc in model.load_cases
            if any(tag in str(lc.id).lower() for tag in ("dead", "sw", "sdl"))
        ]

        # Track elements with explicit dead load
        loaded_elements: Set[Any] = set()

        for dc in dead_cases:
            # Distributed line loads
            for dl in dc.element_loads:
                elem = next((e for e in model.elements if e.id == dl.element_id), None)
                if not elem:
                    continue
                loaded_elements.add(elem.id)
                length = get_beam_length(elem, nodes_map)

                # Total vertical downward magnitude
                w_vert = abs(dl.wz) if is_3d else abs(dl.wy)
                if w_vert < 1e-6:
                    # check all components
                    w_vert = math.sqrt(dl.wx ** 2 + dl.wy ** 2 + dl.wz ** 2)

                load_mag = w_vert * length

                # Distribute to stories:
                if is_horizontal_beam(elem, nodes_map, is_3d=is_3d, tolerance=tolerance):
                    elev = get_beam_elevation(elem, nodes_map, is_3d=is_3d)
                    idx = _closest_story_idx(elev)
                    story_weights[idx] += load_mag
                else:
                    # Vertical or inclined member: split between top and bottom story
                    n_start = nodes_map[elem.start_node]
                    n_end = nodes_map[elem.end_node]
                    z1 = getattr(n_start, "z", 0.0) if is_3d else n_start.y
                    z2 = getattr(n_end, "z", 0.0) if is_3d else n_end.y
                    idx1 = _closest_story_idx(z1)
                    idx2 = _closest_story_idx(z2)
                    story_weights[idx1] += 0.5 * load_mag
                    story_weights[idx2] += 0.5 * load_mag

            # Point loads
            for pl in dc.point_loads:
                node = nodes_map.get(pl.node_id)
                if not node:
                    continue
                node_z = getattr(node, "z", 0.0) if is_3d else node.y
                p_vert = abs(pl.fz) if is_3d else abs(pl.fy)
                if p_vert < 1e-6:
                    p_vert = math.sqrt(pl.fx ** 2 + pl.fy ** 2 + pl.fz ** 2)
                idx = _closest_story_idx(node_z)
                story_weights[idx] += p_vert

        # 2. Check if self-weight was already accounted for in loaded_elements.
        # If not, add section self-weight for remaining elements
        materials_map = {m.id: m for m in model.materials}
        sections_map = {s.id: s for s in model.sections}
        active_units = self.get_units(model)
        g_accel = active_units.gravity_acceleration

        for elem in model.elements:
            if elem.id in loaded_elements:
                continue
            mat = materials_map.get(getattr(elem, "material_id", None))
            sec = sections_map.get(getattr(elem, "section_id", None))
            if not mat or not sec:
                continue

            area = calculate_section_area(sec)
            if area <= 0.0:
                continue

            unit_weight = getattr(mat, "unit_weight", None)
            if unit_weight is not None and float(unit_weight) > 0.0:
                w_unit = float(unit_weight) * area
            else:
                rho = getattr(mat, "rho", None)
                if rho is not None and float(rho) > 0.0:
                    rho_dyn = float(rho) * active_units.consistent_mass_factor
                    w_unit = rho_dyn * g_accel * area
                else:
                    from struct_core import conversion_factor
                    concrete_uw = 24.0 * conversion_factor("kN, m", active_units, "weight_density")
                    w_unit = concrete_uw * area

            length = get_beam_length(elem, nodes_map)
            total_w = float(w_unit) * length

            if is_horizontal_beam(elem, nodes_map, is_3d=is_3d, tolerance=tolerance):
                elev = get_beam_elevation(elem, nodes_map, is_3d=is_3d)
                idx = _closest_story_idx(elev)
                story_weights[idx] += total_w
            else:
                n_start = nodes_map[elem.start_node]
                n_end = nodes_map[elem.end_node]
                z1 = getattr(n_start, "z", 0.0) if is_3d else n_start.y
                z2 = getattr(n_end, "z", 0.0) if is_3d else n_end.y
                idx1 = _closest_story_idx(z1)
                idx2 = _closest_story_idx(z2)
                story_weights[idx1] += 0.5 * total_w
                story_weights[idx2] += 0.5 * total_w

        return [round(w, 2) for w in story_weights]

    def apply_lateral_seismic(
        self,
        model_or_project: Union[StructuralModel, Project],
        sds: float = 0.75,
        sd1: float = 0.40,
        r_factor: float = 8.0,
        system_type: str = "rc_smf",
        importance_factor: float = 1.0,
        tributary_width: float = 5.0,
        direction: str = "both",
        seismic_case_id: str = "EQ_X",
        story_elevations: Optional[List[float]] = None,
    ) -> Tuple[Union[LoadCase, List[LoadCase]], StoryForceResult]:
        """
        Apply ASCE 7 Equivalent Lateral Force (ELF) seismic loads to floor joints.
        Computes exact model-derived story seismic weights from dead loads and element sections.
        Supports 2D and 3D directional seismic ('X', 'Y', or 'both').
        """
        model = self._unwrap_model(model_or_project)
        active_units = self.get_units(model_or_project)
        is_3d = detect_model_dimensionality(model) == "3d"

        elevations = story_elevations or self.detect_story_elevations(model)
        if not elevations:
            raise ValueError(
                "No story elevations detected in model. Specify story_elevations or ensure model geometry has defined floor levels."
            )

        # Calculate exact story weights from model dead loads & sections
        exact_weights = self.calculate_model_seismic_weight(model, elevations)

        # Ensure model has defined dead loads/sections
        if sum(exact_weights) <= 1e-3:
            raise ValueError(
                "Model has zero effective seismic weight W. Ensure dead load cases, member sections, or materials are defined on the model."
            )

        params = SeismicParameters(
            sds=sds,
            sd1=sd1,
            r_factor=r_factor,
            importance_factor=importance_factor,
            system_type=system_type,
        )

        elf_result = calculate_seismic_elf(
            story_elevations=elevations,
            story_weights=exact_weights,
            params=params,
            units=active_units,
        )

        dir_choice = direction.upper().strip()
        do_x = dir_choice in ("X", "BOTH")
        do_y = dir_choice in ("Y", "BOTH") and is_3d

        cases_created: List[LoadCase] = []

        if do_x:
            case_id = seismic_case_id if not do_y else "EQ_X"
            point_loads_x: List[PointLoad] = []
            for idx, elev in enumerate(elevations):
                fx = elf_result.story_forces[idx + 1]
                floor_nodes = [
                    n for n in model.nodes
                    if abs((getattr(n, "z", 0.0) if is_3d else n.y) - elev) <= 0.1
                ]
                if not floor_nodes:
                    continue
                # Apply to left-most node or centroid
                target_node = min(floor_nodes, key=lambda n: n.x)
                point_loads_x.append(PointLoad(node_id=target_node.id, fx=round(fx, 3)))

            case_x = LoadCase(
                id=case_id,
                name="Seismic Load (ASCE 7 ELF - X)",
                point_loads=point_loads_x,
            )
            model.load_cases = [lc for lc in model.load_cases if lc.id != case_id]
            model.load_cases.append(case_x)
            cases_created.append(case_x)

        if do_y:
            case_id = "EQ_Y" if do_x else seismic_case_id
            point_loads_y: List[PointLoad] = []
            for idx, elev in enumerate(elevations):
                fy = elf_result.story_forces[idx + 1]
                floor_nodes = [
                    n for n in model.nodes
                    if abs((getattr(n, "z", 0.0) if is_3d else n.y) - elev) <= 0.1
                ]
                if not floor_nodes:
                    continue
                target_node = min(floor_nodes, key=lambda n: n.y)
                point_loads_y.append(PointLoad(node_id=target_node.id, fy=round(fy, 3)))

            case_y = LoadCase(
                id=case_id,
                name="Seismic Load (ASCE 7 ELF - Y)",
                point_loads=point_loads_y,
            )
            model.load_cases = [lc for lc in model.load_cases if lc.id != case_id]
            model.load_cases.append(case_y)
            cases_created.append(case_y)

        ret_case = cases_created[0] if len(cases_created) == 1 else cases_created
        return ret_case, elf_result

    def apply_lateral_wind(
        self,
        model_or_project: Union[StructuralModel, Project],
        basic_wind_speed: float = 40.0,
        exposure_category: str = "B",
        tributary_width: float = 5.0,
        direction: str = "X",
        wind_case_id: str = "Wind_X",
        story_elevations: Optional[List[float]] = None,
    ) -> Tuple[Union[LoadCase, List[LoadCase]], StoryWindResult]:
        """
        Apply ASCE 7 wind lateral forces to the building frame.
        Supports directional wind ('X', 'Y', or 'both').
        """
        model = self._unwrap_model(model_or_project)
        active_units = self.get_units(model_or_project)
        is_3d = detect_model_dimensionality(model) == "3d"

        elevations = story_elevations or self.detect_story_elevations(model)
        if not elevations:
            raise ValueError(
                "No story elevations detected in model. Specify story_elevations or ensure model geometry has defined floor levels."
            )

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
            units=active_units,
        )

        dir_choice = direction.upper().strip()
        do_x = dir_choice in ("X", "BOTH")
        do_y = dir_choice in ("Y", "BOTH") and is_3d

        cases_created: List[LoadCase] = []

        if do_x:
            case_id = wind_case_id if not do_y else "Wind_X"
            point_loads_x: List[PointLoad] = []
            for idx, elev in enumerate(elevations):
                fw = wind_result.story_forces[idx + 1]
                floor_nodes = [
                    n for n in model.nodes
                    if abs((getattr(n, "z", 0.0) if is_3d else n.y) - elev) <= 0.1
                ]
                if not floor_nodes:
                    continue
                target_node = min(floor_nodes, key=lambda n: n.x)
                point_loads_x.append(PointLoad(node_id=target_node.id, fx=round(fw, 3)))

            case_x = LoadCase(
                id=case_id,
                name="Wind Load (ASCE 7 MWFRS - X)",
                point_loads=point_loads_x,
            )
            model.load_cases = [lc for lc in model.load_cases if lc.id != case_id]
            model.load_cases.append(case_x)
            cases_created.append(case_x)

        if do_y:
            case_id = "Wind_Y" if do_x else wind_case_id
            point_loads_y: List[PointLoad] = []
            for idx, elev in enumerate(elevations):
                fw = wind_result.story_forces[idx + 1]
                floor_nodes = [
                    n for n in model.nodes
                    if abs((getattr(n, "z", 0.0) if is_3d else n.y) - elev) <= 0.1
                ]
                if not floor_nodes:
                    continue
                target_node = min(floor_nodes, key=lambda n: n.y)
                point_loads_y.append(PointLoad(node_id=target_node.id, fy=round(fw, 3)))

            case_y = LoadCase(
                id=case_id,
                name="Wind Load (ASCE 7 MWFRS - Y)",
                point_loads=point_loads_y,
            )
            model.load_cases = [lc for lc in model.load_cases if lc.id != case_id]
            model.load_cases.append(case_y)
            cases_created.append(case_y)

        ret_case = cases_created[0] if len(cases_created) == 1 else cases_created
        return ret_case, wind_result

    def generate_combinations(
        self,
        model_or_project: Union[StructuralModel, Project],
        design_method: str = "LRFD",
        orthogonal_seismic: bool = True,
    ) -> List[LoadCombination]:
        """
        Detect present load cases on the model and generate standard ASCE 7 / ACI 318 combinations.
        Automatically identifies Dead, Live, Roof Live, Snow, and multi-directional Wind / Seismic cases.
        """
        model = self._unwrap_model(model_or_project)
        case_ids = {lc.id for lc in model.load_cases}

        dead_id = "Dead" if "Dead" in case_ids else (next((cid for cid in case_ids if "dead" in str(cid).lower()), None))
        live_id = "Live" if "Live" in case_ids else (next((cid for cid in case_ids if "live" in str(cid).lower() and "roof" not in str(cid).lower()), None))
        roof_live_id = next((cid for cid in case_ids if "roof" in str(cid).lower()), None)
        snow_id = next((cid for cid in case_ids if "snow" in str(cid).lower()), None)

        wind_x_id = next((cid for cid in case_ids if "wind" in str(cid).lower() and "y" not in str(cid).lower()), None)
        wind_y_id = next((cid for cid in case_ids if "wind" in str(cid).lower() and "y" in str(cid).lower()), None)

        eq_x_id = next((cid for cid in case_ids if ("eq" in str(cid).lower() or "seis" in str(cid).lower()) and "y" not in str(cid).lower()), None)
        eq_y_id = next((cid for cid in case_ids if ("eq" in str(cid).lower() or "seis" in str(cid).lower()) and "y" in str(cid).lower()), None)

        if not dead_id:
            raise ValueError(
                "Dead load case not found. ASCE 7 / ACI 318 combinations require a defined dead load case (e.g. 'Dead' or containing 'dead')."
            )

        combs = generate_asce7_combinations(
            dead_case_id=dead_id or "Dead",
            live_case_id=live_id,
            wind_case_id=wind_x_id,
            wind_y_case_id=wind_y_id,
            seismic_case_id=eq_x_id,
            seismic_y_case_id=eq_y_id,
            roof_live_case_id=roof_live_id,
            snow_case_id=snow_id,
            orthogonal_combinations=orthogonal_seismic,
            design_method=design_method,
        )

        model.load_combinations = combs
        return combs

    @staticmethod
    def detect_story_elevations(model: StructuralModel, tolerance: float = 0.05) -> List[float]:
        """
        Inspect model nodes and horizontal beams to identify unique floor elevations.
        """
        return detect_story_elevations(model, tolerance=tolerance)
