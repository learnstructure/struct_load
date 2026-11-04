"""
High-level LoadGenerator orchestrating automated load determination on structural models.
Supports 2D and 3D frames, unit auto-inference, exact model-derived seismic weights,
multi-directional lateral loads, roof snow and live loads, and area load distribution.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Dict, List, Optional, Set, Tuple, Union
from struct_core import (
    DistributedLoad,
    LoadCase,
    LoadCaseFactor,
    LoadCombination,
    Node,
    PointLoad,
    Project,
    StructuralModel,
    Units,
    conversion_factor,
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
from .codes.asce7_tables import get_occupancy_live_load, lookup_occupancy
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
from .engine.mass_source import calculate_mass_source
from .engine.self_weight import calculate_section_area, calculate_self_weight, material_weight_density
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

        raise ValueError("Units unknown: pass units= or use a Project")

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
        *,
        occupancy: str,
        superimposed_dead: float,
        tributary_width: Optional[float] = None,
        one_way_span: Optional[Literal["x", "y"]] = None,
        include_self_weight: bool = True,
        custom_live_load: Optional[float] = None,
        dead_case_id: str = "Dead",
        live_case_id: str = "Live",
        two_way: Optional[bool] = None,
        partition_weight: Optional[float] = None,
        partition_case_id: str = "PartitionWeight",
        allow_partial_floor: bool = False,
    ) -> Tuple[LoadCase, LoadCase]:
        """
        Calculate and assign realistic Dead and Live load cases to the structural model.
        Automatically senses 2D (wy along -Y) vs 3D (wz along -Z) geometry.
        In 3D, defaults to two-way floor panel yield-line distribution unless tributary_width is explicitly provided;
        one-way distribution (every 2D frame) needs ``tributary_width`` (model length unit).

        ``occupancy`` selects the live load (ASCE 7-22 Table 4.3-1) unless ``custom_live_load`` is given;
        ``superimposed_dead`` is the floor dead load in the model pressure unit (0 for none). Neither is assumed.

        Where the floor live load includes a partition allowance (ASCE 7-22 §4.3.2, e.g. offices), a
        separate ``partition_case_id`` case carries the partition weight for the effective seismic
        weight (§12.7.2 item 2): ``partition_weight`` (model pressure unit), or the 10 psf (0.48 kPa)
        minimum when it is not given. The case is a seismic-weight case only; load combinations do
        not include it (the partition allowance is already part of the live load). With
        ``custom_live_load`` the partition weight must be given explicitly to be included.
        """
        model = self._unwrap_model(model_or_project)
        active_units = self.get_units(model_or_project)
        from struct_core import vertical_axis as get_vertical_axis
        v_axis = get_vertical_axis(model)
        is_3d = detect_model_dimensionality(model) == "3d"
        direction = "-z" if (is_3d or v_axis == "z") else "-y"

        if two_way is None:
            use_two_way = is_3d and (one_way_span is None and tributary_width is None)
        else:
            use_two_way = two_way

        if is_3d:
            if tributary_width is not None and not use_two_way:
                raise ValueError("tributary_width is for 2D frames only; in 3D specify one_way_span ('x' or 'y') or use two_way=True.")
            if not use_two_way and one_way_span is None:
                raise ValueError("one_way_span ('x' or 'y') is required for 3D one-way floor load distribution.")
        else:
            if tributary_width is None:
                raise ValueError("tributary_width (model length unit) is required for 2D floor load distribution.")

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
            one_way_span=one_way_span,
            target_case_id=live_case_id,
            case_name=f"Live Load ({occupancy.title()})",
            direction=direction,
            two_way=use_two_way,
            allow_partial_floor=allow_partial_floor,
        )

        # 2. Dead Load: Superimposed + Self-Weight
        sdl_case = apply_floor_load_to_frame(
            model=model,
            pressure=superimposed_dead,
            tributary_width=tributary_width,
            one_way_span=one_way_span,
            target_case_id=dead_case_id,
            case_name="Dead Load",
            direction=direction,
            two_way=use_two_way,
            allow_partial_floor=allow_partial_floor,
        )

        # Self-weight and SDL stay as separate entries so downstream consumers (e.g. mass
        # sources with include_element_mass=True) can tell member self-weight apart via is_self_weight.
        dead_loads: List[DistributedLoad] = list(sdl_case.element_loads)
        if include_self_weight:
            sw_case = calculate_self_weight(
                model=model,
                load_case_id="SW",
                units=active_units,
                direction=direction,
            )
            dead_loads.extend(sw_case.element_loads)

        final_dead_case = LoadCase(
            id=dead_case_id,
            name="Dead Load (Self-Weight + SDL)" if include_self_weight else "Dead Load (SDL)",
            load_type="dead",
            element_loads=dead_loads,
            include_self_weight=False,
        )

        # 3. Partition weight for the effective seismic weight (ASCE 7-22 §12.7.2 item 2)
        if partition_weight is None and custom_live_load is None and lookup_occupancy(occupancy).partition_psf > 0.0:
            if active_units.length in ("in", "ft") or active_units.force in ("kip", "lbf"):
                partition_weight = 10.0 * conversion_factor("lbf, ft", active_units, "pressure")
            else:
                partition_weight = 0.48 * conversion_factor("kN, m", active_units, "pressure")
        partition_case = None
        if partition_weight is not None and partition_weight > 0.0:
            partition_case = apply_floor_load_to_frame(
                model=model,
                pressure=partition_weight,
                tributary_width=tributary_width,
                one_way_span=one_way_span,
                target_case_id=partition_case_id,
                case_name="Partition weight (ASCE 7-22 §12.7.2 item 2, seismic weight only)",
                direction=direction,
                two_way=use_two_way,
                allow_partial_floor=allow_partial_floor,
            )

        # Attach / replace in model (case-insensitive ID match)
        target_ids = {dead_case_id.lower(), live_case_id.lower(), partition_case_id.lower()}
        model.load_cases = [
            lc for lc in model.load_cases if str(lc.id).lower() not in target_ids
        ]
        model.load_cases.extend([final_dead_case, live_case] + ([partition_case] if partition_case else []))

        self.loaded_floor_area = getattr(live_case, "loaded_floor_area", {})
        self.unloaded_beams = getattr(live_case, "unloaded_beams", [])
        final_dead_case.loaded_floor_area = getattr(sdl_case, "loaded_floor_area", {})
        final_dead_case.unloaded_beams = getattr(sdl_case, "unloaded_beams", [])

        return final_dead_case, live_case

    def apply_roof_loads(
        self,
        model_or_project: Union[StructuralModel, Project],
        *,
        tributary_width: Optional[float] = None,
        one_way_span: Optional[Literal["x", "y"]] = None,
        two_way: Optional[bool] = None,
        ground_snow_load: Optional[float] = None,
        roof_live_load: Optional[float] = None,
        tributary_area: Optional[float] = None,
        exposure_factor_ce: float = 1.0,
        thermal_factor_ct: float = 1.0,
        risk_category: str = "II",
        snow_case_id: str = "Snow",
        roof_live_case_id: str = "RoofLive",
    ) -> Dict[str, LoadCase]:
        """
        Calculate and assign ASCE 7-22 Snow and/or Roof Live load cases to roof level beams.

        ``ground_snow_load`` and ``roof_live_load`` are pressures in the model's pressure unit;
        ``tributary_width`` is the loaded width of the roof beams (model length unit); ``risk_category`` selects the
        minimum low-slope roof snow load (Table 7.3-4). Without ``roof_live_load`` the code minimum Lr applies,
        reduced for the members' tributary area ``tributary_area`` (model area unit, Eq. 4.8-1) when it is given,
        else unreduced (R1 = 1).
        """
        model = self._unwrap_model(model_or_project)
        active_units = self.get_units(model_or_project)
        from struct_core import vertical_axis as get_vertical_axis
        v_axis = get_vertical_axis(model)
        is_3d = detect_model_dimensionality(model) == "3d"
        direction = "-z" if (is_3d or v_axis == "z") else "-y"

        elevations = self.detect_story_elevations(model)
        if not elevations:
            raise ValueError(
                "No story elevations detected in model. Ensure horizontal beams or floor nodes exist."
            )
        roof_elevation = elevations[-1]

        if two_way is None:
            use_two_way = is_3d and (one_way_span is None and tributary_width is None)
        else:
            use_two_way = two_way

        if is_3d:
            if tributary_width is not None and not use_two_way:
                raise ValueError("tributary_width is for 2D frames only; in 3D specify one_way_span ('x' or 'y') or use two_way=True.")
            if not use_two_way and one_way_span is None:
                raise ValueError("one_way_span ('x' or 'y') is required for 3D one-way floor load distribution.")
        else:
            if tributary_width is None:
                raise ValueError("tributary_width (model length unit) is required for 2D floor load distribution.")

        created_cases: Dict[str, LoadCase] = {}

        # 1. Snow Load
        if ground_snow_load is not None and ground_snow_load > 0.0:
            pf = calculate_flat_roof_snow_load(
                pg=ground_snow_load,
                ce=exposure_factor_ce,
                ct=thermal_factor_ct,
                risk_category=risk_category,
                units=active_units,
            )
            snow_case = apply_floor_load_to_frame(
                model=model,
                pressure=pf,
                tributary_width=tributary_width,
                one_way_span=one_way_span,
                two_way=use_two_way,
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
                tributary_area=tributary_area if tributary_area is not None else 0.0,
                slope_rise_over_12=0.0,
                units=active_units,
            )

        roof_live_case = apply_floor_load_to_frame(
            model=model,
            pressure=lr_pressure,
            tributary_width=tributary_width,
            one_way_span=one_way_span,
            two_way=use_two_way,
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
        storage_live_cases: Optional[List[str]] = None,
        partition_cases: Optional[List[str]] = None,
        snow_cases: Optional[List[str]] = None,
        flat_roof_snow_pressure: float = 0.0,
        equipment_cases: Optional[List[str]] = None,
        include_general_live: bool = False,
        general_live_cases: Optional[List[str]] = None,
        general_live_factor: float = 0.0,
    ) -> List[float]:
        """
        Compute deterministic story seismic weights W_i per ASCE 7-22 Section 12.7.2:
        1. Dead load (structural self-weight & superimposed dead): 100% (factor = 1.0).
        2. Storage live load: minimum 25% (factor = 0.25) per Section 12.7.2 Item 1.
           Note: Public garages and open parking structures are excluded (0%).
           General floor live load (office, residential, etc.) is 0% per Section 12.7.2.
        3. Partitions: 100% (factor = 1.0) per Section 12.7.2 Item 2.
        4. Permanent operating equipment: 100% (factor = 1.0) per Section 12.7.2 Item 3.
        5. Snow load: 15% where flat roof snow load Pf > 45 psf (2.16 kN/m^2) per Section 12.7.2 Item 4.
        """
        from struct_core import vertical_axis as get_vertical_axis
        v_axis = get_vertical_axis(model)
        is_3d = detect_model_dimensionality(model, tolerance=tolerance) == "3d"
        nodes_map = {n.id: n for n in model.nodes}
        active_units = self.get_units(model)

        from struct_core import conversion_factor
        threshold_snow = 2.16 * conversion_factor("kN, m", active_units, "pressure")
        snow_factor = 0.15 if flat_roof_snow_pressure > (threshold_snow - 1e-6) else 0.0

        n_stories = len(elevations)
        story_weights = [0.0] * n_stories

        # Helper to find closest story index
        def _closest_story_idx(z_val: float) -> int:
            diffs = [abs(z_val - elev) for elev in elevations]
            return int(diffs.index(min(diffs)))

        all_cases = {lc.id: lc for lc in model.load_cases}

        # 1. Dead load cases (factor 1.0)
        dead_cids = [
            cid for cid in all_cases
            if any(tag in str(cid).lower() for tag in ("dead", "sw", "sdl"))
        ]

        # 2. Storage live load cases (factor 0.25)
        if storage_live_cases is not None:
            storage_cids = [cid for cid in storage_live_cases if cid in all_cases]
        else:
            storage_cids = [
                cid for cid in all_cases
                if any(tag in str(cid).lower() for tag in ("storage", "stor"))
            ]

        # 3. Partition cases (factor 1.0)
        if partition_cases is not None:
            part_cids = [cid for cid in partition_cases if cid in all_cases]
        else:
            part_cids = [
                cid for cid in all_cases
                if any(tag in str(cid).lower() for tag in ("partition", "part"))
            ]

        # 4. Equipment cases (factor 1.0)
        if equipment_cases is not None:
            equip_cids = [cid for cid in equipment_cases if cid in all_cases]
        else:
            equip_cids = [
                cid for cid in all_cases
                if any(tag in str(cid).lower() for tag in ("equip", "mep"))
            ]

        # 5. Snow cases (factor 0.15 if Pf > 45 psf / 2.16 kPa, else 0.0)
        if snow_cases is not None:
            snow_cids = [cid for cid in snow_cases if cid in all_cases]
        else:
            snow_cids = [
                cid for cid in all_cases
                if any(tag in str(cid).lower() for tag in ("snow",))
            ]

        # 6. General live cases (0.0 unless include_general_live is True)
        gen_cids = []
        if include_general_live and general_live_factor > 0.0:
            if general_live_cases is not None:
                gen_cids = [cid for cid in general_live_cases if cid in all_cases]
            else:
                storage_set = set(storage_cids)
                gen_cids = [
                    cid for cid in all_cases
                    if any(tag in str(cid).lower() for tag in ("live", "ll"))
                    and cid not in storage_set
                    and not any(tag in str(cid).lower() for tag in ("roof", "lr", "storage", "stor"))
                ]

        weighted_cases: List[Tuple[LoadCase, float, bool]] = []
        for cid in dead_cids:
            weighted_cases.append((all_cases[cid], 1.0, True))
        for cid in storage_cids:
            weighted_cases.append((all_cases[cid], 0.25, False))
        for cid in part_cids:
            weighted_cases.append((all_cases[cid], 1.0, False))
        for cid in equip_cids:
            weighted_cases.append((all_cases[cid], 1.0, False))
        if snow_factor > 0.0:
            for cid in snow_cids:
                weighted_cases.append((all_cases[cid], snow_factor, False))
        if gen_cids and general_live_factor > 0.0:
            for cid in gen_cids:
                weighted_cases.append((all_cases[cid], general_live_factor, False))

        # Track elements with explicit dead load
        loaded_elements: Set[Any] = set()

        for lc, c_scale, marks_dead in weighted_cases:
            if c_scale <= 0.0:
                continue
            # Distributed line loads
            for dl in lc.element_loads:
                elem = next((e for e in model.elements if e.id == dl.element_id), None)
                if not elem:
                    continue
                if marks_dead:
                    loaded_elements.add(elem.id)
                length = get_beam_length(elem, nodes_map)

                # Total vertical downward magnitude
                w_vert = abs(dl.wz) if v_axis == "z" else abs(dl.wy)
                if w_vert < 1e-6:
                    w_vert = math.sqrt(dl.wx ** 2 + dl.wy ** 2 + dl.wz ** 2)

                load_mag = w_vert * length * c_scale

                # Distribute to stories:
                if is_horizontal_beam(elem, nodes_map, is_3d=is_3d, tolerance=tolerance, vertical_axis=v_axis):
                    elev = get_beam_elevation(elem, nodes_map, is_3d=is_3d, vertical_axis=v_axis)
                    idx = _closest_story_idx(elev)
                    story_weights[idx] += load_mag
                else:
                    # Vertical or inclined member: split between top and bottom story
                    n_start = nodes_map[elem.start_node]
                    n_end = nodes_map[elem.end_node]
                    z1 = getattr(n_start, "z", 0.0) if v_axis == "z" else n_start.y
                    z2 = getattr(n_end, "z", 0.0) if v_axis == "z" else n_end.y
                    idx1 = _closest_story_idx(z1)
                    idx2 = _closest_story_idx(z2)
                    story_weights[idx1] += 0.5 * load_mag
                    story_weights[idx2] += 0.5 * load_mag

            # Point loads
            for pl in lc.point_loads:
                node = nodes_map.get(pl.node_id)
                if not node:
                    continue
                node_z = getattr(node, "z", 0.0) if v_axis == "z" else node.y
                p_vert = abs(pl.fz) if v_axis == "z" else abs(pl.fy)
                if p_vert < 1e-6:
                    p_vert = math.sqrt(pl.fx ** 2 + pl.fy ** 2 + pl.fz ** 2)
                idx = _closest_story_idx(node_z)
                story_weights[idx] += p_vert * c_scale

        # 2. Check if self-weight was already accounted for in loaded_elements.
        # If not, add section self-weight for remaining elements
        materials_map = {m.id: m for m in model.materials}
        sections_map = {s.id: s for s in model.sections}
        g_accel = active_units.gravity_acceleration

        for elem in model.elements:
            if elem.id in loaded_elements:
                continue
            if str(getattr(elem, "type", "")).lower() == "spring":
                continue
            sec_id = getattr(elem, "section_id", None)
            sec = sections_map.get(sec_id)
            if sec is None:
                raise ValueError(f"Element {elem.id!r} references missing section {sec_id!r}.")
            mat_id = getattr(sec, "material_id", None)
            mat = materials_map.get(mat_id)
            if mat is None:
                raise ValueError(f"Section {sec.id!r} on element {elem.id!r} references missing material {mat_id!r}.")

            area = calculate_section_area(sec)
            if area <= 0.0:
                continue

            w_unit = material_weight_density(mat, active_units) * area

            length = get_beam_length(elem, nodes_map)
            total_w = float(w_unit) * length

            if is_horizontal_beam(elem, nodes_map, is_3d=is_3d, tolerance=tolerance, vertical_axis=v_axis):
                elev = get_beam_elevation(elem, nodes_map, is_3d=is_3d, vertical_axis=v_axis)
                idx = _closest_story_idx(elev)
                story_weights[idx] += total_w
            else:
                n_start = nodes_map[elem.start_node]
                n_end = nodes_map[elem.end_node]
                z1 = getattr(n_start, "z", 0.0) if v_axis == "z" else n_start.y
                z2 = getattr(n_end, "z", 0.0) if v_axis == "z" else n_end.y
                idx1 = _closest_story_idx(z1)
                idx2 = _closest_story_idx(z2)
                story_weights[idx1] += 0.5 * total_w
                story_weights[idx2] += 0.5 * total_w

        return [round(w, 2) for w in story_weights]

    def apply_lateral_seismic(
        self,
        model_or_project: Union[StructuralModel, Project],
        params: Optional[SeismicParameters] = None,
        direction: str = "both",
        seismic_case_id: Optional[str] = None,
        story_elevations: Optional[List[float]] = None,
        mass_source: Optional[Union[Any, str]] = None,
        story_weights: Optional[List[float]] = None,
        custom_period: Optional[float] = None,
        distribution: str = "face",
        storage_live_cases: Optional[List[str]] = None,
        partition_cases: Optional[List[str]] = None,
        snow_cases: Optional[List[str]] = None,
        flat_roof_snow_pressure: float = 0.0,
        equipment_cases: Optional[List[str]] = None,
    ) -> Tuple[Union[LoadCase, List[LoadCase]], StoryForceResult]:
        """
        Apply ASCE 7 Equivalent Lateral Force (ELF) seismic loads to floor joints.

        Computes exact model-derived story seismic weights from mass sources, dead loads,
        and element sections per ASCE 7-22 Section 12.7.2, and determines design base shear V
        and story lateral forces Fx. Supports 2D and 3D directional seismic ('X', 'Y', or 'both').

        When direction='both' in 3D models, two separate and independent load cases are generated
        (e.g. 'EQ_X' and 'EQ_Y') so that orthogonal seismic loads do not act simultaneously.

        Parameters
        ----------
        model_or_project : StructuralModel or Project
            Target structural model or project container.
        params : SeismicParameters
            Seismic design criteria (sds, sd1, r_factor, importance_factor, tl, system_type, s1).
            Required; site parameters (sds, sd1) have no default.
        direction : str, optional
            Seismic loading direction: 'X', 'Y', or 'both' (default: 'both').
        seismic_case_id : str, optional
            Custom load case ID. If direction='both', suffixes '_X' and '_Y' are appended
            (e.g. 'EQ' -> 'EQ_X', 'EQ_Y'). Defaults to 'EQ_X' and 'EQ_Y'.
        story_elevations : List[float], optional
            Story elevations above base. Inferred automatically from geometry if None.
        mass_source : MassSource, dict, or str, optional
            Mass source specification defining load multipliers and element self-mass.
            If None, uses model.mass_source or derives weights from dead load cases.
        story_weights : List[float], optional
            Directly specified story seismic weights W_i. Overrides automatic calculation if provided.
        custom_period : float, optional
            Calculated or modal fundamental period T (s). Subject to upper limit Cu*Ta per ASCE 7 Section 12.8.2.
        distribution : str, optional
            Horizontal distribution pattern of story lateral forces among floor nodes:
            - 'face' (default): distributes story force equally across nodes on the exterior boundary
              frame face along the loading direction (min x for X-direction, min y for Y-direction).
            - 'uniform': distributes story force equally across all floor nodes at that story elevation.
        storage_live_cases : List[str], optional
            Load case IDs for storage live load (minimum 25% factor per ASCE 7-22 Section 12.7.2 Item 1).
        partition_cases : List[str], optional
            Load case IDs for partition load (100% factor per Section 12.7.2 Item 2).
        snow_cases : List[str], optional
            Load case IDs for snow load (15% factor if Pf > 45 psf / 2.16 kPa per Section 12.7.2 Item 4).
        flat_roof_snow_pressure : float, default=0.0
            Flat roof snow pressure Pf.
        equipment_cases : List[str], optional
            Load case IDs for permanent operating equipment (100% factor per Section 12.7.2 Item 3).

        Returns
        -------
        Tuple[Union[LoadCase, List[LoadCase]], StoryForceResult]
            - Generated LoadCase (or list of [case_x, case_y] if direction='both')
            - StoryForceResult dataclass with base shear, Cs, period, total weight, and story forces.
        """
        from struct_core import vertical_axis as get_vertical_axis
        model = self._unwrap_model(model_or_project)
        active_units = self.get_units(model_or_project)
        v_axis = get_vertical_axis(model)
        is_3d = detect_model_dimensionality(model) == "3d"

        dist_choice = distribution.lower().strip()
        if dist_choice not in ("face", "uniform"):
            raise ValueError(
                f"Invalid distribution '{distribution}'. Must be 'face' or 'uniform'."
            )

        elevations = story_elevations or self.detect_story_elevations(model)
        if not elevations:
            raise ValueError(
                "No story elevations detected in model. Specify story_elevations or ensure model geometry has defined floor levels."
            )

        # Calculate exact story weights from explicit input, mass source, or dead loads
        if story_weights is not None:
            exact_weights = list(story_weights)
        elif mass_source is not None or getattr(model, "mass_source", None) is not None:
            ms_target = mass_source if mass_source is not None else getattr(model, "mass_source", None)
            ms_res = calculate_mass_source(model, mass_source=ms_target, units=active_units)
            exact_weights = [round(ms_res.story_weights.get(elev, 0.0), 2) for elev in elevations]
            if sum(exact_weights) <= 1e-3:
                exact_weights = self.calculate_model_seismic_weight(
                    model,
                    elevations,
                    storage_live_cases=storage_live_cases,
                    partition_cases=partition_cases,
                    snow_cases=snow_cases,
                    flat_roof_snow_pressure=flat_roof_snow_pressure,
                    equipment_cases=equipment_cases,
                )
        else:
            exact_weights = self.calculate_model_seismic_weight(
                model,
                elevations,
                storage_live_cases=storage_live_cases,
                partition_cases=partition_cases,
                snow_cases=snow_cases,
                flat_roof_snow_pressure=flat_roof_snow_pressure,
                equipment_cases=equipment_cases,
            )

        # Ensure model has defined dead loads/sections
        if sum(exact_weights) <= 1e-3:
            raise ValueError(
                "Model has zero effective seismic weight W. Ensure dead load cases, member sections, or materials are defined on the model."
            )

        if params is None:
            raise ValueError(
                "SeismicParameters 'params' is required for apply_lateral_seismic; site parameters (sds, sd1) have no default."
            )
        seismic_params = params

        elf_result = calculate_seismic_elf(
            story_elevations=elevations,
            story_weights=exact_weights,
            params=seismic_params,
            units=active_units,
            custom_period=custom_period,
        )

        dir_choice = direction.upper().strip()
        do_x = dir_choice in ("X", "BOTH")
        do_y = dir_choice in ("Y", "BOTH")

        if do_y and not is_3d:
            if dir_choice == "Y":
                raise ValueError(
                    "Y-direction lateral seismic load is not applicable to 2D models where Y is the vertical gravity axis."
                )
            do_y = False

        if not do_x and not do_y:
            raise ValueError(f"Invalid direction '{direction}'. Must be 'X', 'Y', or 'both'.")

        cases_created: List[LoadCase] = []

        if do_x and do_y:
            base_id = seismic_case_id.rstrip("_XY") if seismic_case_id else "EQ"
            case_id_x = f"{base_id}_X"
            case_id_y = f"{base_id}_Y"
        elif do_x:
            case_id_x = seismic_case_id or "EQ_X"
            case_id_y = None
        else:
            case_id_x = None
            case_id_y = seismic_case_id or "EQ_Y"

        if do_x:
            point_loads_x: List[PointLoad] = []
            for idx, elev in enumerate(elevations):
                fx = elf_result.story_forces[idx + 1]
                floor_nodes = [
                    n for n in model.nodes
                    if abs((getattr(n, "z", 0.0) if v_axis == "z" else n.y) - elev) <= 0.1
                ]
                if not floor_nodes:
                    continue
                if dist_choice == "face":
                    min_x = min(n.x for n in floor_nodes)
                    face_nodes = [n for n in floor_nodes if abs(n.x - min_x) <= 0.05]
                    f_node = fx / len(face_nodes)
                    for target_node in face_nodes:
                        point_loads_x.append(PointLoad(node_id=target_node.id, fx=round(f_node, 4)))
                else:  # uniform
                    f_node = fx / len(floor_nodes)
                    for target_node in floor_nodes:
                        point_loads_x.append(PointLoad(node_id=target_node.id, fx=round(f_node, 4)))

            case_x = LoadCase(
                id=case_id_x,
                name="Seismic Load (ASCE 7 ELF - X)",
                load_type="seismic",
                direction="X",
                point_loads=point_loads_x,
                auto_load="ASCE 7-22",
                seismic_params={
                    "direction": "X",
                    "sds": float(seismic_params.sds),
                    "sd1": float(seismic_params.sd1),
                    "s1": float(seismic_params.s1) if getattr(seismic_params, "s1", None) is not None else None,
                    "tl": float(seismic_params.tl) if getattr(seismic_params, "tl", None) is not None else None,
                    "r_factor": float(getattr(seismic_params, "r_factor", 8.0) or 8.0),
                    "importance_factor": float(getattr(seismic_params, "importance_factor", 1.0) or 1.0),
                    "site_class": str(getattr(seismic_params, "site_class", "D") or "D"),
                    "system_type": str(getattr(seismic_params, "system_type", "rc_smf") or "rc_smf"),
                    "method": int(getattr(seismic_params, "method", 2) or 2),
                    "base_shear": float(elf_result.base_shear),
                    "weight_used": float(elf_result.total_weight),
                    "period_used": float(elf_result.ta or elf_result.period),
                    "coeff_used": float(elf_result.cs),
                },
            )
            model.load_cases = [lc for lc in model.load_cases if lc.id != case_id_x]
            model.load_cases.append(case_x)
            cases_created.append(case_x)

        if do_y:
            point_loads_y: List[PointLoad] = []
            for idx, elev in enumerate(elevations):
                fy = elf_result.story_forces[idx + 1]
                floor_nodes = [
                    n for n in model.nodes
                    if abs((getattr(n, "z", 0.0) if v_axis == "z" else n.y) - elev) <= 0.1
                ]
                if not floor_nodes:
                    continue
                if dist_choice == "face":
                    min_y = min(n.y for n in floor_nodes)
                    face_nodes = [n for n in floor_nodes if abs(n.y - min_y) <= 0.05]
                    f_node = fy / len(face_nodes)
                    for target_node in face_nodes:
                        point_loads_y.append(PointLoad(node_id=target_node.id, fy=round(f_node, 4)))
                else:  # uniform
                    f_node = fy / len(floor_nodes)
                    for target_node in floor_nodes:
                        point_loads_y.append(PointLoad(node_id=target_node.id, fy=round(f_node, 4)))

            case_y = LoadCase(
                id=case_id_y,
                name="Seismic Load (ASCE 7 ELF - Y)",
                load_type="seismic",
                direction="Y",
                point_loads=point_loads_y,
                auto_load="ASCE 7-22",
                seismic_params={
                    "direction": "Y",
                    "sds": float(seismic_params.sds),
                    "sd1": float(seismic_params.sd1),
                    "s1": float(seismic_params.s1) if getattr(seismic_params, "s1", None) is not None else None,
                    "tl": float(seismic_params.tl) if getattr(seismic_params, "tl", None) is not None else None,
                    "r_factor": float(getattr(seismic_params, "r_factor", 8.0) or 8.0),
                    "importance_factor": float(getattr(seismic_params, "importance_factor", 1.0) or 1.0),
                    "site_class": str(getattr(seismic_params, "site_class", "D") or "D"),
                    "system_type": str(getattr(seismic_params, "system_type", "rc_smf") or "rc_smf"),
                    "method": int(getattr(seismic_params, "method", 2) or 2),
                    "base_shear": float(elf_result.base_shear),
                    "weight_used": float(elf_result.total_weight),
                    "period_used": float(elf_result.ta or elf_result.period),
                    "coeff_used": float(elf_result.cs),
                },
            )
            model.load_cases = [lc for lc in model.load_cases if lc.id != case_id_y]
            model.load_cases.append(case_y)
            cases_created.append(case_y)

        ret_case = cases_created[0] if len(cases_created) == 1 else cases_created
        return ret_case, elf_result

    def apply_lateral_wind(
        self,
        model_or_project: Union[StructuralModel, Project],
        *,
        basic_wind_speed: float,
        tributary_width: float,
        exposure_category: str = "B",
        direction: str = "X",
        wind_case_id: str = "Wind_X",
        story_elevations: Optional[List[float]] = None,
    ) -> Tuple[Union[LoadCase, List[LoadCase]], StoryWindResult]:
        """
        Apply ASCE 7 wind lateral forces to the building frame.
        Supports directional wind ('X', 'Y', or 'both'). ``basic_wind_speed`` (the site's mapped V) and
        ``tributary_width`` (the loaded width of the frame, model length unit) are required.
        """
        from struct_core import vertical_axis as get_vertical_axis
        model = self._unwrap_model(model_or_project)
        active_units = self.get_units(model_or_project)
        v_axis = get_vertical_axis(model)
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
                    if abs((getattr(n, "z", 0.0) if v_axis == "z" else n.y) - elev) <= 0.1
                ]
                if not floor_nodes:
                    continue
                target_node = min(floor_nodes, key=lambda n: n.x)
                point_loads_x.append(PointLoad(node_id=target_node.id, fx=round(fw, 3)))

            case_x = LoadCase(
                id=case_id,
                name="Wind Load (ASCE 7 MWFRS - X)",
                load_type="wind",
                direction="X",
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
                    if abs((getattr(n, "z", 0.0) if v_axis == "z" else n.y) - elev) <= 0.1
                ]
                if not floor_nodes:
                    continue
                target_node = min(floor_nodes, key=lambda n: n.y)
                point_loads_y.append(PointLoad(node_id=target_node.id, fy=round(fw, 3)))

            case_y = LoadCase(
                id=case_id,
                name="Wind Load (ASCE 7 MWFRS - Y)",
                load_type="wind",
                direction="Y",
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
        sds: Optional[float] = None,
        rho: float = 1.0,
        include_overstrength: bool = False,
        omega_0: Optional[float] = None,
        seismic_design_category: Optional[str] = None,
        reduce_live_in_seismic: bool = False,
    ) -> List[LoadCombination]:
        """
        Detect present load cases on the model and generate standard ASCE 7 / ACI 318 combinations.
        Automatically identifies Dead, Live, Roof Live, Snow, and multi-directional Wind / Seismic cases.
        All dead-type cases (IDs containing 'dead' or 'sdl', e.g. 'DEAD' and 'SuperDead') receive the dead-load factor.
        Supports vertical seismic effect Ev = 0.2*SDS*D and overstrength combinations per ASCE 7-22 Section 12.4.
        """
        model = self._unwrap_model(model_or_project)

        # Default sds, rho, omega_0, and sdc from project.seismic_basis if present
        if isinstance(model_or_project, Project) and model_or_project.seismic_basis is not None:
            sb = model_or_project.seismic_basis
            if sds is None and getattr(sb, "sds", None) is not None:
                sds = float(sb.sds)
            if rho == 1.0 and getattr(sb, "redundancy_factor", None) is not None:
                rho = float(sb.redundancy_factor)
            if omega_0 is None and getattr(sb, "omega_0", None) is not None:
                omega_0 = float(sb.omega_0)
            if seismic_design_category is None and getattr(sb, "sdc", None) is not None:
                seismic_design_category = str(sb.sdc)

        ordered_cases = list(model.load_cases)
        ordered_ids = [lc.id for lc in ordered_cases]

        from struct_core import classify_load_case

        # Categorize load cases using canonical classifier
        dead_ids: List[Any] = []
        live_ids: List[Any] = []
        roof_live_id: Optional[Any] = None
        snow_id: Optional[Any] = None
        wind_x_id: Optional[Any] = None
        wind_y_id: Optional[Any] = None
        eq_x_id: Optional[Any] = None
        eq_y_id: Optional[Any] = None

        for lc in ordered_cases:
            cat = classify_load_case(lc, allow_unknown=False)
            l_dir = (getattr(lc, "direction", None) or "").upper()
            cid_str = str(lc.id).lower()

            if cat in ("dead", "super_dead"):
                dead_ids.append(lc.id)
            elif cat in ("live", "storage_live"):
                live_ids.append(lc.id)
            elif cat == "roof_live":
                if roof_live_id is None:
                    roof_live_id = lc.id
            elif cat == "snow":
                if snow_id is None:
                    snow_id = lc.id
            elif cat == "wind":
                is_y = l_dir == "Y" or "_y" in cid_str or cid_str.endswith("y")
                if is_y:
                    if wind_y_id is None:
                        wind_y_id = lc.id
                else:
                    if wind_x_id is None:
                        wind_x_id = lc.id
            elif cat == "seismic":
                is_y = l_dir == "Y" or "_y" in cid_str or cid_str.endswith("y")
                if is_y:
                    if eq_y_id is None:
                        eq_y_id = lc.id
                else:
                    if eq_x_id is None:
                        eq_x_id = lc.id
            elif cat == "partition":
                continue  # Partition weight is for seismic mass source only, not load combinations

        if not dead_ids:
            raise ValueError(
                "Dead load case not found. ASCE 7 / ACI 318 combinations require at least one defined dead load case (e.g. 'Dead', 'Gravity', or load_type='dead')."
            )

        combs = generate_asce7_combinations(
            dead_case_id=dead_ids,
            live_case_id=live_ids if live_ids else None,
            wind_case_id=wind_x_id,
            wind_y_case_id=wind_y_id,
            seismic_case_id=eq_x_id,
            seismic_y_case_id=eq_y_id,
            roof_live_case_id=roof_live_id,
            snow_case_id=snow_id,
            orthogonal_combinations=orthogonal_seismic,
            design_method=design_method,
            sds=sds,
            rho=rho,
            include_overstrength=include_overstrength,
            omega_0=omega_0,
            seismic_design_category=seismic_design_category,
            reduce_live_in_seismic=reduce_live_in_seismic,
        )

        model.load_combinations = combs
        return combs

    @staticmethod
    def detect_story_elevations(model: StructuralModel, tolerance: float = 0.05) -> List[float]:
        """
        Inspect model nodes and horizontal beams to identify unique floor elevations.
        """
        return detect_story_elevations(model, tolerance=tolerance)
