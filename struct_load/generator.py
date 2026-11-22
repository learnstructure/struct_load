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
    Story,
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
    detect_stories,
    get_beam_elevation,
    get_beam_length,
    is_horizontal_beam,
    resolve_base_elevation,
)
from .engine.mass_source import (
    MassSourceResult,
    calculate_mass_source,
    create_asce7_mass_source,
    nodal_lumped_masses,
)
from .engine.self_weight import (
    calculate_section_area,
    calculate_self_weight,
    element_mass_per_length,
    shell_self_weights,
    shell_tributary,
)
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
        self.messages: List[str] = []
        self.messages: List[str] = []

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

        stories = self.detect_stories(model)
        if not stories:
            raise ValueError(
                "No stories detected in model. Ensure horizontal beams or floor nodes exist."
            )
        roof_story = stories[-1]

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
                stories=[roof_story],
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
            stories=[roof_story],
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

    def _story_weights_from_mass_source(
        self,
        model: StructuralModel,
        ms_res: MassSourceResult,
        elevations: List[float],
        *,
        base_elevation: Optional[float] = None,
    ) -> List[float]:
        """Aggregate nodal seismic weights from ``ms_res`` onto ``elevations``, excluding restrained base."""
        if not elevations:
            return []
        from struct_core import vertical_axis as get_vertical_axis

        v_axis = get_vertical_axis(model)
        active_units = self.get_units(model)

        base_elev_val: Optional[float] = None
        if base_elevation is not None:
            base_elev_val = float(base_elevation)
        elif getattr(model, "base_elevation", None) is not None:
            base_elev_val = float(model.base_elevation)
        elif getattr(model, "supports", None):
            base_elev_val = resolve_base_elevation(model, messages=self.messages)

        exclude_base_bucket = (
            base_elev_val is not None
            and not any(abs(e - base_elev_val) <= 1e-9 for e in elevations)
        )

        node_tot = nodal_lumped_masses(
            model, ms_res, units=active_units, gravity_accel=ms_res.gravity_accel
        )
        lateral_only = bool(getattr(ms_res.mass_source, "lateral_only", False))
        story_weights = [0.0] * len(elevations)

        for nd in model.nodes:
            m_vec = node_tot.get(nd.id, (0.0, 0.0, 0.0))
            m_eff = float(m_vec[0]) if lateral_only else float(max(m_vec))
            if m_eff <= 0.0:
                continue
            z_val = float(getattr(nd, "z", 0.0)) if v_axis == "z" else float(nd.y)
            diffs = [abs(z_val - elev) for elev in elevations]
            min_diff = min(diffs)
            if exclude_base_bucket and abs(z_val - base_elev_val) <= min_diff + 1e-12:
                continue
            idx = int(diffs.index(min_diff))
            story_weights[idx] += m_eff * ms_res.gravity_accel

        return [round(w, 2) for w in story_weights]

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
        base_elevation: Optional[float] = None,
    ) -> List[float]:
        """
        Compute deterministic story seismic weights W_i per ASCE 7-22 Section 12.7.2 using the
        same seismic mass definition as ``create_asce7_mass_source`` and ``calculate_mass_source``
        (including element self-mass, shell self-mass, assigned nodal/element added masses, and
        qualifying dead, storage live, partition, equipment, and snow loads).

        Mass lumped at the restrained base carries no seismic inertia force and is excluded from
        above-base story weights (unless ``base_elevation`` is itself included in ``elevations``).
        """
        if not elevations:
            return []

        active_units = self.get_units(model)
        base_elev_val: Optional[float] = None
        if base_elevation is not None:
            base_elev_val = float(base_elevation)
        elif getattr(model, "base_elevation", None) is not None:
            base_elev_val = float(model.base_elevation)
        elif getattr(model, "supports", None):
            base_elev_val = resolve_base_elevation(model, messages=self.messages)

        asce7_ms = create_asce7_mass_source(
            model,
            storage_live_case_ids=storage_live_cases,
            partition_load_case_ids=partition_cases,
            snow_load_case_ids=snow_cases,
            flat_roof_snow_pressure=flat_roof_snow_pressure,
            equipment_case_ids=equipment_cases,
            include_general_live=include_general_live,
            general_live_case_ids=general_live_cases,
            general_live_factor=general_live_factor,
            include_element_mass=True,
            include_added_mass=True,
            units=active_units,
        )
        ms_res = calculate_mass_source(
            model,
            mass_source=asce7_ms,
            units=active_units,
            base_elevation=base_elev_val,
        )
        return self._story_weights_from_mass_source(
            model, ms_res, elevations, base_elevation=base_elev_val
        )

    def apply_lateral_seismic(
        self,
        model_or_project: Union[StructuralModel, Project],
        params: Optional[SeismicParameters] = None,
        direction: str = "both",
        seismic_case_id: Optional[str] = None,
        stories: Optional[List[Story]] = None,
        base_elevation: Optional[float] = None,
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
        stories : List[Story], optional
            Named stories above base. Inferred automatically from geometry if None.
        base_elevation : float, optional
            Base datum elevation for ASCE 7-22 §12.8.3 h_x measurements. Uses model.base_elevation
            when omitted, or infers from lowest supports with a message in ``self.messages``.
        mass_source : MassSource, dict, or str, optional
            Mass source specification defining load multipliers and element self-mass.
            If None, uses model.mass_source or derives weights from dead load cases.
        story_weights : List[float], optional
            Directly specified story seismic weights W_i. Overrides automatic calculation if provided.
        custom_period : float, optional
            Calculated or modal fundamental period T (s). Subject to upper limit Cu*Ta per ASCE 7 Section 12.8.2.
        distribution : str, optional
            Horizontal distribution pattern of story lateral forces among floor nodes on floors
            without a rigid diaphragm:
            - 'face' (default): distributes story force equally across nodes on the exterior boundary
              frame face along the loading direction (min x for X-direction, min y for Y-direction).
            - 'uniform': distributes story force equally across all floor nodes at that story elevation.
            Floors constrained by a rigid Z-diaphragm always distribute story lateral force in
            proportion to each node's seismic mass so the resultant passes through the floor center of
            mass per ASCE 7-22 §12.8.4.1.
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

        self.messages = []
        base_elev = (
            float(base_elevation)
            if base_elevation is not None
            else resolve_base_elevation(model, messages=self.messages)
        )
        story_list = list(stories) if stories else self.detect_stories(model)
        if not story_list:
            raise ValueError(
                "No stories detected in model. Specify stories or ensure model geometry has defined floor levels."
            )
        elevations = [float(s.elevation) for s in story_list]

        # Calculate exact story weights from explicit input, mass source, or MS_ASCE7
        ms_res: Optional[MassSourceResult] = None
        if story_weights is not None:
            exact_weights = list(story_weights)
        elif mass_source is not None or getattr(model, "mass_source", None) is not None:
            ms_target = mass_source if mass_source is not None else getattr(model, "mass_source", None)
            ms_res = calculate_mass_source(
                model, mass_source=ms_target, units=active_units, base_elevation=base_elev
            )
            exact_weights = self._story_weights_from_mass_source(
                model, ms_res, elevations, base_elevation=base_elev
            )
            if sum(exact_weights) <= 1e-3:
                asce7_ms = create_asce7_mass_source(
                    model,
                    storage_live_case_ids=storage_live_cases,
                    partition_load_case_ids=partition_cases,
                    snow_load_case_ids=snow_cases,
                    flat_roof_snow_pressure=flat_roof_snow_pressure,
                    equipment_case_ids=equipment_cases,
                    include_element_mass=True,
                    include_added_mass=True,
                    units=active_units,
                )
                ms_res = calculate_mass_source(
                    model, mass_source=asce7_ms, units=active_units, base_elevation=base_elev
                )
                exact_weights = self._story_weights_from_mass_source(
                    model, ms_res, elevations, base_elevation=base_elev
                )
        else:
            asce7_ms = create_asce7_mass_source(
                model,
                storage_live_case_ids=storage_live_cases,
                partition_load_case_ids=partition_cases,
                snow_load_case_ids=snow_cases,
                flat_roof_snow_pressure=flat_roof_snow_pressure,
                equipment_case_ids=equipment_cases,
                include_element_mass=True,
                include_added_mass=True,
                units=active_units,
            )
            ms_res = calculate_mass_source(
                model, mass_source=asce7_ms, units=active_units, base_elevation=base_elev
            )
            exact_weights = self._story_weights_from_mass_source(
                model, ms_res, elevations, base_elevation=base_elev
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
            stories=story_list,
            story_weights=exact_weights,
            params=seismic_params,
            base_elevation=base_elev,
            units=active_units,
            custom_period=custom_period,
        )

        # Resolve per-node lumped seismic masses only if the model defines horizontal rigid diaphragms
        nodes_map = {n.id: n for n in model.nodes}
        diaphs_by_elev: List[Tuple[Any, float]] = []
        node_tot_masses: Optional[Dict[Any, Tuple[float, float, float]]] = None
        if is_3d and v_axis == "z":
            for d in getattr(model, "diaphragms", None) or []:
                if getattr(d, "axis", "Z") != "Z":
                    continue
                zs_d = [float(getattr(nodes_map[nid], "z", 0.0)) for nid in d.node_ids if nid in nodes_map]
                if zs_d:
                    diaphs_by_elev.append((d, sum(zs_d) / len(zs_d)))

        if diaphs_by_elev:
            diaph_ms_res = ms_res
            if diaph_ms_res is None:
                if mass_source is not None or getattr(model, "mass_source", None) is not None:
                    ms_target = mass_source if mass_source is not None else getattr(model, "mass_source", None)
                    diaph_ms_res = calculate_mass_source(
                        model, mass_source=ms_target, units=active_units, base_elevation=base_elev
                    )
                else:
                    asce7_ms = create_asce7_mass_source(
                        model,
                        storage_live_case_ids=storage_live_cases,
                        partition_load_case_ids=partition_cases,
                        snow_load_case_ids=snow_cases,
                        flat_roof_snow_pressure=flat_roof_snow_pressure,
                        equipment_case_ids=equipment_cases,
                        include_element_mass=True,
                        include_added_mass=True,
                        units=active_units,
                    )
                    diaph_ms_res = calculate_mass_source(
                        model, mass_source=asce7_ms, units=active_units, base_elevation=base_elev
                    )
            node_tot_masses = nodal_lumped_masses(
                model, diaph_ms_res, units=active_units, gravity_accel=diaph_ms_res.gravity_accel
            )

        def _story_point_loads(elev: float, story_force: float, axis_dir: str) -> List[PointLoad]:
            floor_nodes = [
                n for n in model.nodes
                if abs((getattr(n, "z", 0.0) if v_axis == "z" else n.y) - elev) <= 0.1
            ]
            story_diaphs = [d for d, z_d in diaphs_by_elev if abs(z_d - elev) <= 0.1]
            if not story_diaphs:
                if not floor_nodes:
                    return []
                if dist_choice == "face":
                    min_coord = min(n.x if axis_dir == "X" else n.y for n in floor_nodes)
                    face_nodes = [
                        n for n in floor_nodes
                        if abs((n.x if axis_dir == "X" else n.y) - min_coord) <= 0.05
                    ]
                    f_node = story_force / len(face_nodes)
                    targets = face_nodes
                else:  # uniform
                    f_node = story_force / len(floor_nodes)
                    targets = floor_nodes
                kw_name = "fx" if axis_dir == "X" else "fy"
                return [PointLoad(node_id=nd.id, **{kw_name: round(f_node, 4)}) for nd in targets]

            # Floor constrained by a rigid Z-diaphragm: distribute by nodal seismic mass (ASCE 7-22 §12.8.4.1)
            seen_ids = {n.id for n in floor_nodes}
            target_nodes = list(floor_nodes)
            for d in story_diaphs:
                for nid in d.node_ids:
                    if nid in nodes_map and nid not in seen_ids:
                        seen_ids.add(nid)
                        target_nodes.append(nodes_map[nid])
            if not target_nodes:
                return []

            axis_idx = 0 if axis_dir == "X" else 1
            kw_name = "fx" if axis_dir == "X" else "fy"
            weights = [
                max(0.0, float((node_tot_masses or {}).get(nd.id, (0.0, 0.0, 0.0))[axis_idx]))
                for nd in target_nodes
            ]
            w_sum = sum(weights)
            if w_sum > 1e-12:
                return [
                    PointLoad(node_id=nd.id, **{kw_name: float(story_force) * (w_i / w_sum)})
                    for nd, w_i in zip(target_nodes, weights)
                    if w_i > 0.0
                ]

            f_node = float(story_force) / len(target_nodes)
            fallback_msg = (
                f"Story at elevation {elev:g} has rigid diaphragm(s) {[d.id for d in story_diaphs]} with zero "
                f"nodal mass; distributed ELF force uniformly across {len(target_nodes)} floor nodes."
            )
            if fallback_msg not in self.messages:
                self.messages.append(fallback_msg)
            return [PointLoad(node_id=nd.id, **{kw_name: f_node}) for nd in target_nodes]

        diaph_story_ids = sorted({
            d.id for elev in elevations for d, z_d in diaphs_by_elev if abs(z_d - elev) <= 0.1
        })
        if diaph_story_ids:
            self.messages.append(
                f"Rigid diaphragm floor(s) {diaph_story_ids} distributed ELF story forces in proportion "
                f"to nodal seismic mass so the resultant acts at the center of mass (ASCE 7-22 §12.8.4.1)."
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
                point_loads_x.extend(_story_point_loads(elev, elf_result.story_forces[idx + 1], "X"))

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
                point_loads_y.extend(_story_point_loads(elev, elf_result.story_forces[idx + 1], "Y"))

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
        stories: Optional[List[Story]] = None,
        base_elevation: Optional[float] = None,
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

        self.messages = []
        base_elev = (
            float(base_elevation)
            if base_elevation is not None
            else resolve_base_elevation(model, messages=self.messages)
        )
        story_list = list(stories) if stories else self.detect_stories(model)
        if not story_list:
            raise ValueError(
                "No stories detected in model. Specify stories or ensure model geometry has defined floor levels."
            )
        elevations = [float(s.elevation) for s in story_list]

        story_heights = []
        prev_h = base_elev
        for elev in elevations:
            story_heights.append(elev - prev_h)
            prev_h = elev

        wind_params = WindParameters(
            basic_wind_speed=basic_wind_speed,
            exposure_category=exposure_category,
        )

        wind_result = calculate_wind_forces(
            stories=story_list,
            tributary_widths=[tributary_width] * len(elevations),
            story_heights=story_heights,
            params=wind_params,
            base_elevation=base_elev,
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

        # Categorize load cases using canonical classifier; record every omitted case in self.messages
        self.messages = []
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
                else:
                    self.messages.append(
                        f"Load case {lc.id!r} (type 'roof_live') was left out of the generated combinations: "
                        f"only one roof_live case is supported (already using {roof_live_id!r})."
                    )
            elif cat == "snow":
                if snow_id is None:
                    snow_id = lc.id
                else:
                    self.messages.append(
                        f"Load case {lc.id!r} (type 'snow') was left out of the generated combinations: "
                        f"only one snow case is supported (already using {snow_id!r})."
                    )
            elif cat == "wind":
                is_y = l_dir == "Y" or "_y" in cid_str or cid_str.endswith("y")
                if is_y:
                    if wind_y_id is None:
                        wind_y_id = lc.id
                    else:
                        self.messages.append(
                            f"Load case {lc.id!r} (type 'wind') was left out of the generated combinations: "
                            f"only one wind case per direction is supported (already using {wind_y_id!r} for Y)."
                        )
                else:
                    if wind_x_id is None:
                        wind_x_id = lc.id
                    else:
                        self.messages.append(
                            f"Load case {lc.id!r} (type 'wind') was left out of the generated combinations: "
                            f"only one wind case per direction is supported (already using {wind_x_id!r} for X)."
                        )
            elif cat == "seismic":
                is_y = l_dir == "Y" or "_y" in cid_str or cid_str.endswith("y")
                if is_y:
                    if eq_y_id is None:
                        eq_y_id = lc.id
                    else:
                        self.messages.append(
                            f"Load case {lc.id!r} (type 'seismic') was left out of the generated combinations: "
                            f"only one seismic case per direction is supported (already using {eq_y_id!r} for Y)."
                        )
                else:
                    if eq_x_id is None:
                        eq_x_id = lc.id
                    else:
                        self.messages.append(
                            f"Load case {lc.id!r} (type 'seismic') was left out of the generated combinations: "
                            f"only one seismic case per direction is supported (already using {eq_x_id!r} for X)."
                        )
            elif cat == "partition":
                self.messages.append(
                    f"Load case {lc.id!r} (type 'partition') was left out of the generated combinations: "
                    f"partition weight enters the seismic mass source only (ASCE 7-22 §12.7.2), not load combinations."
                )
            else:
                self.messages.append(
                    f"Load case {lc.id!r} (type {cat!r}) was left out of the generated combinations: "
                    f"ASCE 7-22 §2.3/§2.4 combination generation does not include load type {cat!r}; "
                    f"assign its load_type or add it to combinations explicitly."
                )

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
    def detect_stories(
        model: StructuralModel,
        tolerance: float = 0.05,
        *,
        messages: Optional[List[str]] = None,
    ) -> List[Story]:
        """
        Inspect model nodes and horizontal beams to identify named floor stories above base.
        """
        return detect_stories(model, tolerance=tolerance, messages=messages)
