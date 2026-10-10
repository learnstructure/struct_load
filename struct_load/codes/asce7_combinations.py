"""
Standard ASCE 7 / ACI 318 Load Combinations Generator.
Produces valid struct_core.LoadCombination instances for LRFD and ASD design.
Fully supports Dead, Live, Roof Live, Snow, and multi-directional Wind and Seismic cases.
"""

from __future__ import annotations

from typing import List, Optional
from struct_core import LoadCaseFactor, LoadCombination


def generate_asce7_combinations(
    dead_case_id: str = "Dead",
    live_case_id: Optional[str] = "Live",
    wind_case_id: Optional[str] = None,
    seismic_case_id: Optional[str] = None,
    roof_live_case_id: Optional[str] = None,
    snow_case_id: Optional[str] = None,
    wind_y_case_id: Optional[str] = None,
    seismic_y_case_id: Optional[str] = None,
    orthogonal_combinations: bool = False,
    design_method: str = "LRFD",
) -> List[LoadCombination]:
    """
    Generate code-compliant factored load combinations per ASCE 7-16/22 Chapter 2
    and ACI 318-19 Chapter 5.

    Parameters
    ----------
    dead_case_id : str
        ID of Dead load case (self-weight + superimposed).
    live_case_id : str, optional
        ID of Live load case.
    wind_case_id : str, optional
        ID of Lateral Wind load case (along X or primary direction).
    seismic_case_id : str, optional
        ID of Lateral Seismic load case (along X or primary direction).
    roof_live_case_id : str, optional
        ID of Roof Live load case (Lr).
    snow_case_id : str, optional
        ID of Snow load case (S).
    wind_y_case_id : str, optional
        ID of Lateral Wind load case along Y.
    seismic_y_case_id : str, optional
        ID of Lateral Seismic load case along Y.
    orthogonal_combinations : bool
        If True, includes ASCE 7 Section 12.5 orthogonal combinations (1.0 Ex +/- 0.3 Ey).
    design_method : str
        'LRFD' (strength design) or 'ASD' (allowable stress design).

    Returns
    -------
    List[LoadCombination]
        List of configured struct_core.LoadCombination objects.
    """
    combs: List[LoadCombination] = []

    def _make_factors(**case_weights) -> List[LoadCaseFactor]:
        factors = []
        for case_id, factor in case_weights.items():
            if case_id is not None and abs(factor) > 1e-6:
                factors.append(LoadCaseFactor(load_case_id=case_id, factor=float(factor)))
        return factors

    method = design_method.upper()

    # Identify companion roof load (either S or Lr)
    roof_case_id = snow_case_id or roof_live_case_id

    # Gather directional cases
    wind_cases = [w for w in (wind_case_id, wind_y_case_id) if w is not None]
    eq_x = seismic_case_id
    eq_y = seismic_y_case_id

    if method == "LRFD":
        # 1. 1.4 D
        combs.append(
            LoadCombination(
                id="COMB_1.4D",
                name="1.4D",
                factors=_make_factors(**{dead_case_id: 1.4}),
            )
        )

        # 2. 1.2 D + 1.6 L + 0.5 (Lr or S)
        comb_2_dict = {dead_case_id: 1.2}
        name_2 = "1.2D"
        if live_case_id:
            comb_2_dict[live_case_id] = 1.6
            name_2 += " + 1.6L"
        if roof_case_id:
            comb_2_dict[roof_case_id] = 0.5
            name_2 += f" + 0.5{roof_case_id}"

        combs.append(
            LoadCombination(
                id="COMB_1.2D+1.6L" if live_case_id else "COMB_1.2D+0.5R",
                name=name_2,
                factors=_make_factors(**comb_2_dict),
            )
        )

        # 3. 1.2 D + 1.6 (Lr or S) + (1.0 L or 0.5 W)
        if roof_case_id:
            comb_3_dict = {dead_case_id: 1.2, roof_case_id: 1.6}
            name_3 = f"1.2D + 1.6{roof_case_id}"
            if live_case_id:
                comb_3_dict[live_case_id] = 1.0
                name_3 += " + 1.0L"
            combs.append(
                LoadCombination(
                    id=f"COMB_1.2D+1.6{roof_case_id}",
                    name=name_3,
                    factors=_make_factors(**comb_3_dict),
                )
            )

        # 4. Wind combinations: 1.2 D + 1.0 W + 1.0 L + 0.5 (Lr or S) and 0.9 D +/- 1.0 W
        if wind_case_id and not wind_y_case_id:
            comb_w_dict = {dead_case_id: 1.2, wind_case_id: 1.0}
            name_w = "1.2D + 1.0L + 1.0W" if live_case_id else "1.2D + 1.0W"
            if live_case_id:
                comb_w_dict[live_case_id] = 1.0
            if roof_case_id:
                comb_w_dict[roof_case_id] = 0.5
                name_w += f" + 0.5{roof_case_id}"

            combs.append(
                LoadCombination(
                    id="COMB_1.2D+1.0L+1.0W" if live_case_id else "COMB_1.2D+1.0W",
                    name=name_w,
                    factors=_make_factors(**comb_w_dict),
                )
            )
            combs.append(
                LoadCombination(
                    id="COMB_0.9D+1.0W",
                    name="0.9D + 1.0W",
                    factors=_make_factors(**{dead_case_id: 0.9, wind_case_id: 1.0}),
                )
            )
            combs.append(
                LoadCombination(
                    id="COMB_0.9D-1.0W",
                    name="0.9D - 1.0W",
                    factors=_make_factors(**{dead_case_id: 0.9, wind_case_id: -1.0}),
                )
            )
        elif wind_cases:
            for w_case in wind_cases:
                w_tag = w_case
                comb_w_dict = {dead_case_id: 1.2, w_case: 1.0}
                name_w = f"1.2D + 1.0{w_tag}"
                if live_case_id:
                    comb_w_dict[live_case_id] = 1.0
                    name_w += " + 1.0L"
                if roof_case_id:
                    comb_w_dict[roof_case_id] = 0.5
                    name_w += f" + 0.5{roof_case_id}"

                combs.append(
                    LoadCombination(
                        id=f"COMB_1.2D+1.0L+1.0{w_tag}",
                        name=name_w,
                        factors=_make_factors(**comb_w_dict),
                    )
                )
                combs.append(
                    LoadCombination(
                        id=f"COMB_0.9D+1.0{w_tag}",
                        name=f"0.9D + 1.0{w_tag}",
                        factors=_make_factors(**{dead_case_id: 0.9, w_case: 1.0}),
                    )
                )
                combs.append(
                    LoadCombination(
                        id=f"COMB_0.9D-1.0{w_tag}",
                        name=f"0.9D - 1.0{w_tag}",
                        factors=_make_factors(**{dead_case_id: 0.9, w_case: -1.0}),
                    )
                )

        # 5. Seismic combinations: 1.2 D + 1.0 E + 1.0 L + 0.2 S and 0.9 D +/- 1.0 E
        if eq_x and not eq_y:
            # Single direction seismic (2D or uncoupled 3D)
            comb_e_dict = {dead_case_id: 1.2, eq_x: 1.0}
            name_e = "1.2D + 1.0L + 1.0E" if live_case_id else "1.2D + 1.0E"
            if live_case_id:
                comb_e_dict[live_case_id] = 1.0
            if snow_case_id:
                comb_e_dict[snow_case_id] = 0.2
                name_e += f" + 0.2{snow_case_id}"

            combs.append(
                LoadCombination(
                    id="COMB_1.2D+1.0L+1.0E" if live_case_id else "COMB_1.2D+1.0E",
                    name=name_e,
                    factors=_make_factors(**comb_e_dict),
                )
            )
            combs.append(
                LoadCombination(
                    id="COMB_0.9D+1.0E",
                    name="0.9D + 1.0E",
                    factors=_make_factors(**{dead_case_id: 0.9, eq_x: 1.0}),
                )
            )
            combs.append(
                LoadCombination(
                    id="COMB_0.9D-1.0E",
                    name="0.9D - 1.0E",
                    factors=_make_factors(**{dead_case_id: 0.9, eq_x: -1.0}),
                )
            )

        elif eq_x and eq_y:
            # 3D seismic with orthogonal directional combinations: 1.0 Ex +/- 0.3 Ey and 0.3 Ex +/- 1.0 Ey
            ortho_pairs = [
                (1.0, 0.0), (-1.0, 0.0),
                (0.0, 1.0), (0.0, -1.0),
            ]
            if orthogonal_combinations:
                ortho_pairs.extend([
                    (1.0, 0.3), (1.0, -0.3), (-1.0, 0.3), (-1.0, -0.3),
                    (0.3, 1.0), (0.3, -1.0), (-0.3, 1.0), (-0.3, -1.0),
                ])

            for fx, fy in ortho_pairs:
                sign_str = ""
                if abs(fx) > 0:
                    sign_str += f"{'+' if fx > 0 else '-'}{abs(fx):.1f}{eq_x}"
                if abs(fy) > 0:
                    sign_str += f"{'+' if fy > 0 else '-'}{abs(fy):.1f}{eq_y}"

                comb_e_dict = {dead_case_id: 1.2, eq_x: fx, eq_y: fy}
                name_e = f"1.2D + 1.0L {sign_str}"
                if live_case_id:
                    comb_e_dict[live_case_id] = 1.0
                if snow_case_id:
                    comb_e_dict[snow_case_id] = 0.2

                cid_clean = sign_str.replace("+", "p").replace("-", "m").replace(".", "")
                combs.append(
                    LoadCombination(
                        id=f"COMB_1.2D+1.0L_{cid_clean}",
                        name=name_e,
                        factors=_make_factors(**comb_e_dict),
                    )
                )
                combs.append(
                    LoadCombination(
                        id=f"COMB_0.9D_{cid_clean}",
                        name=f"0.9D {sign_str}",
                        factors=_make_factors(**{dead_case_id: 0.9, eq_x: fx, eq_y: fy}),
                    )
                )

    else:  # ASD (Allowable Stress / Serviceability)
        # 1. D
        combs.append(
            LoadCombination(
                id="ASD_D",
                name="D",
                factors=_make_factors(**{dead_case_id: 1.0}),
            )
        )

        # 2. D + L
        if live_case_id:
            combs.append(
                LoadCombination(
                    id="ASD_D+L",
                    name="D + L",
                    factors=_make_factors(**{dead_case_id: 1.0, live_case_id: 1.0}),
                )
            )

        # 3. D + (Lr or S)
        if roof_case_id:
            combs.append(
                LoadCombination(
                    id=f"ASD_D+{roof_case_id}",
                    name=f"D + {roof_case_id}",
                    factors=_make_factors(**{dead_case_id: 1.0, roof_case_id: 1.0}),
                )
            )

        # 4. Wind ASD: D + 0.6W, 0.6D + 0.6W
        for w_case in wind_cases:
            w_tag = w_case
            combs.append(
                LoadCombination(
                    id=f"ASD_D+0.6{w_tag}",
                    name=f"D + 0.6{w_tag}",
                    factors=_make_factors(**{dead_case_id: 1.0, w_case: 0.6}),
                )
            )
            combs.append(
                LoadCombination(
                    id=f"ASD_0.6D+0.6{w_tag}",
                    name=f"0.6D + 0.6{w_tag}",
                    factors=_make_factors(**{dead_case_id: 0.6, w_case: 0.6}),
                )
            )

        # 5. Seismic ASD: D + 0.7E, 0.6D + 0.7E
        if eq_x:
            combs.append(
                LoadCombination(
                    id=f"ASD_D+0.7{eq_x}",
                    name=f"D + 0.7{eq_x}",
                    factors=_make_factors(**{dead_case_id: 1.0, eq_x: 0.7}),
                )
            )
            combs.append(
                LoadCombination(
                    id=f"ASD_0.6D+0.7{eq_x}",
                    name=f"0.6D + 0.7{eq_x}",
                    factors=_make_factors(**{dead_case_id: 0.6, eq_x: 0.7}),
                )
            )
        if eq_y:
            combs.append(
                LoadCombination(
                    id=f"ASD_D+0.7{eq_y}",
                    name=f"D + 0.7{eq_y}",
                    factors=_make_factors(**{dead_case_id: 1.0, eq_y: 0.7}),
                )
            )
            combs.append(
                LoadCombination(
                    id=f"ASD_0.6D+0.7{eq_y}",
                    name=f"0.6D + 0.7{eq_y}",
                    factors=_make_factors(**{dead_case_id: 0.6, eq_y: 0.7}),
                )
            )

    return combs
