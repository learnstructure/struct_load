"""
Standard ASCE 7 / ACI 318 Load Combinations Generator.
Produces valid struct_core.LoadCombination instances for LRFD and ASD design.
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
        ID of Lateral Wind load case.
    seismic_case_id : str, optional
        ID of Lateral Seismic load case.
    roof_live_case_id : str, optional
        ID of Roof live load case.
    snow_case_id : str, optional
        ID of Snow load case.
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

    if method == "LRFD":
        # Comb 1: 1.4 D
        combs.append(
            LoadCombination(
                id="COMB_1.4D",
                name="1.4D",
                factors=_make_factors(**{dead_case_id: 1.4}),
            )
        )

        # Comb 2: 1.2 D + 1.6 L
        if live_case_id:
            combs.append(
                LoadCombination(
                    id="COMB_1.2D+1.6L",
                    name="1.2D + 1.6L",
                    factors=_make_factors(**{dead_case_id: 1.2, live_case_id: 1.6}),
                )
            )

        # Comb 3: Wind combinations (1.2D + 1.0W + 1.0L) and (0.9D +/- 1.0W)
        if wind_case_id:
            combs.append(
                LoadCombination(
                    id="COMB_1.2D+1.0L+1.0W",
                    name="1.2D + 1.0L + 1.0W",
                    factors=_make_factors(
                        **{dead_case_id: 1.2, live_case_id: 1.0 if live_case_id else 0.0, wind_case_id: 1.0}
                    ),
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

        # Comb 4: Seismic combinations (1.2D + 1.0L + 1.0E) and (0.9D +/- 1.0E)
        if seismic_case_id:
            combs.append(
                LoadCombination(
                    id="COMB_1.2D+1.0L+1.0E",
                    name="1.2D + 1.0L + 1.0E",
                    factors=_make_factors(
                        **{dead_case_id: 1.2, live_case_id: 1.0 if live_case_id else 0.0, seismic_case_id: 1.0}
                    ),
                )
            )
            combs.append(
                LoadCombination(
                    id="COMB_0.9D+1.0E",
                    name="0.9D + 1.0E",
                    factors=_make_factors(**{dead_case_id: 0.9, seismic_case_id: 1.0}),
                )
            )
            combs.append(
                LoadCombination(
                    id="COMB_0.9D-1.0E",
                    name="0.9D - 1.0E",
                    factors=_make_factors(**{dead_case_id: 0.9, seismic_case_id: -1.0}),
                )
            )

    else:  # ASD (Allowable Stress / Serviceability)
        # Comb 1: D
        combs.append(
            LoadCombination(
                id="ASD_D",
                name="D",
                factors=_make_factors(**{dead_case_id: 1.0}),
            )
        )
        # Comb 2: D + L
        if live_case_id:
            combs.append(
                LoadCombination(
                    id="ASD_D+L",
                    name="D + L",
                    factors=_make_factors(**{dead_case_id: 1.0, live_case_id: 1.0}),
                )
            )
        # Comb 3: D + 0.6W
        if wind_case_id:
            combs.append(
                LoadCombination(
                    id="ASD_D+0.6W",
                    name="D + 0.6W",
                    factors=_make_factors(**{dead_case_id: 1.0, wind_case_id: 0.6}),
                )
            )
            combs.append(
                LoadCombination(
                    id="ASD_0.6D+0.6W",
                    name="0.6D + 0.6W",
                    factors=_make_factors(**{dead_case_id: 0.6, wind_case_id: 0.6}),
                )
            )
        # Comb 4: D + 0.7E
        if seismic_case_id:
            combs.append(
                LoadCombination(
                    id="ASD_D+0.7E",
                    name="D + 0.7E",
                    factors=_make_factors(**{dead_case_id: 1.0, seismic_case_id: 0.7}),
                )
            )
            combs.append(
                LoadCombination(
                    id="ASD_0.6D+0.7E",
                    name="0.6D + 0.7E",
                    factors=_make_factors(**{dead_case_id: 0.6, seismic_case_id: 0.7}),
                )
            )

    return combs
