"""
Standard ASCE 7-22 / ACI 318 Load Combinations Generator.
Produces valid struct_core.LoadCombination instances for LRFD and ASD design.
Fully supports Dead, Live, Roof Live, Snow, and multi-directional Wind and Seismic cases.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union
from struct_core import LoadCaseFactor, LoadCombination


def _fmt(value: float) -> str:
    """Compact load-factor label with at least one decimal (1.0, 1.2, 0.15, 1.105, ...)."""
    text = f"{value:.4g}"
    return text if "." in text or "e" in text else text + ".0"


def _direction_vectors(n_dirs: int, orthogonal: bool) -> List[Tuple[float, ...]]:
    """Signed participation of each lateral direction case, both senses of every direction.

    One direction: (+1), (-1). Two directions: (+-1, 0), (0, +-1) and, with ``orthogonal``,
    the 100-30 combinations of Section 12.5.3.1(a).
    """
    if n_dirs == 1:
        return [(1.0,), (-1.0,)]
    vectors = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
    if orthogonal:
        for major, minor in ((1.0, 0.3), (0.3, 1.0)):
            for sx in (1.0, -1.0):
                for sy in (1.0, -1.0):
                    vectors.append((sx * major, sy * minor))
    return vectors


def _lateral_tag(case_ids: Sequence[str], vector: Tuple[float, ...], scale: float, label: str) -> str:
    """Signed lateral part of a combination name, e.g. '+1.0E' or '+1.0EQ_X-0.3EQ_Y'."""
    parts = []
    for cid, f in zip(case_ids, vector):
        if abs(f) > 1e-12:
            parts.append(f"{'+' if f > 0 else '-'}{_fmt(abs(f) * scale)}{label if len(case_ids) == 1 else cid}")
    return "".join(parts)


def generate_asce7_combinations(
    dead_case_id: Union[str, Sequence[str]] = "Dead",
    live_case_id: Optional[Union[str, Sequence[str]]] = "Live",
    wind_case_id: Optional[str] = None,
    seismic_case_id: Optional[str] = None,
    roof_live_case_id: Optional[str] = None,
    snow_case_id: Optional[str] = None,
    wind_y_case_id: Optional[str] = None,
    seismic_y_case_id: Optional[str] = None,
    orthogonal_combinations: bool = False,
    design_method: str = "LRFD",
    sds: Optional[float] = None,
    rho: float = 1.0,
    include_overstrength: bool = False,
    omega_0: Optional[float] = None,
    seismic_design_category: Optional[str] = None,
    reduce_live_in_seismic: bool = False,
) -> List[LoadCombination]:
    """
    Generate code-compliant factored load combinations per ASCE 7-22 Sections 2.3 and 2.4
    and Section 12.4 (seismic load effects), as adopted by ACI 318-25 Chapter 5.

    Strength design (Section 2.3.1 and 2.3.6):
        1a. 1.4D
        2a. 1.2D + 1.6L + (0.5Lr or 0.3S)
        3a. 1.2D + (1.6Lr or 1.0S) + (L or 0.5W)
        4a. 1.2D + 1.0W + L + (0.5Lr or 0.3S)
        5a. 0.9D + 1.0W
        6.  1.2D + Ev + Eh + L + 0.15S
        7.  0.9D - Ev + Eh
    Allowable stress design (Section 2.4.1 and 2.4.5):
        1a. D;  2a. D + L;  3a. D + (Lr or 0.7S);  4a. D + 0.75L + 0.75(Lr or 0.7S)
        5a. D + 0.6W;  6a. D + 0.75L + 0.75(0.6W) + 0.75(Lr or 0.7S);  7a. 0.6D + 0.6W
        8.  1.0D + 0.7Ev + 0.7Eh
        9.  1.0D + 0.525Ev + 0.525Eh + 0.75L + 0.1S
        10. 0.6D - 0.7Ev + 0.7Eh

    Wind and seismic cases are reversible, so every combination containing them is generated
    for both senses of each direction. Ev = 0.2 SDS D (Section 12.4.2.2) is folded into the dead
    load factor; Eh = rho QE (Section 12.4.2.1), or Emh = Omega0 QE with ``include_overstrength``
    (Section 12.4.3).

    Parameters
    ----------
    dead_case_id : str
        ID of Dead load case (self-weight + superimposed).
    live_case_id : str, optional
        ID of Live load case.
    wind_case_id, wind_y_case_id : str, optional
        IDs of the lateral wind cases along X and Y.
    seismic_case_id, seismic_y_case_id : str, optional
        IDs of the lateral seismic cases along X and Y.
    roof_live_case_id : str, optional
        ID of Roof Live load case (Lr).
    snow_case_id : str, optional
        ID of Snow load case (S).
    orthogonal_combinations : bool
        If True and both seismic directions are given, adds the 100-30 combinations of
        Section 12.5.3.1(a).
    design_method : str
        'LRFD' (strength design) or 'ASD' (allowable stress design).
    sds : float, optional
        Design spectral response acceleration at short periods SDS; enables Ev.
    rho : float, default=1.0
        Redundancy factor rho per Section 12.3.4.
    include_overstrength : bool, default=False
        Generate the seismic combinations with overstrength Emh = Omega0 QE (Section 12.4.3).
    omega_0 : float, optional
        Overstrength factor Omega0 (Table 12.2-1); required with ``include_overstrength``.
    seismic_design_category : str, optional
        Seismic Design Category. In SDC B, Ev = 0 (Section 12.4.2.2 Exception 2).
    reduce_live_in_seismic : bool, default=False
        Use 0.5L in LRFD combination 6 (Section 2.3.6 Exception 1: Lo <= 100 psf,
        not a garage or place of public assembly).

    Returns
    -------
    List[LoadCombination]
        List of configured struct_core.LoadCombination objects.
    """
    if include_overstrength and omega_0 is None:
        raise ValueError("omega_0 (Table 12.2-1) is required when include_overstrength=True.")

    combs: List[LoadCombination] = []
    dead_ids: List[str] = [str(dead_case_id)] if isinstance(dead_case_id, str) else [str(d) for d in dead_case_id]
    live_ids: List[str] = ([str(live_case_id)] if isinstance(live_case_id, str) else [str(l) for l in live_case_id]) if live_case_id else []

    def add(cid: str, name: str, weights: Dict[str, float]) -> None:
        expanded: Dict[str, float] = {}
        for case_id, factor in weights.items():
            if case_id == "__DEAD__":
                for did in dead_ids:
                    expanded[did] = factor
            elif case_id == "__LIVE__":
                for lid in live_ids:
                    expanded[lid] = factor
            elif case_id is not None:
                expanded[case_id] = factor

        factors = [
            LoadCaseFactor(load_case_id=case_id, factor=round(float(factor), 6))
            for case_id, factor in expanded.items()
            if case_id is not None and abs(factor) > 1e-6
        ]
        kind_val = "service" if method == "ASD" else "strength"
        combs.append(LoadCombination(id=cid, name=name, factors=factors, kind=kind_val, method=method))

    method = design_method.upper()
    D = "__DEAD__"
    L = "__LIVE__" if live_ids else None
    wind_ids = [w for w in (wind_case_id, wind_y_case_id) if w is not None]
    eq_ids = [e for e in (seismic_case_id, seismic_y_case_id) if e is not None]
    wind_vectors = _direction_vectors(len(wind_ids), orthogonal=False) if wind_ids else []
    eq_vectors = _direction_vectors(len(eq_ids), orthogonal_combinations and len(eq_ids) == 2) if eq_ids else []

    has_ev = sds is not None and not (
        seismic_design_category and seismic_design_category.strip().upper() == "B"
    )
    ev = 0.2 * sds if has_ev else 0.0

    def lateral(case_ids: Sequence[str], vector: Tuple[float, ...], scale: float) -> Dict[str, float]:
        return {cid: f * scale for cid, f in zip(case_ids, vector) if abs(f) > 1e-12}

    # Companion roof loads: each of Lr and S is investigated as the governing roof load
    roof_options: List[Tuple[str, str]] = []  # (case_id, kind)
    if roof_live_case_id:
        roof_options.append((roof_live_case_id, "Lr"))
    if snow_case_id:
        roof_options.append((snow_case_id, "S"))

    if method == "LRFD":
        # 1a. 1.4D
        add("COMB_1.4D", "1.4D", {D: 1.4})

        # 2a. 1.2D + 1.6L + (0.5Lr or 0.3S)
        companions = [(cid, 0.5 if kind == "Lr" else 0.3) for cid, kind in roof_options] or [(None, 0.0)]
        for roof_id, f_roof in companions:
            w = {D: 1.2}
            name = "1.2D"
            if L:
                w[L] = 1.6
                name += " + 1.6L"
            if roof_id:
                w[roof_id] = f_roof
                name += f" + {_fmt(f_roof)}{roof_id}"
            if L or roof_id:
                add("COMB_" + name.replace(" ", ""), name, w)

        # 3a. 1.2D + (1.6Lr or 1.0S) + (L or 0.5W)
        for roof_id, kind in roof_options:
            f_roof = 1.6 if kind == "Lr" else 1.0
            base = {D: 1.2, roof_id: f_roof}
            name = f"1.2D + {_fmt(f_roof)}{roof_id}"
            if L:
                add("COMB_" + (name + " + 1.0L").replace(" ", ""), name + " + 1.0L", {**base, L: 1.0})
            for vec in wind_vectors:
                tag = _lateral_tag(wind_ids, vec, 0.5, "W")
                add("COMB_" + (name + tag).replace(" ", ""), f"{name} {tag}", {**base, **lateral(wind_ids, vec, 0.5)})

        # 4a. 1.2D + 1.0W + L + (0.5Lr or 0.3S);  5a. 0.9D + 1.0W
        companions = [(cid, 0.5 if kind == "Lr" else 0.3) for cid, kind in roof_options] or [(None, 0.0)]
        for vec in wind_vectors:
            tag = _lateral_tag(wind_ids, vec, 1.0, "W")
            for roof_id, f_roof in companions:
                w = {D: 1.2, **lateral(wind_ids, vec, 1.0)}
                name = "1.2D"
                if L:
                    w[L] = 1.0
                    name += " + 1.0L"
                if roof_id:
                    w[roof_id] = f_roof
                    name += f" + {_fmt(f_roof)}{roof_id}"
                add("COMB_" + (name + tag).replace(" ", ""), f"{name} {tag}", w)
            add("COMB_0.9D" + tag, f"0.9D {tag}", {D: 0.9, **lateral(wind_ids, vec, 1.0)})

        # 6. (1.2 + 0.2SDS)D + Eh + L + 0.15S;  7. (0.9 - 0.2SDS)D + Eh
        if eq_ids:
            f_e = omega_0 if include_overstrength else rho
            e_label = "Em" if include_overstrength else "E"
            f_live = 0.5 if reduce_live_in_seismic else 1.0
            d_add, d_sub = round(1.2 + ev, 4), round(0.9 - ev, 4)
            for vec in eq_vectors:
                tag = _lateral_tag(eq_ids, vec, f_e, e_label)
                w = {D: d_add, **lateral(eq_ids, vec, f_e)}
                name = f"{_fmt(d_add)}D"
                if L:
                    w[L] = f_live
                    name += f" + {_fmt(f_live)}L"
                if snow_case_id:
                    w[snow_case_id] = 0.15
                    name += f" + 0.15{snow_case_id}"
                add("COMB_" + name.replace(" ", "") + tag, f"{name} {tag}", w)
                add(f"COMB_{_fmt(d_sub)}D{tag}", f"{_fmt(d_sub)}D {tag}", {D: d_sub, **lateral(eq_ids, vec, f_e)})

    else:  # ASD
        # 1a. D;  2a. D + L
        add("ASD_D", "D", {D: 1.0})
        if L:
            add("ASD_D+L", "D + L", {D: 1.0, L: 1.0})

        # 3a. D + (Lr or 0.7S);  4a. D + 0.75L + 0.75(Lr or 0.7S)
        for roof_id, kind in roof_options:
            f_roof = 1.0 if kind == "Lr" else 0.7
            add(f"ASD_D+{_fmt(f_roof)}{roof_id}", f"D + {_fmt(f_roof)}{roof_id}", {D: 1.0, roof_id: f_roof})
            if L:
                f_r = 0.75 * f_roof
                add(
                    f"ASD_D+0.75L+{_fmt(f_r)}{roof_id}",
                    f"D + 0.75L + {_fmt(f_r)}{roof_id}",
                    {D: 1.0, L: 0.75, roof_id: f_r},
                )

        # 5a. D + 0.6W;  6a. D + 0.75L + 0.75(0.6W) + 0.75(Lr or 0.7S);  7a. 0.6D + 0.6W
        for vec in wind_vectors:
            tag = _lateral_tag(wind_ids, vec, 0.6, "W")
            add(f"ASD_D{tag}", f"D {tag}", {D: 1.0, **lateral(wind_ids, vec, 0.6)})
            tag_75 = _lateral_tag(wind_ids, vec, 0.45, "W")
            companions = [(cid, 0.75 * (1.0 if kind == "Lr" else 0.7)) for cid, kind in roof_options] or [(None, 0.0)]
            for roof_id, f_roof in companions:
                w = {D: 1.0, **lateral(wind_ids, vec, 0.45)}
                name = "D"
                if L:
                    w[L] = 0.75
                    name += " + 0.75L"
                if roof_id:
                    w[roof_id] = f_roof
                    name += f" + {_fmt(f_roof)}{roof_id}"
                if L or roof_id:
                    add("ASD_" + name.replace(" ", "") + tag_75, f"{name} {tag_75}", w)
            add(f"ASD_0.6D{tag}", f"0.6D {tag}", {D: 0.6, **lateral(wind_ids, vec, 0.6)})

        # 8. (1.0 + 0.14SDS)D + 0.7Eh;  9. (1.0 + 0.105SDS)D + 0.525Eh + 0.75L + 0.1S;
        # 10. (0.6 - 0.14SDS)D + 0.7Eh
        if eq_ids:
            f_e = omega_0 if include_overstrength else rho
            e_label = "Em" if include_overstrength else "E"
            d8, d9, d10 = round(1.0 + 0.7 * ev, 4), round(1.0 + 0.525 * ev, 4), round(0.6 - 0.7 * ev, 4)
            for vec in eq_vectors:
                tag_70 = _lateral_tag(eq_ids, vec, 0.7 * f_e, e_label)
                add(f"ASD_{_fmt(d8)}D{tag_70}", f"{_fmt(d8)}D {tag_70}", {D: d8, **lateral(eq_ids, vec, 0.7 * f_e)})
                tag_525 = _lateral_tag(eq_ids, vec, 0.525 * f_e, e_label)
                w = {D: d9, **lateral(eq_ids, vec, 0.525 * f_e)}
                name = f"{_fmt(d9)}D"
                if L:
                    w[L] = 0.75
                    name += " + 0.75L"
                if snow_case_id:
                    w[snow_case_id] = 0.1
                    name += f" + 0.1{snow_case_id}"
                add("ASD_" + name.replace(" ", "") + tag_525, f"{name} {tag_525}", w)
                add(f"ASD_{_fmt(d10)}D{tag_70}", f"{_fmt(d10)}D {tag_70}", {D: d10, **lateral(eq_ids, vec, 0.7 * f_e)})

    return combs
