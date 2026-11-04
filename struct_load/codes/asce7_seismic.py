"""
ASCE 7-22 Equivalent Lateral Force (ELF) Seismic Design Procedure.
Compliant with ASCE 7-22 Chapter 11 and Section 12.8.

References:
    - ASCE 7-22 Section 12.8.1: Seismic Base Shear (Eq. 12.8-1)
    - ASCE 7-22 Section 12.8.1.1: Calculation of Seismic Response Coefficient (Method 1 & Method 2, Eq. 12.8-2 to 12.8-7)
    - ASCE 7-22 Section 12.8.1.3: Maximum SDS Value in Determination of Cs and Ev
    - ASCE 7-22 Section 12.8.2: Period Determination (Table 12.8-1, Eq. 12.8-8 to 12.8-11, Table 12.8-2)
    - ASCE 7-22 Section 12.8.3: Vertical Distribution of Seismic Forces (Eq. 12.8-12, 12.8-13)
    - ASCE 7-22 Section 12.8.4: Horizontal Distribution of Forces / Story Shear (Eq. 12.8-14)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


@dataclass
class SeismicParameters:
    """
    Design spectral response accelerations and system coefficients per ASCE 7-22.

    Parameters
    ----------
    sds : float
        Design spectral response acceleration at short periods, SDS (g), per Sec 11.4.5.
    sd1 : float
        Design spectral response acceleration at 1-sec period, SD1 (g), per Sec 11.4.5.
    r_factor : float, optional
        Response modification coefficient R; by default that of ``system_type`` in Table 12.2-1 (8.0 for RC SMF).
    importance_factor : float, optional
        Seismic importance factor I_e; by default that of ``risk_category`` in Table 1.5-2 (I/II: 1.0, III: 1.25,
        IV: 1.5).
    tl : float, optional
        Long-period transition period TL (s) per Sec 11.4.6 (a mapped site value). It only matters for periods above
        4 s, the smallest mapped TL; a period beyond that without TL raises ValueError.
    s1 : float, optional
        Mapped MCE_R spectral response acceleration parameter at 1-sec period S1 (g)
        per Sec 11.4.2 or 11.4.4. Used for Eq. (12.8-7) when S1 >= 0.6.
    system_type : str, default='rc_smf'
        Structural system type from Table 12.8-2:
        - 'steel_smf': Steel moment-resisting frame (Ct=0.028 / 0.0724, x=0.8)
        - 'rc_smf' / 'rc_imf' / 'rc_omf': Concrete moment-resisting frame (Ct=0.016 / 0.0466, x=0.9)
        - 'steel_ebf': Steel eccentrically braced frame (Ct=0.030 / 0.0731, x=0.75)
        - 'steel_brbf': Steel buckling-restrained braced frame (Ct=0.030 / 0.0731, x=0.75)
        - 'all_other' / 'rc_shear_wall': All other systems (Ct=0.020 / 0.0488, x=0.75)
    site_class : str, default='D'
        Site soil classification per Sec 11.4.3: 'A', 'B', 'BC', 'C', 'CD', 'D', 'DE', 'E', or 'F'.
    risk_category : str, default='II'
        Risk Category per Table 1.5-1: 'I', 'II', 'III', or 'IV'.
    method : int, default=2
        Method to calculate Cs per Sec 12.8.1.1:
        - 1: Method 1 (Eq. 12.8-2) using multi-period design response spectrum Sa(T) / (R/Ie).
        - 2: Method 2 (Eq. 12.8-3 to 12.8-5) using SDS and SD1 parameters.
    sa_spectrum : callable or dict, optional
        Multi-period response spectrum Sa(T) for Method 1. Can be a callable f(T) -> Sa
        or a dictionary {period: Sa}. Required for Method 1.
    apply_sds_cap : bool, default=False
        Whether to evaluate and apply the Sec 12.8.1.3 cap on SDS (max(1.0, 0.70*SDS)).
    has_irregularities : bool, default=False
        Whether structure has horizontal or vertical irregularities defined in Sec 12.3.2.
    redundancy_factor : float, default=1.0
        Redundancy factor rho per Sec 12.3.4.2 (must be 1.0 to qualify for Sec 12.8.1.3 cap).
    num_stories : int, optional
        Number of stories above the base N. Used for Sec 12.8.1.3 cap (<= 5 stories)
        and approximate period Eq. (12.8-9) (Ta = 0.1 N).
    """
    sds: float
    sd1: float
    site_class: str
    risk_category: str
    system_type: Optional[str] = None
    tl: Optional[float] = None
    damping_ratio: float = 0.05
    r_factor: Optional[float] = None
    importance_factor: Optional[float] = None
    s1: Optional[float] = None
    method: int = 2
    sa_spectrum: Optional[Union[Callable[[float], float], Dict[float, float]]] = None
    apply_sds_cap: bool = False
    has_irregularities: bool = False
    redundancy_factor: float = 1.0
    num_stories: Optional[int] = None

    def __post_init__(self) -> None:
        if self.r_factor is None:
            if not self.system_type:
                raise ValueError("system_type is required when r_factor is not given.")
            self.r_factor = seismic_system_coefficients(self.system_type)[0]
        if self.importance_factor is None:
            self.importance_factor = seismic_importance_factor(self.risk_category)

    def long_period_transition(self, period: float) -> float:
        """TL for a response at ``period`` (s). Without a given TL, periods up to the smallest mapped TL (4 s,
        ASCE 7-22 Sec 11.4.6 maps) lie below every possible TL; a longer period needs the site's TL."""
        if self.tl is not None:
            return self.tl
        if period > MIN_MAPPED_TL:
            raise ValueError(
                f"The long-period transition period TL (Sec 11.4.6) is needed for T = {period:.2f} s > "
                f"{MIN_MAPPED_TL:g} s; give the site's mapped TL."
            )
        return MIN_MAPPED_TL


# Smallest long-period transition period TL on the ASCE 7-22 maps (s)
MIN_MAPPED_TL = 4.0

# Seismic importance factor Ie by Risk Category (ASCE 7-22 Table 1.5-2)
IMPORTANCE_FACTORS: Dict[str, float] = {"I": 1.0, "II": 1.0, "III": 1.25, "IV": 1.5}


def seismic_importance_factor(risk_category: str) -> float:
    """Seismic importance factor Ie of a Risk Category (ASCE 7-22 Table 1.5-2)."""
    key = str(risk_category).strip().upper()
    if key not in IMPORTANCE_FACTORS:
        raise ValueError(f"Unknown Risk Category '{risk_category}'; use one of {list(IMPORTANCE_FACTORS)}.")
    return IMPORTANCE_FACTORS[key]


@dataclass
class StoryForceResult:
    """
    Calculated lateral seismic forces, story shears, and ELF parameters at each level.

    Attributes
    ----------
    base_shear : float
        Total seismic base shear V (kN or kips) per Eq. (12.8-1).
    cs : float
        Seismic response coefficient Cs per Sec 12.8.1.1.
    ta : float
        Fundamental period T used in analysis (s) per Sec 12.8.2.
    total_weight : float
        Effective seismic weight of structure W (kN or kips) per Sec 12.7.2.
    story_forces : Dict[int, float]
        Lateral design force Fx at each story level per Eq. (12.8-12).
    story_weights : Dict[int, float], optional
        Seismic weight wx assigned to each level.
    story_shears : Dict[int, float], optional
        Design story shear Vx at each story per Eq. (12.8-14): Vx = sum_{i=x}^n F_i.
    cvx : Dict[int, float], optional
        Vertical distribution factor Cvx at each story per Eq. (12.8-13).
    k : float, optional
        Vertical distribution exponent k related to period T per Sec 12.8.3.
    cu : float, optional
        Coefficient for upper limit on calculated period Cu from Table 12.8-1.
    t_used : float, optional
        Period T used to evaluate Cs (s), after applying Cu * Ta cap if applicable.
    sds_used : float, optional
        Effective SDS value used (capped per Sec 12.8.1.3 if applicable).
    method_used : int, optional
        ELF calculation method used (1 or 2).
    is_sds_capped : bool, optional
        Whether Sec 12.8.1.3 SDS cap was applied.
    """
    base_shear: float
    cs: float
    ta: float
    total_weight: float
    story_forces: Dict[int, float]
    story_weights: Optional[Dict[int, float]] = None
    story_shears: Optional[Dict[int, float]] = None
    cvx: Optional[Dict[int, float]] = None
    k: Optional[float] = None
    cu: Optional[float] = None
    t_used: Optional[float] = None
    sds_used: Optional[float] = None
    method_used: Optional[int] = None
    is_sds_capped: Optional[bool] = None
    s1_bound_evaluated: bool = True  # False when S1 was not given, so Eq. (12.8-7) could not be checked


def get_cu_coefficient(sd1: float) -> float:
    """
    Coefficient Cu for upper limit on calculated period per ASCE 7-22 Table 12.8-1.

    Parameters
    ----------
    sd1 : float
        Design spectral response acceleration parameter at 1 s.

    Returns
    -------
    float
        Cu coefficient (1.4 to 1.7).
    """
    if sd1 >= 0.3:
        return 1.4
    elif sd1 >= 0.2:
        return round(1.5 - (sd1 - 0.2) / 0.1 * 0.1, 3)
    elif sd1 >= 0.15:
        return round(1.6 - (sd1 - 0.15) / 0.05 * 0.1, 3)
    elif sd1 >= 0.1:
        return round(1.7 - (sd1 - 0.1) / 0.05 * 0.1, 3)
    else:
        return 1.7


def check_sds_cap_eligibility(
    params: SeismicParameters,
    period: float,
    num_stories: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    Verify all 7 criteria of ASCE 7-22 Section 12.8.1.3 for capping SDS at max(1.0, 0.70*SDS).

    Returns
    -------
    Tuple[bool, str]
        (is_eligible, reason_string)
    """
    stories = num_stories if num_stories is not None else params.num_stories

    # Criterion 1: No irregularities (Sec 12.3.2)
    if params.has_irregularities:
        return False, "Structure has horizontal or vertical irregularities (Sec 12.8.1.3 Item 1)."

    # Criterion 2: <= 5 stories above base (Sec 12.8.1.3 Item 2)
    if stories is not None and stories > 5:
        return False, f"Number of stories ({stories}) exceeds five stories (Sec 12.8.1.3 Item 2)."

    # Criterion 3: Period T <= 0.5 s (Sec 12.8.1.3 Item 3)
    if period > 0.5001:
        return False, f"Fundamental period T={period:.3f} s exceeds 0.5 s (Sec 12.8.1.3 Item 3)."

    # Criterion 4: Redundancy factor rho == 1.0 (Sec 12.8.1.3 Item 4)
    if params.redundancy_factor > 1.0001:
        return False, f"Redundancy factor rho={params.redundancy_factor} > 1.0 (Sec 12.8.1.3 Item 4)."

    # Criterion 5: Site Class is not E or F (Sec 12.8.1.3 Item 5)
    site_cls = params.site_class.strip().upper()
    if site_cls in ("E", "F"):
        return False, f"Site Class '{site_cls}' is E or F (Sec 12.8.1.3 Item 5)."

    # Criterion 6: Risk Category I or II (Sec 12.8.1.3 Item 6)
    rc = params.risk_category.strip().upper()
    if rc not in ("I", "II"):
        return False, f"Risk Category '{rc}' is not I or II (Sec 12.8.1.3 Item 6)."

    # Criterion 7: R >= 3.0 (Sec 12.8.1.3 Item 7)
    if params.r_factor < 3.0:
        return False, f"Response modification coefficient R={params.r_factor} < 3.0 (Sec 12.8.1.3 Item 7)."

    return True, "All 7 criteria of ASCE 7-22 Section 12.8.1.3 are satisfied."


def calculate_cw(
    ab: float,
    walls: List[Tuple[float, float]],
    hn: float,
) -> float:
    """
    Calculate shear wall building parameter Cw per ASCE 7-22 Eq. (12.8-11):
    Cw = (100 / AB) * sum_{i=1}^x [ Ai / (1 + 0.83 * (hn / Di)^2) ]

    Parameters
    ----------
    ab : float
        Area of base of structure [ft^2 or m^2].
    walls : List[Tuple[float, float]]
        List of (Ai, Di) tuples for each effective shear wall:
        - Ai: Web area of shear wall i [ft^2 or m^2]
        - Di: Length of shear wall i [ft or m]
    hn : float
        Structural height [ft or m].

    Returns
    -------
    float
        Dimensionless Cw parameter.
    """
    if ab <= 0.0 or not walls:
        return 0.0
    wall_sum = 0.0
    for ai, di in walls:
        if di > 0.0:
            wall_sum += ai / (1.0 + 0.83 * ((hn / di) ** 2))
    return (100.0 / ab) * wall_sum


def approximate_fundamental_period(
    total_height: float,
    *,
    units: Union[str, Any, dict],
    system_type: str = "rc_smf",
    num_stories: Optional[int] = None,
    use_story_count_formula: bool = False,
    cw: Optional[float] = None,
) -> float:
    """
    Calculate approximate fundamental period Ta per ASCE 7-22 Section 12.8.2.1:
    - Eq. (12.8-8): Ta = Ct * hn^x
    - Eq. (12.8-9): Ta = 0.1 * N (for moment frames <= 12 stories, story height >= 10 ft / 3 m)
    - Eq. (12.8-10): Ta = (Cq / sqrt(Cw)) * hn (for shear wall structures <= 120 ft / 36.6 m)

    Parameters
    ----------
    total_height : float
        Building height from base to highest level in active length units.
    system_type : str
        Structural system type from Table 12.8-2:
        - 'rc_smf' / 'rc_imf' / 'rc_omf' / 'concrete_moment': Concrete moment-resisting frame
        - 'steel_smf' / 'steel_moment': Steel moment-resisting frame
        - 'steel_ebf' / 'eccentrically': Steel eccentrically braced frame
        - 'steel_brbf' / 'buckling': Steel buckling-restrained braced frame
        - 'all_other' / 'rc_shear_wall': All other systems
    units : str, Units, or dict
        Active engineering units.
    num_stories : int, optional
        Number of stories N.
    use_story_count_formula : bool, default=False
        If True and structure qualifies, use Eq. (12.8-9): Ta = 0.1 N.
    cw : float, optional
        Cw parameter from Eq. (12.8-11) for shear wall structures.
        If provided and hn <= 120 ft (36.6 m), Eq. (12.8-10) is evaluated.

    Returns
    -------
    float
        Approximate fundamental period Ta in seconds.
    """
    from struct_core import parse_units, conversion_factor, Units
    active_units = parse_units(units)
    is_us = active_units.length in ("in", "ft")
    sys = (system_type or "all_other").lower()

    target_length = "ft" if is_us else "m"
    hn = total_height * conversion_factor(active_units, Units(force="kN", length=target_length), "length")

    # Alternative 1 (Eq. 12.8-9): Ta = 0.1 N, only for concrete or steel moment frames of at most
    # 12 stories with an average story height of at least 10 ft (3 m)
    if use_story_count_formula:
        min_avg_h = 10.0 if is_us else 3.0
        moment_frame = any(tag in sys for tag in ("smf", "imf", "omf", "moment"))
        if not moment_frame or num_stories is None or num_stories > 12 or hn / num_stories < min_avg_h:
            raise ValueError(
                "Eq. 12.8-9 (Ta = 0.1 N) applies only to concrete or steel moment frames of at most 12 stories "
                f"with an average story height of at least {min_avg_h:g} {target_length} "
                f"(system '{system_type}', N = {num_stories}, hn = {hn:.2f} {target_length})."
            )
        return round(0.1 * num_stories, 3)

    # Alternative 2 (Eq. 12.8-10): Ta = (Cq / sqrt(Cw)) * hn
    if cw is not None and cw > 0.0:
        max_h = 120.0 if is_us else 36.6
        if hn <= max_h:
            # Printed ASCE 7-22 values. The SI value is not the converted US one (0.0019 s/ft = 0.0062 s/m),
            # so SI shear-wall periods are about 10x shorter than US ones: an erratum candidate.
            cq = 0.0019 if is_us else 0.00058  # ASCE 7-22 Section 12.8.2.1
            ta_cw = (cq / math.sqrt(cw)) * hn
            return round(ta_cw, 3)

    # General Equation (Eq. 12.8-8): Ta = Ct * hn^x (Table 12.8-2)
    if any(tag in sys for tag in ("steel_smf", "steel_imf", "steel_omf", "steel_moment")):
        ct = 0.028 if is_us else 0.0724
        x = 0.8
    elif any(tag in sys for tag in ("rc_smf", "rc_imf", "rc_omf", "concrete_moment")):
        ct = 0.016 if is_us else 0.0466
        x = 0.9
    elif "steel_ebf" in sys or "eccentrically" in sys:
        ct = 0.030 if is_us else 0.0731
        x = 0.75
    elif "steel_brbf" in sys or "buckling" in sys:
        ct = 0.030 if is_us else 0.0731
        x = 0.75
    else:  # All other structural systems / shear walls
        ct = 0.020 if is_us else 0.0488
        x = 0.75

    ta = ct * (hn ** x)
    return round(ta, 3)


def _eval_single_sa(
    params: SeismicParameters,
    period: float,
    num_stories: Optional[int] = None,
) -> float:
    sds_eff = params.sds
    if params.apply_sds_cap:
        eligible, _ = check_sds_cap_eligibility(params, period, num_stories=num_stories)
        if eligible:
            sds_eff = min(params.sds, max(1.0, 0.70 * params.sds))

    if params.sa_spectrum is not None:
        if callable(params.sa_spectrum):
            return float(params.sa_spectrum(period))
        elif isinstance(params.sa_spectrum, dict):
            t_keys = sorted(params.sa_spectrum.keys())
            if period <= t_keys[0]:
                return float(params.sa_spectrum[t_keys[0]])
            elif period >= t_keys[-1]:
                return float(params.sa_spectrum[t_keys[-1]])
            for i in range(len(t_keys) - 1):
                t_lo, t_hi = t_keys[i], t_keys[i + 1]
                if t_lo <= period <= t_hi:
                    sa_lo, sa_hi = params.sa_spectrum[t_lo], params.sa_spectrum[t_hi]
                    return float(sa_lo + (period - t_lo) / (t_hi - t_lo) * (sa_hi - sa_lo))

    # Standard ASCE 7 Section 11.4.5 design spectrum
    if sds_eff <= 0.0:
        return 0.0

    t0 = 0.2 * params.sd1 / sds_eff
    ts = params.sd1 / sds_eff

    if period < t0:
        val = sds_eff * (0.4 + 0.6 * (period / t0)) if t0 > 0.0 else sds_eff
    elif period <= ts:
        val = sds_eff
    elif period <= params.long_period_transition(period):
        val = params.sd1 / period if period > 0.0 else sds_eff
    else:
        val = (params.sd1 * params.tl) / (period ** 2) if period > 0.0 else sds_eff

    # Damping ratio adjustment if not 5% (ASCE 7 Chapter 18 damping factor B)
    damp = getattr(params, "damping_ratio", 0.05)
    if damp is not None and abs(damp - 0.05) > 1e-4:
        # ASCE 7 Table 18.7-1: B = 4 / (5.6 - ln(100 * damp))
        b_val = 4.0 / max(0.5, 5.6 - math.log(max(1e-4, 100.0 * damp)))
        val = val / b_val

    return float(val)


def evaluate_design_spectral_acceleration(
    params: SeismicParameters,
    period: Union[float, List[float], Any],
    num_stories: Optional[int] = None,
) -> Union[float, Any]:
    """
    Evaluate ASCE 7-22 design spectral response acceleration Sa(T) in (g) at period(s) T.
    Compliant with ASCE 7-22 Section 11.4.5 and Section 12.8.1.3 (SDS cap if applicable).

    Parameters
    ----------
    params : SeismicParameters
        Seismic parameters (SDS, SD1, TL, site_class, damping_ratio, etc.).
    period : float or array-like
        Vibration period T in seconds (scalar or array).
    num_stories : int, optional
        Number of stories (for Sec 12.8.1.3 SDS cap check).

    Returns
    -------
    float or np.ndarray
        Design spectral response acceleration Sa in (g).
    """
    if isinstance(period, (list, tuple)) or hasattr(period, "__iter__") and not isinstance(period, (str, dict)):
        import numpy as np
        return np.array(
            [_eval_single_sa(params, float(t), num_stories=num_stories) for t in period],
            dtype=float,
        )
    return _eval_single_sa(params, float(period), num_stories=num_stories)


def calculate_design_response_spectrum(
    params: SeismicParameters,
    periods: Optional[Union[List[float], Any]] = None,
    num_points: int = 150,
    max_period: Optional[float] = None,
    num_stories: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate the complete ASCE 7-22 design response spectrum curve Sa(T).

    Parameters
    ----------
    params : SeismicParameters
        Seismic parameters (SDS, SD1, TL, site_class, damping_ratio, etc.).
    periods : array-like, optional
        Pre-defined array of periods T (s). If None, an adaptive grid including
        0.0, T0, Ts, and TL is generated up to max_period.
    num_points : int, default=150
        Number of period samples if periods is not specified.
    max_period : float, optional
        Upper bound on period range (seconds). Defaults to max(params.tl + 2.0, 10.0), or the smallest mapped TL
        (4 s) when TL is not given.
    num_stories : int, optional
        Number of stories (for Sec 12.8.1.3 SDS cap check).

    Returns
    -------
    Dict[str, Any]
        - 'periods': List of period values T (s)
        - 'sa': List of spectral accelerations Sa (g)
        - 't0': T0 period (s)
        - 'ts': Ts period (s)
        - 'tl': Long-period transition period TL (s)
        - 'sds': Effective SDS used (g)
        - 'sd1': SD1 used (g)
        - 'site_class': Site soil class (e.g. 'D')
        - 'damping_ratio': Modal damping ratio (default 0.05)
    """
    sds_eff = params.sds
    if params.apply_sds_cap:
        eligible, _ = check_sds_cap_eligibility(params, 0.2, num_stories=num_stories)
        if eligible:
            sds_eff = min(params.sds, max(1.0, 0.70 * params.sds))

    t0 = 0.2 * params.sd1 / sds_eff if sds_eff > 0.0 else 0.0
    ts = params.sd1 / sds_eff if sds_eff > 0.0 else 0.0
    tl = params.tl

    if periods is not None:
        t_vals = [float(t) for t in periods]
    else:
        t_max = max_period if max_period is not None else (max(tl + 2.0, 10.0) if tl is not None else MIN_MAPPED_TL)
        # Create critical points
        crit_pts = {0.0, 0.01, t0, ts, t_max} | ({tl} if tl is not None else set())
        linear_pts = list(math.pow(10, x) for x in (-2 + i * (math.log10(t_max) + 2) / num_points for i in range(num_points + 1)))
        all_pts = sorted(crit_pts.union(linear_pts))
        t_vals = [p for p in all_pts if 0.0 <= p <= t_max]

    sa_vals = [float(_eval_single_sa(params, t, num_stories=num_stories)) for t in t_vals]

    return {
        "periods": t_vals,
        "sa": sa_vals,
        "t0": round(t0, 4),
        "ts": round(ts, 4),
        "tl": round(tl, 4) if tl is not None else None,
        "sds": round(sds_eff, 4),
        "sd1": round(params.sd1, 4),
        "site_class": params.site_class,
        "damping_ratio": getattr(params, "damping_ratio", 0.05),
    }


def calculate_seismic_response_coefficient(
    params: SeismicParameters,
    period: float,
    num_stories: Optional[int] = None,
) -> float:
    """
    Calculate seismic response coefficient Cs per ASCE 7-22 Section 12.8.1.1:
    - Method 1: Eq. (12.8-2) Cs = Sa / (R / Ie)
    - Method 2: Eq. (12.8-3) Cs = SDS / (R / Ie), with upper bounds (Eq. 12.8-4 & 12.8-5)
    - Lower bounds: Eq. (12.8-6) and Eq. (12.8-7) for S1 >= 0.6g.
    - Section 12.8.1.3 SDS cap: max(1.0, 0.70*SDS) if requested and eligible.

    Parameters
    ----------
    params : SeismicParameters
        Seismic parameters defining design spectra and system coefficients.
    period : float
        Fundamental period T in seconds.
    num_stories : int, optional
        Number of stories above base (for checking Sec 12.8.1.3 cap).

    Returns
    -------
    float
        Seismic response coefficient Cs.
    """
    r_ie = params.r_factor / params.importance_factor

    # Section 12.8.1.3 SDS Cap evaluation
    sds_effective = params.sds
    if params.apply_sds_cap:
        eligible, _ = check_sds_cap_eligibility(params, period, num_stories=num_stories)
        if eligible:
            capped_val = max(1.0, 0.70 * params.sds)
            sds_effective = min(params.sds, capped_val)

    # Method Selection: 1 or 2
    method_val = int(params.method) if isinstance(params.method, (int, str)) and str(params.method).isdigit() else 2
    if method_val == 1:
        # Method 1 (Eq. 12.8-2): Cs = Sa / (R / Ie), Sa from the multi-period spectrum of Sec 11.4.5.1
        if params.sa_spectrum is None:
            raise ValueError(
                "Method 1 (ASCE 7-22 Eq. 12.8-2) uses the multi-period design spectrum of Section 11.4.5.1; "
                "supply sa_spectrum or use Method 2."
            )
        sa = _eval_single_sa(params, period, num_stories=num_stories)

        # ASCE 7-22 Section 12.8.1.1 Exception:
        # "Where Equation (12.8-2) is used and the period T is less than the period at which
        # Sa is maximum, the maximum value of Sa shall be used."
        if isinstance(params.sa_spectrum, dict):
            t_peak, sa_peak = max(params.sa_spectrum.items(), key=lambda kv: kv[1])
        else:
            samples = [0.01 * i for i in range(int(100 * max(params.tl or MIN_MAPPED_TL, period)) + 1)]
            t_peak, sa_peak = max(((t, float(params.sa_spectrum(t))) for t in samples), key=lambda kv: kv[1])
        if period < t_peak:
            sa = max(sa, float(sa_peak))

        cs_calc = sa / r_ie

    else:
        # Method 2 (Eq. 12.8-3): Cs = SDS / (R / Ie)
        cs_basic = sds_effective / r_ie

        # Upper bounds:
        # For T <= TL: Eq. (12.8-4) Cs = SD1 / (T * (R / Ie))
        # For T > TL:  Eq. (12.8-5) Cs = (SD1 * TL) / (T^2 * (R / Ie))
        if period <= params.long_period_transition(period):
            cs_max = params.sd1 / (period * r_ie)
        else:
            cs_max = (params.sd1 * params.tl) / ((period ** 2) * r_ie)

        cs_calc = min(cs_basic, cs_max)

    # Lower Bounds for Both Methods:
    # Eq. (12.8-6): Cs >= max(0.044 * SDS * Ie, 0.01)
    cs_min_general = max(0.044 * sds_effective * params.importance_factor, 0.01)

    # Eq. (12.8-7): For structures where S1 >= 0.6g: Cs >= 0.5 * S1 / (R / Ie). S1 is mapped site
    # data; without it the bound is not evaluated (see StoryForceResult.s1_bound_evaluated).
    if params.s1 is not None and params.s1 >= 0.6:
        cs_min_s1 = 0.5 * params.s1 / r_ie
    else:
        cs_min_s1 = 0.0

    cs_min = max(cs_min_general, cs_min_s1)
    cs = max(cs_calc, cs_min)
    return round(cs, 4)


def calculate_seismic_elf(
    story_elevations: List[float],
    story_weights: List[float],
    params: SeismicParameters,
    *,
    units: Union[str, Any, dict],
    custom_period: Optional[float] = None,
    use_story_count_formula: bool = False,
    cw: Optional[float] = None,
) -> StoryForceResult:
    """
    Perform full ASCE 7-22 Equivalent Lateral Force (ELF) analysis:
    Computes fundamental period T, seismic coefficient Cs, base shear V,
    vertical force distribution Fx, and design story shears Vx.

    Parameters
    ----------
    story_elevations : List[float]
        Elevation of each story above base, sorted ascending (e.g. [3.5, 7.0] m).
    story_weights : List[float]
        Effective seismic weight wi assigned to each story level.
    params : SeismicParameters
        Seismic design parameters (SDS, SD1, R, Ie, TL, S1, site_class, method, etc.).
    units : str, Units, or dict, default="kN, m"
        Active engineering units.
    custom_period : float, optional
        Calculated fundamental period T from structural analysis (e.g. modal analysis).
        If provided, T is capped at Cu * Ta per ASCE 7-22 Section 12.8.2.
    use_story_count_formula : bool, default=False
        If True, evaluate Eq. (12.8-9): Ta = 0.1 N for moment frames <= 12 stories.
    cw : float, optional
        Cw parameter for shear wall structures per Eq. (12.8-10) and (12.8-11).

    Returns
    -------
    StoryForceResult
        Complete ELF analysis result containing base shear, Cs, period, story forces Fx,
        and story shears Vx per ASCE 7-22 Section 12.8.
    """
    if len(story_elevations) != len(story_weights):
        raise ValueError("story_elevations and story_weights must have the same length.")
    if len(story_elevations) == 0:
        raise ValueError("story_elevations and story_weights cannot be empty.")

    from struct_core import parse_units
    active_units = parse_units(units)
    hn = story_elevations[-1]
    num_stories = len(story_elevations)

    # 1. Period Determination (Section 12.8.2)
    ta_approx = approximate_fundamental_period(
        total_height=hn,
        system_type=params.system_type,
        units=active_units,
        num_stories=num_stories,
        use_story_count_formula=use_story_count_formula,
        cw=cw,
    )

    cu = get_cu_coefficient(params.sd1)
    if custom_period is not None:
        t_used = min(custom_period, cu * ta_approx)
    else:
        t_used = ta_approx

    # 2. Seismic Response Coefficient Cs (Section 12.8.1.1)
    cs = calculate_seismic_response_coefficient(
        params=params,
        period=t_used,
        num_stories=num_stories,
    )

    # Track effective SDS used (considering Sec 12.8.1.3 cap)
    sds_effective = params.sds
    is_capped = False
    if params.apply_sds_cap:
        eligible, _ = check_sds_cap_eligibility(params, t_used, num_stories=num_stories)
        if eligible:
            capped_val = max(1.0, 0.70 * params.sds)
            if capped_val < params.sds:
                sds_effective = capped_val
                is_capped = True

    # 3. Seismic Base Shear V (Section 12.8.1, Eq. 12.8-1)
    w_total = sum(story_weights)
    base_shear = round(cs * w_total, 3)

    # 4. Vertical Distribution Exponent k (Section 12.8.3)
    if t_used <= 0.5:
        k = 1.0
    elif t_used >= 2.5:
        k = 2.0
    else:
        k = round(1.0 + (t_used - 0.5) / 2.0, 4)

    # 5. Vertical Distribution Factor Cvx and Lateral Force Fx (Eq. 12.8-12 & 12.8-13)
    weighted_heights = [w * (h ** k) for w, h in zip(story_weights, story_elevations)]
    denom = sum(weighted_heights)

    story_forces: Dict[int, float] = {}
    cvx_dict: Dict[int, float] = {}
    for idx, (wh, elev) in enumerate(zip(weighted_heights, story_elevations)):
        cvx = (wh / denom) if denom > 1e-9 else (1.0 / num_stories)
        cvx_dict[idx + 1] = round(cvx, 5)
        fx = round(cvx * base_shear, 3)
        story_forces[idx + 1] = fx

    # Ensure sum of story forces equals base shear exactly per ASCE 7
    diff = round(base_shear - sum(story_forces.values()), 3)
    if abs(diff) > 0 and len(story_forces) > 0:
        top_idx = max(story_forces.keys())
        story_forces[top_idx] = round(story_forces[top_idx] + diff, 3)

    # 6. Horizontal Story Shears Vx (Section 12.8.4, Eq. 12.8-14): Vx = sum_{i=x}^n Fi
    story_shears: Dict[int, float] = {}
    running_shear = 0.0
    for idx in sorted(story_forces.keys(), reverse=True):
        running_shear += story_forces[idx]
        story_shears[idx] = round(running_shear, 3)

    story_weights_dict = {idx + 1: round(w, 3) for idx, w in enumerate(story_weights)}

    return StoryForceResult(
        base_shear=base_shear,
        cs=cs,
        ta=ta_approx,
        total_weight=w_total,
        story_forces=story_forces,
        story_weights=story_weights_dict,
        story_shears=story_shears,
        cvx=cvx_dict,
        k=k,
        cu=cu,
        t_used=round(t_used, 4),
        sds_used=round(sds_effective, 4),
        method_used=params.method,
        is_sds_capped=is_capped,
        s1_bound_evaluated=params.s1 is not None,
    )


def determine_seismic_design_category(
    sds: float,
    sd1: float,
    risk_category: str = "II",
    s1: Optional[float] = None,
) -> str:
    """
    Determine Seismic Design Category (SDC A through F) per ASCE 7-22 Section 11.6.

    Parameters
    ----------
    sds : float
        Design spectral response acceleration parameter at short periods.
    sd1 : float
        Design spectral response acceleration parameter at 1-s period.
    risk_category : str, default="II"
        Risk Category ('I', 'II', 'III', or 'IV') per Table 1.5-1.
    s1 : float, optional
        Mapped maximum considered earthquake spectral response acceleration at 1-s period.

    Returns
    -------
    str
        Seismic Design Category ('A', 'B', 'C', 'D', 'E', or 'F').
    """
    rc = risk_category.strip().upper()
    if rc not in ("I", "II", "III", "IV"):
        rc = "II"

    # ASCE 7-22 Section 11.6: S1 >= 0.75 check
    if s1 is not None and s1 >= 0.75:
        if rc == "IV":
            return "F"
        return "E"

    # Table 11.6-1: SDC Based on Short-Period Response Acceleration SDS
    if rc == "IV":
        if sds < 0.167:
            sdc_sds = "A"
        elif sds < 0.33:
            sdc_sds = "C"
        elif sds < 0.50:
            sdc_sds = "D"
        else:
            sdc_sds = "D"
    else:  # Risk Categories I, II, III
        if sds < 0.167:
            sdc_sds = "A"
        elif sds < 0.33:
            sdc_sds = "B"
        elif sds < 0.50:
            sdc_sds = "C"
        else:
            sdc_sds = "D"

    # Table 11.6-2: SDC Based on 1-s Period Response Acceleration SD1
    if rc == "IV":
        if sd1 < 0.067:
            sdc_sd1 = "A"
        elif sd1 < 0.133:
            sdc_sd1 = "C"
        elif sd1 < 0.20:
            sdc_sd1 = "D"
        else:
            sdc_sd1 = "D"
    else:  # Risk Categories I, II, III
        if sd1 < 0.067:
            sdc_sd1 = "A"
        elif sd1 < 0.133:
            sdc_sd1 = "B"
        elif sd1 < 0.20:
            sdc_sd1 = "C"
        else:
            sdc_sd1 = "D"

    # Governed by the more severe category
    severity_order = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}
    governing = max(sdc_sds, sdc_sd1, key=lambda c: severity_order.get(c, 0))
    return governing


def calculate_accidental_torsion(
    fx: float,
    perpendicular_dimension: float,
    ax: float = 1.0,
) -> float:
    """
    Calculate accidental torsional moment Mta at a given floor level per ASCE 7-22 Section 12.8.4.2:
    Mta = Ax * (0.05 * B) * Fx

    Parameters
    ----------
    fx : float
        Seismic design lateral force applied at the story level [kN or kip].
    perpendicular_dimension : float
        Building plan dimension perpendicular to the direction of applied force B [m or ft].
    ax : float, default=1.0
        Torsional amplification factor Ax (>= 1.0, <= 3.0) per Section 12.8.4.3.

    Returns
    -------
    float
        Accidental torsional moment Mta [kN*m or kip*ft].
    """
    ax_eff = min(3.0, max(1.0, float(ax)))
    eccentricity = 0.05 * abs(perpendicular_dimension)
    return round(ax_eff * eccentricity * abs(fx), 3)


def calculate_torsional_amplification(
    delta_max: float,
    delta_avg: float,
) -> float:
    """
    Calculate accidental torsional amplification factor Ax per ASCE 7-22 Section 12.8.4.3 Eq. (12.8-15):
    Ax = (delta_max / (1.2 * delta_avg))^2, with 1.0 <= Ax <= 3.0.

    Parameters
    ----------
    delta_max : float
        Maximum displacement at level x computed assuming Ax = 1.0 [mm or in].
    delta_avg : float
        Average of displacements at extreme points at level x computed assuming Ax = 1.0 [mm or in].

    Returns
    -------
    float
        Torsional amplification factor Ax (between 1.0 and 3.0).
    """
    if delta_avg <= 1e-9:
        return 1.0
    ratio = delta_max / (1.2 * delta_avg)
    ax = ratio ** 2
    ax_clamped = min(3.0, max(1.0, ax))
    return round(ax_clamped, 3)


def calculate_design_story_drift(
    delta_elastic_top: float,
    delta_elastic_bot: float,
    cd: float,
    importance_factor: float = 1.0,
) -> float:
    """
    Calculate design story drift Delta per ASCE 7-22 Section 12.8.6 Eq. (12.8-16):
    Delta = (Cd * (delta_top - delta_bot)) / Ie

    Parameters
    ----------
    delta_elastic_top : float
        Elastic displacement at upper story joint [mm or in].
    delta_elastic_bot : float
        Elastic displacement at lower story joint [mm or in].
    cd : float
        Deflection amplification factor from Table 12.2-1.
    importance_factor : float, default=1.0
        Seismic importance factor Ie.

    Returns
    -------
    float
        Design story drift Delta [mm or in].
    """
    delta_e = abs(delta_elastic_top - delta_elastic_bot)
    ie = max(1e-4, importance_factor)
    return round((cd * delta_e) / ie, 4)


def check_story_drift_limit(
    story_drift: float,
    story_height: float,
    risk_category: str = "II",
    system_type: str = "other",
) -> Tuple[bool, float, float]:
    """
    Check story drift against allowable drift Delta_a per ASCE 7-22 Table 12.12-1.

    Parameters
    ----------
    story_drift : float
        Design story drift Delta [mm or in].
    story_height : float
        Story height hsx [mm or in].
    risk_category : str, default="II"
        Risk Category ('I', 'II', 'III', or 'IV').
    system_type : str, default="other"
        Structural framing type:
        - 'cantilever': Cantilever column systems
        - '4_stories_or_less': <= 4 stories designed to accommodate drift
        - 'other': All other building structures (default)

    Returns
    -------
    Tuple[bool, float, float]
        (is_compliant, story_drift, allowable_drift)
    """
    rc = risk_category.strip().upper()
    sys = system_type.strip().lower()

    if "cantilever" in sys:
        factor = 0.010 if rc == "IV" else 0.0125
    elif "4_stories" in sys:
        if rc == "IV":
            factor = 0.015
        elif rc == "III":
            factor = 0.020
        else:
            factor = 0.025
    else:
        if rc == "IV":
            factor = 0.010
        elif rc == "III":
            factor = 0.015
        else:
            factor = 0.020

    allowable_drift = round(factor * story_height, 4)
    is_compliant = story_drift <= (allowable_drift + 1e-6)
    return is_compliant, round(story_drift, 4), allowable_drift



# ASCE 7-22 Table 12.2-1 (moment-resisting frame and braced frame systems used by the ecosystem):
# system key -> (R, Omega_0, Cd)
SEISMIC_SYSTEM_COEFFICIENTS: Dict[str, Tuple[float, float, float]] = {
    "rc_smf": (8.0, 3.0, 5.5),       # C.5 Special reinforced concrete moment frames
    "rc_imf": (5.0, 3.0, 4.5),       # C.6 Intermediate reinforced concrete moment frames
    "rc_omf": (3.0, 3.0, 2.5),       # C.7 Ordinary reinforced concrete moment frames
    "steel_smf": (8.0, 3.0, 5.5),    # C.1 Steel special moment frames
    "steel_imf": (4.5, 3.0, 4.0),    # C.3 Steel intermediate moment frames
    "steel_omf": (3.5, 3.0, 3.0),    # C.4 Steel ordinary moment frames
    "steel_ebf": (8.0, 2.0, 4.0),    # B.1 Steel eccentrically braced frames
    "steel_brbf": (8.0, 2.5, 5.0),   # B.26 Steel buckling-restrained braced frames
}

MOMENT_FRAME_SYSTEMS = ("rc_smf", "rc_imf", "rc_omf", "steel_smf", "steel_imf", "steel_omf")

# System keys accepted by the seismic tools: the Table 12.2-1 subset plus 'all_other' (Table 12.8-2 "all other
# structural systems", for the approximate period only)
SEISMIC_SYSTEM_TYPES = (*SEISMIC_SYSTEM_COEFFICIENTS, "all_other")


def normalize_system_type(system_type: str) -> str:
    """Canonical seismic system key (case and surrounding spaces ignored); raises ValueError for any other value."""
    key = str(system_type).strip().lower()
    if key not in SEISMIC_SYSTEM_TYPES:
        raise ValueError(f"Unknown seismic system '{system_type}'; use one of {list(SEISMIC_SYSTEM_TYPES)}.")
    return key


def seismic_system_coefficients(system_type: str) -> Tuple[float, float, float]:
    """
    Design coefficients (R, Omega_0, Cd) of a seismic force-resisting system per ASCE 7-22 Table 12.2-1.

    Raises ValueError for systems that are not tabulated here.
    """
    key = system_type.strip().lower()
    if key not in SEISMIC_SYSTEM_COEFFICIENTS:
        raise ValueError(
            f"System '{system_type}' is not in the ASCE 7-22 Table 12.2-1 subset; "
            f"use one of {sorted(SEISMIC_SYSTEM_COEFFICIENTS)}."
        )
    return SEISMIC_SYSTEM_COEFFICIENTS[key]


def allowable_story_drift(
    story_height: float,
    risk_category: str = "II",
    system_type: str = "rc_smf",
    seismic_design_category: Optional[str] = None,
    redundancy_factor: float = 1.0,
) -> float:
    """
    Allowable story drift Delta_a per ASCE 7-22 Table 12.12-1 ("All other structures"), divided by
    rho for systems solely comprising moment frames in SDC D, E, or F (Section 12.12.1.1).
    """
    _, _, delta_a = check_story_drift_limit(0.0, story_height, risk_category=risk_category, system_type="other")
    sdc = (seismic_design_category or "").strip().upper()
    if system_type.strip().lower() in MOMENT_FRAME_SYSTEMS and sdc in ("D", "E", "F"):
        delta_a /= redundancy_factor
    return delta_a


def stability_coefficient(px: float, delta_xe: float, vx: float, story_height: float) -> float:
    """
    Stability coefficient per ASCE 7-22 Eq. (12.8-18): theta = Px Delta_xe / (Vx hsx), with the
    elastic story drift Delta_xe under the seismic design story shear Vx.
    """
    if vx <= 0.0 or story_height <= 0.0:
        raise ValueError("Story shear Vx and story height hsx must be positive.")
    return px * abs(delta_xe) / (vx * story_height)


def max_stability_coefficient(cd: float, beta: float = 1.0) -> float:
    """theta_max = 0.5 / (beta Cd) <= 0.25 per ASCE 7-22 Eq. (12.8-19), not taken less than 0.10."""
    return max(min(0.5 / (beta * cd), 0.25), 0.10)
