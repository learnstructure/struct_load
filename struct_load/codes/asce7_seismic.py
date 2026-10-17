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
    r_factor : float, default=8.0
        Response modification coefficient R (Table 12.2-1, e.g. 8.0 for RC SMF).
    importance_factor : float, default=1.0
        Seismic importance factor I_e per Sec 11.5.1 (Risk Cat I/II: 1.0, III: 1.25, IV: 1.5).
    tl : float, default=8.0
        Long-period transition period TL (s) per Sec 11.4.6.
    s1 : float, optional
        Mapped MCE_R spectral response acceleration parameter at 1-sec period S1 (g)
        per Sec 11.4.2 or 11.4.4. Used for Eq. (12.8-7) when S1 >= 0.6.
    system_type : str, default='rc_smf'
        Structural system type from Table 12.8-2:
        - 'steel_smf': Steel moment-resisting frame (Ct=0.028 / 0.0724, x=0.8)
        - 'rc_smf': Concrete moment-resisting frame (Ct=0.016 / 0.0466, x=0.9)
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
        or a dictionary {period: Sa}. If None for Method 1, standard Sec 11.4.5 spectrum is used.
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
    tl: float = 8.0
    site_class: str = "D"
    damping_ratio: float = 0.05
    r_factor: float = 8.0
    importance_factor: float = 1.0
    s1: Optional[float] = None
    system_type: str = "rc_smf"
    risk_category: str = "II"
    method: int = 2
    sa_spectrum: Optional[Union[Callable[[float], float], Dict[float, float]]] = None
    apply_sds_cap: bool = False
    has_irregularities: bool = False
    redundancy_factor: float = 1.0
    num_stories: Optional[int] = None


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
    system_type: str = "rc_smf",
    units: Union[str, Any, dict] = "kN, m",
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
        - 'rc_smf' / 'concrete_moment': Concrete moment-resisting frame
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
    sys = system_type.lower()

    target_length = "ft" if is_us else "m"
    hn = total_height * conversion_factor(active_units, Units(force="kN", length=target_length), "length")

    # Alternative 1 (Eq. 12.8-9): Ta = 0.1 N
    if use_story_count_formula and num_stories is not None and num_stories <= 12:
        avg_h = hn / num_stories
        min_avg_h = 10.0 if is_us else 3.0
        if avg_h >= min_avg_h:
            return round(0.1 * num_stories, 3)

    # Alternative 2 (Eq. 12.8-10): Ta = (Cq / sqrt(Cw)) * hn
    if cw is not None and cw > 0.0:
        max_h = 120.0 if is_us else 36.6
        if hn <= max_h:
            cq = 0.0019 if is_us else 0.00058  # ASCE 7-22 Section 12.8.2.1
            ta_cw = (cq / math.sqrt(cw)) * hn
            return round(ta_cw, 3)

    # General Equation (Eq. 12.8-8): Ta = Ct * hn^x (Table 12.8-2)
    if "steel_smf" in sys or "steel_moment" in sys:
        ct = 0.028 if is_us else 0.0724
        x = 0.8
    elif "rc_smf" in sys or "concrete_moment" in sys:
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
    elif period <= params.tl:
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
        Upper bound on period range (seconds). Defaults to max(params.tl + 2.0, 10.0).
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
        t_max = max_period if max_period is not None else max(tl + 2.0, 10.0)
        # Create critical points
        crit_pts = {0.0, 0.01, t0, ts, tl, t_max}
        linear_pts = list(math.pow(10, x) for x in (-2 + i * (math.log10(t_max) + 2) / num_points for i in range(num_points + 1)))
        all_pts = sorted(crit_pts.union(linear_pts))
        t_vals = [p for p in all_pts if 0.0 <= p <= t_max]

    sa_vals = [float(_eval_single_sa(params, t, num_stories=num_stories)) for t in t_vals]

    return {
        "periods": t_vals,
        "sa": sa_vals,
        "t0": round(t0, 4),
        "ts": round(ts, 4),
        "tl": round(tl, 4),
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
        # Method 1 (Eq. 12.8-2): Cs = Sa / (R / Ie)
        if params.sa_spectrum is not None:
            if callable(params.sa_spectrum):
                sa = float(params.sa_spectrum(period))
            elif isinstance(params.sa_spectrum, dict):
                t_keys = sorted(params.sa_spectrum.keys())
                if period <= t_keys[0]:
                    sa = float(params.sa_spectrum[t_keys[0]])
                elif period >= t_keys[-1]:
                    sa = float(params.sa_spectrum[t_keys[-1]])
                else:
                    for i in range(len(t_keys) - 1):
                        t_lo, t_hi = t_keys[i], t_keys[i + 1]
                        if t_lo <= period <= t_hi:
                            sa_lo, sa_hi = params.sa_spectrum[t_lo], params.sa_spectrum[t_hi]
                            sa = float(sa_lo + (period - t_lo) / (t_hi - t_lo) * (sa_hi - sa_lo))
                            break
            else:
                sa = sds_effective
        else:
            # Standard Section 11.4.5 design spectrum
            ts = params.sd1 / sds_effective if sds_effective > 0.0 else 0.0
            if period <= ts:
                sa = sds_effective
            elif period <= params.tl:
                sa = params.sd1 / period
            else:
                sa = (params.sd1 * params.tl) / (period ** 2)

        # ASCE 7-22 Section 12.8.1.1 Exception:
        # "Where Equation (12.8-2) is used and the period T is less than the period at which
        # Sa is maximum, the maximum value of Sa shall be used."
        ts = params.sd1 / sds_effective if sds_effective > 0.0 else 0.0
        if period < ts:
            sa = max(sa, sds_effective)

        cs_calc = sa / r_ie

    else:
        # Method 2 (Eq. 12.8-3): Cs = SDS / (R / Ie)
        cs_basic = sds_effective / r_ie

        # Upper bounds:
        # For T <= TL: Eq. (12.8-4) Cs = SD1 / (T * (R / Ie))
        # For T > TL:  Eq. (12.8-5) Cs = (SD1 * TL) / (T^2 * (R / Ie))
        if period <= params.tl:
            cs_max = params.sd1 / (period * r_ie)
        else:
            cs_max = (params.sd1 * params.tl) / ((period ** 2) * r_ie)

        cs_calc = min(cs_basic, cs_max)

    # Lower Bounds for Both Methods:
    # Eq. (12.8-6): Cs >= max(0.044 * SDS * Ie, 0.01)
    cs_min_general = max(0.044 * sds_effective * params.importance_factor, 0.01)

    # Eq. (12.8-7): For structures where S1 >= 0.6g: Cs >= 0.5 * S1 / (R / Ie)
    s1_check = params.s1 if params.s1 is not None else params.sd1
    if s1_check >= 0.6:
        cs_min_s1 = 0.5 * s1_check / r_ie
    else:
        cs_min_s1 = 0.0

    cs_min = max(cs_min_general, cs_min_s1)
    cs = max(cs_calc, cs_min)
    return round(cs, 4)


def calculate_seismic_elf(
    story_elevations: List[float],
    story_weights: List[float],
    params: SeismicParameters,
    units: Union[str, Any, dict] = "kN, m",
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
    )
