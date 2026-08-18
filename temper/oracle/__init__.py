"""Almgren–Chriss closed forms — Temper's reference engine.

This package is normative (constitution invariant 2): it must reproduce the
vendored FrontierView goldens within the M0 tolerance, and every later milestone
is graded against it rather than against whatever the agent happens to do.

Pure numpy. No torch, no I/O, no network.

Reading order for a new session: :mod:`~temper.oracle.model` fixes the units and
the grid, :mod:`~temper.oracle.impact` the impact functions and the tangent the
closed form is derived at, :mod:`~temper.oracle.schedules` the trajectories
(read its docstring — there are two distinct kappas and the difference matters),
:mod:`~temper.oracle.cost` the moments, :mod:`~temper.oracle.frontier` the
(E, V) locus, and :mod:`~temper.oracle.powerlaw` the optimum of the *vendored*
power-law world — which has no closed form, is solved and certified instead,
and is what M4a grades against.
"""

from .cost import (
    CostMoments,
    cost_moments,
    linear_cost_moments,
    participation,
    permanent_cost_bps,
    schedule_moments,
    shortfall_variance_bps2,
    trades,
)
from .frontier import (
    VENDOR_LAMBDA_GRID,
    FrontierPoint,
    ac_frontier,
    ac_frontier_point,
    optimal_frontier,
    optimal_frontier_point,
)
from .impact import (
    ETA_TILDE_FLOOR,
    PARTICIPATION_FLOOR,
    linearised_eta,
    permanent_drift_bps_per_hour,
    temporary_impact_bps,
)
from .model import (
    BPS,
    ENCODINGS,
    LINEAR_ENCODING,
    POWER_LAW_ENCODING,
    TEMP_EXPONENT,
    TRADING_HOURS_PER_DAY,
    Market,
    SymbolParams,
    default_n_bins,
)
from .powerlaw import (
    KKT_TOLERANCE,
    TemporaryCharge,
    charge_for,
    inventory_penalty_scale,
    kkt_residual,
    local_curvature_floor,
    marginal_costs,
    optimum_by_shooting,
    optimum_for_charge,
    power_law_charge,
    power_law_optimum,
    power_law_optimum_by_shooting,
    tangent_charge,
    varying_objective_bps,
)
from .schedules import (
    KAPPA2_FLOOR,
    SINH_OVERFLOW_KT,
    ac_kappa,
    ac_trajectory,
    objective_curvature,
    objective_curvature_floor,
    objective_hessian,
    optimal_kappa,
    optimal_trajectory,
    optimal_trajectory_by_solve,
    sinh_trajectory,
    twap_trajectory,
)

__all__ = [
    "BPS",
    "CostMoments",
    "ENCODINGS",
    "ETA_TILDE_FLOOR",
    "FrontierPoint",
    "KAPPA2_FLOOR",
    "KKT_TOLERANCE",
    "LINEAR_ENCODING",
    "Market",
    "PARTICIPATION_FLOOR",
    "POWER_LAW_ENCODING",
    "SINH_OVERFLOW_KT",
    "SymbolParams",
    "TEMP_EXPONENT",
    "TRADING_HOURS_PER_DAY",
    "TemporaryCharge",
    "VENDOR_LAMBDA_GRID",
    "ac_frontier",
    "ac_frontier_point",
    "ac_kappa",
    "ac_trajectory",
    "charge_for",
    "cost_moments",
    "default_n_bins",
    "inventory_penalty_scale",
    "kkt_residual",
    "linear_cost_moments",
    "linearised_eta",
    "local_curvature_floor",
    "marginal_costs",
    "objective_curvature",
    "objective_curvature_floor",
    "objective_hessian",
    "optimal_frontier",
    "optimal_frontier_point",
    "optimal_kappa",
    "optimal_trajectory",
    "optimal_trajectory_by_solve",
    "optimum_by_shooting",
    "optimum_for_charge",
    "participation",
    "permanent_cost_bps",
    "permanent_drift_bps_per_hour",
    "power_law_charge",
    "power_law_optimum",
    "power_law_optimum_by_shooting",
    "schedule_moments",
    "shortfall_variance_bps2",
    "sinh_trajectory",
    "tangent_charge",
    "temporary_impact_bps",
    "trades",
    "twap_trajectory",
    "varying_objective_bps",
]
