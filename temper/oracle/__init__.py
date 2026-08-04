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
(E, V) locus.
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
    TEMP_EXPONENT,
    TRADING_HOURS_PER_DAY,
    Market,
    SymbolParams,
    default_n_bins,
)
from .schedules import (
    KAPPA2_FLOOR,
    SINH_OVERFLOW_KT,
    ac_kappa,
    ac_trajectory,
    objective_curvature,
    optimal_kappa,
    optimal_trajectory,
    optimal_trajectory_by_solve,
    sinh_trajectory,
    twap_trajectory,
)

__all__ = [
    "BPS",
    "ETA_TILDE_FLOOR",
    "KAPPA2_FLOOR",
    "PARTICIPATION_FLOOR",
    "SINH_OVERFLOW_KT",
    "TEMP_EXPONENT",
    "TRADING_HOURS_PER_DAY",
    "VENDOR_LAMBDA_GRID",
    "CostMoments",
    "FrontierPoint",
    "Market",
    "SymbolParams",
    "ac_frontier",
    "ac_frontier_point",
    "ac_kappa",
    "ac_trajectory",
    "cost_moments",
    "default_n_bins",
    "linear_cost_moments",
    "linearised_eta",
    "objective_curvature",
    "optimal_frontier",
    "optimal_frontier_point",
    "optimal_kappa",
    "optimal_trajectory",
    "optimal_trajectory_by_solve",
    "participation",
    "permanent_cost_bps",
    "permanent_drift_bps_per_hour",
    "schedule_moments",
    "shortfall_variance_bps2",
    "sinh_trajectory",
    "temporary_impact_bps",
    "trades",
    "twap_trajectory",
]
