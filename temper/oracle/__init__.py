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

M4b adds two more, and they are the only *invented* thing in the package:
:mod:`~temper.oracle.liquidity` is the per-bin multiplier and its closed-form
moments, and :mod:`~temper.oracle.adaptive` is the optimum over adapted policies
in the world that multiplier makes — a dynamic program, so **converged and
bracketed rather than certified**. Read `liquidity`'s docstring before reporting
any number that depends on it: FrontierView has no liquidity process, so §7's
"vendored, not invented" cover does not reach these two.

M5 adds the same pair one rung along, and the same warning applies twice over:
:mod:`~temper.oracle.signal` is a one-step-ahead price signal — invented,
FrontierView has no alpha model — and :mod:`~temper.oracle.alpha` is the optimum
over policies that can see it, again a dynamic program and again **converged
rather than certified**. It is the first module here to use both kinds of
confidence at once: its own value is converged, and the floor it grades execution
against is M4a's *certified* optimum. Read it before reporting any number from it.
"""

from .adaptive import (
    DEFAULT_GRID_POINTS,
    DEFAULT_QUADRATURE_NODES,
    AdaptiveOptimum,
    AugmentedOptimum,
    adaptive_optimum,
    augmented_optimum,
    clairvoyant_trajectories,
    liquidity_charge,
    path_objective_bps,
    richardson_residual,
    static_optimum,
)
from .alpha import (
    CLAIRVOYANT_GRID_POINTS,
    CLAIRVOYANT_PATHS,
    DEFAULT_SIGNAL_GRID_POINTS,
    DEFAULT_SIGNAL_QUADRATURE_NODES,
    AlphaOptimum,
    AugmentedAlphaOptimum,
    alpha_coefficient,
    alpha_optimum,
    augmented_alpha_optimum,
    clairvoyant_price_values,
    execution_floor_bps,
    expected_alpha_bps,
    signal_path_objective_bps,
)
from .cost import (
    CostMoments,
    conditional_alpha_bps,
    conditional_shortfall_variance_bps2,
    cost_moments,
    expected_cost_moments,
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
from .liquidity import (
    DETERMINISTIC_LIQUIDITY,
    LIQUIDITY_MODELS,
    LOGNORMAL_LIQUIDITY,
    DeterministicLiquidity,
    LiquidityLaw,
    LognormalLiquidity,
    liquidity_for,
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
    schedule_invariant_bps,
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
from .signal import (
    NO_SIGNAL,
    ONE_STEP_SIGNAL,
    SIGNAL_MODELS,
    AlphaSignal,
    NoSignal,
    OneStepSignal,
    signal_for,
)

__all__ = [
    "AdaptiveOptimum",
    "AlphaOptimum",
    "AlphaSignal",
    "AugmentedAlphaOptimum",
    "AugmentedOptimum",
    "BPS",
    "CLAIRVOYANT_GRID_POINTS",
    "CLAIRVOYANT_PATHS",
    "CostMoments",
    "DEFAULT_GRID_POINTS",
    "DEFAULT_QUADRATURE_NODES",
    "DEFAULT_SIGNAL_GRID_POINTS",
    "DEFAULT_SIGNAL_QUADRATURE_NODES",
    "DETERMINISTIC_LIQUIDITY",
    "DeterministicLiquidity",
    "ENCODINGS",
    "ETA_TILDE_FLOOR",
    "FrontierPoint",
    "KAPPA2_FLOOR",
    "KKT_TOLERANCE",
    "LINEAR_ENCODING",
    "LIQUIDITY_MODELS",
    "LOGNORMAL_LIQUIDITY",
    "LiquidityLaw",
    "LognormalLiquidity",
    "Market",
    "NO_SIGNAL",
    "NoSignal",
    "ONE_STEP_SIGNAL",
    "OneStepSignal",
    "PARTICIPATION_FLOOR",
    "POWER_LAW_ENCODING",
    "SIGNAL_MODELS",
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
    "adaptive_optimum",
    "alpha_coefficient",
    "alpha_optimum",
    "augmented_alpha_optimum",
    "augmented_optimum",
    "charge_for",
    "clairvoyant_price_values",
    "clairvoyant_trajectories",
    "conditional_alpha_bps",
    "conditional_shortfall_variance_bps2",
    "cost_moments",
    "default_n_bins",
    "execution_floor_bps",
    "expected_alpha_bps",
    "expected_cost_moments",
    "inventory_penalty_scale",
    "kkt_residual",
    "linear_cost_moments",
    "linearised_eta",
    "liquidity_charge",
    "liquidity_for",
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
    "path_objective_bps",
    "permanent_cost_bps",
    "permanent_drift_bps_per_hour",
    "power_law_charge",
    "power_law_optimum",
    "power_law_optimum_by_shooting",
    "richardson_residual",
    "schedule_invariant_bps",
    "schedule_moments",
    "shortfall_variance_bps2",
    "signal_for",
    "signal_path_objective_bps",
    "sinh_trajectory",
    "static_optimum",
    "tangent_charge",
    "temporary_impact_bps",
    "trades",
    "twap_trajectory",
    "varying_objective_bps",
]
