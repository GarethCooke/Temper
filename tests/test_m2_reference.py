"""M2 task 0 — the reference table, the lambda rule, and the derived band.

Every threshold M2 reports against is derived here, from the oracle, before any
agent exists (constitution invariant 3). What this module is actually for is
making that sentence checkable rather than aspirational:

* the committed lambda **is** what the rule selects, re-derived on every run;
* the rule is non-vacuous — the next grid point down fails it, and for the stated
  reason;
* the case parameters the config repeats **are** the vendored FrontierView ones;
* the Hessian the trajectory band is divided by **is** the curvature of the
  objective the agent is graded on, checked against finite differences of
  `schedule_moments` rather than against the algebra it was derived from;
* the band is **tight** — attained along the flattest eigenvector — so reporting
  it beside the observed deviation is reporting a real bound, not a safe one.

No env is constructed here, so this module draws no shock streams at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from temper.eval.experiment import LAMBDA_GRIDS, golden_parameters_match
from temper.eval.reference import (
    LambdaRule,
    NoAdmissibleLambda,
    reference_row,
    select_lambda,
    trajectory_band,
    trajectory_deviation,
    variance_floor_bps2,
)
from temper.oracle import (
    objective_curvature_floor,
    objective_hessian,
    optimal_trajectory,
    schedule_moments,
)

from .conftest import GOLDEN_DOCUMENT, m2_experiment

EXPERIMENT = m2_experiment()
CASE = EXPERIMENT.case
MARKET = CASE.market
ORDER_SIZE = CASE.order_size
LAMBDA = EXPERIMENT.lambda_risk
TABLE = EXPERIMENT.table()


def _raw_golden(case_id: str) -> dict:
    for case in GOLDEN_DOCUMENT["cases"]:
        if case["case_id"] == case_id:
            return case
    raise AssertionError(f"golden case {case_id!r} is not in the vendored fixture")


# ---------------------------------------------------------------------------
# The case is the vendored one
# ---------------------------------------------------------------------------


def test_the_configs_case_is_the_vendored_frontierview_case():
    """The config repeats the symbol parameters; this is why that is safe.

    `temper/` and `tools/` may not read `tests/golden/`, which is test data and
    not package data, so the frontier case's parameters are written into
    `configs/m2_ppo.yaml`. A second home for a parameter set is exactly what this
    repo refuses to leave unguarded — so the guard is here, field by field,
    against the golden case the config's `params_from` names.
    """
    mismatched = golden_parameters_match(CASE, _raw_golden(CASE.params_from))
    assert not mismatched, (
        f"configs/m2_ppo.yaml disagrees with {CASE.params_from} on "
        f"{', '.join(mismatched)}; the vendored fixture is the numeric spec"
    )


def test_the_case_is_on_the_canonical_grid():
    """T = 6.5 h in half-hour bins, N = 13 (ARCHITECTURE.md §9)."""
    assert MARKET.n_bins == 13
    assert MARKET.horizon_hours == pytest.approx(6.5)
    assert MARKET.dt == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# The lambda rule
# ---------------------------------------------------------------------------


def test_the_committed_lambda_is_what_the_rule_selects():
    """Invariant 3, mechanically: the config cannot drift from its own reasoning."""
    selected = EXPERIMENT.verify_lambda_rule()
    assert selected.lambda_risk == LAMBDA
    assert selected.twap_gap >= EXPERIMENT.rule.min_twap_gap
    assert selected.optimal.max_bin_fraction <= EXPERIMENT.rule.max_bin_fraction


def test_the_rule_is_non_vacuous_and_rejects_the_next_lambda_down():
    """Something on the grid must fail, and the *smallest* admissible one wins.

    Without this the selection could be "the first grid point", which would be
    true by accident at any lambda and would say nothing about the testbed being
    discriminative.
    """
    grid = sorted(LAMBDA_GRIDS[EXPERIMENT.lambda_grid])
    index = grid.index(LAMBDA)
    assert index > 0, "the rule selected the smallest grid point; it rejected nothing"

    below = reference_row(MARKET, ORDER_SIZE, grid[index - 1])
    assert not EXPERIMENT.rule.admits(below)
    assert below.twap_gap < EXPERIMENT.rule.min_twap_gap, (
        "the next lambda down should fail the discriminative-testbed condition; "
        f"it has gap {below.twap_gap:.4f} and max bin "
        f"{below.optimal.max_bin_fraction:.4f}"
    )
    for row in TABLE:
        if row.lambda_risk < LAMBDA:
            assert not EXPERIMENT.rule.admits(row)


def test_the_rule_rejects_degenerate_lambdas_at_the_top_of_the_grid():
    """Condition (ii) has to bite somewhere, or it is decoration."""
    rejected = [
        row
        for row in TABLE
        if row.optimal.max_bin_fraction > EXPERIMENT.rule.max_bin_fraction
    ]
    assert rejected, "no grid point liquidates fast enough to test condition (ii)"
    assert all(row.lambda_risk > LAMBDA for row in rejected)


def test_selection_is_a_pure_function_of_the_table():
    """Order in, order out: shuffling the table cannot change the answer."""
    shuffled = list(reversed(TABLE))
    assert select_lambda(shuffled, EXPERIMENT.rule).lambda_risk == LAMBDA


def test_an_impossible_rule_raises_rather_than_returning_the_closest_miss():
    """The brief's instruction on an empty grid is to change the *case*.

    Returning a near-miss would make that instruction skippable, and the skip
    would look like a result.
    """
    with pytest.raises(NoAdmissibleLambda):
        select_lambda(TABLE, LambdaRule(min_twap_gap=10.0, max_bin_fraction=0.5))


def test_the_optimum_is_the_cheapest_of_the_three_at_the_selected_lambda():
    """Sanity, and the ordering every table and figure will show."""
    row = EXPERIMENT.reference()
    assert row.optimal.objective < row.ac.objective < row.twap.objective
    assert row.twap_gap == pytest.approx(
        (row.twap.objective - row.optimal.objective) / row.optimal.objective
    )


# ---------------------------------------------------------------------------
# The variance floor (M1a's finding, and why the table splits it out)
# ---------------------------------------------------------------------------


def test_immediate_liquidation_attains_the_variance_floor_exactly():
    """`V` floors at sigma_bin^2 X^2 because x_0 = X for every schedule.

    The textbook Almgren–Chriss picture, where risk vanishes at instantaneous
    execution, is qualitatively wrong under the convention the goldens pin — the
    shock lands before the first trade can be placed.
    """
    immediate = np.zeros(MARKET.n_bins + 1)
    immediate[0] = ORDER_SIZE
    moments = schedule_moments(immediate, MARKET, order_size=ORDER_SIZE)
    assert moments.variance == pytest.approx(variance_floor_bps2(MARKET), rel=1e-12)


def test_no_schedule_in_the_table_gets_below_the_floor():
    floor = variance_floor_bps2(MARKET)
    for row in TABLE:
        for schedule in row.schedules.values():
            assert schedule.variance >= floor * (1.0 - 1e-12)
            assert schedule.excess_risk >= -1e-12


def test_the_split_adds_up_to_the_objective():
    """E + lambda*V is the objective, and lambda*(V - floor) is what differs.

    A table whose columns did not reconcile would be three numbers rather than a
    decomposition.
    """
    for row in TABLE:
        for schedule in row.schedules.values():
            assert schedule.expected + schedule.risk == pytest.approx(
                schedule.objective, rel=1e-12
            )
            assert schedule.risk - schedule.excess_risk == pytest.approx(
                row.lambda_risk * row.variance_floor, rel=1e-12, abs=1e-15
            )


# ---------------------------------------------------------------------------
# The Hessian and the derived band
# ---------------------------------------------------------------------------


def test_the_hessian_matches_finite_differences_of_the_graded_objective():
    """The band's denominator is the curvature of the metric, not of an algebra.

    `objective_hessian` is derived by hand from the quadratic form. This checks
    it against second differences of `schedule_moments(...).objective` — the
    function the agent is actually graded by — so a divergence between the
    derivation and the metric shows up here rather than as a suspiciously
    generous trajectory band.
    """
    analytic = objective_hessian(MARKET, ORDER_SIZE, LAMBDA)
    optimum = optimal_trajectory(MARKET, ORDER_SIZE, LAMBDA)
    size = MARKET.n_bins - 1
    step = 1e-3 * ORDER_SIZE

    def objective(interior: np.ndarray) -> float:
        path = optimum.copy()
        path[1:-1] = interior
        return schedule_moments(path, MARKET, order_size=ORDER_SIZE).objective(LAMBDA)

    base = optimum[1:-1].copy()
    numerical = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            plus_plus, plus_minus = base.copy(), base.copy()
            minus_plus, minus_minus = base.copy(), base.copy()
            plus_plus[i] += step
            plus_plus[j] += step
            plus_minus[i] += step
            plus_minus[j] -= step
            minus_plus[i] -= step
            minus_plus[j] += step
            minus_minus[i] -= step
            minus_minus[j] -= step
            numerical[i, j] = (
                objective(plus_plus)
                - objective(plus_minus)
                - objective(minus_plus)
                + objective(minus_minus)
            ) / (4.0 * step * step)

    scale = float(np.max(np.abs(analytic)))
    assert np.allclose(numerical, analytic, rtol=0.0, atol=1e-6 * scale), (
        f"worst disagreement {np.max(np.abs(numerical - analytic)) / scale:.2e} of "
        "the Hessian's scale"
    )


def test_the_curvature_floor_is_the_smallest_eigenvalue():
    """The closed form is used because a tolerance is divided by this number."""
    hessian = objective_hessian(MARKET, ORDER_SIZE, LAMBDA)
    smallest = float(np.min(np.linalg.eigvalsh(hessian)))
    assert objective_curvature_floor(MARKET, ORDER_SIZE, LAMBDA) == pytest.approx(
        smallest, rel=1e-12
    )
    assert smallest > 0.0, "the objective is not strictly convex; the band is void"


def test_the_band_is_exactly_the_quadratic_bound():
    """``delta = lambda_min * b^2 / 2`` — the algebra, in one line."""
    band = EXPERIMENT.band()
    assert 0.5 * band.curvature_floor * band.bound_shares**2 == pytest.approx(
        band.delta_objective, rel=1e-12
    )
    assert trajectory_deviation(
        _perturbed(band.bound_shares), optimal_trajectory(MARKET, ORDER_SIZE, LAMBDA)
    ) == pytest.approx(band.bound_shares, rel=1e-12)


def _flattest_direction() -> np.ndarray:
    hessian = objective_hessian(MARKET, ORDER_SIZE, LAMBDA)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    return eigenvectors[:, int(np.argmin(eigenvalues))]


def _perturbed(distance: float) -> np.ndarray:
    path = optimal_trajectory(MARKET, ORDER_SIZE, LAMBDA).copy()
    path[1:-1] += distance * _flattest_direction()
    return path


def test_the_band_is_attained_in_the_graded_objective():
    """Tight, not merely valid — measured on the metric the agent is graded by.

    Along the flattest eigenvector the objective rises by exactly
    ``lambda_min |d|^2 / 2``, so a schedule that meets epsilon can sit at the
    bound and no further. The scale is stepped down until the perturbed schedule
    is still sell-only, because ``U`` is exactly quadratic only there — permanent
    impact is charged on ``|n_k|`` and its telescoping needs monotone inventory.
    That is not a loophole: the env clips every action to ``[0, remaining]``, so
    the non-monotone region is unreachable (the test below).
    """
    band = EXPERIMENT.band()
    optimum = optimal_trajectory(MARKET, ORDER_SIZE, LAMBDA)
    best = schedule_moments(optimum, MARKET, order_size=ORDER_SIZE).objective(LAMBDA)

    for scale in (0.5, 0.25, 0.1, 0.05, 0.02, 0.01):
        perturbed = _perturbed(scale * band.bound_shares)
        if np.any(np.diff(perturbed) > 0.0):
            continue  # a buy-back: outside the quadratic and outside the env
        moved = schedule_moments(
            perturbed, MARKET, order_size=ORDER_SIZE
        ).objective(LAMBDA)
        assert moved - best == pytest.approx(
            scale**2 * band.delta_objective, rel=1e-9
        )
        return
    pytest.fail("no scale of the flattest direction kept the schedule sell-only")


def test_the_band_widens_with_the_tolerance_and_is_a_square_root():
    epsilon = EXPERIMENT.band()
    per_seed = EXPERIMENT.band(EXPERIMENT.tolerances.per_seed_gap_fraction)
    ratio = EXPERIMENT.tolerances.per_seed_gap_fraction / (
        EXPERIMENT.tolerances.epsilon_gap_fraction
    )
    assert per_seed.bound_shares > epsilon.bound_shares
    assert per_seed.bound_shares == pytest.approx(
        epsilon.bound_shares * np.sqrt(ratio), rel=1e-12
    )


def test_a_zero_tolerance_pins_the_trajectory_and_a_negative_one_is_refused():
    zero = trajectory_band(MARKET, ORDER_SIZE, LAMBDA, 0.0)
    assert zero.bound_shares == 0.0
    with pytest.raises(ValueError):
        trajectory_band(MARKET, ORDER_SIZE, LAMBDA, -1.0)


def test_the_deviation_ignores_the_two_endpoints_nobody_controls():
    """Every schedule starts at X and is force-liquidated to zero.

    Including those two identically-zero terms in the norm would make every
    deviation read smaller than it is — by nothing at all, which is precisely why
    it would never be noticed.
    """
    optimum = optimal_trajectory(MARKET, ORDER_SIZE, LAMBDA)
    moved = optimum.copy()
    moved[0] += 1e6
    moved[-1] -= 1e6
    assert trajectory_deviation(moved, optimum) == 0.0
    with pytest.raises(ValueError):
        trajectory_deviation(optimum[:-1], optimum)


# ---------------------------------------------------------------------------
# Provenance and the trace budget — the machinery invariant 1 leans on
# ---------------------------------------------------------------------------


def test_a_regenerated_results_file_does_not_make_the_next_run_dirty():
    """`git_dirty` answers "is the *code* uncommitted?", not "did anything change?".

    A sweep writes into `results/`. Without this distinction the second of two
    sequential sweeps would stamp itself dirty because the first had just written
    its artefact — a false alarm, and false alarms are how a flag stops being
    read. Anything outside `results/` still counts, including a source file moved
    into it.
    """
    from temper.eval.provenance import _source_is_dirty

    assert not _source_is_dirty("")
    assert not _source_is_dirty(" M results/m2_rediscovery.json\n?? results/x.png")
    assert _source_is_dirty(" M temper/agents/ppo.py")
    assert _source_is_dirty("?? configs/new.yaml\n M results/a.json")
    assert _source_is_dirty('R  temper/a.py -> results/a.py')
    assert _source_is_dirty('R  results/a.json -> temper/a.py')


def test_the_trace_thinner_keeps_the_ends_and_respects_its_budget():
    """M3's budget mechanism, checked before M3 needs it.

    Keeping both ends matters: the last point is the converged value every
    summary is read against, and the first is where a learning curve starts.
    """
    from temper.eval.sweep import thin

    trace = list(range(1000))
    assert thin(trace, None) == trace
    assert thin(trace, 5000) == trace
    for budget in (2, 3, 8, 128):
        kept = thin(trace, budget)
        assert len(kept) <= budget
        assert kept[0] == trace[0]
        assert kept[-1] == trace[-1]
        assert kept == sorted(kept), "thinning must preserve order"
    assert thin(trace, 1) == [trace[-1]]
    assert thin([7.0], 128) == [7.0]


def test_both_m2_configs_pin_the_torch_thread_count_and_keep_whole_traces():
    """The committed decisions, asserted rather than left to a comment."""
    from temper.eval.experiment import load_experiment

    from .conftest import REPO_ROOT

    for name in ("m2_ppo.yaml", "m2_ppo_sampled.yaml"):
        experiment = load_experiment(REPO_ROOT / "configs" / name)
        assert experiment.ppo.torch_threads == 8, f"{name} does not pin torch threads"
        assert experiment.trace_points is None, (
            f"{name} thins its traces; M2 commits them whole so the seed spread "
            "stays checkable"
        )
