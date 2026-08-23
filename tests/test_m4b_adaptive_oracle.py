"""M4b task 1 — the adaptive oracle, and the checks that make it a reference.

The DP is the first thing in this repo that is graded against and **not
certified**. M4a's optimum came with a Cholesky factorisation, a 1.2e-15 KKT
residual and an independent solver agreeing to 3.1e-15 of X; a stochastic dynamic
program has no such certificate and pretending otherwise would be the first
dishonest number here. So the reference earns its place by a different route, and
this module is that route:

* **It reduces to a certified number.** At ``sigma_log = 0`` the dynamic program
  must return M4a's ``power_law_optimum`` value. That single test ties the value
  iteration, the quadrature, the stage solve, the inventory grid and the
  schedule-invariant constant to a value that *was* certified — it is the most
  valuable check in the milestone, because every other one is a statement about
  new machinery measured by more new machinery.
* **It is bracketed from both sides.** A feasible policy above and a
  perfect-information relaxation below, both sampled, both paired against a closed
  form so the sampling error is small enough to mean something.
* **Its sufficiency is measured, not assumed.** The DP is the optimum over *all*
  adapted policies only because ``(k, x_k, L_k)`` is sufficient, so the same solve
  is re-run on a state carrying ``L_{k-1}`` and required to agree.
* **Its action map is checked against the failure it could have had.** Solving the
  stage problem by snapping to a grid node instead of searching the interpolant is
  the specific defect the brief's 2 %-of-advantage band exists to catch, and it is
  measured here rather than asserted to be absent.

Everything is oracle-only. No env, no agent, no training loop.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from temper.eval.experiment import LIQUIDITY_READING, load_experiment
from temper.eval.reference import (
    LambdaRule,
    liquidity_trajectories,
    select_lambda,
    static_liquidity_table,
)
from temper.oracle import (
    DeterministicLiquidity,
    LognormalLiquidity,
    Market,
    adaptive_optimum,
    augmented_optimum,
    clairvoyant_trajectories,
    cost_moments,
    expected_cost_moments,
    liquidity_charge,
    liquidity_for,
    optimum_for_charge,
    path_objective_bps,
    power_law_charge,
    power_law_optimum,
    richardson_residual,
    static_optimum,
    trades,
    twap_trajectory,
)
from temper.seeding import M4B_REFERENCE_POOL, pool_rng

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "m4b_liquidity.yaml"
REFERENCE_PATH = REPO_ROOT / "results" / "m4b_reference.json"

#: A coarser grid than the committed reference's, so the whole module stays inside
#: the suite's seconds budget. The claims here are about *agreement between two
#: routes*, and both routes see the same grid, so the resolution is not what is
#: under test — except in the one convergence test, which is marked `deep`.
FAST_POINTS = 401

#: Task 1(c)'s pre-stated band on the feasible upper bound, as a fraction of the
#: adaptive advantage. `docs/briefs/M4b-stochastic-liquidity.md`.
FEASIBLE_BAND = 0.02


@pytest.fixture(scope="module")
def experiment():
    return load_experiment(CONFIG_PATH)


@pytest.fixture(scope="module")
def case(experiment):
    return experiment.case.market, experiment.case.order_size, experiment.lambda_risk


@pytest.fixture(scope="module")
def law(experiment):
    return experiment.liquidity


@pytest.fixture(scope="module")
def committed() -> dict:
    if not REFERENCE_PATH.exists():
        pytest.fail(
            f"{REFERENCE_PATH.relative_to(REPO_ROOT)} is missing; regenerate task 0 "
            "with `python tools/m4b_reference_table.py` before the suite can check "
            "its gates"
        )
    return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The invented process, and its closed-form moments
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sigma", [0.0, 0.25, 0.5, 0.75, 1.2])
def test_the_liquidity_moments_are_the_closed_forms_the_reference_uses(sigma):
    """``E[L] = 1``, and ``E[L^-beta]`` against a large sample of the env's route.

    Two independent routes to the same number: the closed form the static rung is
    priced by, and the draws the env will make. They are deliberately different
    code — this is invariant 6's shape applied to a distribution rather than to a
    market, and it is the check that would catch a multiplier whose mean drifted
    off one because the ``- sigma^2 / 2`` correction was dropped.
    """
    liquidity = LognormalLiquidity(sigma)
    sample = liquidity.draw(pool_rng(20260823, M4B_REFERENCE_POOL, 0), 400_000)

    assert liquidity.mean_multiplier() == 1.0
    # 4 standard errors of the sample mean, which is what "agrees" can mean at
    # this sample size; a dropped Jensen correction would be off by ~sigma^2 / 2.
    tolerance = 4.0 * math.sqrt(max(liquidity.variance(), 1e-30) / sample.size)
    assert abs(sample.mean() - 1.0) <= tolerance + 1e-12
    assert liquidity.variance() == pytest.approx(math.expm1(sigma**2))

    for beta in (0.6, 1.0):
        closed = liquidity.inverse_power_moment(beta)
        assert closed == pytest.approx(math.exp(sigma**2 * beta * (1.0 + beta) / 2.0))
        assert closed >= 1.0, "Jensen: dispersion cannot make a fixed schedule cheaper"
        drawn = float(np.mean(sample ** (-beta)))
        assert drawn == pytest.approx(closed, rel=0.02)


@pytest.mark.parametrize("nodes", [5, 7, 15, 21])
def test_the_quadrature_reproduces_the_moments_it_will_be_used_to_integrate(nodes):
    """The DP's expectation operator, checked on the moment the reference needs."""
    liquidity = LognormalLiquidity(0.5)
    values, weights = liquidity.quadrature(nodes)
    assert float(weights.sum()) == pytest.approx(1.0, abs=1e-12)
    assert float(weights @ values) == pytest.approx(1.0, rel=1e-6)
    assert float(weights @ values ** (-0.6)) == pytest.approx(
        liquidity.inverse_power_moment(0.6), rel=1e-6
    )


def test_the_deterministic_law_is_the_market_every_earlier_milestone_ran_in():
    """``L = 1``, no randomness consumed, and a quadrature that is a point mass.

    The second half matters more than it looks: a law that never touches a
    generator is one independent reason M3's and M4a's committed seeds retrain
    bitwise through the new seam, on top of the liquidity streams being addressed
    away from the price streams.
    """
    liquidity = DeterministicLiquidity()
    generator = pool_rng(1, M4B_REFERENCE_POOL, 0)
    before = generator.bit_generator.state
    assert np.all(liquidity.draw(generator, (7, 13)) == 1.0)
    assert generator.bit_generator.state == before, (
        "the deterministic liquidity law consumed randomness; a Phase-1 env "
        "handed one would no longer be Phase 1"
    )
    values, weights = liquidity.quadrature(15)
    assert values.tolist() == [1.0] and weights.tolist() == [1.0]
    assert liquidity.inverse_power_moment(0.6) == 1.0
    assert not liquidity.stochastic
    assert liquidity_for("deterministic") == liquidity


# ---------------------------------------------------------------------------
# The cost functional: one more argument, and nothing else moved
# ---------------------------------------------------------------------------


def test_cost_moments_without_liquidity_is_bit_identical_to_before(case):
    """No M4a or earlier number may move because a later milestone widened a signature."""
    market, order_size, lambda_risk = case
    for trajectory in (
        twap_trajectory(market, order_size),
        power_law_optimum(market, order_size, lambda_risk),
    ):
        plain = cost_moments(trajectory, market)
        ones = cost_moments(trajectory, market, liquidity=np.ones(market.n_bins))
        assert (plain.temporary, plain.permanent, plain.spread, plain.variance) == (
            ones.temporary,
            ones.permanent,
            ones.spread,
            ones.variance,
        )
        # And the deterministic law's expectation is the same object again.
        expected = expected_cost_moments(trajectory, market, DeterministicLiquidity())
        assert expected.temporary == plain.temporary
        assert expected.objective(lambda_risk) == plain.objective(lambda_risk)


def test_liquidity_enters_expected_cost_and_never_the_graded_variance(case, law):
    """Invariant 7 holds with no amendment, and this is the arithmetic that says so.

    The frozen objective penalises *price*-shortfall variance,
    ``V = sigma_bin^2 sum (x_k / X)^2``. Liquidity dispersion enters ``E[cost]``
    through Jensen and never ``lambda V`` — so one functional, still encoded once.
    The realised-cost variance the differential measures now has two sources while
    the graded ``V`` has one, and that distinction is precisely the kind that
    drifts silently, so it is written down as a test rather than as a sentence.
    """
    market, order_size, lambda_risk = case
    trajectory = power_law_optimum(market, order_size, lambda_risk)
    plain = cost_moments(trajectory, market)
    under_law = expected_cost_moments(trajectory, market, law)

    assert under_law.variance == plain.variance, "liquidity moved the graded V"
    assert under_law.permanent == plain.permanent
    assert under_law.spread == plain.spread
    assert under_law.temporary == plain.temporary * law.inverse_power_moment(
        market.temp_exponent
    )
    assert under_law.temporary > plain.temporary

    # Per *path* the variance is likewise untouched, however extreme the draw.
    extreme = np.full(market.n_bins, 0.2)
    assert cost_moments(trajectory, market, liquidity=extreme).variance == plain.variance


def test_the_vectorised_path_objective_agrees_with_the_grader_route(case, law):
    """``path_objective_bps`` is the fast twin of ``cost_moments(liquidity=...)``.

    Two routes to ``E[cost | L]``: the one the bounds are computed with, over an
    array of paths at once, and the one the grader will call per schedule. If they
    can disagree then the reference and the grade are measuring different things,
    which is the failure M4a's registry rule exists to prevent one level up.
    """
    market, order_size, lambda_risk = case
    trajectory = static_optimum(market, order_size, lambda_risk, law)
    weights = trades(trajectory, market) / order_size
    multipliers = law.draw(pool_rng(20260823, M4B_REFERENCE_POOL, 1), (32, market.n_bins))

    fast = path_objective_bps(weights, multipliers, market, order_size, lambda_risk)
    slow = np.array(
        [
            cost_moments(trajectory, market, liquidity=path).objective(lambda_risk)
            for path in multipliers
        ]
    )
    assert fast == pytest.approx(slow, rel=1e-13)

    # And averaging the conditional expectation over the law returns the closed
    # form the static rung is priced by — the same statement, one level up.
    level = expected_cost_moments(trajectory, market, law).objective(lambda_risk)
    many = path_objective_bps(
        weights,
        law.draw(pool_rng(20260823, M4B_REFERENCE_POOL, 2), (200_000, market.n_bins)),
        market,
        order_size,
        lambda_risk,
    )
    half_width = 1.96 * many.std(ddof=1) / math.sqrt(many.size)
    assert abs(float(many.mean()) - level) <= 3.0 * half_width


# ---------------------------------------------------------------------------
# The two solvers, and the differential between them
# ---------------------------------------------------------------------------


def test_the_batched_clairvoyant_solver_reproduces_the_certified_scalar_one(case, law):
    """M4a's Newton, one axis wider, checked against M4a's Newton.

    A path of *constant* multipliers is exactly the static problem at the
    coefficient ``A L^-beta``, which :func:`optimum_for_charge` already solves and
    ``tests/test_power_law_certificate.py`` already certifies. So the batched
    per-bin solver has a certified answer to be wrong about, on the one family of
    inputs where one exists, and the band is the certificate's own: 1e-10 of X.
    """
    market, order_size, lambda_risk = case
    charge = power_law_charge(market, order_size)
    constants = np.array([0.6, 0.85, 1.0, 1.3, 2.1])
    paths = np.repeat(constants[:, None], market.n_bins, axis=1)

    batched = clairvoyant_trajectories(market, order_size, lambda_risk, paths)
    for index, multiplier in enumerate(constants):
        scalar = optimum_for_charge(
            market,
            order_size,
            lambda_risk,
            type(charge)(
                scale=charge.scale * float(multiplier) ** (-market.temp_exponent),
                exponent=charge.exponent,
                encoding=charge.encoding,
            ),
        )
        assert batched[index] == pytest.approx(scalar, abs=1e-10 * order_size)

    # On genuinely varying paths there is no closed form to check against, so the
    # claim is the one that is checkable: it liquidates, it never buys back, and
    # its value beats the static optimum on every path (it has more information).
    varying = law.draw(pool_rng(20260823, M4B_REFERENCE_POOL, 3), (256, market.n_bins))
    solved = clairvoyant_trajectories(market, order_size, lambda_risk, varying)
    assert np.max(np.abs(solved[:, -1])) <= 1e-9 * order_size
    assert np.all(np.diff(solved, axis=1) <= 0.0), "a clairvoyant path bought back"

    static_weights = trades(
        static_optimum(market, order_size, lambda_risk, law), market
    ) / order_size
    clairvoyant_cost = path_objective_bps(
        -np.diff(solved, axis=1) / order_size, varying, market, order_size, lambda_risk
    )
    static_cost = path_objective_bps(
        static_weights, varying, market, order_size, lambda_risk
    )
    assert np.all(clairvoyant_cost <= static_cost + 1e-12), (
        "perfect information cost more than a fixed schedule on some path; the "
        "relaxation is not a relaxation and the red-flag test has no proof behind it"
    )


def test_the_liquidity_charge_is_m4a_at_an_inflated_coefficient(case, law):
    """The static rung is a *closed form*, and this is why it is allowed to be one."""
    market, order_size, lambda_risk = case
    base = power_law_charge(market, order_size)
    inflated = liquidity_charge(market, order_size, law)
    assert inflated.exponent == base.exponent
    assert inflated.encoding == base.encoding
    assert inflated.scale == base.scale * law.inverse_power_moment(market.temp_exponent)

    solved = static_optimum(market, order_size, lambda_risk, law)
    assert solved == pytest.approx(
        optimum_for_charge(market, order_size, lambda_risk, inflated)
    )
    # A monotone rescaling of M4a's problem, so the schedule is *slower*: impact
    # got more expensive relative to risk and the optimum spreads out.
    m4a = power_law_optimum(market, order_size, lambda_risk)
    assert float(np.max(trades(solved, market))) < float(np.max(trades(m4a, market)))


# ---------------------------------------------------------------------------
# The dynamic program
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("points", [401, 801])
def test_the_dp_returns_m4a_certified_optimum_when_liquidity_is_degenerate(
    case, points
):
    """The single most valuable test in the milestone.

    At ``sigma_log = 0`` there is nothing to adapt to, so the optimum over adapted
    policies collapses onto the optimum over schedules — and that one is
    *certified*. Every piece of new machinery is in the path of this number: the
    backward recursion, the Gauss-Hermite collapse to a point mass, the golden
    section against an interpolated value function, the inventory grid, and
    ``schedule_invariant_bps``. A defect in any of them shows up here as a
    disagreement with a value that has a Cholesky factorisation behind it.
    """
    market, order_size, lambda_risk = case
    certified = cost_moments(
        power_law_optimum(market, order_size, lambda_risk), market
    ).objective(lambda_risk)
    solved = adaptive_optimum(
        market, order_size, lambda_risk, LognormalLiquidity(0.0), points=points
    )
    assert solved.quadrature_nodes == 1, "a point mass needs one node, not fifteen"
    # Linear interpolation of a convex value function converges from *above*, so
    # the DP may only ever overstate — and by less at the finer grid.
    excess = solved.objective_bps - certified
    assert 0.0 <= excess <= 3e-4
    assert DeterministicLiquidity() != LognormalLiquidity(0.0)  # different objects...
    assert adaptive_optimum(
        market, order_size, lambda_risk, DeterministicLiquidity(), points=points
    ).objective_bps == solved.objective_bps  # ...same world


def test_the_dp_beats_the_best_fixed_schedule_and_the_advantage_is_adaptivity(
    case, law
):
    """The milestone's headline quantity, and the rung it is *not* measured from."""
    market, order_size, lambda_risk = case
    solved = adaptive_optimum(market, order_size, lambda_risk, law, points=FAST_POINTS)
    static = expected_cost_moments(
        static_optimum(market, order_size, lambda_risk, law), market, law
    ).objective(lambda_risk)
    m4a = expected_cost_moments(
        power_law_optimum(market, order_size, lambda_risk), market, law
    ).objective(lambda_risk)

    assert solved.objective_bps < static < m4a, (
        "the three rungs are out of order: seeing liquidity must beat knowing its "
        "law, which must beat knowing nothing about it"
    )
    advantage = static - solved.objective_bps
    shift = m4a - static
    assert shift / advantage <= 0.10, "the level shift has taken over the headline"
    # And the level shift is *positive but tiny* — which is the whole reason both
    # rungs are computed in closed form rather than differenced out of simulations.
    assert 0.0 < shift < 0.01


def test_the_augmented_state_does_not_improve_the_value(case, law):
    """Task 1(e): ``(k, x_k, L_k)`` is sufficient, measured rather than asserted.

    A value that *improves* when ``L_{k-1}`` is carried means the process
    implementation is not i.i.d. — a bug in the env, not a discovery about
    markets. The column spread is the sharper half of the check: it says the
    continuations are equal for every previous multiplier, not merely that two
    scalars agreed at one point.
    """
    market, order_size, lambda_risk = case
    plain = adaptive_optimum(
        market, order_size, lambda_risk, law, points=FAST_POINTS
    ).objective_bps
    augmented = augmented_optimum(
        market, order_size, lambda_risk, law, points=FAST_POINTS
    )
    assert augmented.objective_bps == pytest.approx(plain, abs=1e-12)
    assert augmented.column_spread <= 1e-9, (
        f"the continuation depends on the previous multiplier (spread "
        f"{augmented.column_spread:.3e} bps); the liquidity process is not i.i.d."
    )
    # The check would have content if it could fail: the transition matrix is a
    # genuine (nodes, nodes) object, not a broadcast of one row by construction.
    values, transition = law.transition_quadrature(7)
    assert transition.shape == (values.size, values.size)
    assert np.allclose(transition.sum(axis=1), 1.0)


def test_the_feasible_bound_is_a_real_policy_and_its_action_map_is_checked(case, law):
    """Task 1(c) — and a measurement of the failure the 2 % band exists to catch.

    The greedy policy is what an agent could actually execute, so its mean
    conditional cost is an unbiased estimate of an attainable value. The band on
    it is not a test of the DP's convergence — at any affordable path count the
    estimate's own half-width dwarfs the gap — it is a test of the **action map**.
    So the comparison made here is the one that resolves: the same value functions
    rolled out with the stage problem snapped to a coarse grid instead of searched,
    **paired on the same liquidity paths**, where the difference between two
    policies has a hundredth of the variance of either level.
    """
    market, order_size, lambda_risk = case
    solved = adaptive_optimum(market, order_size, lambda_risk, law, points=FAST_POINTS)
    static_trajectory = static_optimum(market, order_size, lambda_risk, law)
    static_level = expected_cost_moments(static_trajectory, market, law).objective(
        lambda_risk
    )
    advantage = static_level - solved.objective_bps

    multipliers = law.draw(
        pool_rng(20260823, M4B_REFERENCE_POOL, 4), (8_000, market.n_bins)
    )
    interpolated = solved.greedy_weights(multipliers)
    assert np.all(interpolated >= -1e-12), "the greedy policy bought back"
    assert interpolated.sum(axis=1) == pytest.approx(1.0, abs=1e-12)

    cost_interpolated = path_objective_bps(
        interpolated, multipliers, market, order_size, lambda_risk
    )
    cost_static = path_objective_bps(
        trades(static_trajectory, market) / order_size,
        multipliers,
        market,
        order_size,
        lambda_risk,
    )
    paired = cost_interpolated - cost_static
    estimate = static_level + float(paired.mean())
    half_width = 1.96 * paired.std(ddof=1) / math.sqrt(paired.size)
    assert abs(estimate - solved.objective_bps) <= FEASIBLE_BAND * advantage + half_width

    # The same value functions with a snapped action map, paired against the
    # interpolated one. Measured at 11 % of the advantage: five times the band, and
    # resolved to 0.15 % because the comparison shares its liquidity paths.
    snapped = _snapped_greedy(solved, multipliers, market, order_size, levels=41)
    degradation = (
        path_objective_bps(snapped, multipliers, market, order_size, lambda_risk)
        - cost_interpolated
    )
    margin = 1.96 * degradation.std(ddof=1) / math.sqrt(degradation.size)
    assert float(degradation.mean()) - margin > FEASIBLE_BAND * advantage, (
        "a coarse, snapped action map is not measurably worse than the "
        "interpolated stage solve, so the 2 % band does not discriminate and "
        "task 1(c) is checking nothing"
    )


def _snapped_greedy(solved, multipliers, market: Market, order_size, *, levels: int):
    """The greedy policy with the stage problem *snapped* rather than solved.

    Deliberately naive: restrict the next inventory to a coarse grid and take the
    best node. This is the implementation task 1(c) tells the milestone not to
    ship, reproduced so that the bar which rejects it can be shown to reject it.
    """
    beta = market.temp_exponent
    coefficients = power_law_charge(market, order_size).scale * multipliers ** (-beta)
    grid = np.linspace(0.0, order_size, levels)
    inventory = np.full(multipliers.shape[0], float(order_size))
    weights = np.empty((multipliers.shape[0], market.n_bins))
    for index in range(market.n_bins):
        if index == market.n_bins - 1:
            trade = inventory
        else:
            remaining = inventory[:, None] - grid[None, :]
            value = np.where(
                remaining >= 0.0,
                coefficients[:, index][:, None]
                * (np.abs(remaining) / order_size) ** (1.0 + beta)
                + np.interp(grid, solved.grid, solved.continuations[index + 1])[None, :],
                np.inf,
            )
            trade = inventory - grid[np.argmin(value, axis=1)]
        weights[:, index] = trade / order_size
        inventory = inventory - trade
    return weights


# ---------------------------------------------------------------------------
# The lambda rule, in the reading task 0 recorded
# ---------------------------------------------------------------------------


def test_the_rule_selects_the_committed_point_in_every_reading(experiment, law):
    """Gate 1: M3's, M4a's and M4b's points are the same lambda, so they compare."""
    market, order_size = experiment.case.market, experiment.case.order_size
    selections = experiment.verify_lambda_rule_agrees_across_worlds()
    assert LIQUIDITY_READING in selections, (
        "a stochastic-liquidity experiment did not have its lambda checked in the "
        "liquidity world"
    )
    assert set(selections.values()) == {experiment.lambda_risk}
    assert experiment.verify_lambda_rule().lambda_risk == experiment.lambda_risk

    row = select_lambda(
        static_liquidity_table(market, order_size, law), experiment.rule
    )
    assert row.lambda_risk == experiment.lambda_risk
    assert row.optimal is row.static, (
        "the rule read something other than the static optimum; task 0 recorded "
        "the static reading and the config's comment says why"
    )


def test_the_adaptive_reading_of_the_rule_is_knife_edge_and_was_not_taken(committed):
    """The finding task 0 recorded, pinned so it cannot quietly change.

    The rule read against the DP's value and its mean schedule selects a
    *different* lambda, and it does so by clearing the 20 % bar at 10^-4 by
    0.011 percentage points. A milestone's lambda turning on the fifth digit of a
    numerically-solved value function is the reason the static reading was
    recorded — not because it agreed.
    """
    gate = committed["gates"]["lambda_agreement"]
    assert gate["reading_rule_applied_to"] == "static"
    alternative = gate["alternative_reading_selects"]
    assert alternative is not None and alternative != committed["config"]["lambda_risk"]

    rule = LambdaRule(**committed["config"]["lambda_rule"])
    row = next(
        entry for entry in committed["table"] if entry["lambda"] == alternative
    )
    margin = row["adaptive_twap_gap"] - rule.min_twap_gap
    assert 0.0 < margin < 1e-3, (
        f"the adaptive reading's margin at the lambda it selects is {margin:.2e}; "
        "task 0 recorded it as knife-edge and the argument for the static reading "
        "rests on that"
    )
    assert row["twap_gap"] < rule.min_twap_gap, (
        "the static reading now admits the same lambda, so the two readings no "
        "longer disagree and task 0's record needs rewriting rather than trusting"
    )


# ---------------------------------------------------------------------------
# The committed artefact
# ---------------------------------------------------------------------------


def test_the_committed_reference_records_four_green_gates(committed, experiment):
    """Definition of done, item 1 — checked in the suite, not only in a log."""
    assert committed["milestone"] == "M4b" and committed["task"] == "0"
    assert committed["all_green"] is True
    gates = committed["gates"]
    assert set(gates) == {"lambda_agreement", "advantage", "level_shift", "bracket"}
    assert all(gate["green"] for gate in gates.values())

    assert gates["advantage"]["advantage_fraction"] >= gates["advantage"]["bar"]
    assert gates["level_shift"]["level_shift_fraction"] <= gates["level_shift"]["bar"]
    assert gates["bracket"]["bracket_fraction"] <= gates["bracket"]["bar"]
    assert gates["bracket"]["ordered_with_cis"] is True

    assert committed["liquidity"]["invented"] is True, (
        "the results file must record that the liquidity process is Temper's own; "
        "constitution §7's vendored-not-invented cover does not reach it"
    )
    assert committed["selected"]["adaptive"]["certified"] is False
    assert committed["config"]["lambda_risk"] == experiment.lambda_risk
    assert committed["config"]["liquidity"] == experiment.liquidity.as_dict()


def test_the_committed_static_rungs_regenerate_from_the_config(committed, experiment):
    """Invariant 1 on the half of the table that has closed forms: exact, not close."""
    market, order_size = experiment.case.market, experiment.case.order_size
    law = experiment.liquidity
    selected = committed["selected"]
    trajectories = liquidity_trajectories(
        market, order_size, experiment.lambda_risk, law
    )
    for name, trajectory in trajectories.items():
        moments = expected_cost_moments(trajectory, market, law)
        assert moments.objective(experiment.lambda_risk) == pytest.approx(
            selected["schedules"][name]["objective_bps"], rel=1e-12
        )
    assert selected["level_shift_bps"] == pytest.approx(
        expected_cost_moments(trajectories["m4a"], market, law).objective(
            experiment.lambda_risk
        )
        - expected_cost_moments(trajectories["static"], market, law).objective(
            experiment.lambda_risk
        ),
        rel=1e-12,
    )


def test_the_brief_and_the_box_agree_on_every_predicted_number(committed):
    """The brief's numbers were predictions on unpinned numpy; these are artefacts.

    A material disagreement would mean the brief is wrong before the code is, and
    the instruction is to stop and report rather than to adjust the brief to fit.
    They agree, so this pins the agreement: the tolerances are generous where the
    predicted quantity is itself a Monte-Carlo estimate and tight where it is a
    closed form.
    """
    predicted = committed["predicted_by_brief"]
    gates = committed["gates"]
    selected = committed["selected"]

    assert selected["adaptive_advantage_bps"] == pytest.approx(
        predicted["advantage"], rel=2e-3
    )
    assert selected["level_shift_bps"] == pytest.approx(
        predicted["level_shift"], rel=5e-3
    )
    assert selected["schedules"]["static"]["objective_bps"] == pytest.approx(
        predicted["j_static"], rel=1e-4
    )
    assert selected["schedules"]["m4a"]["objective_bps"] == pytest.approx(
        predicted["j_m4a"], rel=1e-4
    )
    assert selected["adaptive_bps"] == pytest.approx(predicted["j_dp"], rel=1e-4)
    assert gates["level_shift"]["level_shift_fraction"] == pytest.approx(
        predicted["level_shift_fraction"], rel=0.05
    )
    assert committed["convergence"]["sigma_zero_bps"] == pytest.approx(
        predicted["sigma_zero"], abs=1e-5
    )
    # The bracket and the paired SD are sampled, so they get sampling-sized bands.
    assert gates["bracket"]["bracket_fraction"] == pytest.approx(
        predicted["bracket_fraction"], rel=0.25
    )
    assert selected["feasible_upper"]["paired_sd_bps"] == pytest.approx(
        predicted["paired_sd"], rel=0.1
    )


# ---------------------------------------------------------------------------
# Convergence — minutes, so behind the marker
# ---------------------------------------------------------------------------


@pytest.mark.deep
def test_the_dp_converges_second_order_in_the_inventory_grid(case, law):
    """Grid convergence and the Richardson residual — the reference's uncertainty.

    Linear interpolation of a convex value function is second order in the
    spacing and converges from above, so halving the spacing must quarter the
    error and every value must decrease. The residual this implies is the number
    reported beside ``J_DP`` everywhere, because a reference that is converged
    rather than certified has to say how converged.
    """
    market, order_size, lambda_risk = case
    values = {
        points: adaptive_optimum(
            market, order_size, lambda_risk, law, points=points
        ).objective_bps
        for points in (201, 401, 801, 1601)
    }
    ordered = [values[points] for points in (201, 401, 801, 1601)]
    assert ordered == sorted(ordered, reverse=True), (
        "the DP did not converge monotonically from above; linear interpolation of "
        "a convex value function cannot understate it"
    )
    first = ordered[0] - ordered[1]
    second = ordered[1] - ordered[2]
    third = ordered[2] - ordered[3]
    assert 3.0 <= first / second <= 6.0
    assert 3.0 <= second / third <= 6.0

    extrapolant, residual = richardson_residual(ordered[2], ordered[3])
    assert residual < 1e-5
    assert extrapolant < ordered[3]

    # Quadrature is the cheap axis and it is converged well before the committed
    # node count, which is the claim the default is defended by.
    fine = adaptive_optimum(
        market, order_size, lambda_risk, law, points=401, nodes=41
    ).objective_bps
    for nodes in (7, 15, 21):
        coarse = adaptive_optimum(
            market, order_size, lambda_risk, law, points=401, nodes=nodes
        ).objective_bps
        assert abs(coarse - fine) < 1e-5
