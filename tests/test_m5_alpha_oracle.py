"""M5 task 0 — the alpha oracle, and the checks that make it a reference.

Everything here is oracle-only. No env, no agent, no training loop, and no import
that reaches one: the milestone's own rule is that no training code is written,
imported or run until task 0's four gates are recorded green in the repo, and this
module is part of what records them.

M5 is the first milestone to hold **two kinds of confidence at once**, and most of
what follows exists to keep them apart:

* :func:`~temper.oracle.alpha.alpha_optimum` is a dynamic program — **converged**,
  not certified, and the artefact says so in its own bytes.
* :func:`~temper.oracle.alpha.execution_floor_bps` is M4a's **certified** optimum,
  and it bounds the half of the objective the signal cannot touch.

The reference earns its place by four routes, and they are deliberately different
from each other:

* **It reduces to a certified number.** At ``rho = 0`` the dynamic program must
  return M4a's ``power_law_optimum`` value — with the full quadrature, so the
  expectation is actually taken and the alpha term actually cancels. That single
  check ties the value iteration, the quadrature, the stage solve, the three
  companion value functions, the inventory grid and the schedule-invariant
  constant to a value that *was* certified. Everything else in the milestone is
  new machinery measured by more new machinery.
* **Its conditional cost is checked against sampled reality.** ``E[cost | s]`` is
  a closed form; it is compared with a Monte-Carlo average over *price* draws with
  the signal path pinned. That is what pins the sign of the alpha term and the
  index that says which shock a signal predicts — the two things that would be
  wrong in a way no internal consistency check could see.
* **Its decomposition is checked twice.** The exact backward-pass decomposition is
  required to close as an identity at every node, and then to agree with an
  independent Monte-Carlo decomposition of the same policy rolled out on sampled
  paths.
* **The static reading is checked to the bit.** M4b needed a third *reading* of
  the lambda rule and a recorded decision between two candidates that disagreed.
  M5's static reading is M4a's table exactly, and "exactly" is asserted rather
  than reasoned about.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from temper.eval.experiment import SIGNAL_READING, load_experiment
from temper.eval.reference import (
    LambdaRule,
    ReferenceKind,
    alpha_reference_row,
    reference_table,
    select_lambda,
    signal_static_row,
    signal_static_table,
)
from temper.oracle import (
    POWER_LAW_ENCODING,
    NoSignal,
    OneStepSignal,
    alpha_coefficient,
    alpha_optimum,
    augmented_alpha_optimum,
    clairvoyant_price_values,
    cost_moments,
    execution_floor_bps,
    expected_alpha_bps,
    inventory_penalty_scale,
    power_law_charge,
    power_law_optimum,
    schedule_invariant_bps,
    signal_for,
    signal_path_objective_bps,
    trades,
    twap_trajectory,
)
from temper.seeding import M5_REFERENCE_POOL, pool_rng

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "m5_alpha.yaml"
REFERENCE_PATH = REPO_ROOT / "results" / "m5_reference.json"

#: A coarser grid than the committed reference's, so the whole module stays inside
#: the suite's seconds budget. Every claim here is about *agreement between two
#: routes* and both routes see the same grid, so the resolution is not what is
#: under test — the committed artefact carries the converged numbers.
FAST_POINTS = 401


@pytest.fixture(scope="module")
def experiment():
    return load_experiment(CONFIG_PATH)


@pytest.fixture(scope="module")
def case(experiment):
    return experiment.case.market, experiment.case.order_size, experiment.lambda_risk


@pytest.fixture(scope="module")
def signal(experiment):
    return experiment.signal


@pytest.fixture(scope="module")
def committed() -> dict:
    if not REFERENCE_PATH.exists():
        pytest.fail(
            f"{REFERENCE_PATH.relative_to(REPO_ROOT)} is missing; regenerate task 0 "
            "with `python tools/m5_reference_table.py` before the suite can check "
            "its gates"
        )
    return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The invented process, and the law it says it is
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rho", [0.0, 0.0025, 0.01, 0.05, 0.2])
def test_the_signal_draws_the_joint_law_it_claims(rho):
    """``Corr(s_k, xi_{k+1}) = rho``, every other pair independent, unit variances.

    The whole model is one correlation and one index, and both are the kind of
    thing that is wrong silently: a shock correlated with ``s_k`` instead of
    ``s_{k-1}`` would give an agent a signal about a shock that has already landed,
    which is worth nothing and looks like a training failure rather than a seam
    defect.
    """
    signal = OneStepSignal(rho)
    rng = pool_rng(20260824, M5_REFERENCE_POOL, 11)
    signals, shocks = signal.draw_pair(rng, (400_000, 6))

    assert signals.shape == shocks.shape
    # Unit variance on both, so the signal does not quietly change the market it
    # is a signal about: M1's variance identity and invariant 7 are statements
    # about sigma_bin.
    assert shocks.std() == pytest.approx(1.0, abs=0.01)
    assert signals.std() == pytest.approx(1.0, abs=0.01)
    assert signals.mean() == pytest.approx(0.0, abs=0.01)

    # Exhaustive over the pairs: s_k predicts xi_{k+1} and nothing else — not the
    # shock in its own bin, not one two bins out, and above all not one that has
    # already landed. The draw is seed-addressed, so these are fixed numbers rather
    # than a coin toss; the band is ~5 sampling standard deviations at this M.
    bins = signals.shape[1]
    for k in range(bins):
        for j in range(bins):
            observed = float(np.corrcoef(signals[:, k], shocks[:, j])[0, 1])
            expected = rho if j == k + 1 else 0.0
            assert observed == pytest.approx(expected, abs=0.008), (
                f"corr(s_{k}, xi_{j}) = {observed:.5f}, expected {expected}"
            )


def test_the_signals_mean_is_exactly_zero_and_the_quadratures_is_not():
    """The distinction the lambda claim rests on, written down as a test.

    ``AlphaSignal.mean()`` is an exact float zero and the static pricing route uses
    it, which is why M5's table is bit-identical to M4a's. The *quadrature's* first
    moment is ~1e-17 rather than zero, because Gauss-Hermite nodes are symmetric in
    exact arithmetic and not in floats. Both facts are true and only the first one
    is load-bearing; recording the second here stops a later session discovering it
    and doubting the first.
    """
    signal = OneStepSignal(0.01)
    assert signal.mean() == 0.0
    assert NoSignal().mean() == 0.0
    for nodes in (5, 15, 21):
        values, weights = signal.quadrature(nodes)
        # ~2e-17 at every node count measured, and NOT an exact zero. Bounded
        # rather than pinned: if a numpy release ever made it exact that would be
        # good news, and a test that went red on it would be reporting the wrong
        # thing. What matters is that it is small enough to be irrelevant to the
        # DP and that the bit-identity claim is made about  instead.
        assert abs(float(weights @ values)) < 1e-15
        assert float(weights @ values**2) == pytest.approx(1.0, abs=1e-14)
        assert float(weights.sum()) == pytest.approx(1.0, abs=1e-15)


def test_rho_zero_keeps_the_quadrature_and_no_signal_collapses_it():
    """Two different objects, deliberately, and gate 1 uses both.

    ``NoSignal`` is the absence of the seam; ``OneStepSignal(0)`` is the seam
    carrying an uninformative draw. The second is the stronger differential
    because the expectation is actually taken.
    """
    assert OneStepSignal(0.0).quadrature(15)[0].size == 15
    assert NoSignal().quadrature(15)[0].size == 1
    assert OneStepSignal(0.0).informative is False
    assert OneStepSignal(0.01).informative is True
    assert signal_for("one_step", rho=0.01) == OneStepSignal(0.01)
    assert signal_for("none") == NoSignal()
    with pytest.raises(ValueError, match="unknown signal model"):
        signal_for("hunch", rho=0.5)


# ---------------------------------------------------------------------------
# The conditional cost, against sampled reality
# ---------------------------------------------------------------------------


def test_the_conditional_cost_matches_a_sampled_mean_over_price_draws(case):
    """``E[cost | s]`` is a closed form, and this is what says it is the right one.

    Everything else in this module is an internal-consistency check that a flipped
    sign or a shifted index would sail through. This one is not: it draws
    ``(s, xi)`` from the signal's **own joint law**, prices a signal-reacting
    schedule the way the env would — realised noise ``-A sum_k h_k xi_k``, written
    out here rather than imported so the two routes are genuinely independent — and
    requires the sampled mean to be the conditional closed form.

    It then requires the two convention errors that matter to be **rejected**: the
    alpha term with its sign flipped, and the same term reading ``s_k`` where it
    should read ``s_{k-1}``. A signal about a shock that has already landed is worth
    nothing, and getting that index wrong would look like a training failure rather
    than a seam defect.

    Run at ``rho = 0.2``, which is far outside the milestone's world and is the
    point: the price noise the average has to see through has a per-path standard
    deviation of ~90 bps, so a conventions test needs an effect large enough to
    resolve against it. The convention is the same at every rho; the trained value
    is checked by the identity and decomposition tests above.
    """
    market, order_size, lambda_risk = case
    signal = OneStepSignal(0.2)
    charge = power_law_charge(market, order_size)
    penalty = lambda_risk * inventory_penalty_scale(market)
    amplitude = alpha_coefficient(market)

    # 50 000 rather than more: the discrimination below is twenty-plus half-widths
    # at this M, and rolling the DP policy out is the whole cost of this module.
    paths = 50_000
    signals, shocks = signal.draw_pair(
        pool_rng(20260824, M5_REFERENCE_POOL, 12), (paths, market.n_bins)
    )

    # The DP's own policy, because it is the schedule that reacts to the signal
    # hardest and the discrimination below is a signal-to-noise argument: at
    # rho = 0.2 it monetises ~11 bps, which is twenty-odd half-widths. It reaches
    # this test through `_stage_minimum` and never through the function under
    # test, so the two routes stay independent.
    weights = alpha_optimum(
        market, order_size, lambda_risk, signal, points=FAST_POINTS
    ).greedy_weights(signals)
    # Non-negative rather than strictly positive: at this rho the optimum is
    # nearly a two-bin liquidation, and an empty bin is inside the reachable set.
    assert (weights >= 0.0).all()
    assert weights.sum(axis=1) == pytest.approx(1.0, abs=1e-12)
    holdings = 1.0 - np.cumsum(weights, axis=1) + weights

    realised = (
        np.sum(charge.scale * weights ** (1.0 + market.temp_exponent), axis=1)
        + penalty * np.sum(holdings**2, axis=1)
        + schedule_invariant_bps(market, order_size)
        - amplitude * np.sum(holdings * shocks, axis=1)
    )
    closed = signal_path_objective_bps(
        weights, signals, market, order_size, lambda_risk, signal
    )

    difference = realised - closed
    half_width = 1.96 * difference.std(ddof=1) / math.sqrt(paths)
    assert abs(difference.mean()) < half_width, (
        f"E[cost | s] is biased by {difference.mean():.4f} +/- {half_width:.4f} bps"
    )

    # The two convention errors, each required to be rejected by many half-widths.
    alpha = -amplitude * signal.correlation() * np.sum(
        holdings[:, 1:] * signals[:, :-1], axis=1
    )
    shifted = -amplitude * signal.correlation() * np.sum(
        holdings[:, 1:] * signals[:, 1:], axis=1
    )
    for name, wrong in (
        ("sign flipped", closed - 2.0 * alpha),
        ("index shifted to the shock that already landed", closed - alpha + shifted),
    ):
        bias = float((realised - wrong).mean())
        assert abs(bias) > 10.0 * half_width, (
            f"the {name} convention is not distinguishable: bias {bias:.4f} bps "
            f"against a half-width of {half_width:.4f} ({abs(bias) / half_width:.1f}x)"
        )


def test_a_fixed_schedules_expected_alpha_is_exactly_zero(case, signal):
    """Not approximately. The lambda claim is a statement about floats."""
    market, order_size, lambda_risk = case
    for trajectory in (
        twap_trajectory(market, order_size),
        power_law_optimum(market, order_size, lambda_risk),
    ):
        assert expected_alpha_bps(trajectory, market, order_size, signal) == 0.0
        objective = cost_moments(trajectory, market).objective(lambda_risk)
        assert (
            objective + expected_alpha_bps(trajectory, market, order_size, signal)
            == objective
        )


# ---------------------------------------------------------------------------
# Lambda: the static reading, to the bit
# ---------------------------------------------------------------------------


def test_the_static_reading_is_bit_identical_to_m4a_and_selects_the_same_lambda(
    experiment, signal
):
    """Task 0's recorded result: there is no lambda decision to make here.

    M4b's session had to choose between a static reading that selected 10^-3.5 and
    an adaptive one that selected 10^-4.0 by 0.011 percentage points, and record
    the rejected one with its margin. A zero-mean signal leaves every fixed
    schedule where it was, so this session has nothing to choose — and the honest
    way to say so is to compare every float rather than to explain why they must
    match.
    """
    market, order_size = experiment.case.market, experiment.case.order_size
    mine = signal_static_table(market, order_size, signal)
    theirs = reference_table(
        market, order_size, encoding=POWER_LAW_ENCODING
    )
    assert len(mine) == len(theirs) == 17
    for signal_row, m4a_row in zip(mine, theirs, strict=True):
        assert signal_row.lambda_risk == m4a_row.lambda_risk
        assert signal_row.encoding == m4a_row.encoding == POWER_LAW_ENCODING
        assert set(signal_row.schedules) == set(m4a_row.schedules)
        for name, schedule in m4a_row.schedules.items():
            other = signal_row.schedules[name]
            assert other.expected == schedule.expected
            assert other.variance == schedule.variance
            assert other.risk == schedule.risk
            assert other.excess_risk == schedule.excess_risk
            assert other.objective == schedule.objective
            assert (other.trajectory == schedule.trajectory).all()
        assert signal_row.twap_gap == m4a_row.twap_gap

    assert experiment.rule == LambdaRule(min_twap_gap=0.20, max_bin_fraction=0.50)
    assert select_lambda(mine, experiment.rule).lambda_risk == experiment.lambda_risk
    assert (
        select_lambda(theirs, experiment.rule).lambda_risk == experiment.lambda_risk
    )
    assert SIGNAL_READING == "power_law+signal"


def test_the_signal_never_touches_the_graded_variance(case, signal):
    """Invariant 7 needs no amendment, and this is why. ``V`` is untouched."""
    market, order_size, lambda_risk = case
    trajectory = power_law_optimum(market, order_size, lambda_risk)
    plain = cost_moments(trajectory, market)
    row = signal_static_row(market, order_size, lambda_risk, signal)
    assert row.schedules["optimal"].variance == plain.variance
    strong = signal_static_row(market, order_size, lambda_risk, OneStepSignal(0.2))
    assert strong.schedules["optimal"].variance == plain.variance


# ---------------------------------------------------------------------------
# The dynamic program
# ---------------------------------------------------------------------------


def test_the_dp_at_rho_zero_returns_m4a_certified_value(case):
    """Gate 1 at the fast grid — the differential against a *certified* number.

    The most valuable check in the milestone: everything else here measures new
    machinery with more new machinery, and this measures it against a value with a
    Cholesky factorisation and a 1.2e-15 KKT residual behind it.
    """
    market, order_size, lambda_risk = case
    certified = cost_moments(
        power_law_optimum(market, order_size, lambda_risk), market
    ).objective(lambda_risk)

    full = alpha_optimum(
        market, order_size, lambda_risk, OneStepSignal(0.0), points=FAST_POINTS
    )
    collapsed = alpha_optimum(
        market, order_size, lambda_risk, NoSignal(), points=FAST_POINTS
    )
    # 1e-4 of the advantage is the committed bar; at this coarse grid the
    # discretisation is larger than at 1601 points, so the check here is the
    # weaker one and the artefact carries the committed number.
    assert full.objective_bps == pytest.approx(certified, abs=1e-4)
    assert collapsed.objective_bps == pytest.approx(certified, abs=1e-4)
    assert full.objective_bps == pytest.approx(collapsed.objective_bps, abs=1e-12)
    # The expectation is actually taken and the alpha term actually cancels.
    assert full.quadrature_nodes == 15 and collapsed.quadrature_nodes == 1
    assert full.alpha_bps == 0.0
    # The word discipline, mechanically: a dynamic program is never "certified".
    assert full.as_dict()["certified"] is False
    assert full.as_dict()["reference_kind"].startswith("converged")


def test_the_decomposition_closes_as_an_identity_everywhere(case, signal):
    """``J = impact + risk + alpha + invariant``, asserted at every node."""
    market, order_size, lambda_risk = case
    optimum = alpha_optimum(
        market, order_size, lambda_risk, signal, points=FAST_POINTS
    )
    assert optimum.node_identity_residual_bps < 1e-9
    assert optimum.identity_residual_bps < 1e-9
    assert optimum.objective_bps == pytest.approx(
        optimum.impact_bps
        + optimum.risk_bps
        + optimum.alpha_bps
        + optimum.invariant_bps,
        abs=1e-9,
    )
    assert optimum.invariant_bps == schedule_invariant_bps(market, order_size)
    assert optimum.alpha_bps < 0.0, "the optimum must monetise the signal, not pay it"
    assert optimum.execution_bps == optimum.impact_bps + optimum.risk_bps


def test_the_exact_decomposition_agrees_with_a_sampled_one(case, signal):
    """The second route: roll the DP's own policy out and decompose the samples.

    The backward pass computes the three terms through the same interpolation the
    value iteration uses, which makes the identity nearly free and the *values*
    unchecked. This decomposes the same policy the other way — on sampled signal
    paths, from the realised weights — so a companion recursion that had drifted
    from the policy would show here and nowhere else.
    """
    market, order_size, lambda_risk = case
    optimum = alpha_optimum(
        market, order_size, lambda_risk, signal, points=FAST_POINTS
    )
    paths = 40_000
    signals = signal.draw(pool_rng(20260824, M5_REFERENCE_POOL, 14), (paths, market.n_bins))
    weights = optimum.greedy_weights(signals)
    holdings = 1.0 - np.cumsum(weights, axis=1) + weights

    charge = power_law_charge(market, order_size)
    impact = np.sum(charge.scale * weights ** (1.0 + market.temp_exponent), axis=1)
    risk = lambda_risk * inventory_penalty_scale(market) * np.sum(holdings**2, axis=1)
    alpha = (
        -alpha_coefficient(market)
        * signal.correlation()
        * np.sum(holdings[:, 1:] * signals[:, :-1], axis=1)
    )

    for name, sampled, exact in (
        ("impact", impact, optimum.impact_bps),
        ("risk", risk, optimum.risk_bps),
        ("alpha", alpha, optimum.alpha_bps),
    ):
        half = 1.96 * sampled.std(ddof=1) / math.sqrt(paths)
        assert abs(sampled.mean() - exact) < max(half, 5e-4), (
            f"{name}: sampled {sampled.mean():.6f} +/- {half:.6f} vs exact "
            f"{exact:.6f} bps"
        )

    total = signal_path_objective_bps(
        weights, signals, market, order_size, lambda_risk, signal
    )
    # The greedy policy is feasible, so it cannot beat the optimum it came from by
    # more than sampling error: a feasible upper bound, and the only half of M4b's
    # bracket that survives into M5.
    half = 1.96 * total.std(ddof=1) / math.sqrt(paths)
    assert total.mean() > optimum.objective_bps - half


def test_the_convexity_floor_holds_for_the_policy_and_is_the_red_flag(case, signal):
    """The replacement red-flag test, on the DP and on realised paths.

    ``E[impact + risk] >= J_M4a_varying`` for any policy, by Jensen, with equality
    only at M4a's optimum. Checked on the DP's exact decomposition *and* on the
    sampled rollout, because the flag will be applied to an agent's sampled grade.
    """
    market, order_size, lambda_risk = case
    floor = execution_floor_bps(market, order_size, lambda_risk)
    optimum = alpha_optimum(
        market, order_size, lambda_risk, signal, points=FAST_POINTS
    )
    assert optimum.execution_bps > floor
    assert optimum.execution_bps - floor > 0.5 * (
        cost_moments(power_law_optimum(market, order_size, lambda_risk), market).objective(
            lambda_risk
        )
        - optimum.objective_bps
    ), "the margin has to be large enough to grade against"

    # The floor is M4a's certified optimum on the varying part, and no schedule
    # beats it — including the ones the table carries.
    for trajectory in (
        twap_trajectory(market, order_size),
        power_law_optimum(market, order_size, lambda_risk),
    ):
        weights = trades(trajectory, market) / order_size
        holdings = 1.0 - np.cumsum(weights) + weights
        execution = float(
            power_law_charge(market, order_size).cost_bps(weights)
            + lambda_risk * inventory_penalty_scale(market) * np.sum(holdings**2)
        )
        assert execution >= floor - 1e-12


# ---------------------------------------------------------------------------
# The relaxation that is computed in order to be retired
# ---------------------------------------------------------------------------


def test_the_clairvoyant_solver_reproduces_the_certified_optimum_at_zero_shocks(case):
    """Its own differential: with nothing to foresee it is M4a's problem.

    A per-path grid dynamic program is new machinery too, and this is the one
    place it can be measured against a certified number instead of against an
    opinion. The grid costs a little accuracy from above, which is the direction
    the milestone's argument needs.
    """
    market, order_size, lambda_risk = case
    certified = cost_moments(
        power_law_optimum(market, order_size, lambda_risk), market
    ).objective(lambda_risk)
    quiet = clairvoyant_price_values(
        market, order_size, lambda_risk, np.zeros((3, market.n_bins)), points=801
    )
    assert quiet.shape == (3,)
    assert float(quiet.std()) == 0.0
    assert float(quiet[0]) == pytest.approx(certified, abs=1e-4)
    assert float(quiet[0]) >= certified - 1e-9, (
        "a grid-restricted policy cannot beat the certified optimum; if it does, "
        "the interpolation is not converging from above and the looseness claim "
        "is not conservative"
    )


def test_price_clairvoyance_is_far_too_loose_to_be_a_red_flag(case):
    """Gate 4's evidence, at a path count the suite can afford.

    Retiring an inherited test needs evidence. Three orders of looseness is not a
    number that moves with the draw, so a handful of paths settles it.
    """
    market, order_size, lambda_risk = case
    certified = cost_moments(
        power_law_optimum(market, order_size, lambda_risk), market
    ).objective(lambda_risk)
    shocks = pool_rng(20260824, M5_REFERENCE_POOL, 15).standard_normal(
        (24, market.n_bins)
    )
    values = clairvoyant_price_values(market, order_size, lambda_risk, shocks)
    assert values.mean() < -20.0
    assert (certified - values.mean()) > 300.0 * 0.0808, (
        "the perfect-information bound has to be at least two orders looser than "
        "the advantage for the retirement to be justified"
    )


# ---------------------------------------------------------------------------
# Task 1 — sufficiency, timing, and the two words
# ---------------------------------------------------------------------------


def test_the_augmented_state_does_not_improve_the_value(case, signal):
    """Task 1: ``(k, x_k, s_k)`` is sufficient, measured rather than asserted.

    Carrying ``s_{k-1}`` cannot help, and the reason is sharper than M4b's: that
    signal predicted ``xi_k``, the shock has already landed, and the inventory it
    was charged on was fixed by the previous decision. Its information is
    **spent**.

    The bar is float noise rather than a tolerance. A leak — a signal with memory,
    or a seam that lets a past draw reach a future decision — produces a
    *systematic* improvement, not scatter, so a bar set where a tolerance would
    sit would hide exactly the failure this exists for.
    """
    market, order_size, lambda_risk = case
    plain = alpha_optimum(
        market, order_size, lambda_risk, signal, points=FAST_POINTS
    ).objective_bps
    augmented = augmented_alpha_optimum(
        market, order_size, lambda_risk, signal, points=FAST_POINTS
    )
    assert augmented.objective_bps == plain, (
        f"the augmented state moved the value by "
        f"{augmented.objective_bps - plain:+.3e} bps; under an i.i.d. one-step "
        f"signal it cannot move it at all"
    )
    assert augmented.column_spread <= 1e-14, (
        f"the continuation depends on the previous signal (spread "
        f"{augmented.column_spread:.3e} bps); the seam leaks"
    )
    assert augmented.bins_ahead == 1

    # The check would have content if it could fail: the transition matrix is a
    # genuine (nodes, nodes) object, not a broadcast of one row by construction.
    values, transition = signal.transition_quadrature(7)
    assert transition.shape == (values.size, values.size)
    assert np.allclose(transition.sum(axis=1), 1.0)


def test_a_signal_about_a_shock_that_has_already_landed_is_worth_nothing(case, signal):
    """Task 1's timing check — the off-by-one that would be invisible in the result.

    Everything M5 claims rests on ``s_k`` being about ``xi_{k+1}`` and not
    ``xi_k``. If the seam's timing were one bin out *in the helpful direction*
    every gate would still be green and the advantage would simply be larger, so
    no number the milestone reports would say anything was wrong. M4a caught its
    antithetic mirror charging the wrong world the same way: by running the
    machinery in a configuration whose answer is known in advance.

    Pointed one bin the wrong way, the dynamic program must return the
    *uninformative* value — not merely something small. At ``rho = 0`` the DP
    still carries its own grid discretisation, and calling that residual "a small
    advantage" would be reading noise.
    """
    market, order_size, lambda_risk = case
    landed = OneStepSignal(signal.correlation(), bins_ahead=0)
    blind = OneStepSignal(0.0)
    assert landed.informative is False, (
        "a signal about a shock that has already landed is not informative, "
        "however large rho is"
    )

    already = alpha_optimum(
        market, order_size, lambda_risk, landed, points=FAST_POINTS
    )
    uninformative = alpha_optimum(
        market, order_size, lambda_risk, blind, points=FAST_POINTS
    )
    # Float noise, not a tolerance. At the committed 1601-point grid these land
    # bit-identical; at this coarser one they differ by an ulp, because the lag-0
    # path threads a state term through additions the lag-1 path does not.
    assert abs(already.objective_bps - uninformative.objective_bps) <= 1e-14, (
        f"an already-landed signal is worth "
        f"{uninformative.objective_bps - already.objective_bps:+.3e} bps; it must "
        f"be worth nothing a reader could see"
    )
    assert abs(already.alpha_bps) < 1e-12
    assert already.bins_ahead == 0

    # The sharper half: the *policy* ignores it. Every path gets the same schedule
    # — exactly the same, because the alpha term's coefficient on the action is a
    # float zero — where the model's signal makes the schedule move with the draw.
    # This is what says the collapse is structural and not a cancellation.
    real = alpha_optimum(market, order_size, lambda_risk, signal, points=FAST_POINTS)
    signals = landed.draw(pool_rng(20260824, M5_REFERENCE_POOL, 16), (64, market.n_bins))
    ignored = already.greedy_weights(signals)
    tilted = real.greedy_weights(signals)
    assert float(np.ptp(ignored, axis=0).max()) == 0.0, (
        "the schedule moves with a signal no decision in it could have acted on"
    )
    assert ignored == pytest.approx(
        uninformative.greedy_weights(signals), abs=1e-9
    )
    assert float(np.ptp(tilted, axis=0).max()) > 1e-4, (
        "the model's signal does not move the schedule either, so this test would "
        "pass on a seam that ignored the signal altogether"
    )

    # And the model is worth orders more than the residual it has to be told apart
    # from, so the check has power rather than merely a passing assertion.
    advantage = uninformative.objective_bps - real.objective_bps
    assert advantage > 1e5 * abs(already.objective_bps - uninformative.objective_bps)


def test_the_conditional_cost_reads_the_signal_the_lag_names(case):
    """The same index, one layer down: ``E[cost | s]`` moves with the lag.

    The dynamic program and the grader have to agree about which shock a signal
    predicts, and they reach it by different routes — a stage cost and a vectorised
    sum. This pins the second one against the first's convention directly.
    """
    market, order_size, lambda_risk = case
    weights = trades(twap_trajectory(market, order_size), market) / order_size
    holdings = 1.0 - np.cumsum(weights) + weights
    signals = pool_rng(20260824, M5_REFERENCE_POOL, 17).standard_normal(market.n_bins)
    amplitude = alpha_coefficient(market)

    for lag, expected in (
        (1, -amplitude * 0.01 * float(np.sum(holdings[1:] * signals[:-1]))),
        (0, -amplitude * 0.01 * float(np.sum(holdings * signals))),
    ):
        signal = OneStepSignal(0.01, bins_ahead=lag)
        priced = float(
            signal_path_objective_bps(
                weights, signals, market, order_size, lambda_risk, signal
            )[0]
        )
        blind = float(
            signal_path_objective_bps(
                weights,
                signals,
                market,
                order_size,
                lambda_risk,
                OneStepSignal(0.0),
            )[0]
        )
        assert priced - blind == pytest.approx(expected, rel=1e-12)


def test_the_two_references_carry_their_own_kind(case, signal):
    """Task 1: certified and converged travel with the numbers, not in prose.

    A reader who takes the execution floor for the optimum makes the agent's job
    look seven times larger than it is; one who takes the optimum for a certified
    object claims a Cholesky factorisation for a number with a Richardson residual.
    Both misreadings are one careless sentence away, so the artefact carries the
    difference structurally.
    """
    market, order_size, lambda_risk = case
    row = alpha_reference_row(
        market,
        order_size,
        lambda_risk,
        signal,
        root_seed=20260824,
        paths=256,
        grid_points=FAST_POINTS,
    )
    kinds = row.reference_kinds
    assert set(kinds) == {"execution_floor", "adaptive_optimum"}

    floor, optimum = kinds["execution_floor"], kinds["adaptive_optimum"]
    assert floor.certified is True and floor.kind == "certified"
    assert optimum.certified is False and optimum.kind == "converged"
    assert floor.value_bps == row.execution_floor
    assert optimum.value_bps == row.adaptive_bps
    assert floor.value_bps < optimum.value_bps, (
        "the floor bounds only E[impact + risk]; it is below the whole objective "
        "by the schedule-invariant constant and by the alpha the policy monetises"
    )
    for kind in kinds.values():
        assert kind.evidence and kind.role

    # The word is checked, not merely stored.
    with pytest.raises(ValueError, match="contradicts"):
        ReferenceKind(
            name="wishful",
            value_bps=1.0,
            kind="converged",
            certified=True,
            role="-",
            evidence="-",
        )
    with pytest.raises(ValueError, match="'certified' or 'converged'"):
        ReferenceKind(
            name="wishful",
            value_bps=1.0,
            kind="proven",
            certified=True,
            role="-",
            evidence="-",
        )


# ---------------------------------------------------------------------------
# The committed artefact
# ---------------------------------------------------------------------------


def test_the_committed_reference_records_four_green_gates(committed, experiment):
    """Definition of done, item 1 — checked in the suite, not only in a log."""
    assert committed["milestone"] == "M5" and committed["task"] == "0+1"
    assert committed["all_green"] is True

    gates = committed["gates"]
    assert set(gates) == {
        "rho_zero",
        "advantage",
        "execution_premium",
        "clairvoyant_retired",
    }
    assert all(gate["green"] for gate in gates.values())

    assert gates["advantage"]["advantage_fraction"] >= gates["advantage"]["bar"]
    low, high = gates["execution_premium"]["band"]
    assert low <= gates["execution_premium"]["premium_fraction"] <= high
    assert gates["clairvoyant_retired"]["convexity_holds"] is True
    assert gates["clairvoyant_retired"]["retired"] is True
    assert gates["clairvoyant_retired"]["clairvoyant_looseness_multiple"] > 100.0
    assert (
        abs(gates["rho_zero"]["full_quadrature_difference_bps"])
        <= gates["rho_zero"]["bar_bps"]
    )

    assertions = committed["assertions"]
    assert assertions["lambda_bit_identical"]["green"] is True
    assert assertions["lambda_bit_identical"]["bit_identical_to_m4a"] is True
    assert assertions["lambda_bit_identical"]["mismatches"] == []
    assert assertions["lambda_bit_identical"]["fields_compared"] > 300
    assert assertions["decomposition_identity"]["green"] is True

    assert committed["signal"]["invented"] is True, (
        "the results file must record that the signal is Temper's own; "
        "constitution §7's vendored-not-invented cover does not reach it"
    )
    refinement = committed["feasible_refinement"]
    assert refinement["paths"] >= 200_000
    assert abs(refinement["gap_fraction"]) < 0.02, (
        "the DP's own greedy policy is a *real* policy, so at ten times the "
        "reported paths its value has to be indistinguishable from J_DP; a gap "
        "that survives the path count is a bad action map, not sampling error"
    )

    assert committed["selected"]["adaptive"]["certified"] is False
    assert committed["selected"]["adaptive"]["reference_kind"].startswith("converged")
    assert committed["config"]["lambda_risk"] == experiment.lambda_risk
    assert committed["config"]["signal"] == experiment.signal.as_dict()
    assert committed["provenance"]["git_dirty"] is False


def test_the_committed_reference_records_task_ones_four_answers(committed):
    """Task 1, checked in the suite rather than only in a log.

    Convergence with a Richardson residual, sufficiency to float noise, an
    already-landed signal worth nothing, both references carrying their own word,
    and the premium's cross-lambda range stated over the region where it resolves.
    """
    task_one = committed["task_1"]
    assert task_one["green"] is True

    sufficiency = task_one["sufficiency"]
    assert sufficiency["green"] is True
    assert sufficiency["bar_bps"] <= 1e-14, (
        "the sufficiency bar has to stay float noise: a leak is systematic, so a "
        "bar set where a tolerance would sit would hide it"
    )
    for grid in sufficiency["by_grid"].values():
        assert abs(grid["difference_bps"]) <= sufficiency["bar_bps"]
        assert grid["column_spread_bps"] <= sufficiency["bar_bps"]
        assert grid["bins_ahead"] == 1

    timing = task_one["timing"]
    assert timing["green"] is True
    assert timing["bins_ahead"] == 0
    assert timing["rho"] == committed["signal"]["rho"]
    assert abs(timing["alpha_bps"]) < 1e-12
    assert abs(timing["collapsed_as_fraction_of_advantage"]) < 1e-3, (
        "a signal pointed at a shock that has already landed must be worth "
        "nothing; anything visible here is an off-by-one in the seam's timing"
    )

    kinds = task_one["reference_kinds"]
    assert set(kinds) == {"execution_floor", "adaptive_optimum"}
    assert kinds["execution_floor"]["certified"] is True
    assert kinds["execution_floor"]["kind"] == "certified"
    assert kinds["adaptive_optimum"]["certified"] is False
    assert kinds["adaptive_optimum"]["kind"] == "converged"
    assert kinds["execution_floor"]["value_bps"] < kinds["adaptive_optimum"]["value_bps"]
    for kind in kinds.values():
        assert kind["role"] and kind["evidence"]

    stability = task_one["premium_stability"]
    assert stability["green"] is True
    low, high = stability["resolved_range"]
    assert high - low <= stability["resolved_span_bar"]
    assert stability["resolved_count"] >= 14
    # The three that do not resolve are the degenerate end of the grid, and the
    # measurement says so rather than the prose: escalating the worst shows the
    # fine grid had not converged there either.
    escalation = stability["worst_lambda_escalation"]
    assert len(escalation) >= 4
    coarsest = escalation[min(escalation, key=int)]
    finest = escalation[max(escalation, key=int)]
    assert coarsest - finest > 0.20, (
        "the worst lambda is supposed to be visibly unconverged; if it is not, "
        "the range can be stated over the whole grid and the brief's session note "
        "should say so"
    )
    assert low <= finest <= high or finest < high, (
        "the unresolved lambdas converge *into* the resolved range, which is what "
        "makes the restriction a resolution limit rather than a different regime"
    )

    # The premium at the selected lambda is grid-stable in its own right.
    by_grid = committed["convergence"]["premium_fraction_by_grid"]
    assert committed["convergence"]["premium_fraction_grid_span"] < 0.002
    assert max(by_grid.values()) - min(by_grid.values()) < 0.002


def test_the_committed_fixed_rungs_regenerate_from_the_config(committed, experiment):
    """Invariant 1 on the half of the table that has closed forms: exact, not close."""
    market, order_size = experiment.case.market, experiment.case.order_size
    lambda_risk = experiment.lambda_risk
    row = signal_static_row(market, order_size, lambda_risk, experiment.signal)
    for name, schedule in row.schedules.items():
        assert schedule.objective == pytest.approx(
            committed["selected"]["schedules"][name]["objective_bps"], rel=1e-12
        )
    assert committed["selected"]["execution_floor_bps"] == pytest.approx(
        execution_floor_bps(market, order_size, lambda_risk), rel=1e-12
    )
    assert committed["alpha_coefficient_bps"] == pytest.approx(
        alpha_coefficient(market), rel=1e-12
    )


def test_the_brief_and_the_box_agree_on_every_predicted_number(committed):
    """The brief's numbers were predictions on unpinned numpy; these are artefacts.

    A material disagreement would mean the brief is wrong before the code is, and
    the instruction is to stop and report rather than to adjust the brief to fit.
    They agree, so this pins the agreement: tight where the predicted quantity is a
    closed form or a converged dynamic program, generous where it is itself a
    Monte-Carlo estimate over a few hundred paths.
    """
    predicted = committed["predicted_by_brief"]
    selected = committed["selected"]
    gates = committed["gates"]

    assert selected["schedules"]["optimal"]["objective_bps"] == pytest.approx(
        predicted["j_m4a"], abs=1e-5
    )
    assert selected["adaptive_bps"] == pytest.approx(predicted["j_dp"], abs=5e-5)
    assert selected["signal_advantage_bps"] == pytest.approx(
        predicted["advantage"], rel=1e-3
    )
    assert selected["alpha_available_bps"] == pytest.approx(
        predicted["alpha_available"], rel=1e-3
    )
    assert selected["execution_premium_bps"] == pytest.approx(
        predicted["execution_premium"], rel=2e-3
    )
    assert selected["premium_fraction"] == pytest.approx(
        predicted["premium_fraction"], abs=0.01
    )
    assert selected["execution_floor_bps"] == pytest.approx(
        predicted["execution_floor"], abs=1e-5
    )
    assert committed["alpha_coefficient_bps"] == pytest.approx(
        predicted["alpha_coefficient"], abs=1e-3
    )
    # The clairvoyant number is a 400-path Monte-Carlo estimate with a half-width
    # of ~9 bps in both the brief and the artefact, so agreement is asserted on
    # the *conclusion* rather than on the mean: three orders of looseness.
    assert gates["clairvoyant_retired"]["clairvoyant_looseness_multiple"] == (
        pytest.approx(predicted["clairvoyant_multiple"], rel=0.25)
    )

    for row in committed["value_of_signal"]:
        assert row["advantage_bps"] == pytest.approx(
            row["predicted_advantage_bps"], rel=2e-3
        )
