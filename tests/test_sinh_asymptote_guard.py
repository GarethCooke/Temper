"""M1a task 4 — M0's `sinh-overflow-asymptote` watch item, exercised and closed.

M0's session notes left M1 a watch item: above ``kappa * T = SINH_OVERFLOW_KT``
the vendored trajectory is replaced by its ``exp(-kappa t)`` asymptote, whose
terminal holding is ``X * e^-kT`` rather than a hard zero, and *the env's
``x_N = 0`` force-liquidation must not be written to assume otherwise*. M1 shipped
without ever running a schedule through that branch, so the note stayed a note.

This module turns it into three checked statements.

1. **No cell of the 3 x 3 golden grid reaches the branch** — the largest decay on
   the grid is ``kappa * T = 20.6`` under the vendored convention and 9.0 under
   the exact one, against a threshold of 500. So the branch needs a case of its
   own, which is what ``guard_case`` in ``configs/m1_differential.yaml`` is; it
   is a **guard, not a golden**, and pins no FrontierView number.
2. **The guard case reaches it, and the env costs the result correctly.** The raw
   asymptote trajectory goes through the env *including* its residual terminal
   holding — the policy does not repair it — and the realised schedule's cost
   still matches ``schedule_moments`` of that realised schedule.
3. **The watch item cannot manifest at all, and here is why.** The threshold is
   on ``kappa * T``, so on the canonical 13-bin grid taking the branch forces a
   per-bin decay of at most ``e^(-500/13)`` = 2e-17, which is below half an ulp
   at ``X``. The first bin's trade therefore rounds to the whole order and the
   residual is annihilated before the env ever carries it anywhere: no
   parameterisation on this grid can deliver residual inventory by this route.
   That is a stronger closure than "we checked one case", and it is
   :func:`test_the_branch_can_never_deliver_residual_inventory_on_this_grid`.

What the guard still earns, given (3), is not vacuous: it is the one cell where
the schedule the policy *planned* and the schedule the env *realised* are wildly
different — thirteen decaying bins planned, the entire order in bin 0 realised —
so it pins that the differential compares against what was executed. Task 5(d)'s
under-trader exercises the same rule from the opposite side, by leaving a
remainder rather than by front-loading one away.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from temper.agents import SchedulePolicy, baseline
from temper.eval import run_episode
from temper.oracle import (
    SINH_OVERFLOW_KT,
    ac_kappa,
    ac_trajectory,
    optimal_kappa,
    schedule_moments,
)

from .conftest import M1_CONFIG, build_env, case_by_id, guard_case
from .test_env_identities import assert_identity

GUARD_SPEC = M1_CONFIG["guard_case"]
GUARD_STREAM = int(M1_CONFIG["seeding"]["guard_stream"])
COST_RTOL = float(GUARD_SPEC["cost_rtol"])
PENALTY_RTOL = float(M1_CONFIG["identities"]["penalty_rtol"])
GUARD_EPISODES = int(GUARD_SPEC["episodes"])

GRID_CASES = [case_by_id(cid) for cid in M1_CONFIG["tiers"]["deep"]["cases"]]
CANONICAL_BINS = int(M1_CONFIG["canonical_bins"])


def asymptote_policy(case) -> SchedulePolicy:
    """The raw asymptote schedule, residual terminal holding and all.

    Nothing here rounds the tail down to zero or renormalises the trades to sum
    to ``X``. Whether the env charges a schedule that does not fully liquidate
    itself is the question; repairing the schedule first would answer a different
    one.
    """
    trajectory = ac_trajectory(case.market, case.order_size, case.lambda_risk)
    return SchedulePolicy(trajectory, f"{case.case_id}:ac-asymptote")


@pytest.fixture(scope="module")
def guard():
    """The guard case, its raw schedule, and `GUARD_EPISODES` episodes of it."""
    case = guard_case()
    policy = asymptote_policy(case)
    env = build_env(case, GUARD_STREAM)
    env.reset(seed=GUARD_STREAM)
    return case, policy, [run_episode(env, policy) for _ in range(GUARD_EPISODES)]


# ---------------------------------------------------------------------------
# 1. The grid does not reach the branch, and the guard is not a golden
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", GRID_CASES, ids=str)
def test_no_cell_of_the_golden_grid_reaches_the_asymptote_branch(case):
    """The determination the brief asks for, as an assertion rather than a note.

    Both kappa conventions, because both ship as named baselines: `ac` is what the
    goldens pin and `optimal` is what M2 grades against. If a future re-vendor
    ever moves a golden case past the threshold this fails, and the guard case
    below stops being the only cell that exercises the branch — which is
    something the session that re-vendors needs told, not something to discover
    from a moment mismatch.
    """
    horizon = case.market.horizon_hours
    for name, kappa in (
        ("ac", ac_kappa(case.market, case.order_size, case.lambda_risk)),
        ("optimal", optimal_kappa(case.market, case.order_size, case.lambda_risk)),
    ):
        assert kappa * horizon <= SINH_OVERFLOW_KT, (
            f"{case.case_id}: {name} kappa*T = {kappa * horizon:.1f} is past the "
            f"{SINH_OVERFLOW_KT:g} asymptote threshold; this cell is now on the "
            "guarded branch and the differential is measuring something else"
        )

    for schedule in M1_CONFIG["schedules"]:
        policy = baseline(schedule, case.market, case.order_size, case.lambda_risk)
        assert policy.trajectory[-1] == 0.0, (
            f"{case.case_id}: the {schedule} schedule does not end flat"
        )


def test_the_guard_case_is_labelled_a_guard_and_pins_no_vendored_number():
    """It exercises a branch; it is not evidence about FrontierView.

    A guard presented as a golden would be the worst of both: a number nobody
    exported claiming vendored provenance. The config labels it, `guard_case()`
    refuses to build it if that label is wrong, and the object it returns carries
    no `ac`, `twap` or `derived` block at all — so there is nothing on it a test
    could accidentally compare against.
    """
    case = guard_case()
    assert case.kind == "guard"
    assert GUARD_SPEC["kind"] != "golden"
    for vendored in ("ac", "twap", "derived"):
        assert not hasattr(case, vendored), (
            f"the guard case carries a {vendored!r} block; it must not look like a golden"
        )
    # Its lambda is this repo's choice, not an exported one.
    source = case_by_id(GUARD_SPEC["params_from"])
    assert case.lambda_risk != source.lambda_risk
    assert case.market is source.market  # parameters and grid, though, are vendored


def test_the_guard_case_reaches_the_branch_with_a_representable_residual(guard):
    """Past the threshold, and the residual is a positive double — both needed.

    Past the threshold or the guard guards nothing. Representable because the
    vendored `sinh-overflow-asymptote` case is *not* usable here: at its
    lambda = 100 the terminal holding underflows to exactly 0.0, so it takes the
    branch without ever producing the residual the watch item is about. The
    config's lambda = 1.0 puts kappa*T at ~650 — past 500, under the ~709 where
    e^-kT stops being representable.
    """
    case, policy, _ = guard
    kappa = ac_kappa(case.market, case.order_size, case.lambda_risk)
    kappa_horizon = kappa * case.market.horizon_hours
    assert kappa_horizon > SINH_OVERFLOW_KT, (
        f"the guard case sits at kappa*T = {kappa_horizon:.1f}, below the "
        f"{SINH_OVERFLOW_KT:g} threshold: it does not take the branch it exists for"
    )
    residual = float(policy.trajectory[-1])
    assert residual > 0.0, (
        "the guard case's terminal holding underflowed to zero; it no longer "
        "carries the residual the watch item is about"
    )
    assert residual / case.order_size < 1e-200, (
        "the residual is far larger than e^-500 of X; the branch condition and the "
        "trajectory no longer agree about what the asymptote is"
    )


# ---------------------------------------------------------------------------
# 2. The env costs the realised schedule correctly
# ---------------------------------------------------------------------------


def test_the_env_charges_the_schedule_it_actually_realised(guard):
    """Task 4's requirement: cost vs `schedule_moments` of the *realised* schedule.

    The planned schedule is thirteen bins of exponentially decaying trades. The
    realised schedule is the whole order in bin 0, because the second level is
    2e-17 of X and vanishes into the first subtraction. The oracle is asked about
    what happened, not about what was asked for, and the per-episode noise-removal
    identity has to hold against that.
    """
    case, policy, results = guard
    for index, result in enumerate(results):
        weights = result.shares / case.order_size
        noise = float(np.sum(weights * result.walks))
        moments = schedule_moments(result.trajectory, case.market)
        assert_identity(
            result.cost_bps + noise,
            moments.expected,
            scale=abs(result.cost_bps) + abs(noise) + abs(moments.expected),
            rtol=COST_RTOL,
            what=f"guard episode {index}: realised cost with the price path removed "
            "vs the oracle's E[cost] for the realised schedule",
        )
        assert result.penalty_bps == pytest.approx(
            case.lambda_risk * moments.variance, rel=PENALTY_RTOL
        )
        assert result.trajectory[-1] == 0.0, "the guard episode did not finish flat"
        assert result.shares.sum() == pytest.approx(case.order_size, rel=1e-15)

    # ...and the realised schedule really is not the planned one, or the above
    # would be the same statement the 27 differential cells already make.
    realised = results[0].trajectory
    assert not np.array_equal(realised, policy.trajectory)
    assert realised[1] == 0.0 and policy.trajectory[1] > 0.0


# ---------------------------------------------------------------------------
# 3. Why the watch item can never manifest
# ---------------------------------------------------------------------------


def test_the_branch_can_never_deliver_residual_inventory_on_this_grid():
    """The closure: arithmetic, not a case check.

    Taking the branch requires ``kappa * T > SINH_OVERFLOW_KT``, so on an
    ``N``-bin grid the per-bin decay is at most ``e^(-SINH_OVERFLOW_KT/N)``. At
    the canonical ``N = 13`` that is 2.0e-17 — below half an ulp at ``X``. So the
    first bin's planned trade, ``X - x_1``, rounds to exactly ``X``; the env
    executes the whole order in bin 0 and holds exactly zero thereafter. The
    terminal residual ``X * e^-kT`` — some 200 further orders of magnitude down —
    can never reach the force-liquidation the watch item was written about.

    M0's caution was still right: the env must not *assume* a hard-zero tail, and
    `SchedulePolicy` must not repair one. Both hold, and the tests above run the
    branch through anyway. But the note can be closed rather than carried into M2.
    """
    largest_survivable_decay = math.exp(-SINH_OVERFLOW_KT / CANONICAL_BINS)
    half_ulp = np.finfo(np.float64).eps / 2.0
    assert largest_survivable_decay < half_ulp, (
        f"e^(-{SINH_OVERFLOW_KT:g}/{CANONICAL_BINS}) = {largest_survivable_decay:.2e} is "
        f"no longer below half an ulp ({half_ulp:.2e}); the asymptote branch can now "
        "put residual inventory into the env and the watch item is live again"
    )

    case = guard_case()
    trades = -np.diff(asymptote_policy(case).trajectory)
    assert case.order_size - float(trades.sum()) == 0.0, (
        "the planned trades no longer sum to X in float64, so the residual now "
        "survives into the env — re-open the watch item"
    )
