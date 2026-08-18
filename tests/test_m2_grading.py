"""M2 task 6 — the eval harness, the determinism assertion, and the red flag.

The milestone grades the agent by computing its schedule's objective in closed
form rather than by sampling realised costs. That is only legitimate if the
schedule really is open-loop, so the claim is *tested*, not assumed: the same
policy is rolled out on two unrelated shock streams and the trajectories must be
**bitwise** equal. This module also proves that assertion is not vacuous, by
showing it fails for a policy that is not open-loop.

The other half is the red flag. §4 states the optimum over adapted policies is
the deterministic sinh, and M1's certificate established that
``optimal_trajectory`` is that sinh. So an agent scoring *below* it has not won —
something in the metric, the env or the grading path is broken, and the suite has
to say so rather than print a triumph (``ARCHITECTURE.md`` §1.1).

Everything here also runs the baselines through the identical grading path, which
is the cheapest possible check that the path returns the oracle's own numbers
when handed the oracle's own schedules.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from temper.agents import FractionPolicy, baseline, twap_fractions
from temper.env import ExecutionEnv
from temper.eval import CONTEXT, GRADED, LINEAR, POWER_LAW, sample_costs, standardise
from temper.eval.metrics import WorldMismatch, check_grades_world, metrics_for
from temper.eval.grading import (
    EXPECTED_COST,
    OBJECTIVE,
    RED_FLAG_RTOL,
    SHORTFALL_VARIANCE,
    ScheduleNotDeterministic,
    deterministic_schedule,
    grade_policy,
    grade_trajectory,
    summarise,
)
from temper.oracle import optimal_trajectory, schedule_moments, twap_trajectory
from temper.seeding import M2_DIAGNOSTIC_POOL

from .conftest import SEED_ADDRESS_LEDGER, m2_experiment

EXPERIMENT = m2_experiment()
MARKET = EXPERIMENT.case.market
ORDER_SIZE = EXPERIMENT.case.order_size
LAMBDA = EXPERIMENT.lambda_risk
ROOT_SEED = EXPERIMENT.seeds.root_seed
EVAL_POOL = EXPERIMENT.seeds.eval_pool
EVAL_STREAMS = EXPERIMENT.seeds.eval_streams
REFERENCE = EXPERIMENT.reference()

#: Monte-Carlo cross-check size. Small by M1's standards on purpose: this is not
#: re-running the differential, it is checking that the *analytic grading path*
#: lands where sampling the same schedule lands.
CROSS_CHECK_EPISODES = 20_000


def _grade(policy, name: str | None = None):
    return grade_policy(
        policy,
        MARKET,
        ORDER_SIZE,
        REFERENCE,
        root_seed=ROOT_SEED,
        pool=EVAL_POOL,
        streams=EVAL_STREAMS,
        name=name,
    )


class _DriftingPolicy:
    """A policy whose schedule depends on something other than the observation.

    Stands in for the failure the determinism assertion exists to catch — price
    leaking into the observation — without needing to break the env to produce
    it. What matters is that *something* varies across shock streams; the
    assertion cannot tell, and must not care, what.
    """

    name = "drifting"

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, observation) -> float:
        remaining = float(observation[1]) * ORDER_SIZE
        return float(self._rng.uniform(0.0, 1.0)) * remaining


# ---------------------------------------------------------------------------
# Determinism — what makes analytic grading valid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["twap", "ac", "optimal"])
def test_a_deterministic_policy_realises_the_same_schedule_on_both_eval_streams(name):
    """Bitwise, across the two committed eval streams."""
    policy = baseline(name, MARKET, ORDER_SIZE, LAMBDA)
    schedule = deterministic_schedule(
        policy,
        MARKET,
        ORDER_SIZE,
        LAMBDA,
        root_seed=ROOT_SEED,
        pool=EVAL_POOL,
        streams=EVAL_STREAMS,
    )
    assert schedule.shape == (MARKET.n_bins + 1,)
    assert schedule[0] == ORDER_SIZE
    assert schedule[-1] == 0.0


def test_the_agents_own_parameterisation_is_shock_independent_too():
    """The fraction policy is the shape a trained network takes; same guarantee."""
    policy = FractionPolicy(twap_fractions(MARKET.n_bins), ORDER_SIZE, "fractions")
    schedule = deterministic_schedule(
        policy,
        MARKET,
        ORDER_SIZE,
        LAMBDA,
        root_seed=ROOT_SEED,
        pool=EVAL_POOL,
        streams=EVAL_STREAMS,
    )
    assert schedule == pytest.approx(twap_trajectory(MARKET, ORDER_SIZE), rel=1e-12)


def test_the_determinism_assertion_is_not_vacuous():
    """A policy that is not open-loop must fail it, loudly and by name."""
    with pytest.raises(ScheduleNotDeterministic, match="different"):
        deterministic_schedule(
            _DriftingPolicy(seed=7),
            MARKET,
            ORDER_SIZE,
            LAMBDA,
            root_seed=ROOT_SEED,
            pool=EVAL_POOL,
            streams=EVAL_STREAMS,
        )


def test_one_stream_is_refused_because_it_would_prove_nothing():
    with pytest.raises(ValueError, match="at least two"):
        deterministic_schedule(
            baseline("twap", MARKET, ORDER_SIZE, LAMBDA),
            MARKET,
            ORDER_SIZE,
            LAMBDA,
            root_seed=ROOT_SEED,
            pool=EVAL_POOL,
            streams=(0,),
        )


# ---------------------------------------------------------------------------
# The grading path returns the oracle's numbers
# ---------------------------------------------------------------------------


def test_the_optimum_grades_to_exactly_zero_excess():
    """The reference schedule, through the whole rollout-and-grade path."""
    grade = _grade(baseline("optimal", MARKET, ORDER_SIZE, LAMBDA), "optimal")
    assert grade.objective == pytest.approx(REFERENCE.optimal.objective, rel=1e-12)
    assert grade.excess == pytest.approx(0.0, abs=1e-12)
    assert grade.gap_fraction == pytest.approx(0.0, abs=1e-12)
    assert grade.deviation == pytest.approx(0.0, abs=1e-9)
    assert not grade.red_flag


def test_twap_grades_to_exactly_one_gap_fraction():
    """Zero is the optimum and one is TWAP — that is what the unit *is*."""
    grade = _grade(baseline("twap", MARKET, ORDER_SIZE, LAMBDA), "twap")
    assert grade.gap_fraction == pytest.approx(1.0, rel=1e-12)
    assert grade.objective == pytest.approx(REFERENCE.twap.objective, rel=1e-12)
    assert not grade.red_flag


def test_the_vendored_ac_schedule_lands_between_them():
    """It is a different kappa, so it is neither TWAP nor the optimum.

    Worth stating because the whole reason the oracle carries two kappas is that
    grading against the vendored one would let a correctly-trained agent score
    "better than optimal" (``ARCHITECTURE.md`` §9).
    """
    grade = _grade(baseline("ac", MARKET, ORDER_SIZE, LAMBDA), "ac")
    assert 0.0 < grade.gap_fraction < 1.0
    assert not grade.red_flag


def test_grade_policy_and_grade_trajectory_are_the_same_computation():
    policy = baseline("ac", MARKET, ORDER_SIZE, LAMBDA)
    by_policy = _grade(policy, "ac")
    by_trajectory = grade_trajectory(
        by_policy.trajectory, MARKET, ORDER_SIZE, REFERENCE, name="ac"
    )
    assert by_trajectory.objective == by_policy.objective
    assert by_trajectory.gap_fraction == by_policy.gap_fraction
    assert by_trajectory.deviation == by_policy.deviation


# ---------------------------------------------------------------------------
# The red flag
# ---------------------------------------------------------------------------


def _reference_claiming_optimum(objective: float):
    """A reference whose "optimum" is a value we choose — to test the predicate.

    The real optimum cannot be beaten, which is exactly why the red-flag
    machinery cannot be exercised with a real schedule. Moving the *claimed*
    optimum instead tests the arithmetic and the tolerance without pretending an
    impossible agent exists.
    """
    optimal = dataclasses.replace(REFERENCE.optimal, objective=objective)
    return dataclasses.replace(
        REFERENCE, schedules={**REFERENCE.schedules, "optimal": optimal}
    )


def test_the_red_flag_fires_when_a_schedule_scores_below_the_claimed_optimum():
    optimum = optimal_trajectory(MARKET, ORDER_SIZE, LAMBDA)
    true_objective = REFERENCE.optimal.objective

    flagged = grade_trajectory(
        optimum,
        MARKET,
        ORDER_SIZE,
        _reference_claiming_optimum(true_objective * (1.0 + 1e-6)),
        name="impossible",
    )
    assert flagged.red_flag
    assert flagged.excess < 0.0


def test_the_red_flag_tolerates_float_noise_and_nothing_larger():
    """`RED_FLAG_RTOL` is arithmetic slack, not a margin for being slightly better."""
    optimum = optimal_trajectory(MARKET, ORDER_SIZE, LAMBDA)
    true_objective = REFERENCE.optimal.objective

    inside = grade_trajectory(
        optimum,
        MARKET,
        ORDER_SIZE,
        _reference_claiming_optimum(true_objective * (1.0 + 0.5 * RED_FLAG_RTOL)),
        name="rounding",
    )
    assert not inside.red_flag

    outside = grade_trajectory(
        optimum,
        MARKET,
        ORDER_SIZE,
        _reference_claiming_optimum(true_objective * (1.0 + 10.0 * RED_FLAG_RTOL)),
        name="real",
    )
    assert outside.red_flag


def test_the_configs_red_flag_tolerance_is_the_one_the_grader_uses():
    """One number, not two that agree today."""
    assert EXPERIMENT.tolerances.red_flag_rtol == RED_FLAG_RTOL


# ---------------------------------------------------------------------------
# The grade comes out of the registry, so invariant 7's quarantine binds M2
# ---------------------------------------------------------------------------


def test_the_graded_numbers_come_from_the_linear_registry():
    """M2's metric is a *registered* metric, in the world M2's env charges.

    M4a keyed the registries by encoding, so the property this pins is now the
    sharper one: M2's reference is the linearised world, and asking the registry
    for that world's metrics returns three linear ones and nothing else. Grading
    through the registry is what makes the rule apply to M2; a direct call to the
    closed form would leave it a true statement about a module the grading path
    never touches.
    """
    assert REFERENCE.encoding == LINEAR
    metrics = metrics_for(REFERENCE.encoding)
    for name in (OBJECTIVE, EXPECTED_COST, SHORTFALL_VARIANCE):
        assert name in metrics, f"{name!r} is not a graded metric of this world"
        assert metrics[name].encoding == LINEAR
        assert name not in CONTEXT[LINEAR]

    # ...and the vendored power-law charge is still quarantined, not merely
    # absent: M4a gave it a world of its own without letting it grade this one.
    assert CONTEXT[POWER_LAW], "the context registry is empty; the rule proves nothing"
    assert all(
        metric.encoding == POWER_LAW for metric in CONTEXT[POWER_LAW].values()
    )
    with pytest.raises(WorldMismatch):
        check_grades_world(LINEAR, GRADED[POWER_LAW])


def test_the_registry_route_and_the_closed_form_agree():
    """Non-vacuity for the indirection: it is the same number, computed once."""
    trajectory = optimal_trajectory(MARKET, ORDER_SIZE, LAMBDA)
    moments = schedule_moments(trajectory, MARKET, order_size=ORDER_SIZE)
    grade = grade_trajectory(trajectory, MARKET, ORDER_SIZE, REFERENCE, name="optimal")

    assert grade.objective == pytest.approx(moments.objective(LAMBDA), rel=1e-15)
    assert grade.expected == pytest.approx(moments.expected, rel=1e-15)
    assert grade.variance == pytest.approx(moments.variance, rel=1e-15)


def test_a_schedule_that_does_not_start_at_the_parent_size_is_refused():
    """The tangent `eta_tilde` is a property of the *order*, not of the schedule.

    The registry's metrics read the parent size off the trajectory's first point,
    so a trajectory starting anywhere else would be linearised at the wrong
    participation rate and graded against a different functional — quietly, and
    in the agent's favour or against it depending on the direction.
    """
    trajectory = optimal_trajectory(MARKET, ORDER_SIZE, LAMBDA).copy()
    trajectory[0] *= 0.5
    with pytest.raises(ValueError, match="parent order"):
        grade_trajectory(trajectory, MARKET, ORDER_SIZE, REFERENCE, name="wrong size")


# ---------------------------------------------------------------------------
# Analytic grading really is grading the same thing sampling would
# ---------------------------------------------------------------------------


def test_the_analytic_objective_matches_what_sampling_the_same_schedule_gives():
    """The link back to the simulator, at M2's own case and lambda.

    M1's differential already pins this across a 3 x 3 grid at 200 000 episodes
    per cell. What this adds is the specific claim M2 leans on — that the
    *grading path* computes the objective of the schedule the env actually
    realises — on the frontier case, with the shocks drawn from M2's own
    diagnostic pool.

    The band is M1's: the standardised cost is exactly Gaussian under Phase-1
    dynamics, so 4/sqrt(N) is exact rather than asymptotic.
    """
    policy = baseline("optimal", MARKET, ORDER_SIZE, LAMBDA)
    env = ExecutionEnv(
        MARKET,
        ORDER_SIZE,
        LAMBDA,
        root_seed=ROOT_SEED,
        pool=M2_DIAGNOSTIC_POOL,
        stream_index=100,
    )
    sample = sample_costs(env, policy, CROSS_CHECK_EPISODES, require_fixed_schedule=True)
    moments = schedule_moments(sample.trajectory, MARKET, order_size=ORDER_SIZE)

    z = standardise(sample.costs, moments.expected, moments.variance)
    band = 4.0 / np.sqrt(CROSS_CHECK_EPISODES)
    assert abs(float(np.mean(z))) <= band, (
        f"sampled mean cost is {np.mean(z):.4f} sigma from the analytic E[cost] "
        f"(band {band:.4f})"
    )

    graded = grade_trajectory(
        sample.trajectory, MARKET, ORDER_SIZE, REFERENCE, name="optimal"
    )
    assert graded.objective == pytest.approx(
        moments.objective(LAMBDA), rel=1e-12
    )


# ---------------------------------------------------------------------------
# Reporting across seeds, and where the streams came from
# ---------------------------------------------------------------------------


def test_summarise_reports_median_iqr_and_the_worst_seed():
    summary = summarise("gap_fraction", [0.01, 0.02, 0.03, 0.04, 0.20])
    assert summary.median == pytest.approx(0.03)
    assert summary.q1 == pytest.approx(0.02)
    assert summary.q3 == pytest.approx(0.04)
    assert summary.iqr == pytest.approx(0.02)
    assert summary.worst == pytest.approx(0.20)
    with pytest.raises(ValueError):
        summarise("empty", [])


def test_grading_spent_only_eval_and_diagnostic_streams():
    """Invariant 5, for this module, checked from the ledger rather than by eye.

    The session-wide version runs at teardown; this one attributes a violation to
    grading specifically, and — more usefully — asserts the *positive* fact that
    the eval streams opened are exactly the two the config commits.
    """
    mine = [entry for entry in SEED_ADDRESS_LEDGER if entry[0] == "test_m2_grading.py"]
    assert mine, "the ledger saw no env work from this module"

    pools = {pool for _, _, pool, _ in mine}
    assert pools <= {EVAL_POOL, M2_DIAGNOSTIC_POOL}
    assert "train" not in pools, "evaluation must not touch a training stream"

    eval_streams = {index for _, _, pool, index in mine if pool == EVAL_POOL}
    assert eval_streams == set(EVAL_STREAMS), (
        f"graded on eval streams {sorted(eval_streams)}, config commits "
        f"{list(EVAL_STREAMS)}"
    )
    assert {root for _, root, _, _ in mine} == {ROOT_SEED}
