"""M4a task 4 — the four guarantees the power-law world inherits, checked first.

The brief predicts all four green, and predicts it from a mechanism rather than
from optimism. ``ARCHITECTURE.md`` §9's antithetic entry says the pairing's
exactness "does not exist in Phase 2, where cost stops being affine in the
shocks". That sentence is about a Phase 2 that had not been built, and it is
**wrong about this half of it**: the power law replaces the *temporary* term,
which is a function of the schedule and carries no shock at all. Realised cost
stays

.. code::

    C = f(x)  -  sigma_bin * sum_{k=0}^{N-1} (x_k / X) xi_k

with only ``f`` changed. So all four should survive verbatim:

============================================  ==========================================
Guarantee                                     What red would mean
============================================  ==========================================
Exact per-episode noise identity              cost stopped being affine in the shocks
Antithetic cancellation is exact              the pairing degraded a milestone early
Action identity across the pair               something reached the observation
The schedule is open-loop                     analytic grading is invalid
============================================  ==========================================

Every one of them is a test that already exists for Phase 1, so this module is
minutes of re-running rather than a night of reasoning. It runs **before** the
training point, and if any goes red the milestone's product is that finding and
training does not start.

The corollary is what M4b inherits: what actually ends the pairing's exactness is
a second, *independent* noise source or a price-bearing observation — not
curvature in the cost.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from temper.agents import baseline, execution_env_factory, twap_fractions
from temper.env import EPISODE_KEY, SHOCK_KEY, ExecutionEnv, impact_for
from temper.eval import run_episode
from temper.eval.antithetic import PairLedger, antithetic_reward, mirror_of
from temper.eval.grading import ScheduleNotDeterministic, deterministic_schedule
from temper.eval.variate import deterministic_reward
from temper.oracle import (
    BPS,
    TRADING_HOURS_PER_DAY,
    cost_moments,
    optimal_trajectory,
    power_law_optimum,
    schedule_moments,
)
from temper.seeding import pool_rng

from .conftest import M4A_CONFIG, M4A_ENCODING, build_power_law_env, case_by_id

GUARANTEES = M4A_CONFIG["inherited_guarantees"]
CASES = [case_by_id(case_id) for case_id in GUARANTEES["cases"]]
SCHEDULES = list(M4A_CONFIG["schedules"])
NOISE_RTOL = float(GUARANTEES["noise_rtol"])
CANCELLATION_ATOL = float(GUARANTEES["cancellation_atol"])
EPISODES = int(GUARANTEES["episodes"])
ROOT_SEED = int(M4A_CONFIG["seeding"]["root_seed"])
POOL = str(M4A_CONFIG["seeding"]["pool"])
GUARANTEE_STREAM = int(M4A_CONFIG["seeding"]["guarantee_stream"])

#: Sell side: a positive shock raises the price, a seller does better, the
#: shortfall against arrival falls. Stated, not inferred from the env.
SELL_SIDE_SIGN = -1.0

#: What each guarantee actually observed, so task 4 reports itself rather than
#: only passing — the brief asks for the results to be *recorded* before task 5.
_OBSERVED: dict[str, float] = {}


@pytest.fixture(scope="module", autouse=True)
def report_guarantees(request):
    """Print the four guarantees with the worst number each saw."""
    yield
    if not _OBSERVED:
        return
    writer = request.config.get_terminal_writer()
    writer.line("")
    writer.line("M4a task 4 — the inherited guarantees, in the power-law world:")
    for name, worst in _OBSERVED.items():
        writer.line(f"  {name:34s} green, worst {worst:.3e}")


def _cells():
    """(case, schedule) cells, each on its own stream of M4a's pool."""
    stream = GUARANTEE_STREAM
    cells = []
    for case in CASES:
        for schedule in SCHEDULES:
            cells.append((f"{case.case_id}:{schedule}", case, schedule, stream))
            stream += 1
    return cells


CELLS = _cells()
CELL_IDS = [name for name, _, _, _ in CELLS]


@pytest.fixture(params=CELLS, ids=CELL_IDS)
def cell(request):
    return request.param


def _policy(case, schedule: str):
    return baseline(
        schedule, case.market, case.order_size, case.lambda_risk, encoding=M4A_ENCODING
    )


def _shock_scale_bps(case) -> float:
    """``sigma_bin`` in bps, rebuilt from the raw parameters and the grid.

    Deliberately not ``case.market.sigma_bin``: the functional below is assembled
    from the units contract rather than borrowed from something that already
    knows the answer, exactly as ``tests/test_noise_identity.py`` does it.
    """
    market = case.market
    tau = market.horizon_hours / market.n_bins
    return market.params.sigma * math.sqrt(tau / TRADING_HOURS_PER_DAY) * BPS


def _episode_draws(stream_index: int, n_bins: int, n_episodes: int) -> np.ndarray:
    """The draws the env must have made, regenerated from the seed address alone."""
    generator = pool_rng(ROOT_SEED, POOL, stream_index)
    return generator.standard_normal(n_episodes * n_bins).reshape(n_episodes, n_bins)


# ---------------------------------------------------------------------------
# Guarantee 1 — the exact per-episode noise identity
# ---------------------------------------------------------------------------


def test_the_exact_noise_identity_survives_the_power_law(cell):
    """``C - E[cost] == noise(xi)``, every episode, to round-off — in the new world.

    The right-hand side is *identical* to M1a's: the same
    ``-sigma_bin * sum_k (x_k / X) xi_k``, assembled here from sigma, tau, the
    realised schedule and the draws the seed address resolves to, with nothing
    read out of the env. Only the left-hand side's ``E[cost]`` moved, from
    ``schedule_moments`` to ``cost_moments``.

    That is the whole prediction, and green here is what makes it a measurement:
    the power law changed a term that carries no shock, so the noise functional
    could not have moved. Red would mean realised cost had stopped being affine
    in the price shocks — a genuine surprise, and one that would change M4b's
    design rather than being worked around.
    """
    name, case, schedule, stream = cell
    env = build_power_law_env(case, stream)
    env.reset(seed=stream)
    results = [run_episode(env, _policy(case, schedule)) for _ in range(EPISODES)]
    draws = _episode_draws(stream, case.market.n_bins, EPISODES)

    worst = 0.0
    for index, result in enumerate(results):
        holdings_before_bin = np.asarray(result.trajectory, dtype=np.float64)[:-1]
        terms = (
            SELL_SIDE_SIGN
            * _shock_scale_bps(case)
            * (holdings_before_bin / case.order_size)
            * draws[index]
        )
        expected = cost_moments(result.trajectory, case.market).expected
        residual = result.cost_bps - expected
        scale = (
            abs(result.cost_bps) + abs(expected) + float(np.sum(np.abs(terms)))
        )
        error = abs(residual - float(np.sum(terms))) / scale
        assert error <= NOISE_RTOL, (
            f"{name} episode {index}: realised cost less E[cost] is "
            f"{residual!r}, the noise functional is {float(np.sum(terms))!r}; "
            f"relative error {error:.3e} exceeds {NOISE_RTOL:g}. Cost has stopped "
            "being affine in the shocks."
        )
        worst = max(worst, error)
    _OBSERVED["noise identity"] = max(_OBSERVED.get("noise identity", 0.0), worst)


def test_the_linear_reference_would_break_the_identity(cell):
    """Non-vacuity: the identity is holding against *this* world's E[cost].

    Subtract Phase 1's expectation instead and the residual must miss by orders
    of magnitude. Without this the test above would pass on any env whose noise
    happened to be right, including one still charging the tangent.
    """
    name, case, schedule, stream = cell
    env = build_power_law_env(case, stream + 500)
    env.reset(seed=stream + 500)
    result = run_episode(env, _policy(case, schedule))
    draws = _episode_draws(stream + 500, case.market.n_bins, 1)[0]

    holdings_before_bin = np.asarray(result.trajectory, dtype=np.float64)[:-1]
    terms = (
        SELL_SIDE_SIGN
        * _shock_scale_bps(case)
        * (holdings_before_bin / case.order_size)
        * draws
    )
    wrong = schedule_moments(
        result.trajectory, case.market, order_size=case.order_size
    ).expected
    scale = abs(result.cost_bps) + abs(wrong) + float(np.sum(np.abs(terms)))
    error = abs(result.cost_bps - wrong - float(np.sum(terms))) / scale
    assert error > 1e3 * NOISE_RTOL, (
        f"{name}: the noise identity also holds against the *linear* E[cost] "
        f"(relative error {error:.3e}); it is not testing the world it is in"
    )


# ---------------------------------------------------------------------------
# Guarantees 2 and 3 — the antithetic pair
# ---------------------------------------------------------------------------


def _paired_env(case, stream: int, ledger: PairLedger):
    """One antithetic-wrapped power-law env, built the way the sweep builds it."""
    factory = execution_env_factory(
        case.market,
        case.order_size,
        case.lambda_risk,
        root_seed=ROOT_SEED,
        pool=POOL,
        stream_index=stream,
        reward_wrapper=antithetic_reward(ledger),
        temporary_impact=impact_for(M4A_ENCODING, case.market, case.order_size),
    )
    return factory()


def test_the_antithetic_pair_still_cancels_the_noise_exactly(cell):
    """The averaged reward is the deterministic reward, to floating-point dust.

    ``(xi, -xi)`` averaged kills a term that is *linear* in the shocks. The power
    law is not linear in anything — but it is not a function of the shocks
    either, so it contributes identically to both halves and averages to itself.
    Exactness therefore survives, which is the narrower true version of §9's "does
    not exist in Phase 2".

    Checked against the control variate rather than against zero: the averaged
    reward must equal the reward with M1a's analytic noise form subtracted, which
    is the same statement and is the one M3's gate was written in.
    """
    name, case, schedule, stream = cell
    market, order_size, lam = case.market, case.order_size, case.lambda_risk
    fractions = _fractions_for(case, schedule)

    ledger = PairLedger()
    paired = _paired_env(case, stream + 1500, ledger)
    variate = deterministic_reward(
        ExecutionEnv(
            market,
            order_size,
            lam,
            temporary_impact=impact_for(M4A_ENCODING, market, order_size),
            root_seed=ROOT_SEED,
            pool=POOL,
            stream_index=stream + 1500,
        )
    )

    paired_rewards, paired_trajectory = _run_fractions(paired, fractions, order_size)
    variate_rewards, variate_trajectory = _run_fractions(
        variate, fractions, order_size
    )
    assert np.array_equal(paired_trajectory, variate_trajectory)

    worst = float(np.max(np.abs(paired_rewards - variate_rewards)))
    assert worst <= CANCELLATION_ATOL, (
        f"{name}: the antithetic average and the control variate differ by "
        f"{worst:.3e} bps per step (band {CANCELLATION_ATOL:g}); the pairing's "
        "cancellation is no longer exact in this world"
    )

    # And the sum is the frozen objective, which is invariant 7 restated for this
    # estimator — in the world that now charges it.
    moments = cost_moments(paired_trajectory, market)
    assert float(np.sum(paired_rewards)) == pytest.approx(
        -moments.objective(lam), rel=1e-11
    )
    _OBSERVED["antithetic cancellation"] = max(
        _OBSERVED.get("antithetic cancellation", 0.0), worst
    )


def test_the_action_identity_across_the_pair_is_still_green(cell):
    """Both halves take bitwise-identical actions, because neither can see price.

    The pair wrapper asserts this per step and raises
    :class:`~temper.eval.antithetic.PairDiverged` if it ever fails. M3's §9 entry
    predicted this would go red "when Phase 2 enriches the observation" — M4a
    does not enrich it, so it stays green, and that is the distinction the entry
    needs narrowed to: the break is a price-bearing observation, not a curved
    cost.
    """
    name, case, schedule, stream = cell
    ledger = PairLedger()
    paired = _paired_env(case, stream + 2000, ledger)
    fractions = _fractions_for(case, schedule)

    paired.reset()
    pair = paired.unwrapped if hasattr(paired, "unwrapped") else paired
    for fraction in fractions:
        paired.step(np.array([fraction]))  # PairDiverged if the halves disagree
    assert pair is not None

    # The mirror really is stepping: its shock is the exact negation of the
    # primary's, so the check above is not passing on an unused branch.
    primary = ExecutionEnv(
        case.market,
        case.order_size,
        case.lambda_risk,
        temporary_impact=impact_for(M4A_ENCODING, case.market, case.order_size),
        root_seed=ROOT_SEED,
        pool=POOL,
        stream_index=stream + 2100,
    )
    mirror = mirror_of(primary)
    primary.reset()
    mirror.reset()
    trade = case.order_size / case.market.n_bins
    for _ in range(case.market.n_bins):
        _, _, _, _, info_primary = primary.step(trade)
        _, _, _, _, info_mirror = mirror.step(trade)
        assert info_mirror[SHOCK_KEY] == -info_primary[SHOCK_KEY]
    _OBSERVED["action identity"] = 0.0


def _fractions_for(case, schedule: str) -> np.ndarray:
    """The fraction-of-remaining sequence a named schedule induces."""
    if schedule == "twap":
        return twap_fractions(case.market.n_bins)
    trajectory = _policy(case, schedule).trajectory
    return (trajectory[:-1] - trajectory[1:]) / trajectory[:-1]


def _run_fractions(env, fractions, order_size: float):
    """Step a fraction-wrapped env through one episode; rewards and schedule."""
    from temper.agents import FractionAction

    wrapped = env if isinstance(env, FractionAction) else FractionAction(env)
    wrapped.reset()
    rewards: list[float] = []
    info: dict = {}
    for fraction in fractions:
        _, reward, _, _, info = wrapped.step(np.array([fraction]))
        rewards.append(float(reward))
    return np.array(rewards), info[EPISODE_KEY]["trajectory"]


# ---------------------------------------------------------------------------
# Guarantee 4 — the schedule is still open-loop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_the_schedule_is_still_open_loop_in_the_power_law_world(schedule):
    """Two unrelated eval streams, bitwise-identical trajectories.

    Without this M4a's analytic grading is invalid and nothing downstream of it
    is a number. The observation is untouched by this milestone — that is the
    whole reason M4 was split — so the check should pass, and it is run rather
    than reasoned about because it is the cheapest of the four.
    """
    case = CASES[0]
    trajectory = deterministic_schedule(
        _policy(case, schedule),
        case.market,
        case.order_size,
        case.lambda_risk,
        root_seed=ROOT_SEED,
        pool="eval",
        streams=(0, 1),
        temporary_impact=impact_for(M4A_ENCODING, case.market, case.order_size),
        expect_encoding=M4A_ENCODING,
    )
    assert trajectory[0] == case.order_size
    assert trajectory[-1] == 0.0
    _OBSERVED["open-loop schedule"] = 0.0


def test_the_open_loop_check_still_refuses_a_policy_that_is_not():
    """Non-vacuity, in the new world: a shock-dependent policy must be caught."""

    class Drifting:
        name = "drifting"

        def __init__(self, order_size: float) -> None:
            self._rng = np.random.default_rng(11)
            self.order_size = order_size

        def reset(self) -> None:
            pass

        def act(self, observation) -> float:
            return float(self._rng.uniform()) * float(observation[1]) * self.order_size

    case = CASES[0]
    with pytest.raises(ScheduleNotDeterministic):
        deterministic_schedule(
            Drifting(case.order_size),
            case.market,
            case.order_size,
            case.lambda_risk,
            root_seed=ROOT_SEED,
            pool="eval",
            streams=(0, 1),
            temporary_impact=impact_for(M4A_ENCODING, case.market, case.order_size),
            expect_encoding=M4A_ENCODING,
        )


def test_the_grader_refuses_an_env_from_the_other_world():
    """And the world check itself has teeth.

    ``expect_encoding`` is what stops a power-law reference being fed a schedule
    a Phase-1 env produced. Asserting it fires is what makes it a check rather
    than a parameter nobody passes.
    """
    from temper.eval.metrics import WorldMismatch

    case = CASES[0]
    with pytest.raises(WorldMismatch, match="one world"):
        deterministic_schedule(
            _policy(case, "twap"),
            case.market,
            case.order_size,
            case.lambda_risk,
            root_seed=ROOT_SEED,
            pool="eval",
            streams=(0, 1),
            temporary_impact=None,  # Phase 1
            expect_encoding=M4A_ENCODING,
        )


def test_the_optima_of_the_two_worlds_are_different_schedules():
    """A guard on the whole module: if they agreed there would be no milestone."""
    case = CASES[0]
    power = power_law_optimum(case.market, case.order_size, case.lambda_risk)
    tangent = optimal_trajectory(case.market, case.order_size, case.lambda_risk)
    assert not np.allclose(power, tangent, rtol=1e-6, atol=0.0)
