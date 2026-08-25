"""M5 task 3 — the guard moves, and every case it must still refuse.

The observation-minimality guard has refused a price-bearing observation since
M1a. M5's observation *is* price-bearing, so the line moves for the first time,
and this module is what makes the new line a measurement rather than a sentence in
a docstring. ``tests/observation_guard.py`` states it; this exercises it, on cases
that must pass and cases that must fail.

The three refusals, and why the third is the one worth building
----------------------------------------------------------------
1. **The realised price.** An env whose observation carries the walk. Refused by
   the pinned-signal clause, which is M1a's clause with one axis pinned.
2. **The realised shortfall.** The same clause, because a realised cost is a
   function of the walk. Kept as its own case because "no realised price" and "no
   realised cost" are two sentences a future session could satisfy one of.
3. **A signal about an already-committed shock.** This is the one that has no
   precedent and no natural test. Nothing about it varies with the price stream at
   a pinned signal — it passes clause 1 cleanly — and it is worth exactly zero,
   so no result would look wrong if the seam were built this way. It is refused
   only because the guard measures *which* shocks the observation is correlated
   with, and requires them all to be ones the current decision can still act on.

Case 3 is built out of M5's own timing instrument rather than invented here:
``OneStepSignal(rho, bins_ahead=0)`` is the law task 1 used to prove the advantage
collapses when the signal points one bin short. Pointing the guard at it took one
step more than expected, and the step is worth recording. ``ExecutionEnv`` declines
to wire a lag-0 signal at all — the observation stays two coordinates wide and the
composition gains stay ``(0, 1)`` — so an env holding that law has *nothing* for
the guard to refuse, and publishing the coordinate by hand only shows a draw that
predicts nothing.

The defect had to be modelled where it would actually be written:
:class:`NaivelyInformative` declares ``informative`` on ``rho`` alone, dropping the
condition on the lag. That is one clause of one property, it is the obvious way to
write it, and the real env then does the rest itself — publishes ``s_k``, wires the
predictor to the bin's own draw, and composes ``xi_k = rho s_k + ...``. No env
subclass is involved in case 3, which is what makes it a test of the seam rather
than of a fixture. The same wrapper round the model's lag is permitted, and that
control is below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from temper.env import NO_SIGNAL_STREAM, ExecutionEnv, SignalStream, signal_stream
from temper.oracle import (
    AlphaSignal,
    Market,
    OneStepSignal,
    SymbolParams,
    twap_trajectory,
)
from temper.seeding import M5_DIFFERENTIAL_POOL, SIGNAL_TRAIN_POOL

from .observation_guard import (
    BASE_COORDINATES,
    MinimalityVerdict,
    observation_minimality,
)

#: Big enough that a correlation the guard must catch is unmistakable and a
#: correlation it must ignore is not. The milestone trains at 0.01; a seam defect
#: is a defect at any rho, and testing the *guard* at a rho it can resolve is the
#: difference between a check and a coin toss.
GUARD_RHO = 0.4

#: A shorter forward-only pass than the guard's default, because this module runs
#: it several times and the effect being resolved is 0.4 rather than 1e-2. At
#: 6 000 episodes the sampling standard deviation is 0.013 and the bar below sits
#: at four of them.
GUARD_EPISODES = 6_000
GUARD_TOLERANCE = 0.055


def _market() -> Market:
    return Market(
        params=SymbolParams(
            adv=6e7, sigma=0.0155, half_spread=0.3, eta=0.142, gamma=0.314
        ),
        horizon_hours=6.5,
        n_bins=13,
    )


def _schedule(market: Market) -> np.ndarray:
    return -np.diff(twap_trajectory(market, 100_000.0))


@dataclass(frozen=True)
class NaivelyInformative(AlphaSignal):
    """A signal that calls itself informative on ``rho`` alone, ignoring the lag.

    **No env subclass, and that is the point.** ``ExecutionEnv`` already supports a
    lag-0 signal end to end — it publishes the coordinate, wires the predictor to
    the bin's *own* draw and sets the composition gains — and the only thing
    stopping it is that :class:`~temper.oracle.signal.OneStepSignal` declares
    ``informative`` false at lag 0. So the defect this models is not an exotic
    subclass: it is one condition dropped from one property, which is exactly how
    a real session would write it, and the real env then does everything else.

    Wrapped round the model's lag it reproduces ``ExecutionEnv`` exactly, which is
    the control that keeps the wrapper itself from being what the guard refuses.
    """

    base: AlphaSignal

    @property
    def name(self) -> str:
        return f"naive:{self.base.name}"

    @property
    def informative(self) -> bool:
        """``rho != 0``, and nothing about *which* shock it predicts."""
        return self.base.correlation() != 0.0

    @property
    def lag(self) -> int:
        return self.base.lag

    def mean(self) -> float:
        return self.base.mean()

    def variance(self) -> float:
        return self.base.variance()

    def correlation(self) -> float:
        return self.base.correlation()

    def quadrature(self, nodes: int):
        return self.base.quadrature(nodes)

    def draw(self, rng, size):
        return self.base.draw(rng, size)

    def as_dict(self) -> dict:
        return self.base.as_dict() | {"naive": True}


class ShowsTheWalk(ExecutionEnv):
    """The leak M1a's guard was written for: the realised price, in the observation."""

    def _observation(self) -> np.ndarray:
        return np.append(super()._observation(), self._walk)


class ShowsTheShortfall(ExecutionEnv):
    """The other leak: a realised *cost*, which is a function of the walk."""

    def _observation(self) -> np.ndarray:
        return np.append(super()._observation(), self._shortfall_total)


def _factory(env_class, signal, market: Market):
    """A ``factory(price_stream, pinned_signal)`` the guard can drive."""

    base = NO_SIGNAL_STREAM if signal is None else signal

    def build(price_stream: int, pinned_signal: int | None) -> ExecutionEnv:
        stream = base if pinned_signal is None else base.pinned_to(pinned_signal)
        return env_class(
            market,
            100_000.0,
            1e-4,
            signal=stream,
            root_seed=20260825,
            pool=M5_DIFFERENTIAL_POOL,
            stream_index=price_stream,
        )

    return build


def _run(env_class, signal) -> MinimalityVerdict:
    market = _market()
    return observation_minimality(
        _factory(env_class, signal, market),
        _schedule(market),
        episodes=GUARD_EPISODES,
        tolerance=GUARD_TOLERANCE,
    )


# ---------------------------------------------------------------------------
# What the amendment permits
# ---------------------------------------------------------------------------


def test_the_amended_guard_permits_the_milestone_and_says_what_it_measured():
    """M5's own env, through the guard, with the numbers behind the verdict.

    Both clauses have to be green for the right reasons: price-independent at a
    pinned signal path, no correlation with any committed shock, and a *strong*
    correlation with one the decision can still act on. That last one is what
    stops a green verdict from being a guard that is looking at nothing.
    """
    verdict = _run(
        ExecutionEnv, signal_stream(OneStepSignal(GUARD_RHO), SIGNAL_TRAIN_POOL)
    )
    assert verdict.permitted, verdict.reason
    assert verdict.price_independent
    assert verdict.forward_only
    assert verdict.seam_coordinates == (BASE_COORDINATES,)
    assert verdict.worst_committed_correlation <= GUARD_TOLERANCE
    assert verdict.strongest_actionable_correlation == pytest.approx(
        GUARD_RHO, abs=0.05
    ), (
        "the guard found no forward correlation either, so it would pass an env "
        "whose signal coordinate was a constant"
    )


def test_the_amended_guard_permits_the_worlds_that_predate_it():
    """M0 through M4b, unchanged: two coordinates and nothing to measure."""
    verdict = _run(ExecutionEnv, None)
    assert verdict.permitted, verdict.reason
    assert verdict.price_independent
    assert verdict.seam_coordinates == ()
    assert "carries no seam coordinate" in verdict.reason


def test_the_naive_wrapper_is_not_what_is_being_refused():
    """The control for the negative case below.

    :class:`NaivelyInformative` round the *model's* lag builds an env that is
    ``ExecutionEnv`` in every respect, and is permitted. So when the same wrapper
    round lag 0 is refused, what was refused is the timing and not the wrapper.
    """
    verdict = _run(
        ExecutionEnv,
        signal_stream(NaivelyInformative(OneStepSignal(GUARD_RHO)), SIGNAL_TRAIN_POOL),
    )
    assert verdict.permitted, verdict.reason
    assert verdict.seam_coordinates == (BASE_COORDINATES,)
    assert verdict.strongest_actionable_correlation == pytest.approx(
        GUARD_RHO, abs=0.05
    )


# ---------------------------------------------------------------------------
# What the amendment still refuses
# ---------------------------------------------------------------------------


def test_the_amended_guard_still_refuses_the_realised_price():
    """Clause 1, and the reason the amendment is a narrowing and not a deletion."""
    verdict = _run(ShowsTheWalk, None)
    assert not verdict.permitted
    assert not verdict.price_independent
    assert verdict.forward_only is None, (
        "clause 2 should not have been reached; a guard that has already refused "
        "does not need to measure how"
    )
    assert "realised price" in verdict.reason


def test_the_amended_guard_still_refuses_the_realised_shortfall():
    """Clause 1 again. Two sentences, because a future session could satisfy one."""
    verdict = _run(ShowsTheShortfall, None)
    assert not verdict.permitted
    assert not verdict.price_independent


def test_the_amended_guard_refuses_a_signal_about_an_already_committed_shock():
    """Clause 2 — the case with no precedent, built from task 1's own instrument.

    ``bins_ahead = 0`` makes ``s_k`` predict ``xi_k``, whose cost is charged on
    ``h_k`` — inventory the decision at bin ``k`` has already been made about. It
    is worth exactly nothing (task 1 measured the advantage collapsing to the
    grid's own residual), it sails through clause 1 because it does not move with
    the price stream at a pinned signal, and no number the milestone reports would
    look wrong if the seam were built this way.

    Without this the amendment is a hole of unmeasured width: "a shock that has
    not yet landed" is *true* of ``xi_k`` at decision point ``k``, so the brief's
    own wording admits this case. The guard draws the line at **committed**
    instead, and this is where that distinction stops being a paragraph.
    """
    verdict = _run(
        ExecutionEnv,
        signal_stream(
            NaivelyInformative(OneStepSignal(GUARD_RHO, bins_ahead=0)),
            SIGNAL_TRAIN_POOL,
        ),
    )
    assert verdict.price_independent, (
        "the already-committed case is supposed to pass clause 1 — if it fails "
        "there, this test is not exercising clause 2 at all"
    )
    assert not verdict.permitted, verdict.reason
    assert verdict.forward_only is False
    assert verdict.worst_committed_correlation == pytest.approx(GUARD_RHO, abs=0.05)

    coordinate, k, j = verdict.worst_committed_at
    assert coordinate == BASE_COORDINATES
    assert j == k, (
        f"the correlation should be with the shock of the bin being decided "
        f"(j == k), got j={j} at k={k}"
    )
    assert "already-committed" in verdict.reason


def test_the_guard_refuses_to_run_a_vacuous_comparison():
    """A guard whose two streams drew the same shocks is not a guard."""
    market = _market()
    build = _factory(ExecutionEnv, None, market)
    with pytest.raises(ValueError, match="identical shocks"):
        observation_minimality(
            build, _schedule(market), price_streams=(900, 900), episodes=8
        )
    with pytest.raises(ValueError, match="at least two shock streams"):
        observation_minimality(build, _schedule(market), price_streams=(900,))


def test_the_signal_stream_pin_is_what_makes_clause_one_possible():
    """Without the pin the two rollouts differ in the signal as well as the price.

    Stated as its own check because the pin is easy to drop and its absence looks
    like a *failing guard* rather than a broken one: an unpinned comparison
    refuses a correct env, and the temptation is then to loosen the guard.
    """
    market = _market()
    stream = signal_stream(OneStepSignal(GUARD_RHO), SIGNAL_TRAIN_POOL)
    schedule = _schedule(market)

    def unpinned(price_stream: int, pinned_signal: int | None) -> ExecutionEnv:
        return ExecutionEnv(
            market,
            100_000.0,
            1e-4,
            signal=stream,
            root_seed=20260825,
            pool=M5_DIFFERENTIAL_POOL,
            stream_index=price_stream,
        )

    verdict = observation_minimality(
        unpinned, schedule, episodes=8, tolerance=GUARD_TOLERANCE
    )
    assert not verdict.price_independent, (
        "an unpinned signal moves with the stream index, so this comparison "
        "cannot be clean; if it passed, the pin is not doing anything"
    )
    assert isinstance(stream, SignalStream) and stream.pinned_to(3).index == 3
