"""The observation-minimality guard, amended for M5 — and what it still refuses.

Since M1a the guard has been one sentence: *run the same deterministic schedule
against two shock streams and require the observation sequences to be bitwise
equal*. Under Phase-1 dynamics the shock was the only thing an agent could learn
that TWAP and the sinh cannot see, so any observation that varied with the price
draw was a leak, and §4's "rediscovery must not smuggle in signal" was exactly
that assertion.

M5's observation varies with the price draw **on purpose**. That is the milestone.
So the guard is amended rather than deleted — a guard that stops asserting
anything is worse than no guard — and the amendment has to say precisely where the
line moved, because this is the first time it has moved at all.

The line, stated so a later session can place its own case
-----------------------------------------------------------
**Permitted:** a coordinate that is correlated with a shock whose cost the
*current decision can still change*.

**Refused:** everything else. In particular all three of these, and the third is
the one that is easy to get wrong:

1. The realised price. The walk, a noisy price, an execution price — anything that
   moves when the price stream moves and the signal path does not.
2. The realised shortfall, or any realised cost. Same test catches it: it is a
   function of the walk.
3. **A signal about a shock that is already committed**, however far in the future
   that shock lands. At the decision point for bin ``k`` the inventory ``h_k`` has
   been fixed by the previous decision, so ``xi_k``'s cost is already settled: a
   signal about it is worth exactly nothing, and an observation carrying it is an
   off-by-one in the seam's timing rather than a milestone.

"Committed", not "landed", and the difference is not pedantry
--------------------------------------------------------------
The brief's wording for this amendment is *a signal about a shock that has not yet
landed*. Taken literally that admits case 3, and it should not. Walk the env's own
ordering: ``reset`` returns the decision point for bin 0 with the walk at zero, and
``step(k)`` lands ``xi_k`` **and then** trades. So at the decision point for bin
``k`` the shocks ``xi_0..xi_{k-1}`` have landed and ``xi_k`` has not — yet a signal
about ``xi_k`` is useless, because ``xi_k`` is charged on ``h_k`` and the decision
being made now sets ``h_{k+1}``.

So the permitted set is ``j >= k + 1``, not ``j >= k``. The two differ by exactly
one shock per bin, and that one shock is the whole content of M5's timing
instrument (``temper.oracle.signal.AlphaSignal.lag``): a seam whose signal points
one bin short is worth zero and would be invisible in every number the milestone
reports, because the advantage would simply come back smaller and every gate would
still be green. The guard is what makes that a refusal instead of a disappointment.

Two clauses, and each refuses a different thing
------------------------------------------------
:func:`observation_minimality` returns a :class:`MinimalityVerdict` rather than
asserting, so the same guard can be pointed at a case that must **pass** and a case
that must **fail** and both can be checked.

* **Price-independence at a pinned signal path.** Pin the signal stream, vary the
  price stream, require the observations bitwise equal. This is M1a's clause with
  one axis pinned — the same generalisation
  :func:`~temper.eval.grading.deterministic_schedule` made for liquidity in M4b —
  and it is what refuses cases 1 and 2. Bitwise, not ``allclose``: the claim is
  that the realised price did not enter the observation at all.
* **Forward-only correlation.** Over many episodes, every seam coordinate of the
  observation at decision point ``k`` must be uncorrelated with every shock
  ``xi_j`` for ``j <= k``. This is what refuses case 3, and it is stated on
  *correlation with the realised shocks* rather than on the seam's declared
  parameters so that it measures what the env did rather than what its config said.

The second clause is deliberately not told which coordinate is "the signal". It
checks every coordinate past ``(time left, inventory left)``, so a leak dressed as
a seam coordinate is caught by it as well as by the first clause.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from temper.env import SHOCK_KEY, ExecutionEnv

#: Episodes the forward-only clause averages over. A correlation estimated from
#: ``E`` samples has a standard deviation of about ``1/sqrt(E)``, so 12 000 gives
#: 0.009 and the worst of the ~90 backward pairs sits around 3.2 of those — well
#: inside :data:`DEFAULT_TOLERANCE`, and far below the correlation an off-by-one
#: seam would show, which is ``rho`` itself.
DEFAULT_EPISODES = 12_000

#: The bar on a *backward* correlation. Four sampling standard deviations at
#: :data:`DEFAULT_EPISODES`, and an order below any ``rho`` this guard is pointed
#: at, so the gap between "float noise" and "the seam is one bin out" is not a
#: judgement call.
DEFAULT_TOLERANCE = 0.04

#: The two coordinates every world has had since M1: time left and inventory
#: left. Everything past them is a seam coordinate and is what this guard reads.
BASE_COORDINATES = 2


@dataclass(frozen=True)
class MinimalityVerdict:
    """What the amended guard decided, and on which clause.

    A verdict rather than an assertion, because the guard has to be pointed at
    cases that must be refused as well as cases that must pass, and a guard that
    can only raise cannot be checked for having any teeth.
    """

    permitted: bool
    #: Clause 1 — the observation did not move when only the price stream moved.
    price_independent: bool
    #: Clause 2 — no seam coordinate correlates with an already-committed shock.
    #: ``None`` when the clause was not reached because clause 1 already refused.
    forward_only: bool | None
    #: Largest ``|corr(observation coordinate at k, xi_j)|`` over ``j <= k``.
    worst_committed_correlation: float
    #: The ``(coordinate, k, j)`` that produced it.
    worst_committed_at: tuple[int, int, int] | None
    #: Largest correlation over the *permitted* range ``j >= k + 1``. Reported so
    #: a green verdict cannot be a guard that is simply looking at nothing.
    strongest_actionable_correlation: float
    seam_coordinates: tuple[int, ...]
    episodes: int
    tolerance: float
    reason: str

    def as_dict(self) -> dict:
        return {
            "permitted": self.permitted,
            "price_independent": self.price_independent,
            "forward_only": self.forward_only,
            "worst_committed_correlation": self.worst_committed_correlation,
            "worst_committed_at": self.worst_committed_at,
            "strongest_actionable_correlation": self.strongest_actionable_correlation,
            "seam_coordinates": list(self.seam_coordinates),
            "episodes": self.episodes,
            "tolerance": self.tolerance,
            "reason": self.reason,
        }


def _episode(env: ExecutionEnv, schedule: Sequence[float], seed: int):
    """One full episode: the observations, and the standardised shocks."""
    observation, _ = env.reset(seed=seed)
    seen, walk = [observation], []
    for shares in schedule:
        observation, _, _, _, info = env.step(float(shares))
        seen.append(observation)
        walk.append(info[SHOCK_KEY])
    increments = np.diff(np.concatenate(([0.0], walk)))
    return np.array(seen), increments / (env.market.sigma_bin * 1e4)


def observation_minimality(
    factory: Callable[[int, int | None], ExecutionEnv],
    schedule: Sequence[float],
    *,
    price_streams: Sequence[int] = (900, 901),
    pinned_signal: int = 900,
    episodes: int = DEFAULT_EPISODES,
    tolerance: float = DEFAULT_TOLERANCE,
) -> MinimalityVerdict:
    """Run the amended guard against the env `factory` builds.

    `factory(price_stream, pinned_signal)` returns an
    :class:`~temper.env.ExecutionEnv` on that price stream, with its signal stream
    pinned to `pinned_signal` when that argument is an int and following the env's
    own stream index when it is ``None``. The pin is what makes clause 1 possible
    at all: signal and price normally share the env's stream index, so varying the
    index would move both and the comparison would be vacuous — the same reason
    :class:`~temper.env.liquidity.LiquidityStream` grew a pin in M4b.

    `schedule` is the per-bin share count of a deterministic schedule. Any one
    will do; TWAP is the obvious choice, and what matters is only that it is the
    *same* schedule on both sides so the observation is the only thing that can
    differ.
    """
    if len(price_streams) < 2:
        raise ValueError(
            f"price-independence needs at least two shock streams, got "
            f"{list(price_streams)}"
        )

    # -- clause 1: pin the signal, vary the price -----------------------------
    runs = [_episode(factory(int(s), pinned_signal), schedule, int(s)) for s in price_streams]
    observations = [seen for seen, _ in runs]
    shocks = [drawn for _, drawn in runs]
    if all(np.array_equal(shocks[0], other) for other in shocks[1:]):
        raise ValueError(
            "the price streams drew identical shocks, so clause 1 is vacuous; "
            "choose stream indices that actually differ"
        )
    price_independent = all(
        np.array_equal(observations[0], other) for other in observations[1:]
    )

    width = observations[0].shape[1]
    seam = tuple(range(BASE_COORDINATES, width))

    if not price_independent:
        moved = [
            index
            for index in range(width)
            if not np.array_equal(observations[0][:, index], observations[1][:, index])
        ]
        return MinimalityVerdict(
            permitted=False,
            price_independent=False,
            forward_only=None,
            worst_committed_correlation=float("nan"),
            worst_committed_at=None,
            strongest_actionable_correlation=float("nan"),
            seam_coordinates=seam,
            episodes=0,
            tolerance=tolerance,
            reason=(
                f"the observation moved with the price stream at one pinned signal "
                f"path, on coordinate(s) {moved}. Something about the realised "
                "price — the walk, a price, a realised cost — is reaching the "
                "agent, which no amendment permits"
            ),
        )

    if not seam:
        return MinimalityVerdict(
            permitted=True,
            price_independent=True,
            forward_only=True,
            worst_committed_correlation=0.0,
            worst_committed_at=None,
            strongest_actionable_correlation=0.0,
            seam_coordinates=seam,
            episodes=0,
            tolerance=tolerance,
            reason=(
                "the observation is (time left, inventory left) and carries no "
                "seam coordinate at all, which is M1a's world unchanged"
            ),
        )

    # -- clause 2: the permitted correlation is forward-only ------------------
    env = factory(int(price_streams[0]), None)
    seen, drawn = [], []
    for episode in range(episodes):
        observations_e, shocks_e = _episode(env, schedule, episode)
        seen.append(observations_e)
        drawn.append(shocks_e)
    seen, drawn = np.array(seen), np.array(drawn)

    n_bins = env.market.n_bins
    worst, worst_at, strongest = 0.0, None, 0.0
    for coordinate in seam:
        column = seen[:, :, coordinate]
        for k in range(n_bins):
            if float(np.std(column[:, k])) == 0.0:
                continue
            for j in range(n_bins):
                observed = abs(float(np.corrcoef(column[:, k], drawn[:, j])[0, 1]))
                if j <= k:
                    if observed > worst:
                        worst, worst_at = observed, (coordinate, k, j)
                else:
                    strongest = max(strongest, observed)

    forward_only = worst <= tolerance
    if forward_only:
        reason = (
            f"every seam coordinate is uncorrelated with every already-committed "
            f"shock (worst {worst:.4f} at {worst_at}, bar {tolerance}); the "
            f"strongest correlation it does carry is {strongest:.4f}, and it is "
            "with a shock the current decision can still act on"
        )
    else:
        coordinate, k, j = worst_at  # type: ignore[misc]
        reason = (
            f"observation coordinate {coordinate} at decision point {k} is "
            f"correlated {worst:.4f} with xi_{j}, whose cost is charged on "
            f"inventory the decision at bin {k} cannot change. That is a signal "
            "about an already-committed shock: worth nothing, and an off-by-one "
            "in the seam's timing rather than a milestone"
        )
    return MinimalityVerdict(
        permitted=forward_only,
        price_independent=True,
        forward_only=forward_only,
        worst_committed_correlation=worst,
        worst_committed_at=worst_at,
        strongest_actionable_correlation=strongest,
        seam_coordinates=seam,
        episodes=episodes,
        tolerance=tolerance,
        reason=reason,
    )
