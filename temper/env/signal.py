"""The env's half of the signal seam: a signal, and the stream it draws from.

M5's world adds a **third** injected model to an env that already had two, and
this module is the third instance of the same pattern for the third time the same
reason. :mod:`temper.env.liquidity` exists because a liquidity variate taken out
of the price generator would shift every downstream shock and un-reproduce every
committed number in silence. The signal is the same hazard one turn sharper,
because the signal is *correlated with the shocks by design*: the one thing that
must never happen is for the correlation to be manufactured by the two sharing a
generator rather than by the model.

So a signal never travels alone here either. It travels bound to the seed **pool**
its draws are addressed in, and the acceptance is arithmetic rather than argument:
one M3 seed, one M4a seed and one M4b seed retrain **bitwise** through this seam
(``tests/test_m5_signal_seam.py``). If any of them moves, the seam is reaching into
the price generator — which is exactly what the check exists for, and not a
reproducibility nuisance to be waved through.

How the correlation is actually made
------------------------------------
The env draws ``s_k`` from **this** stream's pool, blockwise at ``reset``, and
``e_k`` from its own price generator, one per step exactly as it always has. The
shock the price walk uses is then

.. code::

    xi_k = rho * s_{k-1} + sqrt(1 - rho^2) * e_k          k >= 1
    xi_0 = e_0                                            nothing predicts it

so the *count and order* of draws taken from the price generator is unchanged, and
at ``rho = 0`` the composition is ``0.0 * s + 1.0 * e``, which is ``e`` to the bit.
That is the whole mechanism behind the three-world regression: prior worlds are not
"close enough", they are untouched.

Where the signal law itself lives, and why it is not here
---------------------------------------------------------
:mod:`temper.oracle.signal`, for the reason
:mod:`~temper.oracle.liquidity` lives there: the analytic reference *needs* the
law — the dynamic program quadratures over it and the conditional grade prices
``E[cost | s]`` with it — and ``temper/oracle`` may not depend on ``temper/env``
(``ARCHITECTURE.md`` §3, the oracle is normative). The split is the same one
:mod:`temper.env.impact` and :mod:`temper.env.liquidity` already make: the oracle
owns the distribution and its closed forms, this module owns the **stream** — the
thing that turns a distribution into draws at an auditable address.

The default is no signal, everywhere
------------------------------------
:data:`NO_SIGNAL_STREAM` carries :class:`~temper.oracle.signal.NoSignal`, draws no
randomness at all, and is what an env gets when nobody names anything. Constitution
§4's "additive alternatives behind the same interface, never silent modifications
of Phase 1", now on a third seam — and the stake is higher than for either earlier
one. A world acquired by omission changes a number; a *predictive observation*
acquired by omission would make every result from M0 to M4b a claim about a market
where the agent could see one step of the future, and every one of them would still
regenerate perfectly from its own config.
"""

from __future__ import annotations

from dataclasses import dataclass

from numpy.random import Generator

from temper.oracle import AlphaSignal, NoSignal
from temper.seeding import M5_DIFFERENTIAL_POOL, POOLS, pool_rng


@dataclass(frozen=True)
class SignalStream:
    """An alpha signal bound to the seed pool its draws are addressed in.

    The two are one object for the reason
    :class:`~temper.env.liquidity.LiquidityStream` makes them one: an informative
    signal with no pool named is the defect this milestone is most exposed to, and
    a separate optional ``pool=`` argument is exactly how that arrives.

    The stream *index* is normally the env's, so seed ordinal ``i``'s price stream
    ``i`` pairs with signal stream ``i`` in a different pool. The two cannot
    collide — the spawn keys differ in their first coordinate — and adding a seed
    cannot move either.

    :attr:`index` pins it, and it exists for the same one caller
    :class:`LiquidityStream`'s does: a check that has to hold the *signal* path
    fixed while varying the price stream. That check is what would say "the
    observation is a function of the signal path and the schedule, and of nothing
    else about the prices", and M5 task 3 is where it lands. Task 2 leaves the
    price-free guard refusing this env, which is the point: the observation is
    price-*bearing* now, and seeing the old guard refuse is the evidence that the
    amendment narrows something real.
    """

    signal: AlphaSignal
    pool: str = M5_DIFFERENTIAL_POOL
    index: int | None = None

    def __post_init__(self) -> None:
        if self.index is not None and self.index < 0:
            raise ValueError(f"a pinned signal index must be >= 0, got {self.index}")
        if not isinstance(self.signal, AlphaSignal):
            raise TypeError(
                f"a signal stream carries an AlphaSignal, got {type(self.signal)!r}"
            )
        if self.pool not in POOLS:
            raise ValueError(
                f"unknown seed pool {self.pool!r}; expected one of {', '.join(POOLS)}"
            )

    @property
    def informative(self) -> bool:
        """Whether this stream tells a policy anything about a shock to come.

        False for the absent seam, and false for a signal pointed at a shock that
        has **already landed** whatever its correlation is — the milestone's
        timing instrument (``temper.oracle.signal.AlphaSignal.lag``). Both cases
        leave the observation at the width it has always had, which is what makes
        the width itself a statement rather than a coincidence.
        """
        return self.signal.informative

    def stream_index(self, env_stream_index: int) -> int:
        """The index this stream draws at, given the env's own."""
        return env_stream_index if self.index is None else self.index

    def generator(self, root_seed: int, stream_index: int) -> Generator:
        """The generator for one episode stream — **never** the price generator.

        Addressed by ``(root_seed, self.pool, index)``: same root seed as the
        env's price stream, different pool (always), possibly a different index.
        That is what makes the signal and the shocks independent *by construction*
        and the correlation between them a property of the model rather than of a
        shared generator — which is the one way a signal seam could look right and
        be meaningless.
        """
        return pool_rng(root_seed, self.pool, self.stream_index(stream_index))

    def pinned_to(self, index: int) -> "SignalStream":
        """The same signal and pool, pinned to one stream whatever the env does."""
        return SignalStream(signal=self.signal, pool=self.pool, index=int(index))

    def as_dict(self) -> dict:
        return {"pool": self.pool, "pinned_index": self.index} | self.signal.as_dict()


#: Phase 1 through M4b, and the default everywhere: no signal, and **no randomness
#: drawn**. The pool is inert for this one — an absent signal never touches a
#: generator, which is one independent reason the earlier milestones' seeds
#: reproduce bitwise through this seam, on top of the pools being disjoint.
NO_SIGNAL_STREAM = SignalStream(signal=NoSignal())


def signal_stream(signal: AlphaSignal, pool: str) -> SignalStream:
    """A stream for `signal` addressed in `pool`. The one sanctioned constructor."""
    return SignalStream(signal=signal, pool=pool)
