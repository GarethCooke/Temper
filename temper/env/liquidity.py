"""The env's half of the liquidity seam: a law, and the stream it draws from.

M4b's world adds a **second, independent noise source**, and the whole of this
module exists to make that second source impossible to get wrong in the one way
that would be silent: by taking its draws out of the price generator. If the
liquidity variate came from the same stream as the shocks, every downstream price
draw would shift — Phase 1 and M4a would stop reproducing, quietly, with every
result still regenerating perfectly well from its own config. So a law never
travels alone: it travels bound to the seed **pool** its draws are addressed in,
and the acceptance for that is arithmetic rather than argument (one M3 seed and
one M4a seed retrain bitwise through this seam,
``tests/test_m4b_phase1_regression.py``).

Where the laws themselves live, and why it is not here
------------------------------------------------------
The M4b brief predicted these classes would live in this module, "mirroring
`impact.py` exactly". They live in :mod:`temper.oracle.liquidity` instead, and the
reason is structural rather than a preference: the analytic reference *needs*
their moments — the static rung is priced by ``E[L^-beta]`` and the dynamic
program quadratures over the law — and ``temper/oracle`` may not depend on
``temper/env`` (``temper/env/impact.py`` already depends the other way, and the
oracle is normative, ``ARCHITECTURE.md`` §3).

The analogy with ``impact.py`` survives intact, and is in fact the same split one
level along: :mod:`temper.oracle.impact` owns the impact *functions* while
:mod:`temper.env.impact` owns the injected *models*; :mod:`temper.oracle.liquidity`
owns the law and its closed-form moments while this module owns the **stream** —
the thing that turns a distribution into draws at an auditable address. The two
routes stay deliberately separate, which is what lets M4b's differential measure
the env's realised draws against the oracle's closed forms instead of against
themselves.

The default is Phase 1, everywhere
----------------------------------
:data:`DETERMINISTIC_LIQUIDITY` is ``L = 1`` and it is what an env, an experiment
loader and a reference table all get when nobody names anything — constitution
§4's "additive alternatives behind the same interface, never silent modifications
of Phase 1", now applied to a *second* seam. M4b hands out two models, so there
are two ways for a config to inherit a Phase-2 world by omission, and
``tests/test_repo_invariants.py`` closes both.
"""

from __future__ import annotations

from dataclasses import dataclass

from numpy.random import Generator

from temper.oracle import DeterministicLiquidity, LiquidityLaw
from temper.seeding import M4B_DIFFERENTIAL_POOL, POOLS, pool_rng


@dataclass(frozen=True)
class LiquidityStream:
    """A liquidity law bound to the seed pool its draws are addressed in.

    The two are one object on purpose. A stochastic law with no pool named is the
    defect this milestone is most exposed to — it would either share the price
    generator (moving every committed shock) or default to some pool a committed
    result is already addressed by — and making the pool a separate optional
    argument is exactly how that happens. Here it cannot: you get a law and its
    address together, or you get Phase 1.

    The stream *index* normally is not here either: it is the env's, and liquidity
    uses the same one, so seed ordinal ``i``'s price stream ``i`` pairs with
    liquidity stream ``i`` in a different pool. The two cannot collide (the spawn
    keys differ in their first coordinate) and adding a seed cannot move either.
    One address, read two ways.

    :attr:`index` **pins** it, and exists for exactly one caller. M4b's successor
    to ``deterministic_schedule`` has to hold the liquidity stream fixed while
    varying the price stream — that is the whole assertion, "the policy is still
    price-free" one axis wider — and with the index following the env there is no
    way to move one without the other. ``None`` is the normal case and means
    "whatever the env's stream index is"; an int means this stream and only this
    stream, however the env is reset.
    """

    law: LiquidityLaw
    pool: str = M4B_DIFFERENTIAL_POOL
    index: int | None = None

    def __post_init__(self) -> None:
        if self.index is not None and self.index < 0:
            raise ValueError(f"a pinned liquidity index must be >= 0, got {self.index}")
        if not isinstance(self.law, LiquidityLaw):
            raise TypeError(
                f"a liquidity stream carries a LiquidityLaw, got {type(self.law)!r}"
            )
        if self.pool not in POOLS:
            raise ValueError(
                f"unknown seed pool {self.pool!r}; expected one of {', '.join(POOLS)}"
            )

    @property
    def stochastic(self) -> bool:
        """Whether this stream is a second noise source at all."""
        return self.law.stochastic

    def stream_index(self, env_stream_index: int) -> int:
        """The index this stream draws at, given the env's own."""
        return env_stream_index if self.index is None else self.index

    def generator(self, root_seed: int, stream_index: int) -> Generator:
        """The generator for one episode stream — never the price generator.

        Addressed by ``(root_seed, self.pool, index)``, which shares the root seed
        with the env's price stream and differs in the pool (always) and possibly
        in the index. That is what makes the two independent by construction
        rather than by a stride nobody re-checks.
        """
        return pool_rng(root_seed, self.pool, self.stream_index(stream_index))

    def pinned_to(self, index: int) -> "LiquidityStream":
        """The same law and pool, pinned to one stream whatever the env does."""
        return LiquidityStream(law=self.law, pool=self.pool, index=int(index))

    def as_dict(self) -> dict:
        return {"pool": self.pool, "pinned_index": self.index} | self.law.as_dict()


#: Phase 1, M4a, and the default everywhere: ``L = 1``, and no randomness drawn.
#: The pool is inert for this law — a deterministic multiplier never touches a
#: generator, which is one independent reason the earlier milestones' seeds
#: retrain bitwise through this seam.
DETERMINISTIC_LIQUIDITY = LiquidityStream(law=DeterministicLiquidity())


def liquidity_stream(law: LiquidityLaw, pool: str) -> LiquidityStream:
    """A stream for `law` addressed in `pool`. The one sanctioned constructor."""
    return LiquidityStream(law=law, pool=pool)
