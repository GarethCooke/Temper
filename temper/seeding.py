"""Deterministic seed pools, disjoint between training and evaluation.

Constitution invariant 5 requires train and eval seeds to be disjoint *by
construction*, not by convention — agents overfit simulators, and a shared seed
between the two would make every Phase-2 claim unfalsifiable. Invariant 1
requires that a committed ``(config, root_seed)`` regenerates a reported number
exactly, on any host.

Both fall out of addressing seeds by path rather than by draw order. Each stream
is ``SeedSequence(entropy=root_seed, spawn_key=(pool_index, index))``: the pool
and the index name the stream, so

* the same ``(root_seed, pool, index)`` always yields the same stream, on any
  machine, in any order, no matter what else the session asked for first;
* streams in different pools have different spawn keys and so cannot collide;
* asking for five eval seeds gives back the same first three as asking for three
  — pools grow without renumbering what is already committed.

``SeedSequence.spawn()`` is deliberately *not* used: it is stateful, so its
output depends on how many children were spawned before, which would make a
config's seeds depend on the order in which the harness happened to build things.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.random import Generator, SeedSequence, default_rng

#: The seed pools. Order fixes the spawn keys, so it is part of the contract:
#: reordering this tuple would silently change every previously reported number.
#: Appending is safe — a new pool cannot move the spawn keys of the old ones.
#:
#: ``m1/differential`` is M1's diagnostic pool: the Monte-Carlo differential draws
#: tens of millions of shocks, and doing that from ``train`` or ``eval`` would
#: burn streams that committed M2 results are addressed by. Diagnostics get their
#: own pool for the same reason train and eval have separate ones (invariant 5).
#: ``m2/diagnostic`` is the same idea one milestone on: M2's fast tests need an
#: env to drive, and driving it from ``eval`` would spend streams the committed
#: rediscovery result is addressed by — on a check that does not care which
#: shocks it gets.
POOLS: tuple[str, ...] = ("train", "eval", "m1/differential", "m2/diagnostic")

#: The pool M1's Monte-Carlo differential draws from. Named here rather than
#: spelled as a literal at the call sites so the quarantine is greppable.
DIFFERENTIAL_POOL = "m1/differential"

#: The pool M2's non-reported checks draw from — the action-space, grading and
#: control-variate tests, none of which report a number.
M2_DIAGNOSTIC_POOL = "m2/diagnostic"

_POOL_INDEX = {name: index for index, name in enumerate(POOLS)}

#: Width in bits of the integers :func:`pool_seeds` returns.
SEED_BITS = 128


def _pool_index(pool: str) -> int:
    try:
        return _POOL_INDEX[pool]
    except KeyError:
        raise ValueError(
            f"unknown seed pool {pool!r}; expected one of {', '.join(POOLS)}"
        ) from None


def pool_sequence(root_seed: int, pool: str, index: int) -> SeedSequence:
    """The :class:`SeedSequence` addressed by ``(root_seed, pool, index)``."""
    if index < 0:
        raise ValueError(f"index must be non-negative, got {index}")
    return SeedSequence(entropy=int(root_seed), spawn_key=(_pool_index(pool), int(index)))


def pool_sequences(root_seed: int, pool: str, count: int) -> tuple[SeedSequence, ...]:
    """The first `count` sequences of `pool`."""
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    return tuple(pool_sequence(root_seed, pool, i) for i in range(count))


def pool_seeds(root_seed: int, pool: str, count: int) -> tuple[int, ...]:
    """The first `count` seeds of `pool` as plain integers.

    For logging, config files and results metadata — anywhere a seed has to be
    written down and read back. :func:`pool_rng` derives generators from the
    sequences directly and never round-trips through these integers.
    """
    return tuple(
        int.from_bytes(
            sequence.generate_state(SEED_BITS // 32, dtype=np.uint32).tobytes(),
            "little",
        )
        for sequence in pool_sequences(root_seed, pool, count)
    )


def pool_rng(root_seed: int, pool: str, index: int) -> Generator:
    """The generator for one stream. Independent of every other stream."""
    return default_rng(pool_sequence(root_seed, pool, index))


def pool_rngs(root_seed: int, pool: str, count: int) -> tuple[Generator, ...]:
    """Generators for the first `count` streams of `pool`."""
    return tuple(default_rng(s) for s in pool_sequences(root_seed, pool, count))


def spawn_pools(root_seed: int, counts: dict[str, int]) -> dict[str, tuple[int, ...]]:
    """Seeds for several pools at once, e.g. ``{"train": 5, "eval": 32}``.

    The convenience entry point for experiment configs: one root seed in, the
    whole experiment's seed allocation out, disjoint by construction.
    """
    unknown = set(counts) - set(POOLS)
    if unknown:
        raise ValueError(
            f"unknown seed pool(s) {', '.join(sorted(unknown))}; "
            f"expected one of {', '.join(POOLS)}"
        )
    return {pool: pool_seeds(root_seed, pool, count) for pool, count in counts.items()}


def disjoint(*pools: Sequence[int]) -> bool:
    """True when no seed appears in more than one of `pools`."""
    total = sum(len(pool) for pool in pools)
    return len(set().union(*(set(pool) for pool in pools))) == total
