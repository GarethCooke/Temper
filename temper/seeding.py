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
#: shocks it gets. ``m3/diagnostic`` continues the pattern for M3's antithetic
#: pairing checks, which are diagnostics of an estimator and report no number.
#: ``m4a/differential`` is M1's differential again in the power-law world. Its
#: own pool rather than more offsets inside ``m1/differential``: a new pool
#: cannot collide with an in-use range, where a new offset can only be *checked*
#: not to. Appended, never inserted — :data:`_POOL_INDEX` is positional and
#: reordering this tuple would re-address every committed result.
#:
#: The last four are M4b's, and the *reason* they exist is the reason the whole
#: module exists. Liquidity is a second, independent noise source, and if its
#: variate came out of the price generator then every downstream price draw would
#: shift — Phase 1 and M4a would stop reproducing, silently, with every result
#: still regenerating from its own config. So liquidity draws from its own
#: address, and the acceptance is arithmetic rather than argued: one M3 seed and
#: one M4a seed retrain **bitwise** through the new seam
#: (``tests/test_m4b_phase1_regression.py``).
#:
#: Two liquidity pools rather than one with a stride, because invariant 5 asks for
#: disjointness *by construction*: a training liquidity path and an evaluation one
#: cannot collide when they are addressed by different spawn keys, where a shared
#: pool split by an offset can only be *checked* not to. ``m4b/reference`` is the
#: oracle's own bound sampling — the clairvoyant relaxation and the feasible upper
#: bound draw tens of thousands of paths, and doing that from ``eval`` would burn
#: streams the trained result is addressed by. ``m4b/differential`` is M1's
#: differential once more, in the liquidity world.
#:
#: ``m5/reference`` is the last of them, and it is ``m4b/reference`` one milestone
#: along: the alpha oracle draws signal paths for its feasible upper bound and
#: shock paths for the price-clairvoyant relaxation, and spending ``eval`` streams
#: on a reference table would burn addresses a trained result is reported at, on a
#: computation with no agent in it. M5's *observation* streams — the signal an env
#: shows a policy — are a separate matter and are deliberately not here yet: a
#: pool nothing addresses is a promise rather than a contract, and this milestone's
#: gates run before any of that exists.
POOLS: tuple[str, ...] = (
    "train",
    "eval",
    "m1/differential",
    "m2/diagnostic",
    "m3/diagnostic",
    "m4a/differential",
    "m4b/liquidity-train",
    "m4b/liquidity-eval",
    "m4b/reference",
    "m4b/differential",
    "m5/reference",
)

#: The pool M1's Monte-Carlo differential draws from. Named here rather than
#: spelled as a literal at the call sites so the quarantine is greppable.
DIFFERENTIAL_POOL = "m1/differential"

#: The pool M2's non-reported checks draw from — the action-space, grading and
#: control-variate tests, none of which report a number.
M2_DIAGNOSTIC_POOL = "m2/diagnostic"

#: The pool M3's non-reported checks draw from — the antithetic pairing's
#: action-identity, shock-negation and zero-variance tests.
M3_DIAGNOSTIC_POOL = "m3/diagnostic"

#: The pool M4a's differential and its inherited-guarantee checks draw from.
M4A_DIFFERENTIAL_POOL = "m4a/differential"

#: M4b's liquidity streams — the *second* noise source, addressed away from the
#: price streams so a shock path cannot move because a multiplier was drawn.
#: Training and evaluation get different pools rather than different offsets in
#: one, so invariant 5's out-of-sample claim is a property of the spawn keys.
LIQUIDITY_TRAIN_POOL = "m4b/liquidity-train"
LIQUIDITY_EVAL_POOL = "m4b/liquidity-eval"

#: The pool the oracle's own Monte-Carlo bounds draw from — the clairvoyant
#: relaxation and the feasible upper bound. Reported numbers, but no agent is
#: involved and no stream a graded result is addressed by may be spent on them.
M4B_REFERENCE_POOL = "m4b/reference"

#: The pool M4b's differential draws from — the liquidity process's own moments
#: and the world's E[cost] at M1's tiers.
M4B_DIFFERENTIAL_POOL = "m4b/differential"

#: The pool M5's oracle draws from — the feasible upper bound's signal paths and
#: the price-clairvoyant relaxation's shock paths. Reported numbers, but no agent
#: is involved and no stream a graded result is addressed by may be spent on them.
M5_REFERENCE_POOL = "m5/reference"

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
