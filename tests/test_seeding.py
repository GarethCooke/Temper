"""Determinism and train/eval disjointness (constitution invariants 1 and 5).

M1's environment depends on these guarantees, so they are pinned before the
environment exists rather than after it starts producing numbers.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from temper.seeding import (
    POOLS,
    disjoint,
    pool_rng,
    pool_rngs,
    pool_seeds,
    pool_sequence,
    spawn_pools,
)

ROOT_SEED = 20260804


def test_identical_root_seed_gives_identical_pools():
    """The regeneration guarantee: same seed in, same seeds out."""
    for pool in POOLS:
        assert pool_seeds(ROOT_SEED, pool, 16) == pool_seeds(ROOT_SEED, pool, 16)


def test_different_root_seeds_give_different_pools():
    """...and the root seed actually does something."""
    for pool in POOLS:
        assert pool_seeds(ROOT_SEED, pool, 8) != pool_seeds(ROOT_SEED + 1, pool, 8)


def test_train_and_eval_pools_are_disjoint():
    """Invariant 5. Disjoint by spawn key, checked as sets for good measure."""
    train = pool_seeds(ROOT_SEED, "train", 64)
    evaluation = pool_seeds(ROOT_SEED, "eval", 64)
    assert disjoint(train, evaluation)
    assert len(set(train)) == 64, "seeds collide within the training pool"
    assert len(set(evaluation)) == 64, "seeds collide within the eval pool"


def test_pools_are_disjoint_across_several_root_seeds():
    """Not a fluke of one root seed."""
    for root in (0, 1, 7, 20260804, 2**63 - 1):
        assert disjoint(pool_seeds(root, "train", 32), pool_seeds(root, "eval", 32))


def test_every_pair_of_pools_is_disjoint_by_construction():
    """Invariant 5, pairwise and exhaustive, over **every** pool there is.

    :func:`disjoint` on the whole set answers "are all of these distinct", which is
    what invariant 5 asks for and is checked below. This asks the sharper
    question — *which* pair collided — over all ``len(POOLS) * (len(POOLS) - 1) / 2``
    of them, and it is written over ``POOLS`` rather than over a list of the pools
    a session happened to add.

    That generality is the point rather than tidiness. M4b's seam arrived with two
    new pools and M5's arrives with two more; a check that compared the newest pair
    against the old ones would have been a construction check exactly until the
    next pair arrived, at which point it would quietly become a check of something
    else. Written this way it covers a pool that does not exist yet.
    """
    seeds = {pool: set(pool_seeds(ROOT_SEED, pool, 64)) for pool in POOLS}
    assert len(POOLS) == len(set(POOLS)), "a pool name appears twice"
    for first, second in itertools.combinations(POOLS, 2):
        shared = seeds[first] & seeds[second]
        assert not shared, (
            f"{first} and {second} share {len(shared)} of 64 streams; the two are "
            "the same generator by another name, and whichever of them was added "
            "later has been silently spending the other's addresses"
        )
    # Non-vacuity: a pool really is *not* disjoint from itself, so the loop above
    # is comparing something.
    assert seeds[POOLS[0]] & seeds[POOLS[0]]


def test_every_pool_is_disjoint_from_every_other():
    """Including M1's diagnostic pool, which draws tens of millions of shocks."""
    assert disjoint(*(pool_seeds(ROOT_SEED, pool, 32) for pool in POOLS))


def test_the_pool_order_is_the_one_committed_results_were_addressed_by():
    """Appending a pool is safe; inserting or reordering one is not.

    The spawn key is `(pool index, stream index)`, so `train` and `eval` keep
    their streams only while they keep positions 0 and 1. This pins that
    directly — a reorder that silently re-randomised every previously reported
    number would otherwise pass every other test in this file.
    """
    assert POOLS[:2] == ("train", "eval"), "existing pools moved; results would change"
    for index, pool in enumerate(POOLS):
        assert pool_sequence(ROOT_SEED, pool, 3).spawn_key == (index, 3)


def test_pools_grow_without_renumbering():
    """Asking for more seeds must not move the ones already committed to results."""
    for pool in POOLS:
        assert pool_seeds(ROOT_SEED, pool, 100)[:7] == pool_seeds(ROOT_SEED, pool, 7)


def test_allocation_does_not_depend_on_request_order():
    """Addressing by spawn key, not by draw order — the reason spawn() is unused."""
    first = pool_seeds(ROOT_SEED, "eval", 5)
    pool_seeds(ROOT_SEED, "train", 999)  # a large intervening allocation
    assert pool_seeds(ROOT_SEED, "eval", 5) == first


def test_generators_are_reproducible_and_independent():
    """Two draws from the same stream match; two streams do not."""
    draws = pool_rng(ROOT_SEED, "train", 3).normal(size=64)
    assert np.array_equal(draws, pool_rng(ROOT_SEED, "train", 3).normal(size=64))

    other_index = pool_rng(ROOT_SEED, "train", 4).normal(size=64)
    other_pool = pool_rng(ROOT_SEED, "eval", 3).normal(size=64)
    assert not np.array_equal(draws, other_index)
    assert not np.array_equal(draws, other_pool)


def test_streams_within_a_pool_are_distinct():
    """No two training runs silently share a random stream."""
    first_draws = {float(rng.normal()) for rng in pool_rngs(ROOT_SEED, "train", 128)}
    assert len(first_draws) == 128


def test_spawn_pools_allocates_a_whole_experiment():
    """The config-level entry point agrees with the per-pool one."""
    allocation = spawn_pools(ROOT_SEED, {"train": 5, "eval": 32})
    assert allocation["train"] == pool_seeds(ROOT_SEED, "train", 5)
    assert allocation["eval"] == pool_seeds(ROOT_SEED, "eval", 32)
    assert disjoint(allocation["train"], allocation["eval"])


def test_seeds_are_plain_json_safe_integers():
    """Seeds get written into results metadata, so they must survive a round trip."""
    for seed in pool_seeds(ROOT_SEED, "train", 4):
        assert isinstance(seed, int) and not isinstance(seed, np.integer)
        assert 0 <= seed < 2**128


def test_unknown_pools_are_rejected():
    """A typo'd pool name must not silently create a third pool."""
    with pytest.raises(ValueError, match="unknown seed pool"):
        pool_seeds(ROOT_SEED, "test", 1)
    with pytest.raises(ValueError, match="unknown seed pool"):
        spawn_pools(ROOT_SEED, {"train": 1, "holdout": 1})


def test_negative_indices_are_rejected():
    """Negative spawn keys would alias other streams."""
    with pytest.raises(ValueError, match="index must be non-negative"):
        pool_sequence(ROOT_SEED, "train", -1)
