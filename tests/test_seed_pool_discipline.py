"""M1a task 5 — M1's draws come out of the diagnostic pool, and only that pool.

M0 pinned that the seed pools are disjoint *by construction*: different spawn
keys, so no stream in one can collide with a stream in another. That is a
property of :mod:`temper.seeding`, and it was never the property invariant 5
actually needs here. The one M1 needs is behavioural — *the harness drew from the
diagnostic pool* — and until now nothing checked it. The failure it admits is
quiet and total: a ``pool=`` default edited in the wrong direction, or one config
key pointing at ``train``, leaves every seeding test in M0 green while M1's tens
of millions of episodes burn streams that M2's committed results will be
addressed by. Invariant 1 would then be false for every number M2 reports, and
nothing would say so.

Two complementary checks:

* **Every configured cell, by address.** Each (case, schedule) cell of both tiers
  is built and reset, and the address it resolves is required to be exactly the
  ``(root_seed, "m1/differential", stream_index)`` the config's addressing rule
  says it should be. That covers the differential exhaustively without running
  it.
* **The whole test path, by recording.** ``tests/conftest.py`` wraps the env's one
  route to randomness for the entire session, so the reserved pools are checked
  against everything the suite does, not against the cells this module
  remembered to drive. The final verdict is asserted at session teardown; what
  is here is the non-vacuity check that the recorder is installed and seeing
  work.
"""

from __future__ import annotations

import numpy as np
import pytest

from temper.env import ExecutionEnv
from temper.seeding import DIFFERENTIAL_POOL, POOLS, pool_rng, pool_sequence

from .conftest import (
    M1_CONFIG,
    RESERVED_POOLS,
    RESOLVED_SEED_ADDRESSES,
    build_env,
    case_by_id,
    differential_pairs,
    guard_case,
)

SEEDING = M1_CONFIG["seeding"]
ROOT_SEED = int(SEEDING["root_seed"])
POOL = SEEDING["pool"]

ALL_PAIRS = differential_pairs("fast") + differential_pairs("deep")


def test_the_config_names_the_diagnostic_pool_and_not_a_reserved_one():
    """The one place the pool is written down, and what it may say."""
    assert POOL == DIFFERENTIAL_POOL == "m1/differential"
    assert POOL in POOLS
    assert POOL not in RESERVED_POOLS
    assert RESERVED_POOLS == {"train", "eval"}


def test_the_envs_default_pool_is_the_diagnostic_one():
    """A caller who forgets to say which pool must not land in a reserved one.

    The default matters more than it looks: M2 will construct envs from a new
    config, and the failure mode is not somebody choosing ``train`` by mistake —
    it is somebody not choosing at all.
    """
    case = case_by_id(M1_CONFIG["tiers"]["fast"]["cases"][0])
    env = ExecutionEnv(case.market, case.order_size, case.lambda_risk, root_seed=1)
    _, pool, _ = env.seed_address
    assert pool == DIFFERENTIAL_POOL


@pytest.mark.parametrize("pair", ALL_PAIRS, ids=str)
def test_every_differential_cell_resolves_the_address_the_config_says(pair):
    """Exhaustive over both tiers: the cell, the pool, and the stream index.

    Checking the address rather than the draws is what makes this affordable —
    all 36 cells, without running 5.4 million episodes to find out where their
    shocks came from.
    """
    before = len(RESOLVED_SEED_ADDRESSES)
    env = build_env(pair.case, pair.stream_index)
    env.reset(seed=pair.stream_index)

    assert env.seed_address == (ROOT_SEED, POOL, pair.stream_index)
    resolved = RESOLVED_SEED_ADDRESSES[before:]
    assert resolved == [(ROOT_SEED, POOL, pair.stream_index)], (
        f"{pair} opened {resolved} rather than its one addressed stream"
    )


def test_the_identity_and_guard_streams_are_addressed_the_same_way():
    """The non-Monte-Carlo cells draw from the same pool as the tiers.

    They are far fewer episodes, which is exactly why they would be the ones to
    escape notice.
    """
    for stream in (int(SEEDING["identity_stream"]), int(SEEDING["guard_stream"])):
        env = build_env(guard_case(), stream)
        env.reset(seed=stream)
        assert env.seed_address == (ROOT_SEED, POOL, stream)


def test_the_recorder_is_installed_and_has_seen_the_env_work():
    """Non-vacuity: the session-wide check would notice if it were bypassed."""
    assert RESOLVED_SEED_ADDRESSES, (
        "no env stream was recorded; the conftest recorder is not wrapping the "
        "env's route to randomness and its teardown assertion proves nothing"
    )
    assert {root for root, _, _ in RESOLVED_SEED_ADDRESSES} == {ROOT_SEED}


def test_no_reserved_pool_has_been_opened():
    """Invariant 5, on everything the suite has run so far.

    The complete statement is made at session teardown, over the whole path
    including whatever ran after this. This is the same assertion made early, so
    a violation is attributed to a test rather than to the session.
    """
    opened = {pool for _, pool, _ in RESOLVED_SEED_ADDRESSES}
    assert not opened & RESERVED_POOLS, (
        f"the env drew from {sorted(opened & RESERVED_POOLS)}; M1 is a diagnostic "
        "and train/eval streams belong to committed M2 results"
    )
    assert opened == {POOL}


def test_the_streams_m1_spends_are_not_train_or_eval_streams():
    """...and even if one had been opened, it would not have been the same numbers.

    M0 pins pool disjointness in general. This pins it for the specific streams
    M1 actually spends, at the level of the generated state rather than the spawn
    key: two pools would still be a bug if they happened to produce the same
    bytes, and "disjoint by construction" is only worth as much as the
    construction.
    """
    streams = sorted({pair.stream_index for pair in ALL_PAIRS}) + [
        int(SEEDING["identity_stream"]),
        int(SEEDING["guard_stream"]),
    ]
    for stream in streams:
        mine = pool_sequence(ROOT_SEED, POOL, stream)
        state = mine.generate_state(4, dtype=np.uint32)
        for reserved in sorted(RESERVED_POOLS):
            other = pool_sequence(ROOT_SEED, reserved, stream)
            assert mine.spawn_key != other.spawn_key
            assert not np.array_equal(state, other.generate_state(4, dtype=np.uint32))
            assert not np.array_equal(
                pool_rng(ROOT_SEED, POOL, stream).standard_normal(8),
                pool_rng(ROOT_SEED, reserved, stream).standard_normal(8),
            )
