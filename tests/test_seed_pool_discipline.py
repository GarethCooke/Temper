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
    DEFAULT_POOL_ALLOWANCE,
    M1_CONFIG,
    POOL_ALLOWANCE,
    RESERVED_POOLS,
    RESOLVED_SEED_ADDRESSES,
    SEED_ADDRESS_LEDGER,
    build_env,
    case_by_id,
    differential_pairs,
    guard_case,
    pool_allowance,
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
    """Non-vacuity: the session-wide check would notice if it were bypassed.

    Scoped to the modules on the default allowance, because M2 runs in the same
    session from its own committed root seed — a whole-session assertion on the
    root would now be asserting that two milestones share one, which is the
    opposite of what a per-experiment config is for.
    """
    assert RESOLVED_SEED_ADDRESSES, (
        "no env stream was recorded; the conftest recorder is not wrapping the "
        "env's route to randomness and its teardown assertion proves nothing"
    )
    m1_roots = {
        root
        for module, root, _, _ in SEED_ADDRESS_LEDGER
        if module not in POOL_ALLOWANCE
    }
    assert m1_roots == {ROOT_SEED}


def test_no_reserved_pool_has_been_opened_by_a_module_that_does_not_own_one():
    """Invariant 5, on everything the suite has run so far.

    Until M2 this was the flat statement *nothing in the suite touches `train` or
    `eval`*, which was true and checkable without knowing who was asking. M2
    trains out of `train` and evaluates out of `eval` — both legitimately — so
    the property became per-module: every draw is attributed, and a module may
    open only the reserved pools ``conftest.RESERVED_POOL_OWNERS`` grants it.
    That is strictly stronger than what it replaced, because it also catches M2
    grading on a stream it trained on, which the flat version could not have
    seen.

    The complete statement is made at session teardown, over the whole path
    including whatever ran after this. This is the same assertion made early, so
    a violation is attributed to a test rather than to the session.
    """
    trespasses = sorted(
        {
            (module, pool)
            for module, _, pool, _ in SEED_ADDRESS_LEDGER
            if pool not in pool_allowance(module)
        }
    )
    assert not trespasses, (
        "modules drew from a pool they are not allowed: "
        + ", ".join(f"{module} -> {pool}" for module, pool in trespasses)
    )


def test_the_m1_differential_path_still_uses_only_the_diagnostic_pool():
    """The original property, kept exactly, for the modules it was written about.

    M1's tens of millions of episodes are a diagnostic; not one of them may be
    charged to a stream a committed M2 result is addressed by. Attributing the
    ledger is what lets this stay an unconditional statement about M1 while M2
    spends `train` and `eval` in the same session.
    """
    m1_modules = sorted(
        {
            module
            for module, _, _, _ in SEED_ADDRESS_LEDGER
            if module not in POOL_ALLOWANCE
        }
    )
    assert m1_modules, "the ledger recorded no M1 module; the recorder is not wired"

    opened = {
        pool
        for module, _, pool, _ in SEED_ADDRESS_LEDGER
        if module not in POOL_ALLOWANCE
    }
    assert opened == {POOL} == set(DEFAULT_POOL_ALLOWANCE), (
        f"the diagnostic path opened {sorted(opened)}, expected only {POOL!r}"
    )


def test_no_module_is_granted_both_training_and_evaluation_streams_by_accident():
    """The allowance table says what it means, and says it about few modules.

    Only the module that regenerates the committed result needs both pools —
    that is what a sweep *is*. Any other module holding both would be able to
    grade on the streams it trained on without a single test noticing, which is
    the failure invariant 5 exists to prevent.
    """
    holders = sorted(
        module
        for module, pools in POOL_ALLOWANCE.items()
        if RESERVED_POOLS <= pools
    )
    assert holders == ["test_m2_rediscovery.py"], (
        f"{holders} hold both train and eval; only the sweep regeneration may"
    )


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
