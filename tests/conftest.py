"""Golden-fixture loading, the M0 tolerances, and the M1 differential config.

The tolerances live here, in code, rather than only in prose: constitution
invariant 3 says thresholds are fixed before the work and changed only by
amending the brief, and a threshold you have to grep the docs for is a threshold
that drifts. M1's numbers go one step further and live in
``configs/m1_differential.yaml``, which this module loads — a committed config the
tests read is also the config a result can be regenerated from (invariant 1).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pytest
import yaml

from temper.env import ExecutionEnv, execution_env, impact_for
from temper.oracle import Market, SymbolParams

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = REPO_ROOT / "tests" / "golden" / "vendor" / "frontierview_goldens.json"
M1_CONFIG_PATH = REPO_ROOT / "configs" / "m1_differential.yaml"
M2_CONFIG_PATH = REPO_ROOT / "configs" / "m2_ppo.yaml"
M4A_CONFIG_PATH = REPO_ROOT / "configs" / "m4a_differential.yaml"

#: Pre-stated M0 tolerances (docs/briefs/M0-oracle-and-goldens.md).
TRAJECTORY_RTOL = 1e-6  # relative to the parent order size X
MOMENTS_RTOL = 1e-6     # relative, on E, V and every decomposition component


@dataclass(frozen=True)
class GoldenCase:
    """One vendored case, with the oracle objects it should be evaluated on."""

    case_id: str
    tag: str
    symbol: str
    order_size: float
    lambda_risk: float
    market: Market
    ac: dict
    twap: dict
    derived: dict

    def __str__(self) -> str:  # keeps pytest ids readable
        return self.case_id


def _load_document() -> dict:
    if not GOLDEN_PATH.exists():
        pytest.fail(
            f"vendored golden fixture missing at {GOLDEN_PATH.relative_to(REPO_ROOT)}.\n"
            "Regenerate it from a FrontierView checkout (read-only there):\n"
            "  make goldens FRONTIERVIEW=/path/to/FrontierView\n"
            "Do not synthesise goldens from this repo's own oracle — that would "
            "collapse the differential into a tautology.",
            pytrace=False,
        )
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _build_case(raw: dict, temp_exponent: float) -> GoldenCase:
    params = SymbolParams(**raw["params"])
    market = Market(
        params=params,
        horizon_hours=raw["horizon_hours"],
        n_bins=raw["n_bins"],
        temp_exponent=temp_exponent,
    )
    return GoldenCase(
        case_id=raw["case_id"],
        tag=raw["tag"],
        symbol=raw["symbol"],
        order_size=raw["X"],
        lambda_risk=raw["lambda"],
        market=market,
        ac=raw["ac"],
        twap=raw["twap"],
        derived=raw["derived"],
    )


GOLDEN_DOCUMENT = _load_document()
GOLDEN_CASES = [
    _build_case(raw, GOLDEN_DOCUMENT["conventions"]["temp_exponent"])
    for raw in GOLDEN_DOCUMENT["cases"]
]
GOLDEN_BY_ID = {case.case_id: case for case in GOLDEN_CASES}


@pytest.fixture(scope="session")
def golden_document() -> dict:
    """The whole vendored fixture, provenance block included."""
    return GOLDEN_DOCUMENT


def pytest_generate_tests(metafunc):
    """Parametrise any test taking a `golden_case` over every vendored case."""
    if "golden_case" in metafunc.fixturenames:
        metafunc.parametrize("golden_case", GOLDEN_CASES, ids=str)


# ---------------------------------------------------------------------------
# M1: the differential config (configs/m1_differential.yaml)
# ---------------------------------------------------------------------------


M1_CONFIG: dict = yaml.safe_load(M1_CONFIG_PATH.read_text(encoding="utf-8"))

#: M4a's committed input: the certificate's pre-stated bars, the power-law
#: differential's tiers, and the four inherited guarantees. A separate file from
#: M1's for the reason its header gives — a Phase-2 milestone editing Phase 1's
#: committed thresholds in place would put both worlds' bars in one blast radius.
M4A_CONFIG: dict = yaml.safe_load(M4A_CONFIG_PATH.read_text(encoding="utf-8"))
M4A_ENCODING: str = str(M4A_CONFIG["world"]["cost_encoding"])


def case_by_id(case_id: str) -> GoldenCase:
    """Resolve a config's `case_id` against the vendored fixture."""
    try:
        return GOLDEN_BY_ID[case_id]
    except KeyError:
        raise ValueError(
            f"{M1_CONFIG_PATH.name} names case {case_id!r}, which is not in the "
            f"vendored goldens. Known ids: {', '.join(sorted(GOLDEN_BY_ID))}"
        ) from None


@dataclass(frozen=True)
class DifferentialPair:
    """One (case, schedule) cell of a tier, with its stream and its bands."""

    tier: str
    case: GoldenCase
    schedule: str
    n_sim: int
    stream_index: int
    mean_band: float
    var_band: float

    def __str__(self) -> str:
        return f"{self.tier}:{self.case.case_id}:{self.schedule}"


def differential_pairs(tier: str) -> list[DifferentialPair]:
    """Every (case, schedule) cell of `tier`, addressed exactly as the config says.

    The stream index is derived, not written down 27 times: `stream_base` plus the
    cell's position in the committed `cases` x `schedules` order. That keeps the
    addressing regenerable from the config while still being a fixed function of
    it — adding a case to the end of a tier cannot move the streams of the cases
    already reported on.
    """
    spec = M1_CONFIG["tiers"][tier]
    schedules = M1_CONFIG["schedules"]
    n_sim = int(spec["n_sim"])
    return [
        DifferentialPair(
            tier=tier,
            case=case_by_id(case_id),
            schedule=schedule,
            n_sim=n_sim,
            stream_index=(
                int(spec["stream_base"]) + case_ordinal * len(schedules) + schedule_ordinal
            ),
            mean_band=float(spec["mean_z_sigmas"]) / math.sqrt(n_sim),
            var_band=float(spec["var_z_sigmas"]) * math.sqrt(2.0 / n_sim),
        )
        for case_ordinal, case_id in enumerate(spec["cases"])
        for schedule_ordinal, schedule in enumerate(schedules)
    ]


def build_env(case, stream_index: int) -> ExecutionEnv:
    """The env for a case, seeded from the config's diagnostic pool.

    Takes a :class:`GoldenCase` or a :class:`GuardCase` — both carry a market, an
    order size and a lambda, and nothing here needs to know which it has.
    """
    seeding = M1_CONFIG["seeding"]
    return ExecutionEnv(
        case.market,
        case.order_size,
        case.lambda_risk,
        root_seed=int(seeding["root_seed"]),
        pool=seeding["pool"],
        stream_index=stream_index,
    )


def build_power_law_env(case, stream_index: int) -> ExecutionEnv:
    """The same env in M4a's world, seeded from M4a's own pool.

    One ``ExecutionEnv`` and one ``step`` loop; the only difference from
    :func:`build_env` is the injected temporary-impact model, which is the whole
    of what "the world changed" means here.
    """
    seeding = M4A_CONFIG["seeding"]
    return ExecutionEnv(
        case.market,
        case.order_size,
        case.lambda_risk,
        temporary_impact=impact_for(M4A_ENCODING, case.market, case.order_size),
        root_seed=int(seeding["root_seed"]),
        pool=seeding["pool"],
        stream_index=stream_index,
    )


def power_law_pairs(tier: str) -> list[DifferentialPair]:
    """Every (case, schedule) cell of M4a's `tier`, addressed as its config says.

    The same derivation as :func:`differential_pairs` over M4a's config, which is
    four schedules wide rather than three: the tangent-derived sinh is a schedule
    like any other in this world, and the certified power-law optimum is a fourth.
    """
    spec = M4A_CONFIG["tiers"][tier]
    schedules = M4A_CONFIG["schedules"]
    n_sim = int(spec["n_sim"])
    return [
        DifferentialPair(
            tier=tier,
            case=case_by_id(case_id),
            schedule=schedule,
            n_sim=n_sim,
            stream_index=(
                int(spec["stream_base"]) + case_ordinal * len(schedules) + schedule_ordinal
            ),
            mean_band=float(spec["mean_z_sigmas"]) / math.sqrt(n_sim),
            var_band=float(spec["var_z_sigmas"]) * math.sqrt(2.0 / n_sim),
        )
        for case_ordinal, case_id in enumerate(spec["cases"])
        for schedule_ordinal, schedule in enumerate(schedules)
    ]


# ---------------------------------------------------------------------------
# M4b: the stochastic-liquidity differential (configs/m4b_differential.yaml)
# ---------------------------------------------------------------------------

M4B_CONFIG_PATH = REPO_ROOT / "configs" / "m4b_differential.yaml"

#: M4b's committed input: the two-seam differential's tiers and the liquidity
#: process's own moment bands. Its own file for the same reason M4a's was —
#: editing an earlier milestone's committed thresholds in place would put two
#: worlds' bars in one blast radius.
M4B_CONFIG: dict = yaml.safe_load(M4B_CONFIG_PATH.read_text(encoding="utf-8"))
M4B_ENCODING: str = str(M4B_CONFIG["world"]["cost_encoding"])


def m4b_liquidity_law():
    """The invented liquidity law M4b's differential runs under."""
    from temper.oracle import liquidity_for

    block = dict(M4B_CONFIG["world"]["liquidity"])
    return liquidity_for(block.pop("model"), **block)


def build_liquidity_env(case, stream_index: int, *, pool: str | None = None):
    """The same env in M4b's world, with **both** seams named and neither default.

    One ``ExecutionEnv`` and one ``step`` loop; the only differences from
    :func:`build_power_law_env` are the injected liquidity stream and the third
    observation coordinate that comes with it. The liquidity pool is the *eval*
    one by default, which is deliberate: the differential is a check on the
    world, and checking it on the streams a graded result is scored over is the
    cheapest way to find out that those streams are what the config says.
    """
    from temper.env import ExecutionEnv, LiquidityStream, impact_for
    from temper.seeding import LIQUIDITY_EVAL_POOL

    seeding = M4B_CONFIG["seeding"]
    return ExecutionEnv(
        case.market,
        case.order_size,
        case.lambda_risk,
        temporary_impact=impact_for(M4B_ENCODING, case.market, case.order_size),
        liquidity=LiquidityStream(
            law=m4b_liquidity_law(), pool=pool or LIQUIDITY_EVAL_POOL
        ),
        root_seed=int(seeding["root_seed"]),
        pool=seeding["pool"],
        stream_index=stream_index,
    )


def liquidity_pairs(tier: str) -> list[DifferentialPair]:
    """Every (case, schedule) cell of M4b's `tier`, addressed as its config says.

    Five schedules wide rather than four: ``static`` is the liquidity world's own
    optimum and ``m4a`` is the power-law one that knows no liquidity, and the
    level shift between them is the milestone's most load-bearing small number.
    """
    spec = M4B_CONFIG["tiers"][tier]
    schedules = M4B_CONFIG["schedules"]
    n_sim = int(spec["n_sim"])
    return [
        DifferentialPair(
            tier=tier,
            case=case_by_id(case_id),
            schedule=schedule,
            n_sim=n_sim,
            stream_index=(
                int(spec["stream_base"])
                + case_ordinal * len(schedules)
                + schedule_ordinal
            ),
            mean_band=float(spec["mean_z_sigmas"]) / math.sqrt(n_sim),
            var_band=float(spec["var_z_sigmas"]) * math.sqrt(2.0 / n_sim),
        )
        for case_ordinal, case_id in enumerate(spec["cases"])
        for schedule_ordinal, schedule in enumerate(schedules)
    ]


# ---------------------------------------------------------------------------
# The task-4 guard case — a case that is deliberately not a golden
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuardCase:
    """A parameter set chosen *here* to reach a branch no golden reaches.

    Structurally distinct from :class:`GoldenCase` on purpose: it has no ``ac``,
    ``twap`` or ``derived`` block, so a test that tried to compare it against a
    vendored number fails with an ``AttributeError`` rather than quietly
    inventing one. The symbol parameters and the grid still come from the
    fixture — there is no second home for a parameter set — but the lambda is
    the config's, and nothing FrontierView exported is pinned by this case.
    """

    case_id: str
    kind: str
    symbol: str
    order_size: float
    lambda_risk: float
    market: Market
    schedule: str

    def __str__(self) -> str:
        return self.case_id


def guard_case() -> GuardCase:
    """The `guard_case` block of the config, resolved against the fixture."""
    spec = M1_CONFIG["guard_case"]
    if spec["kind"] != "guard":
        raise ValueError(
            f"the guard case is labelled {spec['kind']!r}; it is not a golden and "
            "must not be presented as one (see the config's comment)"
        )
    source = case_by_id(spec["params_from"])
    return GuardCase(
        case_id=spec["id"],
        kind=spec["kind"],
        symbol=source.symbol,
        order_size=float(spec["order_size"]),
        lambda_risk=float(spec["lambda_risk"]),
        market=source.market,
        schedule=spec["schedule"],
    )


# ---------------------------------------------------------------------------
# M2: the PPO rediscovery config (configs/m2_ppo.yaml)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def m2_experiment():
    """The committed M2 experiment, resolved once per session.

    Imported inside the function rather than at module scope because
    :mod:`temper.eval.experiment` reaches :mod:`temper.agents.ppo` for its
    hyperparameter dataclass, and that pulls torch onto the import path of every
    test in the suite — including the oracle's, which are meant to prove they do
    not need it.
    """
    from temper.eval.experiment import load_experiment

    return load_experiment(M2_CONFIG_PATH)


# ---------------------------------------------------------------------------
# Seed-pool discipline (M1a task 5)
# ---------------------------------------------------------------------------

#: Every ``(root_seed, pool, stream_index)`` an :class:`ExecutionEnv` resolved
#: during this session, in order. Populated by the autouse fixture below and
#: asserted on by ``tests/test_seed_pool_discipline.py``.
RESOLVED_SEED_ADDRESSES: list[tuple[int, str, int]] = []

#: The same draws, attributed: ``(test module, root_seed, pool, stream_index)``.
#: Attribution is what M2 forced. Until M2 the rule was simply "nothing in the
#: suite may touch `train` or `eval`", which was checkable without knowing who
#: was asking. M2 legitimately trains out of `train` and evaluates out of `eval`,
#: so the rule becomes *which* module may spend *which* pool — and a global
#: counter cannot express that.
SEED_ADDRESS_LEDGER: list[tuple[str, int, str, int]] = []

#: Pools that hold committed results. Only the modules granted them below may
#: spend a stream out of either (constitution invariant 5).
RESERVED_POOLS = frozenset({"train", "eval"})

#: Exactly which pools each test module may open. An allow-list of *pools* per
#: module rather than a list of trusted modules, because the failure worth
#: catching is not a stranger reaching for `train` — it is M2's own evaluation
#: quietly grading on a stream it trained on, which is invariant 5's whole
#: subject and which a blanket "M2 may use the reserved pools" could not see.
#:
#: Everything not named here gets :data:`DEFAULT_POOL_ALLOWANCE`, which is how
#: M1's original property ("the differential draws only from its diagnostic
#: pool") survives M2 landing in the same session.
POOL_ALLOWANCE: dict[str, frozenset[str]] = {
    "test_m2_action_space.py": frozenset({"m2/diagnostic"}),
    "test_m2_grading.py": frozenset({"m2/diagnostic", "eval"}),
    "test_m2_variate.py": frozenset({"m2/diagnostic"}),
    "test_m2_rediscovery.py": frozenset({"m2/diagnostic", "train", "eval"}),
    "test_m3_antithetic.py": frozenset({"m3/diagnostic"}),
    "test_m3_validation.py": frozenset({"m3/diagnostic", "train", "eval"}),
    # M4a: the power-law differential, the certificate's env-side checks, and the
    # four inherited guarantees re-run in the new world. None reports a number.
    "test_m4a_differential.py": frozenset({"m4a/differential"}),
    # `eval` because the fourth guarantee *is* about the eval streams: the
    # open-loop check rolls a policy out on two of them and requires bitwise
    # equality, which is exactly what M2's grading test holds the same pool for.
    "test_m4a_inherited_guarantees.py": frozenset({"m4a/differential", "eval"}),
    "test_m4a_power_law.py": frozenset({"m4a/differential", "eval"}),
    # The Phase-1 bitwise regression retrains a committed M3 seed and regrades
    # it, so it legitimately holds both reserved pools — the same grant, for the
    # same reason, as the two sweep regenerators above.
    "test_m4a_phase1_regression.py": frozenset({"train", "eval"}),
    # M4b's seam regression is M4a's twice over — one committed seed per world —
    # so it holds the same two pools for the same reason. It also builds envs at
    # `train` addresses to assert the *observation shape* and the second seed
    # address without stepping them, which is the cheap half of the same claim.
    "test_m4b_phase1_regression.py": frozenset({"train", "eval"}),
    # M4b's guarantees, and the pools mirror M4a's grant exactly: its own
    # differential pool for the per-step identities, and `eval` because the
    # open-loop successor *is* about the eval streams — pin the liquidity, vary
    # the price, require the trajectory bitwise.
    "test_m4b_inherited_guarantees.py": frozenset({"m4b/differential", "eval"}),
    "test_m4b_differential.py": frozenset({"m4b/differential"}),
    "test_m4b_conditional_grading.py": frozenset({"m4b/differential", "eval"}),
    # The committed policy checkpoint holds M4a's median seed, so verifying it
    # means rolling that policy out on the streams it was *graded* on and
    # nowhere else. `eval` only, and deliberately not `train`: the checkpoint is
    # checked against the grade it earned, never retrained here — a test that
    # needed `train` would be a training run wearing a test's clothes.
    "test_policy_checkpoint.py": frozenset({"eval"}),
    # M5's seam regression is M4b's three times over — one committed seed per
    # world — so it holds the same two reserved pools for the same reason. It also
    # builds envs at `train` addresses to assert the observation *width* and the
    # third seed address without stepping them, and drives its own seam checks
    # from `m5/differential`, which is what that pool is for.
    "test_m5_signal_seam.py": frozenset({"m5/differential", "train", "eval"}),
    # M5 task 3 drives the amended observation guard, which builds envs on its own
    # diagnostic pool and steps them tens of thousands of times. It reports no
    # number and grades nothing, so it holds neither reserved pool.
    "test_m5_observation_guard.py": frozenset({"m5/differential"}),
}

#: The modules that regenerate a committed result and so legitimately hold both
#: reserved pools — one per committed sweep, and `tests/test_seed_pool_discipline.py`
#: asserts the list is exactly this.
SWEEP_REGENERATORS: tuple[str, ...] = (
    "test_m2_rediscovery.py",
    "test_m3_validation.py",
    # M4a's seam regression retrains a committed M3 seed and regrades it, which
    # is the same shape of work as the two above: it reproduces a committed
    # result end to end and therefore needs the pools that result was addressed
    # by. It is not a *new* sweep — it is the invariant-1 check on one.
    "test_m4a_phase1_regression.py",
    # M4b's is the same check across two worlds rather than one, which is what
    # the second seam demands: an env that draws a *second* noise source has to
    # be shown not to have moved either of the milestones that predate it.
    "test_m4b_phase1_regression.py",
    # M5's is that check a third time, and it is the one whose failure mode is
    # worst. The signal is correlated with the price shocks *on purpose*, so a
    # seam that reached into the price generator would not merely move committed
    # numbers — it would manufacture part of the correlation the milestone exists
    # to measure, and every M5 result would be a claim about a world nobody
    # described. Three committed seeds, three worlds, bitwise.
    "test_m5_signal_seam.py",
)

DEFAULT_POOL_ALLOWANCE = frozenset({"m1/differential"})


def pool_allowance(module: str) -> frozenset[str]:
    """The pools `module` may open."""
    return POOL_ALLOWANCE.get(module, DEFAULT_POOL_ALLOWANCE)

#: The module currently running, for the ledger. Set by the hook below; a plain
#: module global because the recorder it feeds is a monkeypatched function and
#: has no other route to pytest's state.
_CURRENT_MODULE = "<session>"


def pytest_runtest_setup(item):
    global _CURRENT_MODULE
    _CURRENT_MODULE = Path(str(item.fspath)).name


@pytest.fixture(scope="session", autouse=True)
def record_env_seed_addresses():
    """Record every pool address the env resolves, and police the reserved pools.

    M0 pins that the pools are disjoint *by construction*. That is a different
    statement from "the harness drew from the pool it was supposed to", which is
    the thing invariant 5 actually needs and which nothing checked: a `pool=`
    default edited in the wrong direction would leave every M0 seeding test green
    while M1's tens of millions of draws quietly burned streams that M2's
    committed results are addressed by.

    The env reaches randomness through exactly one call —
    ``pool_rng(root_seed, pool, stream_index)`` in ``reset`` — so wrapping that
    name records every draw stream the whole suite opens, and the check at
    teardown covers the entire test path rather than the cells one test
    remembered to drive.
    """
    real = execution_env.pool_rng

    def recording(root_seed, pool, index):
        RESOLVED_SEED_ADDRESSES.append((int(root_seed), str(pool), int(index)))
        SEED_ADDRESS_LEDGER.append(
            (_CURRENT_MODULE, int(root_seed), str(pool), int(index))
        )
        return real(root_seed, pool, index)

    execution_env.pool_rng = recording
    try:
        yield RESOLVED_SEED_ADDRESSES
    finally:
        execution_env.pool_rng = real

    trespasses = sorted(
        {
            (module, pool)
            for module, _, pool, _ in SEED_ADDRESS_LEDGER
            if pool not in pool_allowance(module)
        }
    )
    assert not trespasses, (
        "these modules drew env episodes from a pool they are not allowed: "
        + ", ".join(f"{module} -> {pool}" for module, pool in trespasses)
        + ". Diagnostics belong in m1/differential or m2/diagnostic; `train` and "
        "`eval` hold committed M2 results and are separated from each other by "
        "constitution invariant 5."
    )
