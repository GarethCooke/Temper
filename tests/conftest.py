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

from temper.env import ExecutionEnv, execution_env
from temper.oracle import Market, SymbolParams

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = REPO_ROOT / "tests" / "golden" / "vendor" / "frontierview_goldens.json"
M1_CONFIG_PATH = REPO_ROOT / "configs" / "m1_differential.yaml"
M2_CONFIG_PATH = REPO_ROOT / "configs" / "m2_ppo.yaml"

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
}

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
