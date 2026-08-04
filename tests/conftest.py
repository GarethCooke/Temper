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
from pathlib import Path

import pytest
import yaml

from temper.env import ExecutionEnv
from temper.oracle import Market, SymbolParams

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = REPO_ROOT / "tests" / "golden" / "vendor" / "frontierview_goldens.json"
M1_CONFIG_PATH = REPO_ROOT / "configs" / "m1_differential.yaml"

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


def build_env(case: GoldenCase, stream_index: int) -> ExecutionEnv:
    """The env for a golden case, seeded from the config's diagnostic pool."""
    seeding = M1_CONFIG["seeding"]
    return ExecutionEnv(
        case.market,
        case.order_size,
        case.lambda_risk,
        root_seed=int(seeding["root_seed"]),
        pool=seeding["pool"],
        stream_index=stream_index,
    )
