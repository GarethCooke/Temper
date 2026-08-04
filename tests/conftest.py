"""Golden-fixture loading and the tolerances M0 pre-stated.

The tolerances live here, in code, rather than only in prose: constitution
invariant 3 says thresholds are fixed before the work and changed only by
amending the brief, and a threshold you have to grep the docs for is a threshold
that drifts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from temper.oracle import Market, SymbolParams

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = REPO_ROOT / "tests" / "golden" / "vendor" / "frontierview_goldens.json"

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


@pytest.fixture(scope="session")
def golden_document() -> dict:
    """The whole vendored fixture, provenance block included."""
    return GOLDEN_DOCUMENT


def pytest_generate_tests(metafunc):
    """Parametrise any test taking a `golden_case` over every vendored case."""
    if "golden_case" in metafunc.fixturenames:
        metafunc.parametrize("golden_case", GOLDEN_CASES, ids=str)
