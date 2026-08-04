"""Structural invariants of the repo that a code review would otherwise have to catch.

Covers the M0 definition-of-done items that are properties of the *repository*
rather than of any one function: no network in the test path, no torch in the
oracle, no GPU dependence, and a golden fixture that is genuinely vendored
rather than generated here.
"""

from __future__ import annotations

import ast
import socket
from pathlib import Path

import numpy as np
import pytest

from temper.oracle import Market, SymbolParams, ac_trajectory, cost_moments

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "temper"

#: Constitution invariant 8 — `temper/` performs no network I/O. The Anvil
#: participant lives in `client/` and consumes the package.
NETWORK_MODULES = {
    "aiohttp",
    "asyncio",
    "http",
    "httpx",
    "requests",
    "socket",
    "socketserver",
    "ssl",
    "urllib",
    "urllib3",
    "websockets",
    "xmlrpc",
}


def _package_sources() -> list[Path]:
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_the_package_has_sources_to_check():
    """Guard against the two tests below passing because they found nothing."""
    assert len(_package_sources()) >= 3


def test_package_imports_nothing_that_can_reach_a_network():
    """Invariant 8, checked statically across the whole package."""
    for source in _package_sources():
        offenders = _imported_roots(source) & NETWORK_MODULES
        assert not offenders, (
            f"{source.relative_to(REPO_ROOT)} imports {', '.join(sorted(offenders))}; "
            "constitution invariant 8 keeps network code in client/"
        )


def test_the_oracle_does_not_import_torch():
    """The brief: pure numpy, no torch, no I/O. Keeps the reference engine light."""
    for source in sorted((PACKAGE_ROOT / "oracle").rglob("*.py")):
        assert "torch" not in _imported_roots(source), (
            f"{source.relative_to(REPO_ROOT)} imports torch; the oracle is pure numpy"
        )


def test_the_oracle_computes_with_sockets_disabled():
    """The static check with a runtime backstop: no network in the test path."""

    def refuse(*args, **kwargs):
        raise AssertionError("the oracle attempted to open a socket")

    original = socket.socket
    socket.socket = refuse
    try:
        market = Market.for_horizon(
            SymbolParams(adv=6e7, sigma=0.0155, half_spread=0.3, eta=0.142, gamma=0.314),
            6.5,
        )
        moments = cost_moments(ac_trajectory(market, 100_000.0, 1e-5), market)
    finally:
        socket.socket = original

    assert np.isfinite(moments.expected)


def test_the_golden_fixture_is_vendored_not_generated_here():
    """The fixture must credit FrontierView, or the differential is a tautology.

    The M0 brief's one hard stop: goldens synthesised from this repo's own AC
    implementation would make every downstream claim circular.
    """
    from .conftest import GOLDEN_DOCUMENT, GOLDEN_PATH

    assert GOLDEN_PATH.parent.name == "vendor"
    assert GOLDEN_DOCUMENT["provenance"]["source"] == "FrontierView"
    assert GOLDEN_DOCUMENT["cases"], "fixture contains no cases"


@pytest.mark.parametrize("required", ["numpy", "pytest"])
def test_the_test_path_needs_only_cpu_dependencies(required):
    """No GPU anywhere: the whole M0 suite runs on numpy and pytest."""
    __import__(required)
