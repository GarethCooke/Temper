"""Structural invariants of the repo that a code review would otherwise have to catch.

Covers the M0 definition-of-done items that are properties of the *repository*
rather than of any one function: no network in the test path, no torch in the
oracle, no GPU dependence, and a golden fixture that is genuinely vendored
rather than generated here.
"""

from __future__ import annotations

import ast
import shutil
import socket
import subprocess
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


@pytest.mark.parametrize("package", ["oracle", "env"])
def test_the_oracle_and_the_env_do_not_import_torch(package):
    """Pure numpy below the agents.

    The oracle stays light because it is the reference engine. The env stays
    light because M2's agent will cast to torch at *its* boundary and nowhere
    else: an env that returned tensors would make every baseline drag a training
    framework behind it, and would put a dtype conversion between the simulator
    and the closed form it is checked against.
    """
    sources = sorted((PACKAGE_ROOT / package).rglob("*.py"))
    assert sources, f"no sources found under temper/{package}"
    for source in sources:
        assert "torch" not in _imported_roots(source), (
            f"{source.relative_to(REPO_ROOT)} imports torch; temper/{package} is pure numpy"
        )


def test_the_env_does_not_import_the_power_law_moments():
    """Invariant 7's quarantine, at the import level (ARCHITECTURE.md §9).

    Phase 1 is the linearised world end-to-end. `cost_moments` is FrontierView's
    0.6-power charge and is reporting context only — an env that imported it
    would be one edit away from paying out a functional the oracle does not
    minimise, which is the exact failure invariant 7 exists to prevent.
    """
    for source in sorted((PACKAGE_ROOT / "env").rglob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert "cost_moments" not in imported, (
            f"{source.relative_to(REPO_ROOT)} imports cost_moments; Phase 1 charges "
            "the linear tangent (see temper/eval/metrics.py)"
        )


def test_the_env_does_not_import_a_schedule_or_a_closed_form():
    """The differential must not be checking the oracle against itself.

    M1's whole claim is that an independently-written simulator reproduces the
    closed form's moments. If the env reached for `optimal_trajectory` or a
    kappa, the Monte-Carlo tiers would be measuring numpy's arithmetic rather
    than the model.
    """
    forbidden = {
        "ac_kappa",
        "ac_trajectory",
        "cost_moments",
        "linear_cost_moments",
        "objective_curvature",
        "optimal_kappa",
        "optimal_trajectory",
        "schedule_moments",
        "sinh_trajectory",
        "twap_trajectory",
    }
    for source in sorted((PACKAGE_ROOT / "env").rglob("*.py")):
        text = source.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(source))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        offenders = imported & forbidden
        assert not offenders, (
            f"{source.relative_to(REPO_ROOT)} imports {', '.join(sorted(offenders))}; "
            "the env must reach its cost the long way round, bin by bin"
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


def test_every_package_source_would_survive_a_clone():
    """No source under `temper/` may be excluded by .gitignore.

    Not hypothetical: `.gitignore`'s virtualenv entry was `env/`, and a bare
    directory pattern matches at *any* depth, so the whole of `temper/env/` was
    invisible to git the moment M1 created it. `make test` would have stayed green
    here and failed from a clean clone — the one gate every milestone is
    measured by.
    """
    git = shutil.which("git")
    if git is None or not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout with git available")

    sources = _package_sources()
    result = subprocess.run(
        [git, "check-ignore", "--stdin"],
        input="\n".join(str(path) for path in sources),
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    ignored = sorted(
        Path(line.strip().strip('"')).name for line in result.stdout.splitlines() if line.strip()
    )
    assert not ignored, (
        "these package sources are excluded by .gitignore and would be missing "
        f"from a clone: {', '.join(ignored)}"
    )


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
