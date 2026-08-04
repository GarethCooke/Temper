"""The differential: `temper.oracle` against the vendored FrontierView goldens.

Constitution invariant 2 — the oracle is normative, and it earns that status
here. Every vendored case is checked on trajectory, trade list, participation
rates, the derived intermediates, the expected cost, the shortfall variance and
each component of the decomposition, at the tolerances the M0 brief pre-stated.

A failure in this file means Temper's closed forms disagree with FrontierView's
compute core. That is never something to fix by loosening a number here.
"""

from __future__ import annotations

import numpy as np
import pytest

from temper.oracle import (
    ac_kappa,
    ac_trajectory,
    cost_moments,
    default_n_bins,
    linearised_eta,
    participation,
    trades,
    twap_trajectory,
)

from .conftest import MOMENTS_RTOL, TRAJECTORY_RTOL


def _assert_close_to_size(actual, expected, order_size, what):
    """Trajectory / trade-list comparison, relative to the parent order size.

    Absolute scaling matters: trailing inventory levels are legitimately tiny
    (the sinh trajectory ends at zero), and a relative test against a near-zero
    expected value would be a coin flip. The brief states the tolerance against
    X for exactly that reason.
    """
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape, f"{what}: shape {actual.shape} != {expected.shape}"
    worst = float(np.max(np.abs(actual - expected))) / order_size
    assert worst <= TRAJECTORY_RTOL, (
        f"{what}: worst deviation {worst:.3e} of X exceeds {TRAJECTORY_RTOL:.0e}"
    )


def _assert_moments(actual, expected: dict, label: str):
    """Compare E, V and every decomposition component at MOMENTS_RTOL."""
    checks = {
        "expected_cost": (actual.expected, expected["expected_cost"]),
        "variance": (actual.variance, expected["variance"]),
        "temporary": (actual.temporary, expected["decomposition"]["temporary"]),
        "permanent": (actual.permanent, expected["decomposition"]["permanent"]),
        "spread": (actual.spread, expected["decomposition"]["spread"]),
    }
    for name, (got, want) in checks.items():
        assert got == pytest.approx(want, rel=MOMENTS_RTOL, abs=0.0), (
            f"{label} {name}: {got!r} != vendored {want!r} "
            f"(rel {abs(got - want) / abs(want) if want else float('inf'):.3e})"
        )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_provenance_block_present(golden_document):
    """The fixture must say where it came from, or it is not a golden."""
    provenance = golden_document["provenance"]
    assert provenance["source"] == "FrontierView"
    assert len(provenance["commit"]) == 40, "no FrontierView commit sha recorded"
    assert provenance["generated"], "no generation timestamp recorded"
    assert provenance["dirty"] is False, (
        "goldens were exported from a dirty FrontierView working tree; "
        "the commit sha does not identify the code that produced them"
    )


def test_fixture_conventions_match_the_oracle(golden_document):
    """The units contract in the fixture is the one the oracle compiles against."""
    from temper.oracle import TEMP_EXPONENT, TRADING_HOURS_PER_DAY

    conventions = golden_document["conventions"]
    assert conventions["trading_hours_per_day"] == TRADING_HOURS_PER_DAY
    assert conventions["temp_exponent"] == TEMP_EXPONENT

    grid = golden_document["grid"]
    assert default_n_bins(grid["horizon_hours"]) == grid["n_bins"]


def test_every_guarded_branch_is_covered(golden_document):
    """The edge cases the export set out to pin are actually in the fixture."""
    tags = {case["tag"] for case in golden_document["cases"]}
    assert {
        "core",
        "kappa-floor",
        "sinh-overflow-asymptote",
        "two-bin-horizon",
        "participation-floor",
        "high-participation",
        "half-bin-rounding",
    } <= tags


# ---------------------------------------------------------------------------
# Per-case differentials
# ---------------------------------------------------------------------------


def test_derived_quantities_match(golden_case):
    """v_hourly, sigma_bin, eta_tilde and kappa, so a failure localises."""
    market, case = golden_case.market, golden_case
    derived = case.derived

    for name, got, want in (
        ("v_hourly", market.v_hourly, derived["v_hourly"]),
        ("sigma_bin", market.sigma_bin, derived["sigma_bin"]),
        ("eta_tilde", linearised_eta(market, case.order_size), derived["eta_tilde"]),
        (
            "kappa",
            ac_kappa(market, case.order_size, case.lambda_risk),
            derived["kappa"],
        ),
    ):
        assert got == pytest.approx(want, rel=MOMENTS_RTOL, abs=0.0), f"{name} mismatch"


def test_grid_matches(golden_case):
    """The bin count follows from the horizon by the vendored rule."""
    assert default_n_bins(golden_case.market.horizon_hours) == golden_case.market.n_bins


def test_ac_schedule_matches(golden_case):
    """Vendored AC: trajectory, trade list and participation rates."""
    market, case = golden_case.market, golden_case
    trajectory = ac_trajectory(market, case.order_size, case.lambda_risk)

    _assert_close_to_size(trajectory, case.ac["trajectory"], case.order_size, "ac trajectory")
    _assert_close_to_size(
        trades(trajectory, market), case.ac["trades"], case.order_size, "ac trades"
    )
    _assert_close_to_size(
        participation(trajectory, market) * market.dt * market.v_hourly,
        np.asarray(case.ac["participation"]) * market.dt * market.v_hourly,
        case.order_size,
        "ac participation",
    )


def test_ac_moments_match(golden_case):
    """Vendored AC: E, V and the temporary/permanent/spread decomposition."""
    market, case = golden_case.market, golden_case
    trajectory = ac_trajectory(market, case.order_size, case.lambda_risk)
    _assert_moments(cost_moments(trajectory, market), case.ac, f"{case.case_id} ac")


def test_twap_schedule_matches(golden_case):
    """TWAP under the same dynamics: trajectory and trade list."""
    market, case = golden_case.market, golden_case
    trajectory = twap_trajectory(market, case.order_size)

    _assert_close_to_size(
        trajectory, case.twap["trajectory"], case.order_size, "twap trajectory"
    )
    _assert_close_to_size(
        trades(trajectory, market), case.twap["trades"], case.order_size, "twap trades"
    )


def test_twap_moments_match(golden_case):
    """TWAP under the same dynamics: E, V and the decomposition."""
    market, case = golden_case.market, golden_case
    trajectory = twap_trajectory(market, case.order_size)
    _assert_moments(cost_moments(trajectory, market), case.twap, f"{case.case_id} twap")


# ---------------------------------------------------------------------------
# The frontier
# ---------------------------------------------------------------------------


def test_frontier_matches(golden_document):
    """Every (E, V) point on FrontierView's own lambda sweep."""
    from temper.oracle import Market, SymbolParams, ac_frontier_point

    block = golden_document["frontier"]
    symbol_case = next(
        case for case in golden_document["cases"] if case["symbol"] == block["symbol"]
    )
    market = Market(
        params=SymbolParams(**symbol_case["params"]),
        horizon_hours=block["horizon_hours"],
        n_bins=block["n_bins"],
    )

    for point in block["points"]:
        got = ac_frontier_point(market, block["X"], point["lambda"])
        assert got.expected_cost == pytest.approx(
            point["expected_cost"], rel=MOMENTS_RTOL, abs=0.0
        ), f"frontier E at lambda={point['lambda']:g}"
        assert got.variance == pytest.approx(
            point["variance"], rel=MOMENTS_RTOL, abs=0.0
        ), f"frontier V at lambda={point['lambda']:g}"


def test_frontier_recomputation_agrees_with_vendor_rounding(golden_document):
    """The full-precision frontier rounds back to what FrontierView published.

    `generate_frontier` rounds to 4 dp, far coarser than MOMENTS_RTOL, so the
    fixture recomputes the locus at full precision. This checks the
    recomputation really is the same locus and not a differently-parameterised
    one that happens to be smooth.
    """
    for point in golden_document["frontier"]["points"]:
        assert round(point["expected_cost"], 4) == point["expected_cost_rounded_4dp"]
        assert round(point["variance"], 4) == point["variance_rounded_4dp"]
