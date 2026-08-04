#!/usr/bin/env python3
"""Export Almgren-Chriss golden fixtures from the FrontierView compute core.

Read-only with respect to FrontierView (constitution §7: zero upstream changes).
The script lives in Temper so the export is versioned alongside the fixture it
produces, but it imports FrontierView's `api` package and writes only to Temper.

Usage (from the FrontierView repo root, using its own interpreter)::

    cd /path/to/FrontierView
    ./venv/bin/python /path/to/Temper/tools/export_frontierview_goldens.py \
        --out /path/to/Temper/tests/golden/vendor/frontierview_goldens.json

Only the standard library and FrontierView's own modules are imported.

What is pinned
--------------
For every case: the AC-optimal and TWAP inventory trajectories, trade lists and
participation rates, the expected cost, the shortfall variance, and the
temporary/permanent/spread decomposition — plus the derived intermediates
(v_hourly, sigma_bin, eta_tilde, kappa) so a differential failure localises to a
single formula rather than to "somewhere in the model".

A `frontier` block pins the (E, V) locus over FrontierView's own lambda grid.
`generate_frontier` rounds its output to 4 dp, which is far coarser than
Temper's 1e-6 relative tolerance, so the frontier is recomputed here from the
same public functions at full precision; the rounded values are carried
alongside as a cross-check of the recomputation.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Case grid
# --------------------------------------------------------------------------
# Horizons are in hours; bin counts follow FrontierView's own `default_n_bins`
# (half-hour slots, minimum 2), so the grid is FrontierView's, not Temper's.

CORE_SYMBOLS = [
    ("AAPL", 100_000.0),
    ("MSFT", 250_000.0),
    ("JPM", 500_000.0),
]
CORE_LAMBDAS = [1e-7, 1e-5, 1e-3]
CORE_HORIZON = 6.5  # one full trading day

# Cases that deliberately exercise the guarded branches of the compute core.
EDGE_CASES = [
    # (tag, symbol, order_size, horizon_hours, lambda_risk)
    # lambda=1e-20 drives lambda*sigma_bin^2/eta_tilde below the 1e-12 clamp in
    # `_ac_kappa`, so the floor binds and kappa is exactly 1e-6.  (The eta_tilde
    # floor in `_linearised_eta` is unreachable for any realistic parameter set
    # and is therefore not pinned here.)
    ("kappa-floor", "AAPL", 100_000.0, 6.5, 1e-20),
    ("near-twap-lambda", "AAPL", 100_000.0, 6.5, 1e-12),
    ("sinh-overflow-asymptote", "AAPL", 100_000.0, 6.5, 1e2),
    ("two-bin-horizon", "MSFT", 250_000.0, 1.0, 1e-5),
    ("participation-floor", "GOOGL", 1.0, 6.5, 1e-5),
    ("high-participation", "SPY", 4_000_000.0, 6.5, 1e-4),
    ("half-bin-rounding", "JPM", 120_000.0, 2.25, 1e-4),
]

FRONTIER_CASE = ("AAPL", 100_000.0, 6.5)


def _git(args: list[str], cwd: Path) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _decompose(mi, schedule, order_size, params, horizon_hours):
    bd = mi.compute_cost_breakdown(schedule, order_size, params, horizon_hours)
    return {
        "expected_cost": bd.total_bps,
        "variance": bd.variance_bps2,
        "decomposition": {
            "temporary": bd.temporary_bps,
            "permanent": bd.permanent_bps,
            "spread": bd.spread_bps,
        },
    }


def _schedule_block(mi, schedule, order_size, params, horizon_hours):
    """Trajectory / trades / participation plus moments for one schedule."""
    v_hourly = params.adv / mi.TRADING_HOURS_PER_DAY
    dt = horizon_hours / len(schedule)

    participation = [p for _, p in schedule]
    trades = [p * v_hourly * dt for p in participation]

    # Inventory *before* each bin, plus the terminal inventory: N+1 points.
    trajectory = [order_size]
    for n in trades:
        trajectory.append(trajectory[-1] - n)

    block = {
        "participation": participation,
        "trades": trades,
        "trajectory": trajectory,
    }
    block.update(_decompose(mi, schedule, order_size, params, horizon_hours))
    return block


def _case(mi, tag, symbol, order_size, horizon_hours, lambda_risk):
    params = mi.SYMBOL_PARAMS[symbol]
    n_bins = mi.default_n_bins(horizon_hours)
    dt = horizon_hours / n_bins
    v_hourly = params.adv / mi.TRADING_HOURS_PER_DAY
    sigma_bin = params.sigma * (dt / mi.TRADING_HOURS_PER_DAY) ** 0.5
    eta_tilde = mi._linearised_eta(
        params.eta, params.sigma, v_hourly, order_size, horizon_hours
    )
    kappa = mi._ac_kappa(lambda_risk, sigma_bin, eta_tilde)

    ac = mi.schedule_ac_linear(n_bins, order_size, horizon_hours, params, lambda_risk)
    twap = mi.schedule_twap(n_bins, order_size, v_hourly, dt)

    return {
        "case_id": f"{tag}:{symbol}:X{order_size:g}:T{horizon_hours:g}:lam{lambda_risk:g}",
        "tag": tag,
        "symbol": symbol,
        "X": order_size,
        "lambda": lambda_risk,
        "horizon_hours": horizon_hours,
        "n_bins": n_bins,
        "dt_hours": dt,
        "params": {
            "adv": params.adv,
            "sigma": params.sigma,
            "half_spread": params.half_spread,
            "eta": params.eta,
            "gamma": params.gamma,
        },
        "derived": {
            "v_hourly": v_hourly,
            "sigma_bin": sigma_bin,
            "eta_tilde": eta_tilde,
            "kappa": kappa,
            "kappa_T": kappa * horizon_hours,
        },
        "ac": _schedule_block(mi, ac, order_size, params, horizon_hours),
        "twap": _schedule_block(mi, twap, order_size, params, horizon_hours),
    }


def _frontier(mi):
    symbol, order_size, horizon_hours = FRONTIER_CASE
    params = mi.SYMBOL_PARAMS[symbol]
    n_bins = 13  # generate_frontier's own default
    rounded = {
        row["lambda_val"]: row
        for row in mi.generate_frontier(order_size, horizon_hours, params, n_bins)
    }

    points = []
    for lam, row in rounded.items():
        sched = mi.schedule_ac_linear(n_bins, order_size, horizon_hours, params, lam)
        cost, var = mi.compute_cost_variance(sched, order_size, params, horizon_hours)
        points.append(
            {
                "lambda": lam,
                "expected_cost": cost,
                "variance": var,
                "expected_cost_rounded_4dp": row["expected_cost_bps"],
                "variance_rounded_4dp": row["variance_bps2"],
            }
        )

    return {
        "symbol": symbol,
        "X": order_size,
        "horizon_hours": horizon_hours,
        "n_bins": n_bins,
        "points": points,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path, help="destination JSON path")
    ap.add_argument(
        "--frontierview-root",
        type=Path,
        default=Path.cwd(),
        help="FrontierView repo root (default: cwd)",
    )
    args = ap.parse_args()

    root = args.frontierview_root.resolve()
    if not (root / "api" / "market_impact.py").exists():
        ap.error(f"{root} does not look like a FrontierView checkout")
    sys.path.insert(0, str(root))

    from api import market_impact as mi  # noqa: E402  (path set above)

    cases = [
        _case(mi, "core", symbol, order_size, CORE_HORIZON, lam)
        for symbol, order_size in CORE_SYMBOLS
        for lam in CORE_LAMBDAS
    ]
    cases += [_case(mi, *spec) for spec in EDGE_CASES]

    doc = {
        "provenance": {
            "source": "FrontierView",
            "commit": _git(["rev-parse", "HEAD"], root),
            "dirty": bool(_git(["status", "--porcelain"], root)),
            "generated": _dt.datetime.now(_dt.timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
            "exporter": "tools/export_frontierview_goldens.py",
            "modules": ["api.market_impact", "api.parameters"],
            "python": sys.version.split()[0],
        },
        "conventions": {
            "cost_units": "bps of notional",
            "variance_units": "bps^2 (execution shortfall)",
            "trajectory_units": "shares, inventory remaining before each bin, N+1 points",
            "trade_units": "shares executed in each bin, N points",
            "participation_units": "shares per hour / v_hourly (dimensionless)",
            "trading_hours_per_day": mi.TRADING_HOURS_PER_DAY,
            "temp_exponent": 0.6,
            "n_bins_rule": "max(2, round(horizon_hours * 2))",
            "sigma_bin": "sigma_daily * sqrt(dt_hours / trading_hours_per_day)",
        },
        "grid": {
            "horizon_hours": CORE_HORIZON,
            "n_bins": mi.default_n_bins(CORE_HORIZON),
            "dt_hours": CORE_HORIZON / mi.default_n_bins(CORE_HORIZON),
        },
        "cases": cases,
        "frontier": _frontier(mi),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
