"""M2 task 0 — render the oracle-only reference table, and apply the lambda rule.

Run *before* any training code exists, and re-runnable at any time to check the
brief against the oracle:

    python tools/m2_reference_table.py --config configs/m2_ppo.yaml

Everything printed is a closed form evaluated on the committed case. Nothing
here imports an agent, and nothing here can see a training curve — which is the
entire point of doing it first (constitution invariant 3).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# `python tools/x.py` from a clean clone has no `temper` on its path: pytest
# injects it via pyproject's `pythonpath`, and nothing else does. Rather than
# require callers to export PYTHONPATH — which the Makefile targets did not, so
# `make reference` and `make sweep` were broken from a fresh clone while every
# hand-run invocation worked — the tool puts its own repo root on the path.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from temper.eval.experiment import LAMBDA_GRIDS, load_experiment  # noqa: E402
from temper.eval.reference import select_lambda  # noqa: E402


def _fmt(value: float, places: int = 4) -> str:
    return f"{value:.{places}f}"


def render_objectives(table, rule) -> str:
    """Table 1: the objectives, and the two quantities the rule reads."""
    lines = [
        "| λ | J_twap | J_ac | J_optimal | (J_twap−J_opt)/J_opt | max bin, optimal | κT | rule |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in table:
        admits = rule.admits(row)
        why = "✓" if admits else ("i" if row.twap_gap < rule.min_twap_gap else "ii")
        lines.append(
            f"| {row.lambda_risk:.3e} | {_fmt(row.twap.objective)} | "
            f"{_fmt(row.ac.objective)} | {_fmt(row.optimal.objective)} | "
            f"{row.twap_gap * 100:.2f} % | "
            f"{row.optimal.max_bin_fraction * 100:.1f} % | "
            f"{row.kappa_horizon:.2f} | {why} |"
        )
    return "\n".join(lines)


def render_split(table) -> str:
    """Table 2: each objective split into E, λV and λ(V − floor)."""
    header = ["| λ |"]
    rule = ["| --- |"]
    for name in ("twap", "ac", "optimal"):
        header.append(f" E_{name} | λV_{name} | λ(V−floor)_{name} |")
        rule.append(" ---: | ---: | ---: |")
    lines = ["".join(header), "".join(rule)]
    for row in table:
        cells = [f"| {row.lambda_risk:.3e} |"]
        for name in ("twap", "ac", "optimal"):
            schedule = row.schedules[name]
            cells.append(
                f" {_fmt(schedule.expected)} | {_fmt(schedule.risk)} | "
                f"{_fmt(schedule.excess_risk)} |"
            )
        lines.append("".join(cells))
    return "\n".join(lines)


def main() -> int:
    # The table is markdown destined for a UTF-8 document and is full of λ, σ and
    # ≤. Windows consoles default to cp1252, where printing it raises rather than
    # mangling — so the stream is reconfigured instead of the table being spelled
    # in ASCII.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "m2_ppo.yaml"),
        help="the committed experiment config",
    )
    args = parser.parse_args()

    experiment = load_experiment(args.config)
    case, market = experiment.case, experiment.case.market
    table = experiment.table()

    print(f"# M2 task 0 — reference table ({experiment.path.name})\n")
    print(
        f"Case: {case.symbol}, X = {case.order_size:,.0f} shares, "
        f"T = {market.horizon_hours} h, N = {market.n_bins} bins "
        f"(dt = {market.dt} h), σ_bin = {market.sigma_bin * 1e4:.3f} bps.\n"
    )
    print(
        f"Grid: {experiment.lambda_grid} "
        f"({len(LAMBDA_GRIDS[experiment.lambda_grid])} points). "
        f"Variance floor σ_bin²X² = "
        f"{table[0].variance_floor:.2f} bps² — every schedule pays it.\n"
    )

    print("## Objectives, in bps of notional\n")
    print(render_objectives(table, experiment.rule))
    print(
        "\n`rule`: ✓ admissible; `i` fails the discriminative-testbed condition "
        f"(gap ≥ {experiment.rule.min_twap_gap:.0%}); `ii` fails the "
        f"non-degeneracy condition (largest bin ≤ "
        f"{experiment.rule.max_bin_fraction:.0%}).\n"
    )

    print("## The split: E, λV, and λ(V − floor)\n")
    print(render_split(table))
    print()

    selected = select_lambda(table, experiment.rule)
    committed = experiment.lambda_risk
    agrees = selected.lambda_risk == committed
    print("## Selection\n")
    print(
        f"- Rule selects **λ = {selected.lambda_risk:.6e}** "
        f"(10^{math.log10(selected.lambda_risk):.1f} on the grid); config commits "
        f"{committed:.6e} — {'agree' if agrees else 'DISAGREE'}."
    )
    gap = selected.twap_gap
    print(
        f"- TWAP gap {gap * 100:.2f} %; largest optimal bin "
        f"{selected.optimal.max_bin_fraction * 100:.1f} % of X; κT "
        f"{selected.kappa_horizon:.2f}."
    )
    epsilon = experiment.tolerances.epsilon_gap_fraction
    per_seed = experiment.tolerances.per_seed_gap_fraction
    j_opt = selected.optimal.objective
    print(
        f"- ε = {epsilon:.0%} of the gap = {epsilon * gap * 100:.3f} % of "
        f"J_optimal = {epsilon * gap * j_opt:.5f} bps (median across seeds)."
    )
    print(
        f"- Per-seed floor = {per_seed:.0%} of the gap = "
        f"{per_seed * gap * 100:.3f} % = {per_seed * gap * j_opt:.5f} bps."
    )

    band = experiment.band()
    print(
        f"- Derived trajectory band at ε: λ_min(H) = "
        f"{band.curvature_floor:.4e} bps/share², ΔU = "
        f"{band.delta_objective:.5f} bps ⇒ ‖δ‖₂ ≤ "
        f"{band.bound_shares:,.0f} shares ({band.bound_fraction:.1%} of X)."
    )
    per_seed_band = experiment.band(per_seed)
    print(
        f"- ...and at the per-seed floor: ‖δ‖₂ ≤ "
        f"{per_seed_band.bound_shares:,.0f} shares "
        f"({per_seed_band.bound_fraction:.1%} of X)."
    )
    print(
        "\nThe band is derived, not chosen: the objective is flat near its "
        "minimum by exactly the amount the Hessian says, so an independently "
        "picked trajectory tolerance would be either vacuous or unmeetable for "
        "reasons unrelated to the agent."
    )
    return 0 if agrees else 1


if __name__ == "__main__":
    raise SystemExit(main())
