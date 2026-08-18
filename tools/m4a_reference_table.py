"""M4a task 0 — the power-law reference table, and the three gates it decides.

Oracle only. No agent exists when this runs, which is the entire point
(constitution invariant 3): a threshold derived from closed forms and one
certified solve, *before* any training code, cannot have been chosen to fit a
training curve.

    python tools/m4a_reference_table.py --config configs/m4a_power_law.yaml

Three gates, all readable off the oracle in minutes, and all three must be green
or the milestone re-shapes:

1. **λ agreement.** M2's selection rule — smallest λ whose TWAP gap clears 20 %
   and whose optimum's largest bin is inside 50 % — applied to *each* world's
   table, must pick the same λ. If it does not, M4a's point is not comparable to
   the two committed milestones it is being put beside, and the resolution is a
   decision rather than a rescue.
2. **The advantage is worth an evening.** The available advantage
   ``J_opt(tangent) − J_pow*`` must be at least 1 % of ``J_pow*`` at the selected
   λ. Below that the honest move is to report it and let M4b lead.
3. **The testbed discriminates in trajectory space too.** The band implied by the
   median tolerance must sit comfortably inside the distance the Almgren–Chriss
   schedule already is from the optimum — otherwise "the agent got inside the
   band" is a claim the closed form also satisfies.

Exit status is 0 only if all three are green *and* the config's committed λ is
the one the rule selects.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# `python tools/x.py` from a clean clone has no `temper` on its path.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from temper.eval.experiment import LAMBDA_GRIDS, load_experiment  # noqa: E402
from temper.eval.reference import (  # noqa: E402
    reference_table,
    select_lambda,
    trajectory_band,
    trajectory_deviation,
)
from temper.oracle import (  # noqa: E402
    ENCODINGS,
    LINEAR_ENCODING,
    POWER_LAW_ENCODING,
    kkt_residual,
    power_law_charge,
    trades,
)

#: The gate on the available advantage, as a fraction of ``J_pow*``.
MIN_ADVANTAGE_FRACTION = 0.01

#: How much clear air the AC separation must have over the median band. The brief
#: predicted 3.6; "comfortably inside" is read as at least double.
MIN_SEPARATION_RATIO = 2.0


def _fmt(value: float, places: int = 4) -> str:
    return f"{value:.{places}f}"


def render_table(table, rule) -> str:
    """The power-law analogue of M2 task 0's table 1."""
    lines = [
        "| λ | J_twap | J_ac | J_opt(tangent) | J_pow* | TWAP gap | available advantage "
        "| avail / gap | max bin, pow* | rule |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in table:
        admits = rule.admits(row)
        why = "✓" if admits else ("i" if row.twap_gap < rule.min_twap_gap else "ii")
        lines.append(
            f"| {row.lambda_risk:.3e} | {_fmt(row.twap.objective)} | "
            f"{_fmt(row.ac.objective)} | {_fmt(row.tangent.objective)} | "
            f"{_fmt(row.optimal.objective)} | {row.twap_gap:.4f} | "
            f"{row.advantage_fraction * 100:.3f} % | "
            f"{row.advantage_fraction / row.twap_gap:.4f} | "
            f"{row.optimal.max_bin_fraction * 100:.1f} % | {why} |"
        )
    return "\n".join(lines)


def render_certificate(row, market, order_size) -> str:
    """The solve's own numbers at the selected λ, beside the table it produced."""
    optimum = row.optimal.trajectory
    charge = power_law_charge(market, order_size)
    residual = kkt_residual(optimum, market, order_size, row.lambda_risk, charge)
    weights = trades(optimum, market) / order_size
    return "\n".join(
        [
            f"- Relative KKT residual **{residual:.3e}** (bar 1e-12); "
            f"smallest bin weight {float(np.min(weights)):.4f}, largest "
            f"{float(np.max(weights)):.4f} — interior, so the equal-marginal "
            "condition is sufficient.",
        ]
    )


def main() -> int:
    # The table is markdown destined for a UTF-8 document and is full of the same
    # Greek and arrows M2's is. Windows consoles default to cp1252, where printing
    # it raises rather than mangling, so the stream is reconfigured instead of the
    # table being spelled in ASCII.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "configs" / "m4a_power_law.yaml")
    )
    args = parser.parse_args()

    experiment = load_experiment(args.config)
    market = experiment.case.market
    order_size = experiment.case.order_size
    grid = LAMBDA_GRIDS[experiment.lambda_grid]
    rule = experiment.rule

    print(
        f"# M4a task 0 — {experiment.case.symbol}, X = {order_size:,.0f}, "
        f"T = {market.horizon_hours} h, N = {market.n_bins}; the "
        f"{experiment.lambda_grid} λ grid, {len(grid)} points\n"
    )

    tables = {
        encoding: reference_table(market, order_size, grid, encoding=encoding)
        for encoding in ENCODINGS
    }
    power = tables[POWER_LAW_ENCODING]

    print("## The power-law table\n")
    print(render_table(power, rule))
    print()
    print(
        "`J_opt(tangent)` is `optimal_trajectory` — the exact minimiser of the "
        "*tangent's* objective, which is what the vendored closed form produces "
        "— priced under the power law. `J_pow*` is the certified optimum of the "
        "power-law world. Their difference is the available advantage: what "
        "there was to be beaten.\n"
    )

    # --- gate 1: the rule selects the same λ in both worlds ------------------
    selections = {
        encoding: select_lambda(table, rule).lambda_risk
        for encoding, table in tables.items()
    }
    agree = len(set(selections.values())) == 1
    selected = select_lambda(power, rule)
    committed = experiment.lambda_risk

    print("## Gate 1 — both encodings' tables select the same λ\n")
    for encoding, lam in sorted(selections.items()):
        print(f"- `{encoding}` selects **{lam:.6e}** (10^{math.log10(lam):.1f}).")
    print(f"- {'**GREEN** — they agree.' if agree else '**RED** — they disagree.'}")
    print(
        f"- The config commits {committed:.6e} — "
        f"{'agree' if committed == selected.lambda_risk else '**DISAGREE**'}.\n"
    )

    # --- gate 2: the advantage is worth an evening --------------------------
    advantage = selected.available_advantage
    fraction = selected.advantage_fraction
    worth_it = fraction >= MIN_ADVANTAGE_FRACTION

    print("## Gate 2 — the available advantage is ≥ 1 % of J_pow*\n")
    print(
        f"- `J_opt(tangent) − J_pow*` = **{advantage:.5f} bps** = "
        f"**{fraction:.3%}** of `J_pow*` = {selected.optimal.objective:.4f} bps."
    )
    print(
        f"- {'**GREEN**' if worth_it else '**RED**'} against the "
        f"{MIN_ADVANTAGE_FRACTION:.0%} bar."
    )
    epsilon = experiment.tolerances.epsilon_fraction
    per_seed = experiment.tolerances.per_seed_fraction
    print(
        f"- Median bar: {epsilon:.0%} of it = **{epsilon * advantage:.5f} bps** "
        f"⇒ capture fraction c ≥ {1 - epsilon:.2f}. Per seed: "
        f"{per_seed:.0%} = {per_seed * advantage:.5f} bps ⇒ c ≥ {1 - per_seed:.2f}."
    )
    linear_row = next(
        row for row in tables[LINEAR_ENCODING] if row.lambda_risk == committed
    )
    m3_epsilon = 0.05 * (linear_row.twap.objective - linear_row.optimal.objective)
    power_epsilon = 0.05 * (selected.twap.objective - selected.optimal.objective)
    print(
        f"- Why the denominator moved: 5 % of this λ's TWAP gap is "
        f"{m3_epsilon:.5f} bps as M3 computed it (linear encoding) and "
        f"{power_epsilon:.5f} bps re-derived here — "
        f"{m3_epsilon / advantage:.1f}×–{power_epsilon / advantage:.1f}× the "
        "*entire* available advantage. An agent graded to M3's ε in this world "
        "would pass while capturing none of the mis-specification.\n"
    )

    # --- gate 3: the band is inside the AC separation -----------------------
    band = trajectory_band(
        market, order_size, committed, epsilon * advantage, encoding=POWER_LAW_ENCODING
    )
    separation = trajectory_deviation(
        selected.tangent.trajectory, selected.optimal.trajectory
    )
    ratio = separation / band.bound_shares
    discriminates = ratio >= MIN_SEPARATION_RATIO

    print("## Gate 3 — the band discriminates in trajectory space\n")
    print(
        f"- λ_min(H) at `x*` = {band.curvature_floor:.4e} bps/share² "
        f"(**local** — the Hessian is not constant in x under the power law)."
    )
    print(
        f"- Median tolerance ⇒ ‖δ‖₂ ≤ **{band.bound_shares:,.0f} shares** "
        f"({band.bound_fraction:.2%} of X)."
    )
    print(
        f"- The tangent-derived schedule sits **{separation:,.0f} shares** from "
        f"`x*` — {separation / order_size:.1%} of the parent order — "
        f"a factor of **{ratio:.2f}**."
    )
    print(
        f"- {'**GREEN**' if discriminates else '**RED**'} against the "
        f"{MIN_SEPARATION_RATIO:.0f}× bar."
    )
    tangent_bins = trades(selected.tangent.trajectory, market) / order_size
    optimum_bins = trades(selected.optimal.trajectory, market) / order_size
    print(
        f"- In bin terms: the tangent front-loads {tangent_bins[0]:.1%} of the "
        f"order in bin 0 where the power-law optimum front-loads "
        f"{optimum_bins[0]:.1%}. The power law charges `Σ n^1.6` where the "
        "tangent charges `Σ n²`, so concentrating is cheaper than the closed "
        "form believes and the correct schedule is faster.\n"
    )

    print("## The certified optimum at the selected λ\n")
    print(render_certificate(selected, market, order_size))
    print()

    green = agree and worth_it and discriminates and committed == selected.lambda_risk
    print(
        f"## Verdict — {'all three gates GREEN' if green else 'NOT all gates green'}\n"
    )
    if not green:
        print(
            "The brief's instruction when a gate is red is to report it and "
            "re-shape the milestone — leading with M4b if the advantage is the "
            "problem — not to relax the bar."
        )
    return 0 if green else 1


if __name__ == "__main__":
    raise SystemExit(main())
