"""M4a task 6 — the degradation figure: how far the closed form is, and what learning recovered.

    python tools/m4a_degradation.py --config configs/m4a_power_law.yaml

Excess over the *certified* power-law optimum against lambda, for TWAP, the
vendored Almgren–Chriss schedule and the tangent-derived sinh, across the whole
committed grid — oracle only, so the curves are free — with the agent's ten seeds
drawn individually at the one lambda that was trained.

This is the ROADMAP's "AC-schedule degradation quantified", and it is the
milestone in one picture: the vertical distance from the tangent's curve down to
the seed markers *is* the capture fraction.

Nothing here trains. The oracle half is closed forms and one certified solve per
grid point; the agent half is read out of ``results/m4a_power_law.json``. So the
figure is a view of a committed result and redraws byte-identically from a clean
clone, which is what invariant 1 asks of it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from temper.eval.experiment import LAMBDA_GRIDS, load_experiment  # noqa: E402
from temper.eval.figures import degradation_figure  # noqa: E402
from temper.eval.provenance import Provenance  # noqa: E402
from temper.eval.reference import reference_table  # noqa: E402
from temper.oracle import POWER_LAW_ENCODING  # noqa: E402

#: The schedules the curves are drawn for. ``optimal`` is the reference every
#: excess is measured against, so it is the zero line and is not drawn.
CURVE_SCHEDULES = ("twap", "ac", "tangent")


def build_curves(experiment, document: dict) -> dict:
    """The oracle table and the agent's seeds, in the shape the figure takes."""
    market = experiment.case.market
    order_size = experiment.case.order_size
    grid = LAMBDA_GRIDS[experiment.lambda_grid]
    table = reference_table(market, order_size, grid, encoding=POWER_LAW_ENCODING)

    lambdas = [row.lambda_risk for row in table]
    excess = {
        name: [
            (row.schedules[name].objective - row.optimal.objective)
            / row.optimal.objective
            for row in table
        ]
        for name in CURVE_SCHEDULES
    }

    trained = experiment.lambda_risk
    row = next(r for r in table if r.lambda_risk == trained)
    seeds = [record["grade"]["relative_excess"] for record in document["seeds"]]
    capture = document["summary"]["capture_fraction"]["median"]

    return {
        "lambdas": lambdas,
        "excess": excess,
        "trained_lambda": trained,
        "tangent_at_trained": (
            (row.tangent.objective - row.optimal.objective) / row.optimal.objective
        ),
        "seed_excess": seeds,
        "median_capture": capture,
        "available_advantage_bps": row.available_advantage,
    }


def caption(experiment, document: dict, curves: dict) -> str:
    """The claim, in the figure, in the words the config committed."""
    summary = document["summary"]
    verdict = document["verdict"]
    capture = summary["capture_fraction"]
    optimum = document["reference"]["schedules"]["optimal"]
    advantage = curves["available_advantage_bps"]
    return (
        f"Excess over the certified optimum of FrontierView's 0.6-power world, "
        f"{experiment.case.symbol} X={experiment.case.order_size:,.0f} "
        f"T={experiment.case.market.horizon_hours}h N={experiment.case.market.n_bins}. "
        f"The closed form is derived at the tangent to this world's impact "
        f"function, so it does not solve it: at $\\lambda=10^{{-3.5}}$ that costs "
        f"{advantage:.4f} bps — "
        f"{100 * advantage / optimum['objective_bps']:.2f}% of the objective "
        f"$E+\\lambda V$, {100 * advantage / optimum['expected_bps']:.2f}% of "
        f"expected cost alone. The agent captured a median {capture['median']:.1%} "
        f"of it (IQR {capture['iqr']:.1%}, worst seed {capture['worst']:.1%}), a "
        f"median absolute excess of {verdict['median_excess_bps']:+.5f} bps. "
        f"{len(document['seeds'])} seeds, each drawn. Graded analytically — the "
        f"dispersion here is across seeds, not a sampling interval. Curves are "
        f"clipped at 1e-7: below $\\lambda\\approx10^{{-7}}$ all three schedules "
        f"and the optimum agree to four decimal places in bps, so the flat left "
        f"end is the clip, not a floor in the objective."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "configs" / "m4a_power_law.yaml")
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output stem; defaults to results/m4a_degradation",
    )
    args = parser.parse_args()

    experiment = load_experiment(args.config)
    result = experiment.results_metrics
    if not result.exists():
        print(
            f"{result.relative_to(REPO_ROOT)} does not exist; run the sweep first "
            "(`make m4a`)"
        )
        return 1
    document = json.loads(result.read_text(encoding="utf-8"))

    curves = build_curves(experiment, document)
    provenance = Provenance(**document["provenance"])
    stem = Path(args.out) if args.out else REPO_ROOT / "results" / "m4a_degradation"
    written = degradation_figure(
        stem,
        curves=curves,
        provenance=provenance,
        caption=caption(experiment, document, curves),
        formats=experiment.figure_formats,
    )
    for path in written:
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
