"""M4b task 6 — the adaptivity figure, redrawn from committed artefacts.

Two committed files in, one figure out, and **nothing computed here**. The rungs
come off ``results/m4b_liquidity.json`` (the trained sweep) and the value-of-sight
curve off ``results/m4b_reference.json`` (task 0's oracle table), so the picture is
a *view* of results rather than a second route to them — which is what lets it
redraw byte-identically from a clean clone without a training run.

    python tools/m4b_adaptivity.py --config configs/m4b_liquidity.yaml

The caption is assembled here rather than written into the figure module, for the
same reason M4a's is: it carries numbers, and a caption with numbers in it is a
claim. Three of them are non-negotiable and appear every time this figure is
drawn — the **denominator**, the fact that ``sigma_L`` is **invented**, and the
**bracket width** on a reference that is converged rather than certified.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from temper.eval.experiment import load_experiment  # noqa: E402
from temper.eval.figures import adaptivity_figure  # noqa: E402
from temper.eval.provenance import Provenance  # noqa: E402


def build_rungs(document: dict) -> dict:
    """The left panel's four levels and its ten seeds, read off the sweep."""
    reference = document["reference"]
    return {
        "sigma_log": document["liquidity"]["sigma_log"],
        "m4a": reference["schedules"]["m4a"]["objective_bps"],
        "static": reference["schedules"]["static"]["objective_bps"],
        "adaptive": reference["adaptive_bps"],
        "clairvoyant": reference["clairvoyant"]["value_bps"],
        "clairvoyant_half_width": reference["clairvoyant"]["half_width_bps"],
        "seeds": [record["grade"]["objective_bps"] for record in document["seeds"]],
    }


def build_curve(table: dict) -> dict:
    """The right panel: the oracle's value of sight at each reported sigma_L."""
    rows = sorted(table["value_of_sight"], key=lambda row: row["sigma_log"])
    return {
        "sigma_log": [row["sigma_log"] for row in rows],
        "advantage_bps": [row["adaptive_advantage_bps"] for row in rows],
        "level_shift_bps": [row["level_shift_bps"] for row in rows],
    }


#: Characters per caption line. Measured against the 11.6-inch canvas at 7.6 pt:
#: matplotlib silently draws text past the figure edge, so the width is bounded
#: where the string is built rather than trusted to fit. The house note records a
#: caption running off the canvas as a real failure that reached a committed
#: artefact once already.
CAPTION_WIDTH = 168


def caption(experiment, document: dict, rungs: dict) -> str:
    """The three things this figure may never be shown without, hard-wrapped."""
    reference = document["reference"]
    summary = document["summary"]
    verdict = document["verdict"]
    control = document["shuffled_control"]
    advantage = reference["adaptive_advantage_bps"]
    capture = summary["capture_fraction"]

    lines = [
        f"The liquidity model is INVENTED — a one-parameter i.i.d. lognormal "
        f"multiplier on v_hourly, E[L] = 1, sigma_L = {rungs['sigma_log']:g}, "
        f"Temper's own and not FrontierView's. The claim is what seeing THAT "
        f"process is worth, not what seeing real market liquidity would be.",
        f"Denominator: the ADAPTIVE advantage J_static* - J_DP = "
        f"{advantage:.5f} bps ({reference['advantage_fraction']:.2%} of J_DP). "
        f"NOT J_M4a - J_DP: {verdict['level_shift_bps']:.5f} bps of that "
        f"({verdict['level_shift_fraction_of_advantage']:.1%} of the advantage) is a "
        f"level shift any static solver picks up free by re-solving.",
        f"Capture fraction median {capture['median']:.3f} "
        f"(IQR {capture['q1']:.3f}-{capture['q3']:.3f}, worst seed "
        f"{capture['worst']:.3f}); median excess over J_DP "
        f"{verdict['median_excess_bps']:.5f} bps. Liquidity-shuffled control: "
        f"median capture {control['median']:.3f} against a "
        f"{verdict['shuffled_capture_bar']:.2f} bar.",
        f"J_DP is CONVERGED AND BRACKETED, not certified: the clairvoyant "
        f"relaxation and the DP's own greedy policy bracket it to "
        f"{reference['bracket_fraction']:.1%} of the advantage. Grading is "
        f"E[cost | L] — exact given the liquidity path, no price sampling — "
        f"averaged over {document['seeds'][0]['grade']['paths']:,} held-out paths, "
        f"paired against the static optimum's closed form.",
    ]
    return "\n".join(
        textwrap.fill(line, width=CAPTION_WIDTH, subsequent_indent="  ")
        for line in lines
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "configs" / "m4b_liquidity.yaml")
    )
    parser.add_argument(
        "--table", default=str(REPO_ROOT / "results" / "m4b_reference.json")
    )
    args = parser.parse_args()

    experiment = load_experiment(args.config)
    document = json.loads(experiment.results_metrics.read_text(encoding="utf-8"))
    table = json.loads(Path(args.table).read_text(encoding="utf-8"))

    rungs = build_rungs(document)
    curve = build_curve(table)
    stamp = document["provenance"]
    written = adaptivity_figure(
        experiment.results_figure,
        rungs=rungs,
        curve=curve,
        provenance=Provenance(
            config=stamp["config"],
            config_sha256=stamp["config_sha256"],
            git_rev=stamp["git_rev"],
            git_dirty=stamp["git_dirty"],
            python=stamp["python"],
        ),
        caption=caption(experiment, document, rungs),
        formats=experiment.figure_formats,
    )
    for path in written:
        # `relative_to` raises for anything outside the tree, and a driver that
        # dies while *reporting* where it wrote a file has thrown away the file's
        # only mention. The figure had already been written when this fired.
        try:
            shown = path.resolve().relative_to(REPO_ROOT)
        except ValueError:
            shown = path
        print(f"wrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
