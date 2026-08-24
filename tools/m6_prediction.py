"""M6's figure — the live leg redrawn from the five committed run artefacts.

Five committed files in, one figure out, and **nothing recomputed here**. The
per-bin series come off each run's own ``realised``/``predicted`` bins and the
tier rows off each run's ``measurement`` block, so the picture is a *view* of
results rather than a second route to them — which is what lets it redraw
byte-identically from a clean clone with no venue, no server and no network.

    python tools/m6_prediction.py

There is no ``--config``. M6 has two of them (``m6_anvil.yaml`` for the four
local runs, ``m6_anvil_deployment.yaml`` for the public one), and neither is the
input to this figure: the artefacts are. ``--out`` exists only so the drawing
path can be exercised without overwriting the committed PNG.

The caption is assembled here rather than in the figure module, for the reason
M4a's and M4b's are: it carries numbers, and a caption with numbers in it is a
claim. Four of them are non-negotiable and appear every time this figure is
drawn — the **three tiers** and what each one is worth, the fact that every run
here is a **demo and not an evaluation** (``ARCHITECTURE.md`` §7), that the
deployment's number is **withheld rather than taken**, and that all three tiers
filled the whole parent and attributed every share.
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

from client.book import slippage_bps  # noqa: E402
from temper.eval.figures import prediction_figure  # noqa: E402
from temper.eval.provenance import Provenance  # noqa: E402

#: The five committed runs. Named here rather than read out of the two configs
#: because the artefacts *are* this tool's input: a figure that resolved its
#: sources through a config could be pointed at a different set of results
#: without the diff showing it.
ARTEFACTS = {
    "ladder": "results/m6_anvil_ladder.json",
    "thin": "results/m6_anvil_thin.json",
    "wide": "results/m6_anvil_wide.json",
    "feeder": "results/m6_anvil_feeder.json",
    "deployment": "results/m6_anvil_deployment.json",
}

#: The runs that build their own book and are therefore predictable in closed
#: form. Only these three can appear on the per-bin panel — the other two have
#: ``predicted: null`` and, as their own artefacts say, there cannot be one.
LADDER_RUNS = ("ladder", "thin", "wide")

#: The run whose provenance stamps the figure. Four of the five ran from
#: ``m6_anvil.yaml`` at one revision and the deployment ran from its own config
#: at another, so one footer cannot cover all five; the headline run's stamp goes
#: in the footer and every source is named with its revision in the caption.
STAMP_RUN = "ladder"

#: The feeder's four seed-pinned attempts, in bps. **Brief prose, not artefact
#: data** — ``results/m6_anvil_feeder.json`` records one run, and the spread was
#: measured across four server restarts that were never merged into a file. So it
#: is stated in the caption and may not be drawn: an axis is for numbers a reader
#: can go and check, and this one is only checkable against
#: ``docs/briefs/M6-anvil-live-leg.md``.
FEEDER_ATTEMPTS_BPS = (12.3820, 12.3752, 12.3752, 12.3587)

def max_abs_residual(ladders: dict) -> tuple[float, int]:
    """``(worst |realised - predicted|, how many comparisons)`` over the strip.

    Differenced from the same two series the figure draws, so the number the
    caption carries is the number the strip shows. The tests assert it against
    a pre-stated bound rather than trusting it: an unchecked number in a
    caption is a claim nobody has read, which is the defect class the house
    note names about a driver's last line.
    """
    worst = 0.0
    comparisons = 0
    for row in ladders["runs"]:
        for realised, predicted in zip(row["realised_bps"], row["predicted_bps"]):
            worst = max(worst, abs(realised - predicted))
            comparisons += 1
    return worst, comparisons


#: Characters per line of the one annotation on the per-bin panel, which
#: shares the caption's problem and needs its own bound: it is drawn inside
#: the left panel rather than across the canvas, so it gets less room.
ANNOTATION_WIDTH = 100

#: Characters per caption line. The same 11.6-inch canvas at the same 7.6 pt as
#: M4b's, so its measured bound transfers unchanged: matplotlib draws text
#: straight past the figure edge without a word of complaint, and the house note
#: records a caption doing exactly that on a committed artefact.
CAPTION_WIDTH = 168


def _cumulative_bps(document: dict, source: str) -> list[float]:
    """Arrival slippage in bps of everything filled up to and including each bin.

    Cumulative rather than per bin: a per-bin VWAP says nothing about the parent
    order, and the last point of this series *is* the run's reported number — so
    the panel ends where ``measurement.realised_slippage_bps`` says it should,
    which is a check the reader can make with a ruler.

    A bin before anything has filled has no VWAP, so it is ``nan`` and leaves a
    gap. None of the committed runs has one; a run that did would show the hole
    rather than a fabricated zero.
    """
    arrival = float(document["measurement"]["arrival_mid_ticks"])
    side = document["order"]["side"]
    notional = 0.0
    filled = 0
    series: list[float] = []
    for record in document[source]["bins"]:
        notional += float(record["notional_tick_shares"])
        filled += int(record["filled"])
        series.append(
            slippage_bps(arrival, notional / filled, side) if filled else float("nan")
        )
    return series


def build_ladders(documents: dict[str, dict]) -> dict:
    """The left panel: three predictable runs, predicted against realised, per bin."""
    runs = []
    for name in LADDER_RUNS:
        document = documents[name]
        ladder = document["ladder"]
        runs.append(
            {
                "run": name,
                "label": (
                    f"{ladder['name']} — {ladder['depth_per_side']:,}/side, "
                    f"{ladder['half_spread_ticks']}-tick half-spread"
                ),
                "predicted_bps": _cumulative_bps(document, "predicted"),
                "realised_bps": _cumulative_bps(document, "realised"),
            }
        )

    thin = documents["thin"]
    first = thin["realised"]["bins"][0]
    short = int(first["unfilled"])
    would_have_held = int(first["inventory_before"]) - int(first["requested"])
    return {
        "bins": [record["bin"] + 1 for record in thin["realised"]["bins"]],
        "runs": runs,
        "highlight": {
            "run": "thin",
            "bin": first["bin"] + 1,
            "text": textwrap.fill(
                f"bin {first['bin'] + 1}, thin ladder: asked {first['requested']} "
                f"into {thin['ladder']['depth_per_side']} of depth. Swept all "
                f"{len(first['fills'])} levels, filled {first['filled']}, {short} "
                f"unfilled — and ACCEPTED IS NOT FILLED: Anvil has no market orders, "
                f"so the remainder was cancelled and the shortfall carried. Bin "
                f"{first['bin'] + 2} opened on {first['inventory_after']} shares "
                f"where a full fill would have left {would_have_held} — a state "
                f"ExecutionEnv has never produced.",
                width=ANNOTATION_WIDTH,
            ),
        },
    }


#: What each tier is, in the words the panel has room for. The full statement of
#: each is in the caption; these are the labels that keep the three groups
#: distinguishable at a glance.
TIER_CAPTIONS = {
    1: "Tier 1 — client-built book: predicted, then measured",
    2: "Tier 2 — another's book: measured, no closed form",
    3: "Tier 3 — public floor: WITHHELD, void",
}


def build_tiers(documents: dict[str, dict]) -> dict:
    """The right panel: five runs, three tiers, one bps axis.

    The number each row carries is taken from the run's own ``measurement``
    block and from nowhere else. Tier 3's is ``unreported_bps`` — the field
    exists precisely so a void run can say what it would have reported without
    reporting it — and the row is flagged ``void`` so the figure draws it as a
    measurement withheld rather than one taken.
    """
    rows = []
    for name, tier in (
        ("ladder", 1),
        ("thin", 1),
        ("wide", 1),
        ("feeder", 2),
        ("deployment", 3),
    ):
        document = documents[name]
        measurement = document["measurement"]
        reconciliation = document["reconciliation"]
        void = bool(measurement["void"])
        value = measurement["unreported_bps"] if void else measurement["realised_slippage_bps"]
        if value is None:
            raise ValueError(
                f"the {name} run reports neither a slippage nor an unreported "
                "figure; there is nothing honest to draw for it"
            )
        third_party = int(reconciliation["third_party_fills"])
        if tier == 1:
            note = f"predicted = realised, all {len(document['realised']['bins'])} bins"
        elif tier == 2:
            note = f"no closed form exists · {third_party:,} third-party fills"
        else:
            note = f"VOID · {third_party:,} third-party fills"
        rows.append(
            {
                "run": name,
                "label": document["ladder"]["name"] if tier == 1 else name,
                "tier": tier,
                "bps": float(value),
                "void": void,
                "value": f"{value:.2f} bps withheld" if void else f"{value:.2f} bps",
                "note": note,
            }
        )
    return {"rows": rows, "captions": TIER_CAPTIONS}


def caption(documents: dict[str, dict], ladders: dict, tiers: dict) -> str:
    """The five things this figure may never be shown without, hard-wrapped."""
    by_run = {row["run"]: row for row in tiers["rows"]}
    worst, comparisons = max_abs_residual(ladders)
    worst_text = "exactly 0.0" if worst == 0.0 else f"{worst:.3e}"
    deployment = documents["deployment"]
    feeder = documents["feeder"]
    reason = deployment["measurement"]["reasons"][0]
    parent = int(documents["ladder"]["order"]["parent"])
    attempts = " / ".join(f"{value:.4f}" for value in FEEDER_ATTEMPTS_BPS)

    lines = [
        f"Tier 1 — reference, thin and wide. The client builds the book, so a "
        f"committed ladder plus a deterministic policy plus deterministic matching "
        f"makes every fill computable BEFORE the run. Predicted and realised agree "
        f"level for level and bin for bin, at {by_run['ladder']['bps']:.2f} / "
        f"{by_run['thin']['bps']:.2f} / {by_run['wide']['bps']:.2f} bps, with zero "
        f"third-party fills. Worst |realised - predicted| across all "
        f"{comparisons} per-bin comparisons: {worst_text} bps, drawn in the "
        f"residual strip under the panel on an axis narrow enough to show it — "
        f"the panel above spans 10 to 39 bps, where any of this would be "
        f"invisible.",
        f"Tier 2 — feeder. Anvil's feeder builds the book and keeps trading it, so "
        f"book state is wall-clock dependent and there is no closed-form prediction "
        f"to check it against — there cannot be. Realised "
        f"{by_run['feeder']['bps']:.4f} bps through "
        f"{feeder['reconciliation']['third_party_fills']:,} third-party fills. "
        f"Reproducibility is weak, and measured rather than asserted: {attempts} bps "
        f"across four seed-pinned attempts (docs/briefs/M6-anvil-live-leg.md — brief "
        f"prose, not a committed artefact, so it is stated here and not drawn).",
        f"Tier 3 — deployment, on a shared public floor. {reason}, so the "
        f"measurement is VOID and the number is WITHHELD rather than taken: it is "
        f"measurement.unreported_bps = {by_run['deployment']['bps']:.6f}, drawn "
        f"outlined and struck through for that reason. It is not a data point beside "
        f"the other four.",
        f"All three tiers filled the whole {parent:,}-share parent and attributed "
        f"every share to this client's own order ids. What changes down the ladder is "
        f"the strength of the claim, not the behaviour of the client. Every number "
        f"here is a demo measurement against Anvil — a matching engine from the "
        f"same portfolio, consumed through its public contract with zero upstream "
        f"changes — and is NOT an evaluation (ARCHITECTURE.md §7): feeder flow is "
        f"synthetic and non-adversarial, and performance claims live in the "
        f"simulator.",
        "Sources: "
        + "; ".join(
            f"{ARTEFACTS[name]} at git {documents[name]['provenance']['git_rev'][:8]} "
            f"({documents[name]['provenance']['config']})"
            for name in (*LADDER_RUNS, "feeder", "deployment")
        )
        + ".",
    ]
    return "\n".join(
        textwrap.fill(line, width=CAPTION_WIDTH, subsequent_indent="  ")
        for line in lines
    )


def load_documents(root: Path) -> dict[str, dict]:
    """The five artefacts, or a clear refusal naming the one that is missing."""
    documents: dict[str, dict] = {}
    for name, relative in ARTEFACTS.items():
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(
                f"{relative} is missing, so the {name} run cannot be drawn. All "
                "five committed runs are inputs to this figure; see the Makefile's "
                "M6 block for how each was produced."
            )
        documents[name] = json.loads(path.read_text(encoding="utf-8"))
    return documents


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "results" / "m6_prediction"),
        help="figure stem, without a suffix (default: results/m6_prediction)",
    )
    args = parser.parse_args()

    documents = load_documents(REPO_ROOT)
    ladders = build_ladders(documents)
    tiers = build_tiers(documents)
    stamp = documents[STAMP_RUN]["provenance"]
    written = prediction_figure(
        Path(args.out),
        ladders=ladders,
        tiers=tiers,
        provenance=Provenance(
            config=stamp["config"],
            config_sha256=stamp["config_sha256"],
            git_rev=stamp["git_rev"],
            git_dirty=stamp["git_dirty"],
            python=stamp["python"],
        ),
        caption=caption(documents, ladders, tiers),
    )
    for path in written:
        # `relative_to` raises for anything outside the tree, and a driver that
        # dies while *reporting* where it wrote a file has thrown away the file's
        # only mention. M4b's figure tool failed exactly here the first time it
        # ran, after the figure had already been written.
        try:
            shown = path.resolve().relative_to(REPO_ROOT)
        except ValueError:
            shown = path
        print(f"wrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
