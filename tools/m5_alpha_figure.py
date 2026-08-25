"""M5 wrap-up — the alpha figure, redrawn from committed artefacts.

Two committed files in, one figure out, and **nothing computed here**. The plane
comes off ``results/m5_alpha.json`` (the trained sweep) and the rho curve off
``results/m5_reference.json`` (task 0's oracle table), so the picture is a *view*
of results rather than a second route to them — which is what lets it redraw
byte-identically from a clean clone without a training run.

    python tools/m5_alpha_figure.py

Neither panel reports a single capture fraction, deliberately. M5's own finding is
that the net number is a *difference* — the gross alpha a policy monetised, minus
the premium it paid to monetise it — and that the difference alone cannot separate
a policy that traded the signal well from one that traded it badly and executed
well. A one-number figure would say in pictures the thing the milestone spent six
tasks disproving.

The caption is assembled here rather than in the figure module, for the reason
M4a's and M4b's are: it carries numbers, and a caption with numbers in it is a
claim. Four of them are non-negotiable and appear every time this figure is drawn
— the **denominator**, the fact that ``rho`` is **invented**, the reference being
**converged rather than certified**, and the **shared-path offset** that puts the
DP above the 1.00 contour.
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

from temper.eval.figures import alpha_figure  # noqa: E402
from temper.eval.provenance import Provenance  # noqa: E402

SWEEP = REPO_ROOT / "results" / "m5_alpha.json"
REFERENCE = REPO_ROOT / "results" / "m5_reference.json"
TARGET = REPO_ROOT / "results" / "m5_alpha"

#: Characters per caption line, measured against the 11.9-inch canvas at 7.6 pt.
#: Bounded where the string is built rather than hoped for at draw time:
#: matplotlib will not tell you that text ran off the edge.
CAPTION_WIDTH = 178


def _shown(path: Path) -> str:
    """A path as a reader would name it: repo-relative when it is in the repo.

    `Path.relative_to` RAISES rather than declining when the path is elsewhere, so
    a `--out` under a temp directory would turn a working figure into a traceback
    on the line that reports where it was written. That is the M4b defect (the
    tool's `main` died reporting where it wrote the figure) with a different cause,
    which is reason enough not to write it the same way twice.
    """
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def build_plane(document: dict) -> dict:
    """The left panel: every graded policy in (alpha capture, premium ratio).

    The plane's own geometry comes with it. Net capture is linear in these two
    coordinates, and the figure draws its contours, so the two slopes and the
    intercept are computed here — from the artefact's numbers, once — rather than
    inside a figure that is supposed to compute nothing.

    The intercept is not zero and that is the point. It is read off the *graded*
    M4a schedule, which monetises no alpha by construction and records an alpha
    capture of -0.0035: the empirical mean of the 200,000 shared signal paths is
    not exactly zero, at the 1/sqrt(M) ~ 2e-3 scale it should not be. That offset
    cancels out of net capture, which is a paired difference against that same
    schedule, and does not cancel out of alpha capture, which is a level.
    """
    reference = document["reference"]
    baselines = {
        name: {
            "alpha_capture": grade["alpha_capture"],
            "premium_ratio": grade["premium_ratio"],
            "net_capture": grade["net_capture"],
        }
        for name, grade in document["baselines"].items()
    }
    alpha_available = reference["alpha_available_bps"]
    premium = reference["execution_premium_bps"]
    advantage = reference["signal_advantage_bps"]
    anchor = baselines["optimal"]
    # net = intercept + (A a - P p) / D, pinned on the schedule whose net capture
    # is zero by definition. Asserted against every grade by the tests.
    intercept = anchor["net_capture"] - (
        alpha_available * anchor["alpha_capture"] - premium * anchor["premium_ratio"]
    ) / advantage
    return {
        "alpha_available_bps": alpha_available,
        "premium_bps": premium,
        "advantage_bps": advantage,
        "net_intercept": intercept,
        "seed_alpha_capture": [r["grade"]["alpha_capture"] for r in document["seeds"]],
        "seed_premium_ratio": [r["grade"]["premium_ratio"] for r in document["seeds"]],
        "shuffled_alpha_capture": [
            r["shuffled"]["alpha_capture"] for r in document["seeds"]
        ],
        "shuffled_premium_ratio": [
            r["shuffled"]["premium_ratio"] for r in document["seeds"]
        ],
        "baselines": baselines,
    }


def build_curve(table: dict, document: dict) -> dict:
    """The right panel: the oracle's value of the signal at each reported rho."""
    rows = sorted(table["value_of_signal"], key=lambda row: row["rho"])
    signal = document["signal"]
    return {
        "rho": [row["rho"] for row in rows],
        "alpha_available_bps": [row["alpha_available_bps"] for row in rows],
        "execution_premium_bps": [row["execution_premium_bps"] for row in rows],
        "advantage_bps": [row["advantage_bps"] for row in rows],
        "trained_rho": signal["rho"],
        "trained_explained": signal["explained_variance_fraction"],
    }


def caption(document: dict, plane: dict, curve: dict) -> str:
    """The four claims that travel with this figure, wrapped to the canvas."""
    summary = document["summary"]
    headline = document["verdict"]["headline"]
    control = document["shuffled_control"]["net_capture"]
    offset = plane["baselines"]["optimal"]["alpha_capture"]
    text = (
        "THE THREE NUMBERS, medians over ten seeds, never one of them: alpha capture "
        f"{headline['alpha_capture_median']:+.4f} ({headline['alpha_bps_median']:+.5f} "
        f"of {headline['reference_alpha_bps']:.5f} bps gross); execution premium "
        f"{headline['premium_ratio_median']:.4f}x "
        f"({headline['execution_premium_bps_median']:+.5f} of "
        f"{headline['reference_premium_bps']:.5f} bps); net capture "
        f"{headline['net_capture_median']:+.4f} ({headline['median_excess_bps']:+.5f} "
        f"bps over J_DP, on an advantage of {headline['advantage_bps']:.5f} bps). "
        f"Worst seed {summary['net_capture']['worst']:+.4f}. Signal-shuffled control "
        f"{control['median']:+.4f} median. "
        "DENOMINATOR: every fraction is of the net signal advantage A - P, what the "
        "converged DP gains over M4a's certified optimum in the same world - not of "
        "the gross alpha, and not of the objective. "
        "INVENTED: rho is Temper's own. FrontierView vendored an impact law and no "
        "signal, so section 7's 'vendored, not invented' cover does not reach the "
        "right-hand panel's x axis, which is why it is drawn as a curve. "
        "CONVERGED, NOT CERTIFIED: J_DP is a Richardson-extrapolated dynamic program "
        "with a certified floor under its execution half, not a closed form. "
        f"THE OFFSET: the graded M4a schedule monetises no alpha and reads "
        f"{offset:+.4f} rather than 0 - the 200,000 shared signal paths have a "
        "non-zero empirical mean at the 1/sqrt(M) scale. It cancels out of net "
        "capture, a paired difference against that schedule, and not out of alpha "
        "capture, a level; so the DP sits just above the 1.00 contour, not on it. "
        "TWAP (22.1x premium) and AC (1.95x) are off the top of the left panel."
    )
    return textwrap.fill(text, width=CAPTION_WIDTH)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, default=SWEEP)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--out", type=Path, default=TARGET)
    parser.add_argument("--formats", default="png")
    args = parser.parse_args(argv)

    for path in (args.sweep, args.reference):
        if not path.exists():
            print(
                f"{_shown(path)} is missing. This tool is a view of committed "
                "results and computes nothing itself, so it has nothing to draw."
            )
            return 1

    document = json.loads(args.sweep.read_text(encoding="utf-8"))
    table = json.loads(args.reference.read_text(encoding="utf-8"))
    plane = build_plane(document)
    curve = build_curve(table, document)

    written = alpha_figure(
        args.out,
        plane=plane,
        curve=curve,
        provenance=Provenance(**{
            key: value
            for key, value in document["provenance"].items()
            if key in Provenance.__dataclass_fields__
        }),
        caption=caption(document, plane, curve),
        formats=tuple(f.strip() for f in args.formats.split(",") if f.strip()),
    )
    for path in written:
        print(f"  wrote {_shown(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
