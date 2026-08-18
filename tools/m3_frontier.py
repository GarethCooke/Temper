"""M3 — the frontier sweep: generate the point configs, run them serially, draw the frontier.

    python tools/m3_frontier.py configs      # write configs/m3_frontier/*.yaml from the manifest
    python tools/m3_frontier.py check        # the committed point configs are what the generator writes
    python tools/m3_frontier.py run          # every point, serially, through tools/train.py — a day
    python tools/m3_frontier.py figure       # aggregate results/m3_frontier/*.json -> results/m3_frontier.*
    python tools/m3_frontier.py figure --redraw   # redraw from the committed aggregate, byte-identical
    python tools/m3_frontier.py status       # which points have results, and their verdicts

The manifest is ``configs/m3_frontier.yaml``: which frontier grid, which template
config every point is stamped from, and exactly which fields a point may change.
Each point is an ordinary experiment run by ``tools/train.py`` and graded exactly
as M2 and the validation run were; the aggregate and the figure are views of the
per-point results files (``temper/eval/frontier.py``), so ``figure`` redraws
byte-identically without training anything.

``run`` is strictly serial and runs each point in a fresh process — 512 envs at
8 threads saturates the reference box, and two concurrent sweeps truncate each
other (M2's first discarded run). It runs **M2's rule-selected lambda first**
(:func:`~temper.eval.frontier.run_order`), because that point is comparable to
two committed results and is therefore where the amended update budget is
checked against a known answer; ``--only <lambda>`` runs one point on its own for
exactly that check, and ``--skip-done`` then continues the rest. It passes
``--expect any``: a lambda point that misses its per-lambda epsilon is a
*finding* the frontier reports, not a reason to abandon the remaining points; a
red flag anywhere still stops the run, because that is a defect rather than a
result.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "m3_frontier.yaml"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from temper.eval.frontier import (  # noqa: E402
    aggregate,
    load_manifest,
    point_experiments,
    run_order,
    stale_point_configs,
    write_point_configs,
)


def _stdout_utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def cmd_configs(manifest) -> int:
    written = write_point_configs(manifest, REPO_ROOT)
    for path in written:
        print(f"  wrote {path.relative_to(REPO_ROOT)}")
    # Load every one back through the experiment loader, and verify its lambda
    # against the frontier-grid rule, so a generated config that would refuse to
    # run is caught at generation rather than at the top of a night's sweep.
    for experiment in point_experiments(manifest):
        experiment.verify_lambda_rule()
    print(f"  {len(written)} point configs on the {manifest.grid} grid, all verified")
    return 0


def cmd_check(manifest) -> int:
    stale = stale_point_configs(manifest, REPO_ROOT)
    if stale:
        print("point configs differ from what the generator writes:")
        for path in stale:
            print(f"  {path.relative_to(REPO_ROOT)}")
        print("regenerate with `python tools/m3_frontier.py configs`, then commit")
        return 1
    for experiment in point_experiments(manifest):
        experiment.verify_lambda_rule()
    print(f"  {len(manifest.lambdas)} point configs match the generator and verify")
    return 0


def cmd_status(manifest) -> int:
    for experiment in point_experiments(manifest):
        path = experiment.results_metrics
        if not path.exists():
            print(f"  λ = {experiment.lambda_risk:.3e}  {experiment.path.name:<22}  — not run")
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        summary, verdict = document["summary"], document["verdict"]
        print(
            f"  λ = {experiment.lambda_risk:.3e}  {experiment.path.name:<22}  "
            f"median gap {summary['gap_fraction']['median']:.4f} "
            f"(IQR {summary['gap_fraction']['iqr']:.4f}, worst "
            f"{summary['gap_fraction']['worst']:.4f})  "
            f"ε {'met' if verdict['epsilon_met'] else 'MISSED'} · "
            f"floor {'met' if verdict['per_seed_met'] else 'MISSED'} · "
            f"red flags {len(verdict['red_flags'])} · "
            f"{verdict['sweep_seconds']:.0f}s"
            f"{' · dirty' if document['provenance']['git_dirty'] else ''}"
        )
    return 0


def cmd_run(manifest, *, quiet: bool, skip_done: bool, only: float | None = None) -> int:
    if stale_point_configs(manifest, REPO_ROOT):
        print("point configs are stale; run `configs` and commit before sweeping")
        return 1
    order = run_order(manifest)
    if only is not None:
        order = [p for p in order if p.lambda_risk == only]
        if not order:
            print(f"no point at lambda = {only!r} on the {manifest.grid} grid")
            return 1
    for experiment in order:
        if skip_done and experiment.results_metrics.exists():
            print(f"  λ = {experiment.lambda_risk:.3e}: results exist, skipping")
            continue
        command = [
            sys.executable,
            str(REPO_ROOT / "tools" / "train.py"),
            "--config",
            str(experiment.path),
            "--expect",
            "any",
        ]
        if quiet:
            command.append("--quiet")
        print(f"=== λ = {experiment.lambda_risk:.3e}  ({experiment.path.name})", flush=True)
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if completed.returncode != 0:
            print(
                f"point {experiment.path.name} exited {completed.returncode}; "
                "stopping the sweep (a red flag or a crash is a defect, not a result)"
            )
            return completed.returncode
    return 0


def _caption(document: dict) -> str:
    verdict = document["verdict"]
    grid = document["grid"]["lambdas"]
    n_seeds = document["points"][0]["n_seeds"] if document["points"] else 0
    updates = document["points"][0]["updates"] if document["points"] else 0
    floor = document["variance_floor_bps2"]
    met = len(verdict["epsilon_met_at"])
    # Lines kept under ~85 characters: at 8 inches wide a longer title runs off
    # the canvas, and matplotlib will not say so.
    return (
        f"{document['milestone']} frontier — {document['case']['symbol']}, "
        f"X = {document['case']['order_size']:,.0f}, "
        f"{len(grid)} λ from {min(grid):.0e} to {max(grid):.0e}, "
        f"{n_seeds} seeds per λ, {updates} updates per seed\n"
        f"estimator: antithetic pairing — each episode as (ξ, −ξ), rewards averaged; "
        f"graded analytically\n"
        f"top: E[cost] vs V − σ²_bin·X² (floor {floor:,.0f} bps², paid by every schedule);\n"
        f"x is linear below 0.5, so 0 is on the axis — where the agent sits once it "
        f"liquidates in bin 0\n"
        f"AC lies ON the optimal curve, displaced: AC(λ) = optimal(cλ) exactly, "
        f"c ≈ 5 at low λ\n"
        f"bottom: gap fraction per λ, every seed drawn · ε met at {met} of "
        f"{len(document['points'])} λ · red flags: "
        f"{'none' if verdict['red_flag_free'] else verdict['red_flags']} · "
        f"sweep {verdict['total_sweep_seconds'] / 3600:.1f} h"
    )


def cmd_figure(manifest, *, allow_partial: bool, redraw: bool = False) -> int:
    """Aggregate the points and draw the frontier; with `redraw`, draw only.

    The distinction is the same one M2 drew with ``--figure-only`` and it exists
    for the same reason. Aggregating **re-stamps**: the document records the
    revision it was built at, so committing it necessarily creates a newer
    revision, and re-aggregating from the commit that contains it can never
    reproduce its bytes. ``--redraw`` reads the committed aggregate and draws
    from it, provenance included, so the figure is a pure function of a
    committed result — which is what makes "the figure redraws byte-identically"
    a property one can check from a clean clone rather than a hope.
    """
    from temper.eval.figures import frontier_figure
    from temper.eval.provenance import Provenance

    if redraw:
        if not manifest.results_metrics.exists():
            print(
                f"{manifest.results_metrics.relative_to(REPO_ROOT)} does not exist; "
                "run the sweep and aggregate first"
            )
            return 1
        document = json.loads(manifest.results_metrics.read_text(encoding="utf-8"))
    else:
        document = aggregate(manifest, REPO_ROOT, require_complete=not allow_partial)
        if document["provenance"]["git_dirty"]:
            print(
                "  WARNING: the source tree is dirty; the aggregate's recorded revision "
                "does not contain the code that produced it (invariant 1)."
            )
        manifest.results_metrics.parent.mkdir(parents=True, exist_ok=True)
        manifest.results_metrics.write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )
        print(f"  wrote {manifest.results_metrics.relative_to(REPO_ROOT)}")
    written = frontier_figure(
        manifest.results_figure,
        aggregate=document,
        provenance=Provenance(**document["provenance"]),
        caption=_caption(document),
        formats=manifest.figure_formats,
    )
    for path in written:
        print(f"  wrote {path.relative_to(REPO_ROOT)}")
    verdict = document["verdict"]
    print(
        f"  {verdict['points']} points · ε met at {len(verdict['epsilon_met_at'])} · "
        f"per-seed floor missed at {len(verdict['per_seed_missed_at'])} · "
        f"red flags {'none' if verdict['red_flag_free'] else verdict['red_flags']} · "
        f"total {verdict['total_sweep_seconds'] / 3600:.2f} h"
    )
    return 0


def main() -> int:
    _stdout_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("configs", "check", "run", "figure", "status"))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--quiet", action="store_true", help="run: no per-update trace")
    parser.add_argument(
        "--skip-done", action="store_true", help="run: skip points whose results file exists"
    )
    parser.add_argument(
        "--partial", action="store_true", help="figure: draw whatever points exist"
    )
    parser.add_argument(
        "--redraw",
        action="store_true",
        help=(
            "figure: draw from the committed aggregate without re-aggregating or "
            "re-stamping it — the byte-identical redraw"
        ),
    )
    parser.add_argument(
        "--only",
        type=float,
        default=None,
        help="run: a single lambda point (must be on the grid) — the budget check",
    )
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    if args.command == "configs":
        return cmd_configs(manifest)
    if args.command == "check":
        return cmd_check(manifest)
    if args.command == "status":
        return cmd_status(manifest)
    if args.command == "run":
        return cmd_run(
            manifest, quiet=args.quiet, skip_done=args.skip_done, only=args.only
        )
    return cmd_figure(manifest, allow_partial=args.partial, redraw=args.redraw)


if __name__ == "__main__":
    raise SystemExit(main())
