"""M2 — train the PPO agent on ``ExecutionEnv`` and grade it against the oracle.

    python tools/m2_train.py --config configs/m2_ppo.yaml

Runs the committed number of training seeds, evaluates each deterministically,
grades the schedule each induced *analytically* through
:func:`~temper.oracle.cost.schedule_moments`, and writes the metrics JSON and the
trajectory-overlay figure named by the config. Both carry the config digest and
the git revision (constitution invariant 1).

The work is in :mod:`temper.eval.sweep`; this file is argument parsing, printing
and the figure — so the suite can run the same sweep without a subprocess, and
so matplotlib stays off the package's import path.

Exit status is whether the run reached the verdict ``--expect`` says it should,
which is not the same as whether it passed: the sampled-reward sweep is M2's
recorded *miss*, and it clearing epsilon would be the surprising outcome.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "m2_ppo.yaml"


def _config_path(argv: list[str]) -> Path:
    """`--config` out of raw argv, before argparse and before torch exists."""
    for index, item in enumerate(argv):
        if item == "--config" and index + 1 < len(argv):
            return Path(argv[index + 1])
        if item.startswith("--config="):
            return Path(item.split("=", 1)[1])
    return DEFAULT_CONFIG


def _pin_thread_env(argv: list[str]) -> None:
    """Pin the OpenMP/BLAS pools from the config, *before* torch is imported.

    ``ppo.torch_threads`` fixes torch's intra-op pool, which is what decides
    reduction order and therefore the trained weights — that is the
    reproducibility half, and it is applied inside ``train``. This is the
    *runtime* half, and it has to happen before the first torch import because
    the OpenMP pool is sized once, at load.

    It is here rather than in the Makefile because leaving it to the environment
    is a trap with teeth. Run without it on this 8-physical/16-logical host, the
    surrounding pools default to 16, oversubscribe, and every seed takes ~25 %
    longer — which pushed one seed of an acceptance sweep from ~1 300 s to
    1 801 s, one second past the brief's 30-minute per-seed cap, truncating it
    and costing the whole run. The cap was right; the environment was the bug.
    ``setdefault`` so an operator who exports these deliberately still wins.
    """
    import yaml  # cheap, and pulls in nothing heavy

    try:
        document = yaml.safe_load(_config_path(argv).read_text(encoding="utf-8"))
        threads = int(document["ppo"]["torch_threads"])
    except (OSError, KeyError, TypeError, ValueError):
        return  # a bad config is argparse's problem to report, not this hook's
    if threads > 0:
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(name, str(threads))


_pin_thread_env(sys.argv[1:])

from temper.eval.experiment import Experiment, load_experiment  # noqa: E402
from temper.eval.figures import trajectory_overlay  # noqa: E402
from temper.eval.sweep import build_document, run_sweep  # noqa: E402


def _stdout_utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def _header(experiment: Experiment) -> None:
    reference = experiment.reference()
    case = experiment.case
    estimator = (
        "control variate" if experiment.estimator.control_variate else "sampled rewards"
    )
    print(
        f"M2 — {case.symbol}, X = {case.order_size:,.0f}, "
        f"λ = {experiment.lambda_risk:.6e} (rule-selected), "
        f"{experiment.seeds.n_seeds} seeds, {estimator}"
    )
    print(
        f"  J_optimal {reference.optimal.objective:.6f} bps · "
        f"J_ac {reference.ac.objective:.6f} · "
        f"J_twap {reference.twap.objective:.6f} · "
        f"TWAP gap {reference.twap_gap:.2%}"
    )


def _progress(experiment: Experiment):
    every = max(1, experiment.ppo.num_updates // 10)

    def report(update: int, metrics: dict) -> None:
        if update % every and update != experiment.ppo.num_updates:
            return
        print(
            f"    update {update:5d}/{experiment.ppo.num_updates}  "
            f"step {metrics['global_step']:>10,}  "
            f"train return {metrics['train_return']:9.4f}  "
            f"KL {metrics['approx_kl']:.4f}  {metrics['seconds']:6.1f}s",
            flush=True,
        )

    return report


def _on_seed(ordinal: int, grade, result) -> None:
    print(
        f"  seed {ordinal}: J {grade.objective:.6f} bps · "
        f"excess {grade.relative_excess:+.4%} = {grade.gap_fraction:.1%} of the "
        f"TWAP gap · ‖δ‖₂ {grade.deviation:,.0f} shares · {result.seconds:.0f}s"
        f"{' (TIMED OUT)' if result.timed_out else ''}"
        f"{' · RED FLAG' if grade.red_flag else ''}",
        flush=True,
    )


def write_figure(experiment: Experiment, document: dict) -> None:
    """The overlay figure, drawn from `document` and nothing else.

    Separated from the metrics so `--figure-only` can redraw a committed result
    without retraining. That is not a convenience: the figure is a *view* of the
    committed JSON, and a plotting change that could only be applied by spending
    two hours of CPU would either not be applied or would silently produce a
    figure and a metrics file from different runs.
    """
    summary = document["summary"]
    estimator = (
        "control variate — the deterministic (noise-free) reward"
        if experiment.estimator.control_variate
        else "sampled rewards — the full Phase-1 price noise"
    )
    verdict = document["verdict"]
    # Three short lines rather than two long ones: at 8 inches wide a title much
    # past ~75 characters runs off the canvas, and matplotlib will not tell you.
    caption = (
        f"M2 rediscovery — {experiment.case.symbol}, "
        f"X = {experiment.case.order_size:,.0f} shares, "
        f"λ = {experiment.lambda_risk:.3e}, {len(document['seeds'])} seeds\n"
        f"estimator: {estimator}\n"
        f"median excess {summary['relative_excess']['median']:+.3%} of J_optimal "
        f"= {summary['gap_fraction']['median']:.1%} of the TWAP gap "
        f"(ε = {experiment.tolerances.epsilon_gap_fraction:.0%}, "
        f"{'met' if verdict['epsilon_met'] else 'MISSED'}; "
        f"worst seed {summary['gap_fraction']['worst']:.1%})"
    )

    written = trajectory_overlay(
        experiment.results_figure,
        hours=experiment.case.market.times,
        agent_trajectories=[record["grade"]["trajectory"] for record in document["seeds"]],
        reference=experiment.reference(),
        order_size=experiment.case.order_size,
        band=experiment.band(),
        provenance=experiment.provenance(REPO_ROOT),
        caption=caption,
        formats=experiment.figure_formats,
    )
    for path in written:
        print(f"  wrote {path.relative_to(REPO_ROOT)}")


def write_outputs(experiment: Experiment, document: dict) -> None:
    """The metrics JSON and the overlay figure, both provenance-stamped."""
    metrics_path = experiment.results_metrics
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {metrics_path.relative_to(REPO_ROOT)}")
    write_figure(experiment, document)


def main() -> int:
    _stdout_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "m2_ppo.yaml"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="verify the config and the lambda rule, then stop",
    )
    parser.add_argument("--quiet", action="store_true", help="no per-update trace")
    parser.add_argument(
        "--no-write", action="store_true", help="grade and print, but write nothing"
    )
    parser.add_argument(
        "--figure-only",
        action="store_true",
        help="redraw the figure from the committed metrics JSON; train nothing",
    )
    parser.add_argument(
        "--expect",
        choices=("pass", "miss"),
        default="pass",
        help=(
            "the verdict this run is expected to reach. Exit status is whether "
            "the run met that expectation, not whether it passed — the "
            "sampled-reward sweep is M2's *recorded miss* and a pass from it "
            "would be the surprising outcome"
        ),
    )
    args = parser.parse_args()

    experiment = load_experiment(args.config)
    if args.figure_only:
        path = experiment.results_metrics
        if not path.exists():
            print(f"{path.relative_to(REPO_ROOT)} does not exist; run the sweep first")
            return 1
        write_figure(experiment, json.loads(path.read_text(encoding="utf-8")))
        return 0
    if args.dry_run:
        reference = experiment.verify_lambda_rule()
        print(
            f"config OK · λ = {experiment.lambda_risk:.6e} matches the rule · "
            f"J_optimal {reference.optimal.objective:.6f} bps · "
            f"ε = {experiment.tolerances.epsilon_gap_fraction:.0%} of a "
            f"{reference.twap_gap:.2%} gap"
        )
        return 0

    _header(experiment)
    sweep = run_sweep(
        experiment,
        repo_root=REPO_ROOT,
        on_seed=_on_seed,
        progress=None if args.quiet else _progress(experiment),
    )
    if sweep.provenance.git_dirty:
        print(
            "  WARNING: the source tree was dirty when this run started, so its "
            "recorded revision does not contain the code that produced it "
            "(invariant 1). Commit before an acceptance run."
        )
    document = build_document(sweep)

    baselines = ", ".join(
        f"{name} {g.gap_fraction:+.4f}" for name, g in sweep.baselines.items()
    )
    print(f"  baselines graded through the same rollout: {baselines}")

    if not args.no_write:
        write_outputs(experiment, document)

    verdict, summary = document["verdict"], document["summary"]
    print()
    print(
        f"median gap fraction {summary['gap_fraction']['median']:.4f} "
        f"(IQR {summary['gap_fraction']['iqr']:.4f}, worst "
        f"{summary['gap_fraction']['worst']:.4f}) against ε = "
        f"{experiment.tolerances.epsilon_gap_fraction} and a per-seed floor of "
        f"{experiment.tolerances.per_seed_gap_fraction}"
    )
    print(
        f"median excess {summary['relative_excess']['median']:+.4%} of J_optimal · "
        f"median ‖δ‖₂ {summary['deviation']['median']:,.0f} shares against a "
        f"derived bound of {experiment.band().bound_shares:,.0f}"
    )
    if verdict["red_flags"]:
        print(
            f"RED FLAG — {', '.join(verdict['red_flags'])} scored below the "
            "certified optimum. This is a defect, not a result "
            "(ARCHITECTURE.md §1.1)."
        )
    reached = "pass" if verdict["passed"] else "miss"
    print(f"sweep {verdict['sweep_seconds']:.0f}s · verdict: {reached.upper()}")
    if reached != args.expect:
        print(
            f"...but --expect {args.expect} was stated. A run that does not reach "
            "the verdict it was expected to reach is a finding either way: if a "
            "recorded miss starts passing, the amendment that rests on it needs "
            "revisiting (docs/briefs/M2-ppo-rediscovery.md, amendment 1)."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
