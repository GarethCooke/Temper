"""Train the PPO agent on ``ExecutionEnv`` from a committed config; grade it against the oracle.

    python tools/train.py --config configs/m2_ppo.yaml
    python tools/train.py --config configs/m3_antithetic_validation.yaml

Runs the committed number of training seeds, evaluates each deterministically,
grades the schedule each induced *analytically* through
:func:`~temper.oracle.cost.schedule_moments`, and writes the metrics JSON and the
trajectory-overlay figure named by the config. Both carry the config digest and
the git revision (constitution invariant 1). Which milestone, which reward regime
and which lambda are all the config's to say; this driver reads them and prints
them.

The work is in :mod:`temper.eval.sweep`; this file is argument parsing, printing
and the figure — so the suite can run the same sweep without a subprocess, and
so matplotlib stays off the package's import path.

Exit status is whether the run reached the verdict ``--expect`` says it should,
which is not the same as whether it passed: the sampled-reward sweep is M2's
recorded *miss*, and it clearing epsilon would be the surprising outcome.

History: this file was ``tools/m2_train.py`` through M2, and the two M2 configs
still name it that way in their comments — those files are frozen by the
digests their committed results carry, so the comment stays stale rather than
the results going unverifiable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "m2_ppo.yaml"

# `python tools/x.py` from a clean clone has no `temper` on its path: pytest
# injects it via pyproject's `pythonpath`, and nothing else does. Rather than
# require callers to export PYTHONPATH — which the Makefile targets did not, so
# `make reference` and `make sweep` were broken from a fresh clone while every
# hand-run invocation worked — the tool puts its own repo root on the path.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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

import numpy as np  # noqa: E402

from temper.eval.experiment import Experiment, load_experiment  # noqa: E402
from temper.eval.figures import trajectory_overlay  # noqa: E402
from temper.eval.provenance import Provenance  # noqa: E402
from temper.eval.sweep import build_document, run_sweep  # noqa: E402

#: How each reward regime reads in a header and a figure caption.
REGIME_LABELS = {
    "sampled": "sampled rewards — the full Phase-1 price noise",
    "control_variate": "control variate — the deterministic (noise-free) reward",
    "antithetic": "antithetic pairing — each episode as (ξ, −ξ), rewards averaged",
}


def _stdout_utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


class _KeepAwake:
    """Hold the host awake for the run's lifetime, on Windows; a no-op elsewhere.

    Process-scoped: ``SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)``
    on entry, ``ES_CONTINUOUS`` alone on exit, which reverts it. No power setting
    is changed. M2 lost a seed of a four-hour sweep to the host sleeping mid-run,
    and the fix belongs with the run rather than in the operator's memory. Best
    effort: if the call is unavailable, the run proceeds and says so.
    """

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __init__(self) -> None:
        self._kernel = None

    def __enter__(self):
        if sys.platform == "win32":
            try:
                import ctypes

                self._kernel = ctypes.windll.kernel32
                self._kernel.SetThreadExecutionState(
                    self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED
                )
                print("  host held awake for the run (SetThreadExecutionState)")
            except Exception as error:  # pragma: no cover - platform dependent
                self._kernel = None
                print(f"  could not hold the host awake: {error!r}")
        return self

    def __exit__(self, *exc):
        if self._kernel is not None:
            self._kernel.SetThreadExecutionState(self.ES_CONTINUOUS)
        return False


def _header(experiment: Experiment) -> None:
    reference = experiment.reference()
    case = experiment.case
    estimator = REGIME_LABELS[experiment.estimator.regime]
    where = (
        "rule-selected"
        if experiment.frontier_grid is None
        else f"{experiment.frontier_grid} frontier grid"
    )
    print(
        f"{experiment.milestone} — {case.symbol}, X = {case.order_size:,.0f}, "
        f"λ = {experiment.lambda_risk:.6e} ({where}), "
        f"{experiment.seeds.n_seeds} seeds, {estimator}, "
        f"{experiment.cost_encoding} world"
    )
    print(
        f"  J_optimal {reference.optimal.objective:.6f} bps · "
        f"J_ac {reference.ac.objective:.6f} · "
        f"J_twap {reference.twap.objective:.6f} · "
        f"TWAP gap {reference.twap_gap:.2%}"
    )
    advantage = reference.available_advantage
    if advantage is not None:
        print(
            f"  J_tangent {reference.tangent.objective:.6f} bps · available "
            f"advantage {advantage:.5f} bps ({reference.advantage_fraction:.3%} "
            f"of J_optimal) · bar {experiment.tolerances.epsilon_fraction:.0%} of "
            f"it = {experiment.tolerances.epsilon_fraction * advantage:.5f} bps"
        )
    _reward_scale_check(experiment, reference)


def _reward_scale_check(experiment: Experiment, reference) -> None:
    """The per-step reward range the scale actually produces, measured not derived.

    M4a's brief asks for this before the run rather than after. The committed
    ``reward.scale`` was set against a Phase-1 objective and carries at the
    *episode* level — the two worlds' objectives are 1.2 % apart at this lambda —
    but the per-step charge is more front-loaded under the power law, and PPO's
    value head and advantage normalisation were tuned on a range.

    The range is *measured*, by stepping the real env with the control variate on
    top so the shocks are subtracted exactly. That is the deterministic per-step
    reward the antithetic pair averages to, which is what the agent actually
    sees — and it is measured rather than reassembled from the moments because
    re-deriving the per-bin split of permanent, temporary and spread by hand is
    precisely the kind of arithmetic that is wrong in a way nobody notices in a
    diagnostic line.
    """
    from temper.env import ExecutionEnv, impact_for
    from temper.eval.variate import deterministic_reward
    from temper.seeding import DIFFERENTIAL_POOL

    market, order_size = experiment.case.market, experiment.case.order_size
    scale = experiment.reward_scale
    lines = []
    for name in ("optimal", "twap"):
        schedule = np.asarray(reference.schedules[name].trajectory, dtype=float)
        env = deterministic_reward(
            ExecutionEnv(
                market,
                order_size,
                experiment.lambda_risk,
                temporary_impact=impact_for(
                    experiment.cost_encoding, market, order_size
                ),
                root_seed=experiment.seeds.root_seed,
                pool=DIFFERENTIAL_POOL,
                stream_index=0,
            )
        )
        env.reset()
        rewards = [
            scale * float(env.step(trade)[1]) for trade in -np.diff(schedule)
        ]
        lines.append(f"{name} [{min(rewards):+.4f}, {max(rewards):+.4f}]")
    print(
        f"  reward scale {scale:g} · scaled per-step reward range, measured on the "
        f"noise-free reward: " + " · ".join(lines)
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


def _on_seed(ordinal: int, grade, result, pairs=()) -> None:
    capture = (
        ""
        if grade.capture_fraction is None
        else f" · capture {grade.capture_fraction:.4f}"
    )
    print(
        f"  seed {ordinal}: J {grade.objective:.6f} bps · "
        f"excess {grade.excess:+.5f} bps = {grade.relative_excess:+.4%} = "
        f"{grade.gap_fraction:.1%} of the TWAP gap{capture} · "
        f"‖δ‖₂ {grade.deviation:,.0f} shares · {result.seconds:.0f}s"
        f"{' (TIMED OUT)' if result.timed_out else ''}"
        f"{' · RED FLAG' if grade.red_flag else ''}",
        flush=True,
    )
    if pairs:
        sampled = float(np.nanmedian([u.sampled_variance for u in pairs]))
        averaged = float(np.nanmedian([u.averaged_variance for u in pairs]))
        ratio = float(np.nanmedian([u.variance_ratio for u in pairs]))
        print(
            f"          reward variance per update (median over updates): "
            f"sampled half {sampled:.1f} bps² · averaged {averaged:.3e} bps² · "
            f"ratio {ratio:.2e}",
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
    estimator = REGIME_LABELS[experiment.estimator.regime]
    verdict = document["verdict"]
    gate = document.get("gate")
    # Short lines rather than long ones: at 8 inches wide a title much past ~75
    # characters runs off the canvas, and matplotlib will not tell you. The
    # no-gate form is M2's, character for character, so M2's committed figures
    # still redraw byte-identically; a gated run gets a fourth line.
    capture = summary.get("capture_fraction")
    if capture is not None:
        # M4a is not a rediscovery run and its caption must not say it is: the
        # agent is beating a closed form, not reproducing one, and the number it
        # leads with is the capture fraction with the absolute excess beside it.
        caption = (
            f"{experiment.milestone} — {experiment.case.symbol}, "
            f"X = {experiment.case.order_size:,.0f} shares, "
            f"λ = {experiment.lambda_risk:.3e}, {len(document['seeds'])} seeds\n"
            f"{experiment.cost_encoding} world · estimator: {estimator}\n"
            f"captured {capture['median']:.1%} of the "
            f"{verdict['denominator_bps']:.4f} bps the tangent left on the table "
            f"(worst seed {capture['worst']:.1%})\n"
            f"median excess over the certified optimum "
            f"{verdict['median_excess_bps']:+.5f} bps"
        )
        written = trajectory_overlay(
            experiment.results_figure,
            hours=experiment.case.market.times,
            agent_trajectories=[
                record["grade"]["trajectory"] for record in document["seeds"]
            ],
            reference=experiment.reference(),
            order_size=experiment.case.order_size,
            band=experiment.band(),
            provenance=Provenance(**document["provenance"]),
            caption=caption,
            formats=experiment.figure_formats,
        )
        for path in written:
            print(f"  wrote {path.relative_to(REPO_ROOT)}")
        return

    caption = (
        f"{experiment.milestone} rediscovery — {experiment.case.symbol}, "
        f"X = {experiment.case.order_size:,.0f} shares, "
        f"λ = {experiment.lambda_risk:.3e}, {len(document['seeds'])} seeds\n"
        f"estimator: {estimator}\n"
    )
    if gate is None:
        caption += (
            f"median excess {summary['relative_excess']['median']:+.3%} of J_optimal "
            f"= {summary['gap_fraction']['median']:.1%} of the TWAP gap "
            f"(ε = {experiment.tolerances.epsilon_fraction:.0%}, "
            f"{'met' if verdict['epsilon_met'] else 'MISSED'}; "
            f"worst seed {summary['gap_fraction']['worst']:.1%})"
        )
    else:
        caption += (
            f"median excess {summary['relative_excess']['median']:+.4%} of J_optimal "
            f"= {summary['gap_fraction']['median']:.3%} of the TWAP gap; "
            f"worst seed {summary['gap_fraction']['worst']:.3%}\n"
            f"gate: median ≤ {gate['median_gap_fraction_max']:g} against the "
            f"{gate['reference_regime'].replace('_', ' ')}'s "
            f"{gate['reference_median_gap_fraction']:.4f} — "
            f"{'met' if gate['met'] else 'MISSED'}"
        )

    written = trajectory_overlay(
        experiment.results_figure,
        hours=experiment.case.market.times,
        agent_trajectories=[record["grade"]["trajectory"] for record in document["seeds"]],
        reference=experiment.reference(),
        order_size=experiment.case.order_size,
        band=experiment.band(),
        # The *run's* stamp, read back out of the document — not a fresh one. A
        # figure is a view of a result, so its footer must name the revision that
        # produced the result, not the revision that happened to be checked out
        # when someone redrew it. Re-stamping here would put today's git state on
        # a picture of last week's run, which is the precise failure the footer
        # exists to prevent.
        provenance=Provenance(**document["provenance"]),
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
        choices=("pass", "miss", "any"),
        default="pass",
        help=(
            "the verdict this run is expected to reach. Exit status is whether "
            "the run met that expectation, not whether it passed — the "
            "sampled-reward sweep is M2's *recorded miss* and a pass from it "
            "would be the surprising outcome. `any` is for a frontier sweep "
            "point, where missing the per-lambda epsilon is a finding rather "
            "than an error; a red flag still exits non-zero under `any`"
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
        gate_reference = experiment.verify_gate_reference()
        where = (
            "matches the rule"
            if experiment.frontier_grid is None
            else f"is a point of the {experiment.frontier_grid} frontier grid"
        )
        denominator = experiment.denominator_bps(reference)
        print(
            f"config OK · λ = {experiment.lambda_risk:.6e} {where} · "
            f"{experiment.cost_encoding} world · "
            f"J_optimal {reference.optimal.objective:.6f} bps · "
            f"ε = {experiment.tolerances.epsilon_fraction:.0%} of the "
            f"{experiment.tolerances.denominator} = {denominator:.5f} bps "
            f"⇒ {experiment.tolerances.epsilon_fraction * denominator:.5f} bps"
        )
        if gate_reference is not None:
            print(
                f"gate OK · median gap ≤ {experiment.gate.median_gap_fraction:g} "
                f"against {experiment.gate.reference.name} "
                f"(median {gate_reference:.5f})"
            )
        return 0

    _header(experiment)
    with _KeepAwake():
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
    graded = summary[experiment.tolerances.graded_attribute]
    print()
    print(
        f"median {experiment.tolerances.graded_attribute} {graded['median']:.4f} "
        f"(IQR {graded['iqr']:.4f}, worst {graded['worst']:.4f}) against ε = "
        f"{experiment.tolerances.epsilon_fraction} and a per-seed floor of "
        f"{experiment.tolerances.per_seed_fraction}, both fractions of the "
        f"{experiment.tolerances.denominator} "
        f"({verdict['denominator_bps']:.5f} bps)"
    )
    if "capture_fraction" in summary:
        capture = summary["capture_fraction"]
        # The fraction leads and the absolute excess goes beside it every time:
        # a capture fraction near 1 on an advantage of 0.037 bps is a small
        # absolute claim and should read as one (ARCHITECTURE.md §9).
        print(
            f"capture fraction: median {capture['median']:.4f} "
            f"(IQR {capture['iqr']:.4f}, worst {capture['worst']:.4f}) · "
            f"median absolute excess over the certified optimum "
            f"{verdict['median_excess_bps']:+.5f} bps"
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
    if summary.get("reward_variance") is not None:
        rv = summary["reward_variance"]
        print(
            f"reward variance per update, median across seeds: sampled half "
            f"{rv['sampled_median']:.1f} bps² · averaged {rv['averaged_median']:.3e} "
            f"bps² · ratio {rv['variance_ratio_median']:.2e}"
        )
    gate = document.get("gate")
    if gate is not None:
        print(
            f"gate: median gap {gate['median_gap_fraction']:.5f} against "
            f"≤ {gate['median_gap_fraction_max']:g} (reference "
            f"{gate['reference']} {gate['reference_regime']}: "
            f"{gate['reference_median_gap_fraction']:.5f}) · "
            f"{'MET' if gate['met'] else 'MISSED'}"
        )
    reached = "pass" if verdict["passed"] else "miss"
    print(f"sweep {verdict['sweep_seconds']:.0f}s · verdict: {reached.upper()}")
    if args.expect == "any":
        return 1 if verdict["red_flags"] else 0
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
