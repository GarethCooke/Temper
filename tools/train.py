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
from dataclasses import replace
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

from temper.agents.checkpoint import network_description, save_policy  # noqa: E402
from temper.agents.execution import fraction_to_shares  # noqa: E402
from temper.eval.experiment import Experiment, load_experiment  # noqa: E402
from temper.eval.figures import trajectory_overlay  # noqa: E402
from temper.env import ExecutionEnv, impact_for  # noqa: E402
from temper.eval.sweep import (  # noqa: E402
    ALPHA_CAPTURE_BAR,
    PREMIUM_RATIO_BAR,
    format_alpha_headline,
)
from temper.eval.grading import median_ordinal  # noqa: E402
from temper.eval.provenance import Provenance, config_digest  # noqa: E402
from temper.eval.sweep import build_document, grade, run_sweep, train_seed  # noqa: E402
from temper.seeding import pool_seeds  # noqa: E402

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


def _liquidity_header(experiment: Experiment) -> None:
    """The liquidity world's header, from the *static* rungs only.

    Deliberately no dynamic program here. The DP takes minutes and ``run_sweep``
    computes it once for the whole sweep a moment later; solving it twice would be
    two chances for the run and its own banner to disagree about the number every
    grade is an excess over. What this can say cheaply it says: the invented
    parameter, the level shift, and the fact that the reference is not certified.
    """
    from temper.eval.reference import static_liquidity_row

    case = experiment.case
    law = experiment.liquidity
    row = static_liquidity_row(
        case.market, case.order_size, experiment.lambda_risk, law
    )
    print(
        f"  liquidity: INVENTED — {law.name}, sigma_log = {law.sigma_log:g}, "
        f"E[L] = {law.mean_multiplier():g}, "
        f"E[L^-{case.market.temp_exponent:g}] = "
        f"{law.inverse_power_moment(case.market.temp_exponent):.5f}. "
        "FrontierView has no liquidity process; this one is Temper's own."
    )
    print(
        f"  J_static* {row.static.objective:.6f} bps (best fixed schedule that "
        f"knows the law) · J_M4a {row.m4a.objective:.6f} · level shift "
        f"{row.level_shift:.5f} bps — a re-solve, and NOT the agent's"
    )
    print(
        "  J_DP is solved by the sweep, once: converged and bracketed by a "
        "perfect-information relaxation, NOT certified."
    )


def _header(experiment: Experiment) -> None:
    if experiment.liquidity.stochastic:
        case = experiment.case
        estimator = REGIME_LABELS[experiment.estimator.regime]
        print(
            f"{experiment.milestone} — {case.symbol}, X = {case.order_size:,.0f}, "
            f"λ = {experiment.lambda_risk:.6e} (rule-selected), "
            f"{experiment.seeds.n_seeds} seeds, {estimator}, "
            f"{experiment.cost_encoding} world + stochastic liquidity"
        )
        _liquidity_header(experiment)
        _reward_scale_check(experiment, experiment.reference())
        return
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


def _seed_line(ordinal: int, grade, result) -> str:
    """One seed's line, in whichever grade shape this world produces.

    Two grades exist and they are not interchangeable: a deterministic policy has
    a trajectory, a deviation from the optimum and a TWAP gap, and a
    liquidity-observing one has a *distribution* of schedules and a confidence
    interval. Reading the second through the first's fields is an
    ``AttributeError`` that only a training run can reach, which is exactly the
    class of defect ``docs/house-notes.md``'s *The artefact writer is tested on
    fabricated data* exists for — so the branch is here, in a pure function, and
    ``tests/test_sweep_document.py`` calls it on a fabricated grade.
    """
    common = (
        f"{result.seconds:.0f}s"
        f"{' (TIMED OUT)' if result.timed_out else ''}"
        f"{' · RED FLAG' if grade.red_flag else ''}"
    )
    if hasattr(grade, "premium_ratio"):  # AlphaGrade — three numbers, never one
        return (
            f"  seed {ordinal}: J {grade.objective:.6f} ± "
            f"{grade.half_width_bps:.6f} bps · {format_alpha_headline(grade)} · "
            f"{grade.paths:,} paths"
            f"{' · soft flag' if grade.soft_flag else ''} · {common}"
        )
    if hasattr(grade, "paths"):  # LiquidityGrade — graded by conditional expectation
        below = (
            ""
            if not grade.paths_below_clairvoyant
            else f" · {grade.paths_below_clairvoyant} PATHS BELOW CLAIRVOYANT"
        )
        return (
            f"  seed {ordinal}: J {grade.objective:.6f} ± {grade.half_width:.6f} bps "
            f"· excess over J_DP {grade.excess:+.5f} bps · capture "
            f"{grade.capture_fraction:.4f} · {grade.paths:,} paths{below}"
            f"{' · soft flag' if grade.soft_flag else ''} · {common}"
        )
    capture = (
        ""
        if grade.capture_fraction is None
        else f" · capture {grade.capture_fraction:.4f}"
    )
    return (
        f"  seed {ordinal}: J {grade.objective:.6f} bps · "
        f"excess {grade.excess:+.5f} bps = {grade.relative_excess:+.4%} = "
        f"{grade.gap_fraction:.1%} of the TWAP gap{capture} · "
        f"‖δ‖₂ {grade.deviation:,.0f} shares · {common}"
    )


def _on_seed(ordinal: int, grade, result, pairs=()) -> None:
    print(_seed_line(ordinal, grade, result), flush=True)
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
    if experiment.signal.informative:
        # M5's figure is the wrap-up session's, deliberately. Said out loud here
        # rather than silently skipped, because a `write_figure` that quietly
        # returns is exactly how a reporting path goes unexercised — and
        # `tests/test_m5_sweep_document.py` asserts that this branch is still the
        # one that runs, so writing the figure without wiring it into the pre-run
        # fabricated pass turns that test red.
        print("  (no figure: M5's is the wrap-up session's, not this run's)")
        return
    if experiment.liquidity.stochastic:
        _write_liquidity_figure(experiment, document)
        return
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


def _write_liquidity_figure(experiment: Experiment, document: dict) -> None:
    """M4b's two-panel figure, from the sweep and task 0's committed table.

    The value-of-sight curve lives in ``results/m4b_reference.json`` because it is
    oracle surface and has nothing to do with the agent; if that file is absent
    the figure is skipped with a message rather than half-drawn, because a curve
    against an invented parameter is the half of this figure that keeps the claim
    honest and a version without it would be the more flattering picture.
    """
    from temper.eval.figures import adaptivity_figure

    from tools.m4b_adaptivity import build_curve, build_rungs, caption

    table_path = REPO_ROOT / "results" / "m4b_reference.json"
    if not table_path.exists():
        print(
            f"  skipped the figure: {table_path.relative_to(REPO_ROOT)} is missing, "
            "and the value-of-sight curve against the invented sigma_L is not "
            "optional (run tools/m4b_reference_table.py)"
        )
        return
    rungs = build_rungs(document)
    written = adaptivity_figure(
        experiment.results_figure,
        rungs=rungs,
        curve=build_curve(json.loads(table_path.read_text(encoding="utf-8"))),
        provenance=Provenance(**document["provenance"]),
        caption=caption(experiment, document, rungs),
        formats=experiment.figure_formats,
    )
    for path in written:
        print(f"  wrote {_shown(path)}")


def _shown(path: Path) -> Path:
    """A path as the log should read it: repo-relative when it is inside the repo.

    Not decoration. ``--figure-only`` and the tests both point the figure
    somewhere else, and ``relative_to`` raises on a path outside the tree — a
    driver that dies while *reporting* where it wrote a file is the same shape of
    defect as one that dies while printing a grade.
    """
    try:
        return path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return path


def write_outputs(experiment: Experiment, document: dict) -> None:
    """The metrics JSON and the overlay figure, both provenance-stamped."""
    metrics_path = experiment.results_metrics
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {metrics_path.relative_to(REPO_ROOT)}")
    write_figure(experiment, document)


# ---------------------------------------------------------------------------
# Checkpoint export: one named seed, retrained and written down
# ---------------------------------------------------------------------------

#: How the exported policy is named beside its results JSON —
#: ``results/m4a_power_law.json`` -> ``results/m4a_power_law_policy.npz``.
CHECKPOINT_SUFFIX = "_policy.npz"

#: The rule that decides *which* seed gets exported, stated here rather than
#: chosen at export time. See :func:`~temper.eval.grading.median_ordinal`: the
#: seed at the upper of the two central ranks, which at an even seed count is
#: the worse of the pair. Recorded verbatim into the checkpoint's metadata, so
#: the artefact carries the rule it was selected by and not merely its result.
MEDIAN_RULE = (
    "the sweep's seeds sorted ascending by graded objective (a cost, so "
    "best-first), index n // 2 — the upper of the two central ranks, which at "
    "an even seed count is the worse of the two. A tie-break that can only "
    "cost, so the exported policy is at or below the sweep's median and never "
    "above it."
)


class _Recorder:
    """A policy that remembers every ``(observation, fraction)`` it was asked for.

    Wrapped around the trained :class:`~temper.agents.execution.PPOPolicy` for
    the grading rollout, so the observations and actions written into the
    checkpoint are the ones the *graded* schedule was actually produced from
    rather than a second rollout that ought to agree with it. It delegates the
    fraction and re-does the same conversion
    (:func:`~temper.agents.execution.fraction_to_shares`), so the env sees
    exactly what it would have seen without the wrapper.
    """

    def __init__(self, policy) -> None:
        self._policy = policy
        self.name = policy.name
        self.order_size = policy.order_size
        self.episodes: list[list[tuple[np.ndarray, float]]] = []

    def reset(self) -> None:
        self.episodes.append([])

    def act(self, observation) -> float:
        fraction = self._policy.fraction(observation)
        self.episodes[-1].append(
            (np.asarray(observation, dtype=np.float64).copy(), float(fraction))
        )
        return fraction_to_shares(fraction, observation, self.order_size)


def _checkpoint_path(experiment: Experiment, given: str | None) -> Path:
    if given:
        return Path(given)
    metrics = experiment.results_metrics
    return metrics.with_name(metrics.stem + CHECKPOINT_SUFFIX)


def _selected_ordinal(document: dict, requested: str) -> tuple[int, str]:
    """``(ordinal, how)`` — the seed to export, and the rule that chose it."""
    objectives = [record["grade"]["objective_bps"] for record in document["seeds"]]
    if requested == "median":
        return median_ordinal(objectives), MEDIAN_RULE
    ordinal = int(requested)
    if not 0 <= ordinal < len(objectives):
        raise SystemExit(
            f"--ordinal {ordinal} is outside the {len(objectives)} seeds the "
            "committed sweep recorded"
        )
    return ordinal, "named explicitly on the command line, not by the median rule"


def run_export(experiment: Experiment, args) -> int:
    """Retrain one seed of a committed sweep and write its policy to ``.npz``.

    Not a sweep: a sweep's number is a median over ten seeds and takes hours,
    while an *artefact* is one policy and takes one seed. The seed is chosen by
    :data:`MEDIAN_RULE` off the committed results JSON, so the export names a
    seed the sweep already reported rather than a fresh draw, and the grade the
    reloaded weights earn can be compared with the grade that seed was
    committed under.

    What is checked before the file is written is the *verdict*, not the digits.
    Torch's CPU reductions depend on the thread count and PPO compounds that
    over 751 updates, so bitwise agreement is a property of this host rather
    than of the pipeline (``tests/test_m2_rediscovery.py`` records the
    measurement). Whether it reproduced bitwise is recorded in the metadata
    either way; what would stop the write is the retrained seed missing M4a's
    per-seed bar or raising a red flag, which would mean the artefact does not
    represent the result it claims to.
    """
    import torch  # local: the top of this file stays importable without it

    metrics_path = experiment.results_metrics
    if not metrics_path.exists():
        print(
            f"{metrics_path.relative_to(REPO_ROOT)} does not exist; the export "
            "picks its seed off a committed sweep, so run the sweep first"
        )
        return 1
    document = json.loads(metrics_path.read_text(encoding="utf-8"))
    ordinal, how = _selected_ordinal(document, args.ordinal)
    committed = document["seeds"][ordinal]["grade"]
    target = _checkpoint_path(experiment, args.export_checkpoint)

    # The lambda rule, re-derived before anything trains — the same refusal
    # `run_sweep` opens with, and for a stronger reason here. This was missing:
    # the export was the *only* training entry point in the repo that trained
    # without it, and it is the one that writes the single artefact leaving the
    # repo, read by `client/` on a live venue. Unchecked, an edited `lambda_risk`
    # in a committed config yields a policy stamped with this milestone's name,
    # cross-referencing the sweep by digest, graded against the edited world's
    # own optimum — with `reproduced_bitwise: false` as the only trace, printed
    # as a diagnostic rather than raised as a refusal. Found by reconciling this
    # driver against the `run_sweep(ordinals=...)` factoring that reached the
    # same conclusion from the other direction.
    experiment.verify_lambda_rule()
    experiment.verify_gate_reference()

    provenance = experiment.provenance(REPO_ROOT)
    if provenance.git_dirty:
        print(
            "  WARNING: the source tree is dirty, so this checkpoint's recorded "
            "revision will not contain the code that produced it (invariant 1). "
            "Commit before the export run."
        )

    _header(experiment)
    print(
        f"  exporting seed {ordinal} of {len(document['seeds'])} — {how}\n"
        f"  committed grade: J {committed['objective_bps']:.9f} bps · capture "
        f"{committed['capture_fraction']:.6f} · excess "
        f"{committed['excess_bps']:+.8f} bps"
    )

    with _KeepAwake():
        result, policy = train_seed(
            experiment, ordinal, progress=None if args.quiet else _progress(experiment)
        )
    recorder = _Recorder(policy)
    regrade = grade(experiment, recorder, name=f"seed{ordinal}")

    # `grade` rolls the policy out once per eval stream and has already required
    # the two trajectories to be bitwise equal; this requires the same of the
    # *actions*, which is the stronger statement and the one `client/` will lean
    # on when it replays these observations without an env at all.
    episodes = recorder.episodes
    head = episodes[0]
    for index, other in enumerate(episodes[1:], start=1):
        if [f for _, f in other] != [f for _, f in head]:
            print(
                f"  eval stream {index} produced different actions from stream 0 "
                "— the policy is not open-loop and nothing here is gradable"
            )
            return 1
    observations = np.array([obs for obs, _ in head], dtype=np.float64)
    fractions = np.array([f for _, f in head], dtype=np.float64)

    bitwise = regrade.objective == committed["objective_bps"]
    drift = regrade.objective - committed["objective_bps"]
    print(
        f"  retrained grade:  J {regrade.objective:.9f} bps · capture "
        f"{regrade.capture_fraction:.6f} · excess {regrade.excess:+.8f} bps · "
        f"‖δ‖₂ {regrade.deviation:,.0f} shares · {result.seconds:.0f}s"
    )
    print(
        "  reproduced the committed objective bitwise"
        if bitwise
        else f"  differs from the committed objective by {drift:+.3e} bps "
        "(not bitwise — recorded, not hidden)"
    )

    per_seed_bar = experiment.tolerances.per_seed_fraction
    graded_value = getattr(regrade, experiment.tolerances.graded_attribute)
    if regrade.red_flag or graded_value > per_seed_bar:
        print(
            f"  REFUSING to write: the retrained seed scored "
            f"{graded_value:.4f} of the "
            f"{experiment.tolerances.denominator.replace('_', ' ')} against a "
            f"per-seed floor of {per_seed_bar:g}"
            + (" and raised a RED FLAG" if regrade.red_flag else "")
            + ". An artefact that misses the bar its sweep was reported under "
            "does not represent that result."
        )
        return 1

    ordering = np.argsort(
        [record["grade"]["objective_bps"] for record in document["seeds"]],
        kind="stable",
    ).tolist()
    metadata = {
        "name": f"{experiment.milestone.lower()}-seed{ordinal}",
        "milestone": experiment.milestone,
        "brief": "docs/briefs/M6-anvil-live-leg.md (prerequisite)",
        "written_by": "tools/train.py --export-checkpoint",
        "torch": torch.__version__,
        "provenance": provenance.as_dict(),
        "source_result": {
            "path": str(metrics_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "sha256": config_digest(metrics_path),
            "git_rev": document["provenance"]["git_rev"],
        },
        "selection": {
            "rule": how,
            "ordinal": ordinal,
            "n_seeds": len(document["seeds"]),
            "rank_from_best": ordering.index(ordinal) + 1,
            "sweep_median_objective_bps": document["summary"]["objective"]["median"],
        },
        "seed": {
            "root_seed": experiment.seeds.root_seed,
            "train_pool": experiment.seeds.train_pool,
            "eval_pool": experiment.seeds.eval_pool,
            "n_seeds": experiment.seeds.n_seeds,
            "ordinal": ordinal,
            "torch_seed": int(
                pool_seeds(
                    experiment.seeds.root_seed,
                    experiment.seeds.train_pool,
                    experiment.seeds.n_seeds,
                )[ordinal]
            ),
            "env_streams": list(
                experiment.seeds.env_streams(ordinal, experiment.ppo.num_envs)[:1]
            ),
            "eval_streams": list(experiment.seeds.eval_streams),
        },
        "world": {
            "cost_encoding": experiment.cost_encoding,
            "lambda_risk": experiment.lambda_risk,
            "order_size": experiment.case.order_size,
            "n_bins": experiment.case.market.n_bins,
            "horizon_hours": experiment.case.market.horizon_hours,
            "symbol": experiment.case.symbol,
            "case_id": experiment.case.case_id,
            "params_from": experiment.case.params_from,
            "note": (
                "the world the weights were trained in, recorded so a reader "
                "knows what they are. Nothing about it enters inference: the "
                "observation is (time remaining fraction, inventory remaining "
                "fraction) and no market parameter reaches the network, which "
                "is why this policy is portable to a venue it has never seen."
            ),
        },
        "network": network_description(result.agent, experiment.ppo.hidden_sizes),
        "grade": regrade.as_dict(),
        "committed_grade": committed,
        "reproduced_bitwise": bool(bitwise),
        "training": {
            "updates": result.updates,
            "global_step": result.global_step,
            "seconds": result.seconds,
            "timed_out": result.timed_out,
            "torch_threads": experiment.ppo.torch_threads,
        },
        "eval": {
            "streams": list(experiment.seeds.eval_streams),
            "arrays": {
                "eval_observations": "(n_bins, 2) float64 — one per decision",
                "eval_fractions": "(n_bins,) float64 — the squashed mean action",
                "eval_trajectory": "(n_bins + 1,) float64 — inventory, shares",
            },
            "note": (
                "one deterministic evaluation episode; every eval stream "
                "produced the same observations and actions bitwise. A second "
                "implementation of the forward pass is pinned by replaying "
                "`eval_observations` and matching `eval_fractions`, which needs "
                "neither an env nor torch."
            ),
        },
    }

    written = save_policy(
        target,
        result.agent,
        metadata,
        extra={
            "eval_observations": observations,
            "eval_fractions": fractions,
            "eval_trajectory": np.asarray(regrade.trajectory, dtype=np.float64),
        },
    )
    resolved = written.resolve()
    inside = resolved.is_relative_to(REPO_ROOT)
    print(f"  wrote {resolved.relative_to(REPO_ROOT) if inside else resolved}")
    return 0


def print_alpha_verdict(experiment: Experiment, document: dict) -> None:
    """M5's verdict, and the rule it exists to enforce: **all three, always**.

    The headline cannot appear alone. At the optimum 45 % of the gross effect is
    paid back, so a net capture of 0.9 is consistent with a policy that monetises
    everything and executes badly *and* with one that monetises little and
    executes well, and those are different results. The three come out of one
    formatter (:func:`~temper.eval.sweep.format_alpha_headline`) so that is a
    property of the code rather than of everyone's discipline, and every fraction
    carries the bps it is a fraction of.

    A pure function of ``(experiment, document)``, for the reason
    :func:`print_verdict` is one — and this milestone gave that reason a third
    airing, so the pre-run pass calls this on fabricated data before a seed runs.
    """
    verdict, summary = document["verdict"], document["summary"]
    head = verdict["headline"]
    graded = summary["advantage_fraction"]
    print()
    print(
        f"median advantage_fraction {graded['median']:.4f} (IQR {graded['iqr']:.4f}, "
        f"worst {graded['worst']:.4f}) against ε = "
        f"{experiment.tolerances.epsilon_fraction} and a per-seed floor of "
        f"{experiment.tolerances.per_seed_fraction}, both fractions of the net "
        f"signal advantage ({verdict['denominator_bps']:.5f} bps)"
    )
    print()
    print("THE THREE NUMBERS, and never one of them:")
    print(
        f"  alpha capture     {head['alpha_capture_median']:+.4f}  "
        f"({head['alpha_bps_median']:+.5f} of {head['reference_alpha_bps']:.5f} bps "
        f"gross) against a bar of {ALPHA_CAPTURE_BAR:.2f} — "
        f"{'met' if verdict['alpha_capture_met'] else 'MISSED'}"
    )
    print(
        f"  execution premium {head['premium_ratio_median']:.4f}x  "
        f"({head['execution_premium_bps_median']:+.5f} of "
        f"{head['reference_premium_bps']:.5f} bps the optimum pays) against a bar "
        f"of {PREMIUM_RATIO_BAR:.2f}x — "
        f"{'met' if verdict['premium_ratio_met'] else 'MISSED'}"
    )
    print(
        f"  net capture       {head['net_capture_median']:+.4f}  "
        f"({head['median_excess_bps']:+.5f} bps over J_DP, on an advantage of "
        f"{head['advantage_bps']:.5f} bps)"
    )
    print(
        "  The optimum gives back "
        f"{head['reference_premium_bps'] / head['reference_alpha_bps']:.1%} of the "
        "gross effect, so the net figure alone cannot distinguish a policy that "
        "trades the signal well from one that trades it badly and executes well."
    )

    control = document["shuffled_control"]["net_capture"]
    print()
    if control is None:
        print("signal-shuffled control: NOT RUN")
    else:
        print(
            f"signal-shuffled control: median net capture {control['median']:+.4f} "
            f"(worst {control['worst']:+.4f}) against a bar of "
            f"{verdict['shuffled_net_capture_bar']:+.2f} — "
            f"{'met' if verdict['shuffled_control_met'] else 'MISSED'}. The gap "
            f"between {head['net_capture_median']:+.4f} and {control['median']:+.4f} "
            "is the actual claim: a policy tilting on a signal unrelated to the "
            "prices pays the premium and monetises nothing."
        )

    if verdict["red_flags"]:
        print(
            f"RED FLAG (rigorous, and CERTIFIED): {verdict['red_flags']} scored "
            f"E[impact + risk] below M4a's certified optimum "
            f"({document['reference']['execution_floor_bps']:.6f} bps). Impact and "
            "risk are convex and carry no signal, so no policy can be below that."
        )
    if verdict["soft_flags"]:
        print(
            f"soft flag: {verdict['soft_flags']} scored below J_DP beyond the DP's "
            "own residual and this grade's half-width — reported and investigated, "
            "never auto-failed, because J_DP is converged and not certified"
        )
    if verdict["timed_out"]:
        print(
            f"BUDGET BOUND: seeds {verdict['timed_out']} stopped on wall-clock "
            "rather than on their update budget, so this sweep compares different "
            "amounts of training and does not pass whatever else it says"
        )
    print()
    print(
        f"verdict: {'PASSED' if verdict['passed'] else 'NOT PASSED'} · "
        f"{verdict['sweep_seconds']:.0f}s"
    )


def print_verdict(experiment: Experiment, document: dict) -> None:
    """Everything the run says after the last seed, in whichever world it ran in.

    A pure function of ``(experiment, document)`` and nothing else, extracted for
    the reason ``docs/house-notes.md``'s *The artefact writer is tested on
    fabricated data* gives — and extracted *after* it had already cost a run.
    This block was inline in ``main`` and therefore reachable only by training
    ten seeds first; it read ``summary['relative_excess']``, which is a
    deterministic-world key that a liquidity summary does not have, and M4b's
    sweep died on it **after** grading all ten seeds and writing both artefacts.
    Nothing was lost that time only because ``write_outputs`` runs before the
    printing does.

    Two worlds, two shapes, and the branches are on *what the summary contains*
    rather than on a milestone name: a liquidity-observing policy has a
    distribution of schedules, so there is no single trajectory deviation to
    quote and no derived band to quote it against.
    """
    if experiment.signal.informative:
        print_alpha_verdict(experiment, document)
        return
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
    if document.get("shuffled_control") is not None:
        control = document["shuffled_control"]
        print(
            f"level shift {verdict['level_shift_bps']:.5f} bps "
            f"({verdict['level_shift_fraction_of_advantage']:.1%} of the "
            f"advantage) — reported on its own line, and not the agent's: an "
            f"agent measured against M4a's schedule would appear to gain the "
            f"naive {verdict['naive_gap_bps']:.5f} bps"
        )
        print(
            f"liquidity-shuffled control: median capture {control['median']:.4f} "
            f"(worst {control['worst']:.4f}) against a bar of "
            f"{verdict['shuffled_capture_bar']:.2f} — "
            f"{'met' if verdict['shuffled_control_met'] else 'MISSED'}. The gap "
            f"between the real and shuffled capture is the actual claim."
        )
        if verdict["seeds_below_clairvoyant"]:
            print(
                f"RED FLAG (rigorous): {verdict['seeds_below_clairvoyant']} scored "
                "below the perfect-information relaxation on at least one path"
            )
    if "capture_fraction" in summary:
        capture = summary["capture_fraction"]
        # The fraction leads and the absolute excess goes beside it every time:
        # a capture fraction near 1 on an advantage of 0.037 bps is a small
        # absolute claim and should read as one (ARCHITECTURE.md §9). The
        # reference is only called *certified* where it is one — M4b's is a
        # dynamic program and gets the word it earned instead.
        reference_name = (
            "J_DP (converged, bracketed)"
            if document.get("reference_kind")
            else "the certified optimum"
        )
        print(
            f"capture fraction: median {capture['median']:.4f} "
            f"(IQR {capture['iqr']:.4f}, worst {capture['worst']:.4f}) · "
            f"median absolute excess over {reference_name} "
            f"{verdict['median_excess_bps']:+.5f} bps"
        )
    # Trajectory-space reporting, and only where a trajectory is the answer. A
    # liquidity-observing policy has a *distribution* of schedules, so there is
    # no single deviation to quote and no band to quote it against — the
    # summary carries neither key and asking for them is how this line died
    # after a full sweep had been graded and written.
    if "relative_excess" in summary:
        print(
            f"median excess {summary['relative_excess']['median']:+.4%} of J_optimal · "
            f"median ‖δ‖₂ {summary['deviation']['median']:,.0f} shares against a "
            f"derived bound of {experiment.band().bound_shares:,.0f}"
        )
    if verdict["red_flags"]:
        print(
            f"RED FLAG — {', '.join(verdict['red_flags'])} scored below the "
            "reference. This is a defect, not a result (ARCHITECTURE.md §1.1)."
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


class _TiltedSchedule:
    """A fabricated policy that actually *reads* the signal coordinate.

    The reporting pass needs one, and TWAP will not do. A fixed schedule ignores
    the observation, so the shuffled re-grade returns the identical number and the
    control's whole code path is exercised without the control being exercised —
    which is the shape of check that passes and means nothing.

    This tilts each bin's TWAP slice by the observed signal, clipped to the
    reachable set. Not a good policy and not meant to be: what it has to do is
    react, so that shuffling the signal it reacts to *changes the answer*, and the
    pass can assert that it did.
    """

    name = "fabricated_tilt"

    def __init__(self, market, order_size: float, strength: float = 4.0) -> None:
        self._market = market
        self._order_size = float(order_size)
        self._strength = float(strength)
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def act(self, observation) -> float:
        remaining_bins = self._market.n_bins - self._step
        self._step += 1
        inventory = float(observation[1]) * self._order_size
        if remaining_bins <= 1:
            return inventory
        slice_size = inventory / remaining_bins
        tilt = 1.0 - self._strength * float(observation[2])
        return min(max(slice_size * tilt, 0.0), inventory)


def assert_training_and_grading_agree(experiment: Experiment) -> tuple:
    """The training env and the graded env must show the policy the same thing.

    Cheap, and it cost twenty minutes to learn. M5 task 6's first sweep trained
    seed 0 on a **two**-coordinate observation — `train_seed` had never been handed
    the signal stream — and then graded it on a three-coordinate one. The network
    is built at the training width, so the first forward pass in grading died on
    ``mat1 and mat2 shapes cannot be multiplied (1x3 and 2x64)``.

    Loud is the only reason that cost a seed rather than a milestone. An agent
    trained blind and graded sighted, had the widths happened to match, would have
    produced a perfectly plausible capture fraction about a world it never traded
    in — and every identity, differential and guard in this repo would have stayed
    green, because each of them checks one env at a time and this is a statement
    about **two**.

    Returns the two observation spaces, so a caller can print what it checked.
    """
    from temper.agents.execution import execution_env_factory
    from temper.eval.sweep import (
        evaluation_liquidity,
        evaluation_signal,
        training_liquidity,
        training_signal,
    )

    case = experiment.case
    impact = impact_for(experiment.cost_encoding, case.market, case.order_size)
    training = execution_env_factory(
        case.market,
        case.order_size,
        experiment.lambda_risk,
        root_seed=experiment.seeds.root_seed,
        pool=experiment.seeds.train_pool,
        stream_index=0,
        temporary_impact=impact,
        liquidity=training_liquidity(experiment),
        signal=training_signal(experiment),
    )()
    graded = ExecutionEnv(
        case.market,
        case.order_size,
        experiment.lambda_risk,
        temporary_impact=impact,
        liquidity=evaluation_liquidity(experiment),
        signal=evaluation_signal(experiment),
        root_seed=experiment.seeds.root_seed,
        pool=experiment.seeds.eval_pool,
    )
    if training.observation_space.shape != graded.observation_space.shape:
        raise AssertionError(
            f"the training env shows {training.observation_space.shape} and the "
            f"graded env shows {graded.observation_space.shape}. A policy trained "
            "on one and graded on the other is being scored in a world it never "
            "traded in — and if the widths had matched, nothing would have said so"
        )
    # And the two must be *different streams* of the same shape: same width,
    # disjoint pools. Invariant 5's out-of-sample claim is a property of the spawn
    # keys, and this is the one place both halves are in scope at once.
    if experiment.signal.informative:
        assert training.unwrapped.signal_address[1] != graded.signal_address[1], (
            "training and evaluation draw their signal from the same pool"
        )
    return training.observation_space, graded.observation_space


def alpha_reporting_pass(experiment: Experiment, *, paths: int = 256) -> dict:
    """Exercise **every** path that runs after the sweep, on fabricated data.

    ``docs/house-notes.md``, *No code path may be reachable only at the end of a
    long run*, applied as M5's ROADMAP row asks: to the whole reporting path
    rather than to the producer. M4b obeyed that note where its brief named the
    function and was bitten **four more times** in code the brief had not named,
    twice after a full sweep had already been graded.

    So the list is explicit and each entry is named rather than implied:

    1. the document assembly (:func:`~temper.eval.sweep.build_alpha_document`);
    2. the three-number computation — alpha capture, execution premium, net
       capture — through the formatter every call site uses;
    3. the **shuffled-control re-grade**, which is the one path here that no fast
       test reaches and that in a real run executes only after a seed has trained
       (its own line, below, for exactly that reason);
    4. the red-flag evaluation, on a fabricated grade that *trips* it, because a
       flag that has never fired is a branch that has never run;
    5. the verdict block (:func:`print_alpha_verdict`);
    6. the per-seed line (:func:`_seed_line`);
    7. every line that reports where a file was written
       (:func:`write_outputs`, into a scratch directory).

    Returns the fabricated document, so a test can assert on the same object the
    run exercised rather than on a second one built to look like it.
    """
    from temper.eval.sweep import (
        SweepResult,
        alpha_reference,
        build_alpha_document,
        grade_alpha,
        grade_alpha_baselines,
    )

    case = experiment.case
    training_space, graded_space = assert_training_and_grading_agree(experiment)
    print(
        f"  reporting pass · widths     training {training_space.shape} == "
        f"graded {graded_space.shape}"
    )
    reference = alpha_reference(experiment)
    policy = _TiltedSchedule(case.market, case.order_size)

    # (2) and (3): a real grade and a real *shuffled* re-grade, at a path count
    # that costs seconds. The control is the one path that only executes after a
    # seed exists in a real run, which is the exposure the house note describes.
    grade, detail = grade_alpha(
        experiment, policy, reference, name="fabricated", paths=paths
    )
    shuffled, _ = grade_alpha(
        experiment,
        policy,
        reference,
        name="fabricated_shuffled",
        paths=paths,
        shuffled=True,
    )
    print(f"  reporting pass · grade      {format_alpha_headline(grade)}")
    print(f"  reporting pass · shuffled   {format_alpha_headline(shuffled)}")
    # The control has to *do* something, or its code path is exercised while the
    # control is not. A reacting policy shown an unrelated signal must land
    # somewhere else.
    if shuffled.alpha_bps == grade.alpha_bps:
        raise AssertionError(
            "the shuffled re-grade returned the same alpha as the real one, so "
            "the control changed nothing. Either the fabricated policy is not "
            "reading the signal or ShuffledSignal is not replacing it"
        )

    # (4) the red-flag branch, fired rather than merely present.
    tripped = replace(grade, red_flag=True, soft_flag=True, name="fabricated_flagged")
    assert tripped.red_flag and tripped.soft_flag

    # The baselines run at the *start* of `run_sweep` rather than after it, so
    # they are not on the reporting path — and they are on the sweep's critical
    # path, which a missing import in this very function proved by failing the
    # first launch. Cheap to include, and the note's rule reads "no code path
    # reachable only inside a long run" as easily as "only at the end of one".
    fabricated_baselines = grade_alpha_baselines(experiment, reference, paths)
    print(
        "  reporting pass · baselines  "
        + ", ".join(
            f"{name} {g.objective:.5f}" for name, g in fabricated_baselines.items()
        )
    )

    fabricated = _fabricated_training(experiment)
    sweep = SweepResult(
        experiment=experiment,
        baselines={},
        grades=(),
        training=tuple(fabricated),
        seconds=1.0,
        provenance=experiment.provenance(REPO_ROOT),
        pairs=tuple(() for _ in fabricated),
        ordinals=tuple(range(experiment.seeds.n_seeds)),
        alpha_grades=tuple(
            replace(grade, name=f"seed{i}") for i in range(experiment.seeds.n_seeds)
        ),
        shuffled_alpha_grades=tuple(
            replace(shuffled, name=f"seed{i}_shuffled")
            for i in range(experiment.seeds.n_seeds)
        ),
        alpha_detail=tuple(dict(detail) for _ in range(experiment.seeds.n_seeds)),
        alpha_reference_row=reference,
        alpha_baselines=fabricated_baselines,
    )

    # (1) the document, (5) the verdict, (6) the per-seed line, (7) the writes.
    document = build_alpha_document(sweep)
    for ordinal in range(experiment.seeds.n_seeds):
        print(_seed_line(ordinal, sweep.alpha_grades[ordinal], fabricated[ordinal]))
    print(_seed_line(0, tripped, fabricated[0]))
    print_alpha_verdict(experiment, document)

    scratch = REPO_ROOT / "results" / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    write_outputs(replace(experiment, results_metrics=scratch / "m5_pass.json"), document)
    return document


def _fabricated_training(experiment: Experiment):
    """One :class:`TrainResult` per seed, with nothing trained.

    Deliberately *not* a mock: the reporting path reads `config.num_updates`,
    `timed_out` and the traces off these, and a stub that answered by attribute
    would exercise the reporting rather than the reporting's contract with the
    trainer. One of them is marked budget-bound, so the branch that refuses a
    bound sweep runs here too.
    """
    from temper.agents.ppo import TrainResult

    updates = experiment.ppo.num_updates
    return [
        TrainResult(
            agent=None,
            config=experiment.ppo,
            seed=ordinal,
            global_step=experiment.ppo.total_timesteps,
            updates=updates,
            seconds=1.0,
            timed_out=False,
            returns=[0.0] * min(updates, 8),
            episode_counts=[1] * min(updates, 8),
            approx_kls=[0.0] * min(updates, 8),
            entropies=[0.0] * min(updates, 8),
            value_losses=[0.0] * min(updates, 8),
            return_variances=[0.0] * min(updates, 8),
        )
        for ordinal in range(experiment.seeds.n_seeds)
    ]


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
        "--export-checkpoint",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help=(
            "train one seed of the committed sweep and write its policy as a "
            "provenance-stamped .npz beside the results JSON; no sweep, no "
            "figure. PATH overrides the default location"
        ),
    )
    parser.add_argument(
        "--ordinal",
        default="median",
        help=(
            "which seed --export-checkpoint exports. `median` (the default) "
            "reads the committed sweep and applies the median rule; an integer "
            "names a seed explicitly and is recorded as such in the artefact"
        ),
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
    if args.export_checkpoint is not None:
        return run_export(experiment, args)
    if args.dry_run:
        reference = experiment.verify_lambda_rule()
        gate_reference = experiment.verify_gate_reference()
        if experiment.liquidity.stochastic:
            # The bar cannot be printed without the dynamic program, and printing
            # the deterministic world's number instead would understate it by
            # 1.7x — in the flattering direction. So the dry run says what it can
            # check (the rule, the world, both seams) and says plainly what it
            # cannot.
            print(
                f"config OK · λ = {experiment.lambda_risk:.6e} matches the rule in "
                f"every reading · {experiment.cost_encoding} world + "
                f"{experiment.liquidity.name} liquidity "
                f"(sigma_log = {experiment.liquidity.sigma_log:g}, INVENTED)"
            )
            _liquidity_header(experiment)
            print(
                f"ε = {experiment.tolerances.epsilon_fraction:.0%} of the ADAPTIVE "
                f"advantage J_static* - J_DP, which needs the dynamic program and "
                f"is computed once by the sweep — deliberately not printed here "
                f"from the deterministic world's tangent advantage, which is a "
                f"different and smaller number"
            )
            return 0
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
    if experiment.signal.informative:
        # BEFORE the first seed, not after the last. Every path that runs after
        # the sweep, exercised on fabricated data — including the shuffled
        # control, which in a real run cannot execute until a seed has trained.
        print("\n-- pre-run reporting pass (fabricated data, no training) --")
        alpha_reporting_pass(experiment)
        print("-- reporting pass complete; starting seed 0 --\n")
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

    if sweep.liquidity_baselines:
        baselines = ", ".join(
            f"{name} {g.objective:.5f}"
            for name, g in sweep.liquidity_baselines.items()
        )
        print(f"  baselines graded through the same rollout (bps): {baselines}")
    elif sweep.alpha_baselines:
        baselines = ", ".join(
            f"{name} {g.objective:.5f}" for name, g in sweep.alpha_baselines.items()
        )
        print(f"  baselines graded through the same rollout (bps): {baselines}")
    else:
        baselines = ", ".join(
            f"{name} {g.gap_fraction:+.4f}" for name, g in sweep.baselines.items()
        )
        print(f"  baselines graded through the same rollout: {baselines}")

    if not args.no_write:
        write_outputs(experiment, document)

    print_verdict(experiment, document)

    verdict = document["verdict"]
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
