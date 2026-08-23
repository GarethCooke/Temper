"""The multi-seed sweep: train, grade analytically, summarise. No figures, no CLI.

Package surface rather than driver surface, because the suite has to be able to
run it. ``tests/test_m2_rediscovery.py`` regenerates one seed and checks it
reproduces its committed grade — invariant 1 end to end — and a sweep that lived
only inside ``tools/`` could not be asked to do that without a subprocess and a
parsed stdout.

Three refusals are built in rather than left to the caller:

* the lambda is re-derived from task 0's rule before anything trains
  (:meth:`~temper.eval.experiment.Experiment.verify_lambda_rule`);
* every seed's grade comes from :mod:`temper.eval.grading`, which asserts the
  eval schedule is shock-independent before it computes anything;
* a seed scoring below the certified optimum is recorded as a red flag and makes
  the verdict fail, rather than being reported as the agent winning
  (``ARCHITECTURE.md`` §1.1).

Nothing here imports matplotlib. The figure is the driver's job, so the core
stays headless.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from temper.agents.execution import PPOPolicy, execution_env_factory
from temper.agents.ppo import TrainResult, train
from temper.env import impact_for
from temper.eval.antithetic import PairLedger, PairUpdateStats, antithetic_reward
from temper.eval.experiment import ANTITHETIC, CONTROL_VARIATE, SAMPLED, Experiment
from temper.eval.grading import Grade, grade_policy, summarise
from temper.eval.provenance import Provenance
from temper.eval.variate import deterministic_reward
from temper.seeding import pool_seeds

#: The quantities reported with median and IQR across seeds. All are costs, so
#: larger is worse for every one of them without exception — which is why the
#: capture fraction is *not* here and ``advantage_fraction``, its complement, is:
#: :func:`~temper.eval.grading.summarise` takes ``worst`` to be ``max`` for
#: everything it is handed, and one quantity where bigger is better would get
#: that backwards exactly once, silently, in a verdict.
SUMMARISED = ("gap_fraction", "relative_excess", "objective", "deviation")

#: Reported as well, but only in a world where the closed form left something on
#: the table (M4a). ``advantage_fraction`` is ``None`` elsewhere.
ADVANTAGE_SUMMARISED = ("advantage_fraction",)


def reward_wrapper(experiment: Experiment, ledger: PairLedger | None = None):
    """The estimator's env wrapper for this experiment's regime, or ``None``.

    One place that maps a regime name to the code that implements it, so a
    results file's ``estimator.regime`` and the wrapper that produced its
    numbers cannot drift apart. The antithetic wrapper records into `ledger`.
    """
    regime = experiment.estimator.regime
    if regime == SAMPLED:
        return None
    if regime == CONTROL_VARIATE:
        return deterministic_reward
    if regime == ANTITHETIC:
        return antithetic_reward(ledger)
    raise ValueError(f"no reward wrapper for estimator regime {regime!r}")


def train_seed(
    experiment: Experiment,
    ordinal: int,
    *,
    progress: Callable[[int, dict], None] | None = None,
    ledger: PairLedger | None = None,
) -> tuple[TrainResult, PPOPolicy]:
    """Train one seed at the address the config gives it.

    The seed address does two jobs at once and they are deliberately the same
    address: ``pool_seeds(root, train_pool, n)[ordinal]`` seeds torch, and
    ``seeds.env_streams(ordinal, num_envs)`` names the shock streams. "Seed 3"
    is therefore one reproducible object rather than two things that happen to
    be numbered alike (invariant 5).

    Under the antithetic regime every env of the seed records both halves'
    episode returns into `ledger` (one is made if none is given), and the ledger
    is closed once per update from the training loop's progress hook — so
    ``ledger.updates`` is the per-update reward-variance trace when this
    returns. Other regimes leave the ledger untouched.
    """
    case, seeds, ppo = experiment.case, experiment.seeds, experiment.ppo
    if ledger is None:
        ledger = PairLedger()
    wrapper = reward_wrapper(experiment, ledger)
    impact = impact_for(experiment.cost_encoding, case.market, case.order_size)

    def hook(update: int, metrics: dict) -> None:
        if experiment.estimator.antithetic:
            ledger.close_update()
        if progress is not None:
            progress(update, metrics)

    factories = [
        execution_env_factory(
            case.market,
            case.order_size,
            experiment.lambda_risk,
            root_seed=seeds.root_seed,
            pool=seeds.train_pool,
            stream_index=stream,
            reward_scale=experiment.reward_scale,
            reward_wrapper=wrapper,
            temporary_impact=impact,
        )
        for stream in seeds.env_streams(ordinal, ppo.num_envs)
    ]
    torch_seed = pool_seeds(seeds.root_seed, seeds.train_pool, seeds.n_seeds)[ordinal]

    result = train(factories, ppo, seed=torch_seed, progress=hook)
    return result, PPOPolicy(result.agent, case.order_size, name=f"ppo_seed{ordinal}")


def grade(experiment: Experiment, policy, *, name: str) -> Grade:
    """Grade a policy on the committed eval streams, analytically."""
    return grade_policy(
        policy,
        experiment.case.market,
        experiment.case.order_size,
        experiment.reference(),
        root_seed=experiment.seeds.root_seed,
        pool=experiment.seeds.eval_pool,
        streams=experiment.seeds.eval_streams,
        name=name,
    )


def grade_baselines(experiment: Experiment) -> dict[str, Grade]:
    """TWAP, the vendored AC schedule and the optimum, through the same path.

    Not read off the oracle. Running them through the rollout and the grader is
    the cheapest possible check that the path returns the oracle's own numbers
    when it is handed the oracle's own schedules — and it is what puts the three
    baselines on every table, as invariant 4 requires.
    """
    from temper.agents.baselines import WORLD_BASELINES, baseline

    case = experiment.case
    encoding = experiment.cost_encoding
    return {
        name: grade(
            experiment,
            baseline(
                name,
                case.market,
                case.order_size,
                experiment.lambda_risk,
                encoding=encoding,
            ),
            name=name,
        )
        for name in WORLD_BASELINES[encoding]
    }


@dataclass(frozen=True)
class SweepResult:
    """Everything the sweep produced, before it becomes JSON or a figure."""

    experiment: Experiment
    baselines: dict[str, Grade]
    grades: tuple[Grade, ...]
    training: tuple[TrainResult, ...]
    seconds: float
    provenance: Provenance
    #: Per seed, the per-update pair statistics; empty tuples outside the
    #: antithetic regime.
    pairs: tuple[tuple[PairUpdateStats, ...], ...] = ()
    #: The seed ordinals this run actually trained, in the order it trained
    #: them. Carried rather than re-derived, because everything downstream that
    #: names a seed needs its *address* and not its position in a list: a subset
    #: run's third result is not seed 2, and `build_document` would otherwise
    #: stamp it with seed 2's `env_stream_base`. Empty means the full range,
    #: which is what every committed sweep so far was.
    ordinals: tuple[int, ...] = ()

    @property
    def addresses(self) -> tuple[int, ...]:
        """The ordinal of each trained seed, positionally aligned with `grades`."""
        return self.ordinals or tuple(range(len(self.grades)))

    @property
    def trajectories(self) -> list[list[float]]:
        return [[float(x) for x in g.trajectory] for g in self.grades]


def run_sweep(
    experiment: Experiment,
    *,
    repo_root: Path | None = None,
    on_seed: Callable[..., None] | None = None,
    progress: Callable[[int, dict], None] | None = None,
    ordinals: Sequence[int] | None = None,
) -> SweepResult:
    """Every seed, trained and graded. Verifies the lambda rule first.

    `ordinals` restricts the run to a subset of the config's seeds, addressed
    exactly as the full sweep addresses them — seed 9 of a ten-seed config is the
    same object whether it is run first, ninth or alone, because its torch seed
    and its env streams both come from its ordinal and neither depends on what
    ran before it (:func:`train_seed`). ``tests/test_m4a_phase1_regression.py``
    is the standing evidence: one seed, retrained in isolation, reproduces its
    committed grade bitwise.

    The result of a subset run is **not** an acceptance artefact — a median over
    one seed is not a median, and invariant 4 asks for dispersion. It exists so a
    single seed's *policy* can be re-derived without spending the whole sweep
    again; the caller is expected to write a checkpoint rather than a metrics
    file, and ``tools/train.py --export-checkpoint`` writes exactly that and
    never builds a metrics document at all.

    `build_document` refuses a subset run outright. A metrics file is a claim
    about a *sweep*, and there is no honest way to write one from one seed: the
    median, the IQR and the worst-seed verdict are all statements about ten.
    """
    experiment.verify_lambda_rule()
    experiment.verify_gate_reference()
    # Stamped *before* any training, not after. A sweep runs for hours; the
    # question the stamp answers is "which source tree produced this?", and that
    # is the tree the run started from. An end-of-run stamp would silently
    # attribute the result to whatever the repo looked like two hours later —
    # including edits made while it ran, and including its own sibling sweep's
    # freshly written artefacts.
    provenance = experiment.provenance(repo_root)
    started = time.perf_counter()

    baselines = grade_baselines(experiment)
    grades: list[Grade] = []
    training: list[TrainResult] = []
    pairs: list[tuple[PairUpdateStats, ...]] = []
    selected = (
        range(experiment.seeds.n_seeds) if ordinals is None else tuple(ordinals)
    )
    for ordinal in selected:
        if not 0 <= ordinal < experiment.seeds.n_seeds:
            raise ValueError(
                f"seed ordinal {ordinal} is outside the config's "
                f"{experiment.seeds.n_seeds} seeds; an ordinal is an address into "
                "the committed seed pool, not a free-running counter"
            )
        ledger = PairLedger()
        result, policy = train_seed(
            experiment, ordinal, progress=progress, ledger=ledger
        )
        seed_grade = grade(experiment, policy, name=f"seed{ordinal}")
        grades.append(seed_grade)
        training.append(result)
        pairs.append(tuple(ledger.updates))
        if on_seed is not None:
            on_seed(ordinal, seed_grade, result, tuple(ledger.updates))

    return SweepResult(
        experiment=experiment,
        baselines=baselines,
        grades=tuple(grades),
        training=tuple(training),
        seconds=time.perf_counter() - started,
        provenance=provenance,
        pairs=tuple(pairs),
        ordinals=tuple(selected),
    )


def thin(trace: list[float], points: int | None) -> list[float]:
    """Uniformly subsample `trace` to at most `points`, keeping both ends.

    ``None`` keeps the trace whole, which is what M2 commits: at five seeds the
    per-update traces are what make the seed spread *checkable* rather than
    asserted, and 1.2 MB is a fair price for that. M3 is a different arithmetic —
    17 lambdas times five seeds is ~20 MB of the same thing — so the budget is a
    committed field rather than a decision deferred to whoever notices the repo
    has grown (``ROADMAP.md``, M3 row).
    """
    if points is None or points <= 0 or len(trace) <= points:
        return list(trace)
    if points == 1:
        return [trace[-1]]
    last = len(trace) - 1
    indices = sorted({round(i * last / (points - 1)) for i in range(points)})
    return [trace[i] for i in indices]


#: Per-update traces in a training record, subject to the trace budget.
TRACES = (
    "train_returns",
    "train_return_variance",
    "approx_kl",
    "entropy",
    "value_loss",
)

#: Per-update traces in an antithetic pair record, subject to the same budget.
PAIR_TRACES = (
    "sampled_variance",
    "mirror_variance",
    "averaged_variance",
    "cancelled_mean_square",
    "variance_ratio",
)


def _training_record(result: TrainResult, points: int | None) -> dict:
    """A seed's training record, with its per-update traces thinned to budget."""
    record = result.as_dict()
    for name in TRACES:
        record[name] = thin(record[name], points)
    return record


def _nanmedian(values) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0 or np.all(np.isnan(array)):
        return float("nan")
    return float(np.nanmedian(array))


def _pair_record(updates: tuple[PairUpdateStats, ...], points: int | None) -> dict:
    """A seed's antithetic evidence: the per-update traces, and their medians.

    The medians are over updates and are what the brief's "realised reward
    variance per update against the sampled regime's" reads as one number per
    seed; the traces are what make it checkable.
    """
    columns = {name: [getattr(u, name) for u in updates] for name in PAIR_TRACES}
    return {
        "updates": len(updates),
        "episodes_per_update": (
            int(np.median([u.episodes for u in updates])) if updates else 0
        ),
        "median": {name: _nanmedian(values) for name, values in columns.items()},
        "traces": {name: thin(values, points) for name, values in columns.items()},
    }


def _gate_block(experiment: Experiment, median_gap: float) -> dict | None:
    """The validation gate's verdict, against the committed reference result."""
    gate = experiment.gate
    if gate is None:
        return None
    if not gate.reference.exists():
        raise FileNotFoundError(
            f"the gate references {gate.reference}, which does not exist; the "
            "committed result it validates against must be present"
        )
    reference = json.loads(gate.reference.read_text(encoding="utf-8"))
    reference_median = float(reference["summary"]["gap_fraction"]["median"])
    return {
        "median_gap_fraction_max": gate.median_gap_fraction,
        "reference": gate.reference.name,
        "reference_median_gap_fraction": reference_median,
        "reference_regime": reference["config"]["estimator"].get(
            "regime",
            "control_variate"
            if reference["config"]["estimator"].get("control_variate")
            else "sampled",
        ),
        "median_gap_fraction": median_gap,
        "met": bool(median_gap <= gate.median_gap_fraction),
    }


def build_document(sweep: SweepResult) -> dict:
    """The results JSON: the claim, the provenance, the numbers, the verdict.

    Refuses a subset run. Every number below is a statement about the *sweep* —
    the median, the IQR, the worst seed, the epsilon verdict — and none of them
    means anything computed over one seed; invariant 4 asks for dispersion and a
    median over one value is not a median. The refusal is here rather than in the
    caller because this is the function that would write the file.

    The sharper reason is provenance. Until `ordinals` existed, the seed records
    took their address from their *position* in the list, which was correct
    exactly because the list was always the full range. A subset run breaks that
    silently: `run_sweep(ordinals=[9])` would have written seed 9's grade under
    ``"ordinal": 0`` with seed 0's ``env_stream_base``, which is a false
    provenance stamp in the one file the whole repo's invariant 1 rests on.
    :attr:`SweepResult.addresses` fixes the labelling; this refusal removes the
    question.
    """
    experiment = sweep.experiment
    if sweep.ordinals and tuple(sweep.ordinals) != tuple(
        range(experiment.seeds.n_seeds)
    ):
        raise ValueError(
            f"this sweep trained ordinals {list(sweep.ordinals)} of the config's "
            f"{experiment.seeds.n_seeds}; a metrics document is a claim about a "
            "sweep and cannot be written from a subset. Export the policy "
            "instead (tools/train.py --export-checkpoint)."
        )
    tolerances = experiment.tolerances
    reference = experiment.reference()

    names = list(SUMMARISED)
    if reference.available_advantage is not None:
        names += list(ADVANTAGE_SUMMARISED)
    summary = {
        name: summarise(name, [getattr(g, name) for g in sweep.grades]).as_dict()
        for name in names
    }
    # The capture fraction is the number M4a leads with, so it is written out
    # rather than left to the reader to subtract — from the *same* summary, so
    # the two can never disagree about which seed was worst.
    if "advantage_fraction" in summary:
        advantage = summary["advantage_fraction"]
        summary["capture_fraction"] = {
            "name": "capture_fraction",
            "values": [1.0 - v for v in advantage["values"]],
            "median": 1.0 - advantage["median"],
            "q1": 1.0 - advantage["q3"],
            "q3": 1.0 - advantage["q1"],
            "iqr": advantage["iqr"],
            "worst": 1.0 - advantage["worst"],
        }
    # Which fraction the pre-stated bars are read on. The denominator is the
    # committed decision (`tolerances.denominator`); this is the field it names.
    graded_on = summary[tolerances.graded_attribute]
    red_flags = [g.name for g in sweep.grades if g.red_flag]
    verdict = {
        "tolerance_denominator": tolerances.denominator,
        "graded_attribute": tolerances.graded_attribute,
        "epsilon_met": bool(
            graded_on["median"] <= tolerances.epsilon_fraction
        ),
        "per_seed_met": bool(
            graded_on["worst"] <= tolerances.per_seed_fraction
        ),
        "red_flags": red_flags,
        "timed_out": [
            ordinal
            for ordinal, result in zip(sweep.addresses, sweep.training)
            if result.timed_out
        ],
        "sweep_seconds": sweep.seconds,
        "within_sweep_budget": bool(
            sweep.seconds <= experiment.runtime.sweep_seconds
        ),
    }
    verdict["passed"] = bool(
        verdict["epsilon_met"] and verdict["per_seed_met"] and not red_flags
    )
    gate = _gate_block(experiment, summary["gap_fraction"]["median"])
    verdict["denominator_bps"] = experiment.denominator_bps(reference)
    verdict["median_excess_bps"] = summarise(
        "excess", [g.excess for g in sweep.grades]
    ).median
    if gate is not None:
        verdict["gate_met"] = gate["met"]

    points = experiment.trace_points
    antithetic = experiment.estimator.antithetic
    pairs = sweep.pairs if sweep.pairs else tuple(() for _ in sweep.grades)
    if antithetic:
        summary["reward_variance"] = {
            "unit": "bps^2 of episode return, unscaled",
            "sampled_median": _nanmedian(
                _nanmedian(u.sampled_variance for u in seed_pairs)
                for seed_pairs in pairs
            ),
            "averaged_median": _nanmedian(
                _nanmedian(u.averaged_variance for u in seed_pairs)
                for seed_pairs in pairs
            ),
            "variance_ratio_median": _nanmedian(
                _nanmedian(u.variance_ratio for u in seed_pairs)
                for seed_pairs in pairs
            ),
        }

    return {
        "milestone": experiment.milestone,
        "claim": experiment.estimator.claim,
        "provenance": sweep.provenance.as_dict(),
        "config": experiment.as_dict(),
        "reference": reference.as_dict(),
        "bands": {
            "epsilon": experiment.band().as_dict(),
            "per_seed": experiment.band(
                tolerances.per_seed_fraction
            ).as_dict(),
        },
        "baselines": {name: g.as_dict() for name, g in sweep.baselines.items()},
        "trace_points": points,
        "seeds": [
            {
                "ordinal": ordinal,
                "env_stream_base": experiment.seeds.env_streams(
                    ordinal, experiment.ppo.num_envs
                )[0],
                "training": _training_record(result, points),
                **(
                    {"pair": _pair_record(seed_pairs, points)} if antithetic else {}
                ),
                "grade": seed_grade.as_dict(),
            }
            for ordinal, seed_grade, result, seed_pairs in zip(
                sweep.addresses, sweep.grades, sweep.training, pairs
            )
        ],
        "summary": summary,
        "gate": gate,
        "verdict": verdict,
    }
