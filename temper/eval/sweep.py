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

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from temper.agents.execution import PPOPolicy, execution_env_factory
from temper.agents.ppo import TrainResult, train
from temper.eval.experiment import Experiment
from temper.eval.grading import Grade, grade_policy, summarise
from temper.eval.variate import deterministic_reward
from temper.seeding import pool_seeds

#: The quantities reported with median and IQR across seeds. All are costs, so
#: larger is worse for every one of them without exception.
SUMMARISED = ("gap_fraction", "relative_excess", "objective", "deviation")


def train_seed(
    experiment: Experiment,
    ordinal: int,
    *,
    progress: Callable[[int, dict], None] | None = None,
) -> tuple[TrainResult, PPOPolicy]:
    """Train one seed at the address the config gives it.

    The seed address does two jobs at once and they are deliberately the same
    address: ``pool_seeds(root, train_pool, n)[ordinal]`` seeds torch, and
    ``seeds.env_streams(ordinal, num_envs)`` names the shock streams. "Seed 3"
    is therefore one reproducible object rather than two things that happen to
    be numbered alike (invariant 5).
    """
    case, seeds, ppo = experiment.case, experiment.seeds, experiment.ppo
    wrapper = deterministic_reward if experiment.estimator.control_variate else None

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
        )
        for stream in seeds.env_streams(ordinal, ppo.num_envs)
    ]
    torch_seed = pool_seeds(seeds.root_seed, seeds.train_pool, seeds.n_seeds)[ordinal]

    result = train(factories, ppo, seed=torch_seed, progress=progress)
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
    from temper.agents.baselines import baseline

    case = experiment.case
    return {
        name: grade(
            experiment,
            baseline(name, case.market, case.order_size, experiment.lambda_risk),
            name=name,
        )
        for name in ("twap", "ac", "optimal")
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

    @property
    def trajectories(self) -> list[list[float]]:
        return [[float(x) for x in g.trajectory] for g in self.grades]


def run_sweep(
    experiment: Experiment,
    *,
    repo_root: Path | None = None,
    on_seed: Callable[[int, Grade, TrainResult], None] | None = None,
    progress: Callable[[int, dict], None] | None = None,
) -> SweepResult:
    """Every seed, trained and graded. Verifies the lambda rule first."""
    experiment.verify_lambda_rule()
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
    for ordinal in range(experiment.seeds.n_seeds):
        result, policy = train_seed(experiment, ordinal, progress=progress)
        seed_grade = grade(experiment, policy, name=f"seed{ordinal}")
        grades.append(seed_grade)
        training.append(result)
        if on_seed is not None:
            on_seed(ordinal, seed_grade, result)

    return SweepResult(
        experiment=experiment,
        baselines=baselines,
        grades=tuple(grades),
        training=tuple(training),
        seconds=time.perf_counter() - started,
        provenance=provenance,
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
TRACES = ("train_returns", "approx_kl", "entropy", "value_loss")


def _training_record(result: TrainResult, points: int | None) -> dict:
    """A seed's training record, with its per-update traces thinned to budget."""
    record = result.as_dict()
    for name in TRACES:
        record[name] = thin(record[name], points)
    return record


def build_document(sweep: SweepResult) -> dict:
    """The results JSON: the claim, the provenance, the numbers, the verdict."""
    experiment = sweep.experiment
    tolerances = experiment.tolerances
    reference = experiment.reference()

    summary = {
        name: summarise(name, [getattr(g, name) for g in sweep.grades]).as_dict()
        for name in SUMMARISED
    }
    red_flags = [g.name for g in sweep.grades if g.red_flag]
    verdict = {
        "epsilon_met": bool(
            summary["gap_fraction"]["median"] <= tolerances.epsilon_gap_fraction
        ),
        "per_seed_met": bool(
            summary["gap_fraction"]["worst"] <= tolerances.per_seed_gap_fraction
        ),
        "red_flags": red_flags,
        "timed_out": [
            ordinal
            for ordinal, result in enumerate(sweep.training)
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

    points = experiment.trace_points
    return {
        "milestone": "M2",
        "claim": experiment.estimator.claim,
        "provenance": sweep.provenance.as_dict(),
        "config": experiment.as_dict(),
        "reference": reference.as_dict(),
        "bands": {
            "epsilon": experiment.band().as_dict(),
            "per_seed": experiment.band(
                tolerances.per_seed_gap_fraction
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
                "grade": seed_grade.as_dict(),
            }
            for ordinal, (seed_grade, result) in enumerate(
                zip(sweep.grades, sweep.training)
            )
        ],
        "summary": summary,
        "verdict": verdict,
    }
