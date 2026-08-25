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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from temper.agents.execution import PPOPolicy, execution_env_factory
from temper.agents.ppo import TrainResult, train
from temper.env import LiquidityStream, SignalStream, impact_for
from temper.eval.antithetic import PairLedger, PairUpdateStats, antithetic_reward
from temper.eval.conditional import (
    DEFAULT_EVAL_PATHS,
    DEFAULT_SIGNAL_PATHS,
    AlphaGrade,
    LiquidityGrade,
    conditional_costs,
    conditional_rollouts,
    fixed_schedule_grade,
    grade_conditional,
    grade_signal,
    signal_rollouts,
    trajectory_quantiles,
)
from temper.eval.experiment import ANTITHETIC, CONTROL_VARIATE, SAMPLED, Experiment
from temper.eval.grading import (
    Grade,
    deterministic_schedule,
    grade_policy,
    summarise,
)
from temper.eval.provenance import Provenance
from temper.eval.reference import (
    REFERENCE_SCHEDULES,
    AlphaReferenceRow,
    LiquidityReferenceRow,
    alpha_reference_row,
    liquidity_reference_row,
)
from temper.eval.variate import deterministic_reward
from temper.oracle import (
    AlphaSignal,
    Market,
    alpha_coefficient,
    clairvoyant_trajectories,
)
from temper.seeding import (
    LIQUIDITY_EVAL_POOL,
    LIQUIDITY_TRAIN_POOL,
    M4B_REFERENCE_POOL,
    M5_REFERENCE_POOL,
    SIGNAL_EVAL_POOL,
    SIGNAL_TRAIN_POOL,
    pool_seeds,
)

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

#: M4b's pre-stated bar on the liquidity-shuffled control: a policy re-graded
#: with the observed multiplier drawn independently of the charged one must
#: capture at most this much of the adaptive advantage. The *gap* between the
#: real and shuffled capture fractions is the actual claim — a headline that
#: survived the shuffle would be measuring something other than adaptivity.
SHUFFLED_CAPTURE_BAR = 0.15


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


def training_liquidity(experiment: Experiment) -> LiquidityStream:
    """The liquidity stream training draws from — the *train* pool, always.

    Disjoint from evaluation by construction rather than by a stride: the two
    pools have different spawn keys, so a training liquidity path and an
    evaluation one cannot be the same object however the indices are chosen. That
    is invariant 5 doing M4b's out-of-sample work, and it is the whole reason
    :data:`~temper.seeding.LIQUIDITY_TRAIN_POOL` and
    :data:`~temper.seeding.LIQUIDITY_EVAL_POOL` are two pools instead of one.
    """
    return LiquidityStream(law=experiment.liquidity, pool=LIQUIDITY_TRAIN_POOL)


def evaluation_liquidity(experiment: Experiment) -> LiquidityStream:
    """The liquidity stream every graded rollout draws from — the *eval* pool."""
    return LiquidityStream(law=experiment.liquidity, pool=LIQUIDITY_EVAL_POOL)


def training_signal(experiment: Experiment) -> SignalStream:
    """The signal stream training draws from — the *train* pool, always.

    The third seam under the second seam's rule, and invariant 5 does the same
    out-of-sample work: a training signal path and an evaluation one cannot be the
    same object when their spawn keys differ, which is why
    :data:`~temper.seeding.SIGNAL_TRAIN_POOL` and
    :data:`~temper.seeding.SIGNAL_EVAL_POOL` are two pools rather than one split by
    a stride nobody re-checks.

    These are the sanctioned constructors, and they exist from the moment the pools
    do: a pool nobody addresses by name is a pool the next session addresses by
    literal.
    """
    return SignalStream(signal=experiment.signal, pool=SIGNAL_TRAIN_POOL)


def evaluation_signal(experiment: Experiment) -> SignalStream:
    """The signal stream every graded rollout draws from — the *eval* pool."""
    return SignalStream(signal=experiment.signal, pool=SIGNAL_EVAL_POOL)


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
    # M4a's §9 lesson, one seam later: *every* env the estimator constructs has to
    # be handed every injected per-episode property. The factory below builds the
    # primary and `mirror_of` builds the mirror from it, so both routes carry the
    # liquidity stream — and the pair asserts per step that they saw the same
    # multiplier rather than trusting that they did.
    liquidity = training_liquidity(experiment)

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
            liquidity=liquidity,
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
    #: M4b's half. ``grades`` stays empty in the liquidity world: a
    #: liquidity-observing policy's schedule is not open-loop, so there is no
    #: single trajectory for the analytic grader to score and the honest thing is
    #: an empty tuple rather than a number computed from one arbitrary path.
    liquidity_reference: LiquidityReferenceRow | None = None
    #: M5's, and the same rule: a signal-observing policy is graded by
    #: conditional expectation and has no analytic `Grade` to summarise.
    alpha_grades: tuple = ()
    shuffled_alpha_grades: tuple = ()
    alpha_detail: tuple = ()
    alpha_reference_row: AlphaReferenceRow | None = None
    alpha_baselines: dict = field(default_factory=dict)
    liquidity_grades: tuple[LiquidityGrade, ...] = ()
    shuffled_grades: tuple[LiquidityGrade, ...] = ()
    liquidity_baselines: dict[str, LiquidityGrade] = field(default_factory=dict)
    schedule_quantiles: tuple[dict, ...] = ()

    @property
    def scored(self) -> tuple:
        """Whichever grade list this world populated — exactly one of the two.

        A liquidity-observing policy has no analytic ``Grade`` and a
        deterministic one has no ``LiquidityGrade``, so "the seeds this sweep
        scored" is one question with one answer and the callers that only need to
        count them should not have to know which world they are in.
        """
        return self.grades or self.liquidity_grades

    @property
    def addresses(self) -> tuple[int, ...]:
        """The ordinal of each trained seed, positionally aligned with `scored`."""
        return self.ordinals or tuple(range(len(self.scored)))

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

    stochastic = experiment.liquidity.stochastic
    informative = experiment.signal.informative
    if stochastic and informative:
        raise ValueError(
            "this config stacks M4b's stochastic liquidity under M5's signal. "
            "That is a real milestone and it is BACKLOG: bundled, a red result "
            "cannot be attributed, and the two adaptivities respond to different "
            "randomness and compete for the same schedule shape"
        )
    alpha_row = alpha_reference(experiment) if informative else None
    reference = liquidity_reference(experiment) if stochastic else None
    multipliers = (
        liquidity_evaluation_paths(experiment) if stochastic else None
    )
    clairvoyant = (
        conditional_costs(
            clairvoyant_trajectories(
                experiment.case.market,
                experiment.case.order_size,
                experiment.lambda_risk,
                multipliers,
            ),
            multipliers,
            experiment.case.market,
            experiment.lambda_risk,
        )
        if stochastic
        else None
    )
    alpha_baselines = (
        grade_alpha_baselines(experiment, alpha_row, DEFAULT_SIGNAL_PATHS)
        if informative
        else {}
    )
    baselines = {} if stochastic or informative else grade_baselines(experiment)
    liquidity_baselines = (
        grade_liquidity_baselines(experiment, reference, multipliers)
        if stochastic
        else {}
    )
    grades: list[Grade] = []
    liquidity_grades: list[LiquidityGrade] = []
    shuffled_grades: list[LiquidityGrade] = []
    alpha_grades: list[AlphaGrade] = []
    shuffled_alpha: list[AlphaGrade] = []
    alpha_detail: list[dict] = []
    quantiles: list[dict] = []
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
        if informative:
            seed_grade, detail = grade_alpha(
                experiment, policy, alpha_row, name=f"seed{ordinal}"
            )
            # The control is part of the milestone, not an extra, and it runs
            # here rather than in a later pass because it costs a re-grade rather
            # than a re-train and because a control run after the verdict is a
            # control nobody would have to publish. It is also the one path in
            # this sweep that no fast test reaches, which is why the pre-run
            # fabricated pass gives it its own line.
            shuffled_grade, _ = grade_alpha(
                experiment,
                policy,
                alpha_row,
                name=f"seed{ordinal}_shuffled",
                shuffled=True,
            )
            alpha_grades.append(seed_grade)
            shuffled_alpha.append(shuffled_grade)
            alpha_detail.append(detail)
        elif stochastic:
            seed_grade, seed_quantiles = grade_liquidity(
                experiment,
                policy,
                reference,
                name=f"seed{ordinal}",
                clairvoyant_costs=clairvoyant,
            )
            # The control is part of the milestone, not an extra, and it is run
            # here rather than in a later pass because it costs a re-grade rather
            # than a re-train and because a control run after the verdict is a
            # control nobody would have to publish.
            shuffled_grade, _ = grade_liquidity(
                experiment,
                policy,
                reference,
                name=f"seed{ordinal}_shuffled",
                shuffled=True,
            )
            liquidity_grades.append(seed_grade)
            shuffled_grades.append(shuffled_grade)
            quantiles.append(seed_quantiles)
        else:
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
        liquidity_reference=reference,
        liquidity_grades=tuple(liquidity_grades),
        shuffled_grades=tuple(shuffled_grades),
        liquidity_baselines=liquidity_baselines,
        schedule_quantiles=tuple(quantiles),
        alpha_grades=tuple(alpha_grades),
        shuffled_alpha_grades=tuple(shuffled_alpha),
        alpha_detail=tuple(alpha_detail),
        alpha_reference_row=alpha_row,
        alpha_baselines=alpha_baselines,
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


class BudgetBound(RuntimeError):
    """A training run stopped on wall-clock rather than on its update budget.

    ``ppo.max_seconds`` is a **runaway guard, not the budget** — M3's config says
    so in as many words — and until M5 nothing checked that it had not bound. A
    run that hits it has trained fewer updates than its config named, so it is a
    different result from one that did not, and every comparison drawn against it
    is a comparison between two different amounts of training.

    M5 task 2 is the reason this exists rather than the reason it might. Two
    sweeps were left contending for one eight-core box; each was getting a
    fraction of the machine, each would have bound at 90 minutes with a few
    hundred updates missing, and the bitwise regression they were feeding would
    have come back **RED** — a defect reported about a seam that was fine, from a
    comparison that was never valid, at the end of an hour. The run this protects
    is ten seeds against a committed artefact, which is the run one least wants to
    repeat.

    Raised rather than warned, for the reason
    :class:`~temper.eval.grading.ScheduleNotDeterministic` is: a result whose
    budget bound has not been compared badly, it has not been compared at all.
    """


def budget_record(result) -> dict:
    """What the budget did, as a block a result file and a message can share."""
    return {
        "timed_out": bool(result.timed_out),
        "updates": int(result.updates),
        "target_updates": int(result.config.num_updates),
        "bound_at_update": int(result.updates) if result.timed_out else None,
        "seconds": float(result.seconds),
        "max_seconds": result.config.max_seconds,
    }


def refuse_if_budget_bound(results, *, comparison: str, labels=None) -> None:
    """Raise :class:`BudgetBound` if any of `results` stopped on wall-clock.

    `results` are :class:`~temper.agents.ppo.TrainResult` objects or the dicts a
    committed artefact stores them as — both, because the comparisons this
    guards have one of each on their two sides and the *committed* half is
    exactly as capable of having bound as the fresh one.

    `comparison` names what would have been compared, because the message has to
    say why the run stopped rather than what the assertion was called.
    """
    bound = []
    for index, result in enumerate(results):
        if isinstance(result, dict):
            timed_out = bool(result.get("timed_out"))
            done = result.get("updates")
            target = result.get("target_updates")
        else:
            timed_out = bool(result.timed_out)
            done = result.updates
            target = result.config.num_updates
        if timed_out:
            name = labels[index] if labels is not None else f"result {index}"
            bound.append(f"{name} (stopped at update {done} of {target})")
    if bound:
        raise BudgetBound(
            f"{comparison} would compare against a run whose wall-clock budget "
            f"bound early: {', '.join(bound)}. `max_seconds` is a runaway guard "
            "and not the budget, so this is not a tolerance to widen — it is two "
            "different amounts of training. Re-run it on an idle box."
        )


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
    if experiment.signal.informative:
        # M5's, dispatched on the world for M4b's reason: a signal-observing
        # policy is graded by conditional expectation, reports three numbers
        # rather than one, and shares only a skeleton with either of the others.
        return build_alpha_document(sweep)
    if experiment.liquidity.stochastic:
        # M4b's document, and the dispatch is on the *world* rather than on which
        # fields happen to be populated: a liquidity-observing policy is graded by
        # conditional expectation and has no analytic `Grade` to summarise, so the
        # two documents answer different questions and share only a skeleton.
        return build_liquidity_document(sweep)
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
    verdict["budgets"] = [budget_record(r) for r in sweep.training]
    verdict["passed"] = bool(
        verdict["epsilon_met"]
        and verdict["per_seed_met"]
        and not red_flags
        # A sweep whose wall-clock guard bound trained fewer updates than its
        # config named on at least one seed, so its median and its worst seed are
        # summaries over different amounts of training. Not a tolerance.
        and not verdict["timed_out"]
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


# ---------------------------------------------------------------------------
# M4b — the liquidity world's grading and its document
# ---------------------------------------------------------------------------

#: The liquidity summary's quantities, all costs, so larger is worse for every
#: one without exception — the same rule :data:`SUMMARISED` is written under, and
#: the reason ``capture_fraction`` is derived from ``advantage_fraction`` rather
#: than summarised directly.
LIQUIDITY_SUMMARISED = ("advantage_fraction", "objective", "excess")


def liquidity_reference(experiment: Experiment) -> LiquidityReferenceRow:
    """The committed reference row: five fixed rungs, the DP, and both bounds.

    Minutes, and computed **once** per sweep rather than once per seed: it does
    not depend on the agent, and ten independent solves of the same dynamic
    program would be ten chances for them to disagree.
    """
    return liquidity_reference_row(
        experiment.case.market,
        experiment.case.order_size,
        experiment.lambda_risk,
        experiment.liquidity,
        root_seed=experiment.seeds.root_seed,
    )


# ---------------------------------------------------------------------------
# M5 — the alpha-aware sweep
# ---------------------------------------------------------------------------

#: M5's shuffled-signal control bar, pre-stated in the brief. **Negative**, and
#: that is the point: a policy tilting on a signal unrelated to the prices pays
#: the execution premium and monetises nothing, so it should do *worse* than not
#: tilting at all. M4b's control bar was 0.15 — "does not survive"; M5's is a
#: prediction with a sign, which is a stronger thing to be wrong about.
SHUFFLED_NET_CAPTURE_BAR = -0.50

#: The other two of M5's three, pre-stated in the brief's numbers table. An agent
#: may pay more for its alpha than the optimum does, but not half as much again.
ALPHA_CAPTURE_BAR = 0.85
PREMIUM_RATIO_BAR = 1.30

#: The oracle's own paths for M5's grading, and the pool they come from.
SIGNAL_SHUFFLE_ADDRESS = (M5_REFERENCE_POOL, 500)


def alpha_reference(experiment: Experiment) -> AlphaReferenceRow:
    """The alpha-aware row every policy on the chart is scored against."""
    return alpha_reference_row(
        experiment.case.market,
        experiment.case.order_size,
        experiment.lambda_risk,
        experiment.signal,
        root_seed=experiment.seeds.root_seed,
    )


def per_bin_alpha_bps(
    trajectories: np.ndarray,
    signals: np.ndarray,
    market: Market,
    signal: AlphaSignal,
) -> list[float]:
    """The alpha term split by the bin whose inventory it is charged on.

    ``-A rho * mean_p[h_k(p) s_{k-lag}(p)]`` for each ``k``, summing to the total
    the grade reports. Recorded per seed because **three separate defects in this
    milestone lived at the first bin** — the alpha sum starting one bin too early,
    the conditional variance losing the first bin's whole share, and the seam's
    own timing being one bin out — and a per-bin attribution turns the fourth from
    a hunt into a glance. Entries below ``lag`` are exactly zero and are kept in
    the list rather than trimmed, because a shifted index shows up as a non-zero
    first entry and an absent entry cannot be non-zero.
    """
    lag = signal.lag
    holdings = trajectories[:, :-1] / trajectories[:, :1]
    scale = -alpha_coefficient(market) * signal.correlation()
    contributions = [0.0] * market.n_bins
    for k in range(lag, market.n_bins):
        contributions[k] = scale * float(
            np.mean(holdings[:, k] * signals[:, k - lag])
        )
    return contributions


def grade_alpha(
    experiment: Experiment,
    policy,
    reference: AlphaReferenceRow,
    *,
    name: str,
    paths: int = DEFAULT_SIGNAL_PATHS,
    shuffled: bool = False,
) -> tuple[AlphaGrade, dict]:
    """Roll a policy out on held-out signal paths and score the conditional cost.

    The licence runs first and in this order, as M4b's does.
    :func:`~temper.eval.grading.deterministic_schedule` pins the signal stream and
    varies the *price* stream and requires the trajectory bitwise identical —
    what makes ``E[cost | s]`` a closed form at all — and
    :func:`~temper.eval.conditional.check_conditioning_matches_observation` runs
    inside the rollout, so the grade cannot condition on more or less than the
    policy could see.

    `shuffled` runs the overfit control: the observation's signal comes from an
    independent address while the env charges its own. It is part of the claim
    rather than an extra.
    """
    case = experiment.case
    impact = impact_for(experiment.cost_encoding, case.market, case.order_size)
    signal = evaluation_signal(experiment)

    if not shuffled:
        deterministic_schedule(
            policy,
            case.market,
            case.order_size,
            experiment.lambda_risk,
            root_seed=experiment.seeds.root_seed,
            pool=experiment.seeds.eval_pool,
            streams=experiment.seeds.eval_streams,
            temporary_impact=impact,
            signal=signal,
            expect_encoding=experiment.cost_encoding,
        )

    trajectories, signals = signal_rollouts(
        policy,
        case.market,
        case.order_size,
        experiment.lambda_risk,
        temporary_impact=impact,
        signal=signal,
        root_seed=experiment.seeds.root_seed,
        pool=experiment.seeds.eval_pool,
        stream_index=experiment.seeds.eval_streams[0],
        paths=paths,
        shuffle=SIGNAL_SHUFFLE_ADDRESS if shuffled else None,
    )
    grade = grade_signal(
        trajectories,
        signals,
        case.market,
        case.order_size,
        reference,
        experiment.signal,
        name=name,
        soft_slack=reference.feasible.half_width_bps,
    )
    detail = {
        "per_bin_alpha_bps": per_bin_alpha_bps(
            trajectories, signals, case.market, experiment.signal
        ),
        "schedule_quantiles": trajectory_quantiles(trajectories),
    }
    return grade, detail


def grade_alpha_baselines(
    experiment: Experiment, reference: AlphaReferenceRow, paths: int
) -> dict:
    """The four fixed schedules, scored through the same conditional route.

    Every one of them has an alpha term of exactly zero in expectation — they are
    fixed, and ``E[s] = 0`` — so what this measures is the *sampling* half: on the
    same paths a fixed schedule's realised alpha is not zero, and scoring it here
    rather than quoting its closed form is what makes the agent's row and the
    baseline rows comparable path by path.
    """
    return {
        name: grade_alpha(
            experiment,
            baseline(
                name,
                experiment.case.market,
                experiment.case.order_size,
                experiment.lambda_risk,
                encoding=experiment.cost_encoding,
            ),
            reference,
            name=name,
            paths=paths,
        )[0]
        for name in REFERENCE_SCHEDULES[experiment.cost_encoding]
    }


def grade_liquidity(
    experiment: Experiment,
    policy,
    reference: LiquidityReferenceRow,
    *,
    name: str,
    paths: int = DEFAULT_EVAL_PATHS,
    shuffled: bool = False,
    clairvoyant_costs=None,
) -> tuple[LiquidityGrade, dict]:
    """Roll a policy out on held-out liquidity paths and score the conditional cost.

    Two assertions run before the number is computed, in this order and for
    different reasons. :func:`~temper.eval.grading.deterministic_schedule` pins
    the liquidity stream and varies the *price* stream and requires the trajectory
    to be bitwise identical — the successor to M2's open-loop check, and what
    licenses ``E[cost | L]`` as a closed form at all. Then the rollouts happen on
    the eval pool, which is disjoint from training by construction.

    `shuffled` runs the liquidity-shuffled control: the observation's multiplier
    comes from an independent stream while the env charges its own. It is part of
    the milestone rather than an extra — if the advantage survives it, the agent
    is not using the signal and the headline is measuring something else.
    """
    case = experiment.case
    impact = impact_for(experiment.cost_encoding, case.market, case.order_size)
    liquidity = evaluation_liquidity(experiment)

    if not shuffled:
        # The licence for the conditional expectation, checked before it is used.
        deterministic_schedule(
            policy,
            case.market,
            case.order_size,
            experiment.lambda_risk,
            root_seed=experiment.seeds.root_seed,
            pool=experiment.seeds.eval_pool,
            streams=experiment.seeds.eval_streams,
            temporary_impact=impact,
            liquidity=liquidity,
            expect_encoding=experiment.cost_encoding,
        )

    trajectories, multipliers = conditional_rollouts(
        policy,
        case.market,
        case.order_size,
        experiment.lambda_risk,
        temporary_impact=impact,
        liquidity=liquidity,
        root_seed=experiment.seeds.root_seed,
        pool=experiment.seeds.eval_pool,
        stream_index=experiment.seeds.eval_streams[0],
        paths=paths,
        shuffle=(M4B_REFERENCE_POOL, 500) if shuffled else None,
    )
    grade = grade_conditional(
        trajectories,
        multipliers,
        case.market,
        case.order_size,
        reference,
        name=name,
        clairvoyant_costs=clairvoyant_costs,
        soft_slack=reference.feasible.half_width_bps,
    )
    return grade, trajectory_quantiles(trajectories)


def liquidity_evaluation_paths(experiment: Experiment, paths: int = DEFAULT_EVAL_PATHS):
    """The eval liquidity paths every policy on the chart is scored on.

    Common random numbers, and *these* are the paths: a policy graded through
    :func:`grade_liquidity` draws the same blocks off the same stream in the same
    order, so a fixed schedule priced here and an agent rolled out there are
    comparable path by path. Returned so the clairvoyant relaxation can be solved
    once for the whole sweep rather than once per seed.
    """
    case = experiment.case
    stream = evaluation_liquidity(experiment)
    generator = stream.generator(
        experiment.seeds.root_seed, experiment.seeds.eval_streams[0]
    )
    return experiment.liquidity.draw(generator, (paths, case.market.n_bins))


def grade_liquidity_baselines(
    experiment: Experiment,
    reference: LiquidityReferenceRow,
    multipliers,
) -> dict[str, LiquidityGrade]:
    """Every fixed rung, priced on the same paths the agent is scored on.

    Not read off the oracle: running them through the *grader* is the cheapest
    check that the grading path returns the closed form when handed a schedule
    whose closed form is known — and it is what puts the baselines on the chart,
    as invariant 4 requires. ``static`` is the control variate, so it must come
    back exactly; the rest must land within their own half-widths.
    """
    case = experiment.case
    return {
        name: fixed_schedule_grade(
            row.trajectory,
            multipliers,
            case.market,
            case.order_size,
            reference,
            name=name,
        )
        for name, row in reference.schedules.items()
    }


#: The three numbers M5 reports, and it reports them **together**. Named in one
#: place so no call site can quote one of them: at the optimum 45 % of the gross
#: effect is paid back, so `net_capture` alone cannot tell a policy that trades
#: the signal well from one that trades it badly and executes well. Each carries
#: its absolute bps beside it (§9's denominator entry).
ALPHA_HEADLINE = ("alpha_capture", "premium_ratio", "net_capture")


def alpha_headline(grade) -> dict:
    """One policy's three numbers, with the bps each fraction is a fraction of."""
    return {
        "alpha_capture": grade.alpha_capture,
        "alpha_bps": grade.alpha_bps,
        "reference_alpha_bps": grade.reference_alpha_bps,
        "premium_ratio": grade.premium_ratio,
        "execution_premium_bps": grade.execution_premium_bps,
        "reference_premium_bps": grade.reference_premium_bps,
        "net_capture": grade.net_capture,
        "excess_bps": grade.excess_bps,
        "advantage_bps": (
            grade.deterministic_objective - grade.reference_objective
        ),
    }


def format_alpha_headline(grade) -> str:
    """The one line a console and a caption both use. All three, never one.

    Formatted here rather than at the call sites so that "never the net capture
    alone" is a property of the code rather than of everyone's discipline.
    """
    numbers = alpha_headline(grade)
    return (
        f"alpha {numbers['alpha_capture']:+.3f} "
        f"({numbers['alpha_bps']:+.5f} of {numbers['reference_alpha_bps']:.5f} bps) · "
        f"premium {numbers['premium_ratio']:.3f}x "
        f"({numbers['execution_premium_bps']:+.5f} of "
        f"{numbers['reference_premium_bps']:.5f} bps) · "
        f"net {numbers['net_capture']:+.3f} "
        f"({numbers['excess_bps']:+.5f} bps over J_DP)"
    )


def build_alpha_document(sweep: SweepResult) -> dict:
    """M5's results JSON: three numbers per seed, the control, and the per-bin alpha."""
    experiment = sweep.experiment
    if sweep.ordinals and tuple(sweep.ordinals) != tuple(
        range(experiment.seeds.n_seeds)
    ):
        raise ValueError(
            f"this sweep trained ordinals {list(sweep.ordinals)} of the config's "
            f"{experiment.seeds.n_seeds}; a metrics document is a claim about a "
            "sweep and cannot be written from a subset"
        )
    tolerances = experiment.tolerances
    reference = sweep.alpha_reference_row
    grades = sweep.alpha_grades
    shuffled = sweep.shuffled_alpha_grades

    summary = {
        name: summarise(name, [getattr(g, name) for g in grades]).as_dict()
        for name in ("objective", "excess_bps", "alpha_bps", "execution_premium_bps")
    }
    for name in ALPHA_HEADLINE:
        summary[name] = summarise(
            name, [getattr(g, name) for g in grades]
        ).as_dict()
    # The bar is stated on the excess as a fraction of the advantage, which is
    # 1 - net_capture. Derived from the SAME summary so the two can never
    # disagree about which seed was worst.
    net = summary["net_capture"]
    summary["advantage_fraction"] = {
        "name": "advantage_fraction",
        "values": [1.0 - v for v in net["values"]],
        "median": 1.0 - net["median"],
        "q1": 1.0 - net["q3"],
        "q3": 1.0 - net["q1"],
        "iqr": net["iqr"],
        "worst": 1.0 - net["worst"],
    }
    graded_on = summary["advantage_fraction"]

    shuffled_summary = (
        summarise("net_capture", [g.net_capture for g in shuffled]).as_dict()
        if shuffled
        else None
    )

    red_flags = [g.name for g in grades if g.red_flag]
    verdict = {
        "tolerance_denominator": tolerances.denominator,
        "graded_attribute": "advantage_fraction",
        "epsilon_met": bool(graded_on["median"] <= tolerances.epsilon_fraction),
        "per_seed_met": bool(graded_on["worst"] <= tolerances.per_seed_fraction),
        # Rigorous and CERTIFIED, not numerical: impact and risk are convex and
        # carry no signal, so E[impact + risk] >= M4a's certified optimum for any
        # policy at all. An agent below it is a defect with a proof.
        "red_flags": red_flags,
        "soft_flags": [g.name for g in grades if g.soft_flag],
        "shuffled_control_met": (
            None
            if shuffled_summary is None
            else bool(shuffled_summary["median"] <= SHUFFLED_NET_CAPTURE_BAR)
        ),
        "shuffled_net_capture_bar": SHUFFLED_NET_CAPTURE_BAR,
        "timed_out": [
            ordinal
            for ordinal, result in zip(sweep.addresses, sweep.training)
            if result.timed_out
        ],
        "budgets": [budget_record(r) for r in sweep.training],
        "sweep_seconds": sweep.seconds,
        "within_sweep_budget": bool(
            sweep.seconds <= experiment.runtime.sweep_seconds
        ),
    }
    verdict["passed"] = bool(
        verdict["epsilon_met"]
        and verdict["per_seed_met"]
        and not red_flags
        and verdict["shuffled_control_met"] is not False
        and not verdict["timed_out"]
    )
    # The three, at the sweep level, in one block. Never one of them.
    verdict["headline"] = {
        "alpha_capture_median": summary["alpha_capture"]["median"],
        "alpha_bps_median": summary["alpha_bps"]["median"],
        "reference_alpha_bps": reference.alpha_available,
        "premium_ratio_median": summary["premium_ratio"]["median"],
        "execution_premium_bps_median": summary["execution_premium_bps"]["median"],
        "reference_premium_bps": reference.execution_premium,
        "net_capture_median": summary["net_capture"]["median"],
        "median_excess_bps": summary["excess_bps"]["median"],
        "advantage_bps": reference.signal_advantage,
    }
    verdict["denominator_bps"] = reference.signal_advantage
    verdict["median_excess_bps"] = summary["excess_bps"]["median"]
    verdict["alpha_capture_met"] = bool(
        summary["alpha_capture"]["median"] >= ALPHA_CAPTURE_BAR
    )
    verdict["premium_ratio_met"] = bool(
        summary["premium_ratio"]["median"] <= PREMIUM_RATIO_BAR
    )

    points = experiment.trace_points
    pairs = sweep.pairs if sweep.pairs else tuple(() for _ in grades)
    seeds = []
    for index, ordinal in enumerate(sweep.addresses):
        seeds.append(
            {
                "ordinal": ordinal,
                "env_stream_base": ordinal * experiment.seeds.env_stream_stride,
                "training": _training_record(sweep.training[index], points),
                "budget": budget_record(sweep.training[index]),
                **({"pair": _pair_record(pairs[index], points)} if experiment.estimator.antithetic else {}),
                "grade": grades[index].as_dict(),
                "headline": alpha_headline(grades[index]),
                # Per bin, for the reported seeds. Three separate defects in this
                # milestone lived at the first bin; a per-bin attribution turns
                # the fourth from a hunt into a glance.
                "per_bin_alpha_bps": sweep.alpha_detail[index]["per_bin_alpha_bps"],
                "schedule_quantiles": sweep.alpha_detail[index][
                    "schedule_quantiles"
                ],
                "shuffled": (
                    shuffled[index].as_dict() if index < len(shuffled) else None
                ),
                "shuffled_headline": (
                    alpha_headline(shuffled[index])
                    if index < len(shuffled)
                    else None
                ),
            }
        )

    return {
        "milestone": experiment.milestone,
        "claim": experiment.estimator.claim,
        "provenance": sweep.provenance.as_dict(),
        "config": experiment.as_dict(),
        "signal": experiment.signal.as_dict(),
        "reference": reference.as_dict(),
        "reference_kind": {
            name: kind.as_dict()
            for name, kind in reference.reference_kinds.items()
        },
        "baselines": {
            name: grade.as_dict() for name, grade in sweep.alpha_baselines.items()
        },
        "trace_points": points,
        "seeds": seeds,
        "summary": summary,
        "shuffled_control": {
            "bar": SHUFFLED_NET_CAPTURE_BAR,
            "net_capture": shuffled_summary,
            "address": list(SIGNAL_SHUFFLE_ADDRESS),
        },
        "verdict": verdict,
    }


def build_liquidity_document(sweep: "SweepResult") -> dict:
    """M4b's results JSON. Same skeleton, a different question underneath.

    The verdict is read on ``advantage_fraction`` against the *adaptive*
    advantage, and the **level shift is a line of its own** — reported beside the
    headline everywhere so nobody credits the agent with a constant any static
    solver picks up by re-solving at an inflated coefficient.

    Three things this file says that no earlier one had to. The reference is
    ``converged and bracketed``, not certified, and it says so in the field a
    reader would look for the word in. The liquidity process is **invented**, and
    says that too. And the headline carries a *confidence interval*, because M4b
    is the first milestone whose grade is an average rather than a closed form —
    over liquidity alone, with no price sampling anywhere.
    """
    experiment = sweep.experiment
    if sweep.ordinals and tuple(sweep.ordinals) != tuple(
        range(experiment.seeds.n_seeds)
    ):
        raise ValueError(
            f"this sweep trained ordinals {list(sweep.ordinals)} of the config's "
            f"{experiment.seeds.n_seeds}; a metrics document is a claim about a "
            "sweep and cannot be written from a subset."
        )
    tolerances = experiment.tolerances
    reference = sweep.liquidity_reference
    if reference is None:
        raise ValueError(
            "a stochastic-liquidity sweep has no reference row; the dynamic "
            "program is what its grades are excesses over"
        )

    grades = sweep.liquidity_grades
    summary = {
        name: summarise(name, [getattr(g, name) for g in grades]).as_dict()
        for name in LIQUIDITY_SUMMARISED
    }
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
    graded_on = summary[tolerances.graded_attribute]

    shuffled = sweep.shuffled_grades
    shuffled_summary = (
        summarise(
            "capture_fraction", [g.capture_fraction for g in shuffled]
        ).as_dict()
        if shuffled
        else None
    )

    red_flags = [g.name for g in grades if g.red_flag]
    below = [g.name for g in grades if g.paths_below_clairvoyant]
    verdict = {
        "tolerance_denominator": tolerances.denominator,
        "graded_attribute": tolerances.graded_attribute,
        "epsilon_met": bool(graded_on["median"] <= tolerances.epsilon_fraction),
        "per_seed_met": bool(graded_on["worst"] <= tolerances.per_seed_fraction),
        "red_flags": red_flags,
        # The rigorous form, and it needs no interval: perfect information is the
        # per-path minimum over all schedules, so a policy below it on *any* path
        # is a defect with a proof rather than a discovery.
        "seeds_below_clairvoyant": below,
        "soft_flags": [g.name for g in grades if g.soft_flag],
        "shuffled_control_met": (
            None
            if shuffled_summary is None
            else bool(shuffled_summary["median"] <= SHUFFLED_CAPTURE_BAR)
        ),
        "shuffled_capture_bar": SHUFFLED_CAPTURE_BAR,
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
    verdict["budgets"] = [budget_record(r) for r in sweep.training]
    verdict["passed"] = bool(
        verdict["epsilon_met"]
        and verdict["per_seed_met"]
        and not red_flags
        and not below
        and verdict["shuffled_control_met"] is not False
        and not verdict["timed_out"]
    )
    verdict["denominator_bps"] = reference.adaptive_advantage
    verdict["median_excess_bps"] = summary["excess"]["median"]
    # Its own line, every time. 3.8 % of the naive gap at the trained sigma_log,
    # and none of it the agent's.
    verdict["level_shift_bps"] = reference.level_shift
    verdict["level_shift_fraction_of_advantage"] = reference.level_shift_fraction
    verdict["naive_gap_bps"] = reference.m4a.objective - reference.adaptive_bps

    points = experiment.trace_points
    antithetic = experiment.estimator.antithetic
    pairs = sweep.pairs if sweep.pairs else tuple(() for _ in grades)
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
        "liquidity": experiment.liquidity.as_dict(),
        "reference": reference.as_dict(),
        "reference_kind": "converged and bracketed, not certified",
        "baselines": {
            name: g.as_dict() for name, g in sweep.liquidity_baselines.items()
        },
        "trace_points": points,
        "seeds": [
            {
                "ordinal": ordinal,
                "env_stream_base": experiment.seeds.env_streams(
                    ordinal, experiment.ppo.num_envs
                )[0],
                "training": _training_record(result, points),
                **({"pair": _pair_record(seed_pairs, points)} if antithetic else {}),
                "grade": seed_grade.as_dict(),
                "schedule_quantiles": quantiles,
                **(
                    {"shuffled": shuffled_grade.as_dict()}
                    if shuffled_grade is not None
                    else {}
                ),
            }
            for ordinal, seed_grade, result, seed_pairs, quantiles, shuffled_grade in zip(
                sweep.addresses,
                grades,
                sweep.training,
                pairs,
                sweep.schedule_quantiles or [{} for _ in grades],
                shuffled or [None for _ in grades],
            )
        ],
        "summary": summary,
        "shuffled_control": shuffled_summary,
        "verdict": verdict,
    }
