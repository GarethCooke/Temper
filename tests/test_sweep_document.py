"""``build_document`` assembles a results file, in both worlds, in milliseconds.

This module exists because of a specific two-hour mistake, and it is cheaper to
write it down than to make it again. M4a's verdict block was edited to read its
bars off a world-dependent field; the edit dropped the line that computes
``red_flags``, and nothing noticed — because every test that exercises
``build_document`` does so by *training first*. So the ``NameError`` surfaced
after ten seeds had trained and been graded, at the moment the document was
assembled, and the run wrote nothing at all. Ten correct answers, discarded by a
missing assignment.

The gap was structural rather than careless. ``run_sweep`` and ``build_document``
were only ever reachable behind a training loop, so the assembly of the artefact —
which is pure data manipulation and takes microseconds — inherited the cost of
the thing that produces its input. This module severs that: it grades a nudged
copy of each world's own optimum through the *real* grader, pairs it with a real
:class:`~temper.agents.ppo.TrainResult` (whose ``as_dict`` never touches the
network, so no training is needed to build one), and runs the real
``build_document`` over the pair in every encoding — asserting the shape a
committed result must have. From M4b it does the same for every *reporting* path
as well: the verdict block, the per-seed line in both grade shapes, the figure
tool's ``main`` end to end, and the two failure modes that have to stay visible.
The portable form of the rule is ``docs/house-notes.md``, *No code path may be
reachable only at the end of a long run* — which is this note's second title,
and M4b is why: naming ``build_document`` made ``build_document`` safe and did
nothing for the twenty lines under it, one of which then died with ten graded
seeds in hand.

What it deliberately does **not** do is check the numbers are right — that is
``tests/test_m2_rediscovery.py``, ``tests/test_m3_validation.py`` and
``tests/test_m4a_power_law.py``, each against a committed artefact. This is the
cheap half: every key a reader and a figure depend on exists, in every world,
before an evening is spent producing one.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from temper.agents.ppo import TrainResult
from temper.eval.experiment import AVAILABLE_ADVANTAGE, TWAP_GAP, load_experiment
from temper.eval.grading import grade_trajectory
from temper.eval.provenance import Provenance
from temper.eval.sweep import SweepResult, build_document
from temper.oracle import LINEAR_ENCODING, POWER_LAW_ENCODING

from .conftest import REPO_ROOT

CONFIGS = {
    LINEAR_ENCODING: REPO_ROOT / "configs" / "m3_frontier" / "lambda_1e-3.5.yaml",
    POWER_LAW_ENCODING: REPO_ROOT / "configs" / "m4a_power_law.yaml",
}

PROVENANCE = Provenance(
    config="fabricated.yaml",
    config_sha256="0" * 64,
    git_rev="0" * 40,
    git_dirty=False,
    python="3.12.2",
)


def _training_result(config, updates: int) -> TrainResult:
    """A real :class:`TrainResult` with no trained network in it.

    ``agent`` is the only heavy field and ``as_dict`` never touches it, so the
    record can be the *real* class rather than a stand-in — which matters,
    because a stand-in is exactly the thing that drifts out of step with what a
    run actually produces and leaves this module testing itself.
    """
    return TrainResult(
        agent=None,
        config=config,
        seed=1,
        global_step=updates * 13 * 512,
        updates=updates,
        seconds=700.0,
        timed_out=False,
        returns=[0.0] * updates,
        episode_counts=[512] * updates,
        approx_kls=[0.0] * updates,
        entropies=[0.0] * updates,
        value_losses=[0.0] * updates,
        return_variances=[0.0] * updates,
    )


def _sweep(encoding: str, n_seeds: int = 10) -> SweepResult:
    """A sweep whose seeds are the world's own optimum, nudged.

    Perturbing the optimum rather than inventing a schedule keeps every grade a
    real one: it goes through ``grade_trajectory``, the registry and the world's
    reference, so the document is assembled from the same objects a training run
    would hand it.
    """
    experiment = load_experiment(CONFIGS[encoding])
    reference = experiment.reference()
    market, order_size = experiment.case.market, experiment.case.order_size
    optimum = np.asarray(reference.optimal.trajectory, dtype=float)

    rng = np.random.default_rng(4)
    grades = []
    for ordinal in range(n_seeds):
        schedule = optimum.copy()
        # Monotone-preserving nudge: scale the interior toward the optimum's own
        # shape, so the perturbed schedule stays inside the reachable set.
        schedule[1:-1] *= 1.0 + 1e-4 * rng.normal(size=schedule.size - 2)
        schedule[1:-1] = np.minimum.accumulate(schedule[1:-1])
        grades.append(
            grade_trajectory(
                schedule, market, order_size, reference, name=f"seed{ordinal}"
            )
        )

    return SweepResult(
        experiment=experiment,
        baselines={},
        grades=tuple(grades),
        training=tuple(
            _training_result(experiment.ppo, experiment.ppo.num_updates)
            for _ in grades
        ),
        seconds=7000.0,
        provenance=PROVENANCE,
        pairs=tuple(() for _ in grades),
    )


def test_the_training_record_is_the_real_one_the_sweep_writes():
    """Every trace ``build_document`` thins is present and the right length.

    The record is a genuine ``TrainResult``, so this is really asking whether the
    document's trace handling still matches the class — which is the coupling a
    hand-rolled stand-in would have hidden.
    """
    from temper.eval.sweep import TRACES

    record = _training_result(load_experiment(CONFIGS[LINEAR_ENCODING]).ppo, 7).as_dict()
    assert set(TRACES) <= set(record)
    for name in TRACES:
        assert len(record[name]) == 7
    assert record["timed_out"] is False


@pytest.mark.parametrize("encoding", sorted(CONFIGS))
def test_the_document_assembles_in_both_worlds(encoding):
    """The whole artefact, built from real grades, with no training anywhere."""
    document = build_document(_sweep(encoding))

    assert document["config"]["cost_encoding"] == encoding
    assert document["reference"]["encoding"] == encoding
    assert len(document["seeds"]) == 10
    assert document["provenance"]["git_dirty"] is False
    for record in document["seeds"]:
        grade = record["grade"]
        assert grade["encoding"] == encoding
        assert set(grade) >= {
            "objective_bps",
            "excess_bps",
            "gap_fraction",
            "advantage_fraction",
            "capture_fraction",
            "deviation_shares",
            "red_flag",
            "trajectory",
        }


@pytest.mark.parametrize("encoding", sorted(CONFIGS))
def test_the_verdict_carries_every_key_a_reader_depends_on(encoding):
    """The keys the driver prints, the tests read and the figures draw.

    ``red_flags`` is on this list for the reason at the top of the module: it was
    dropped in an edit, and the only thing that would have caught it before a
    night of training is an assertion that it is there.
    """
    verdict = build_document(_sweep(encoding))["verdict"]
    assert set(verdict) >= {
        "tolerance_denominator",
        "graded_attribute",
        "epsilon_met",
        "per_seed_met",
        "red_flags",
        "timed_out",
        "sweep_seconds",
        "within_sweep_budget",
        "passed",
        "denominator_bps",
        "median_excess_bps",
    }
    assert verdict["red_flags"] == []
    assert verdict["passed"] is True
    assert verdict["denominator_bps"] > 0.0


def test_the_denominator_is_the_one_the_config_names():
    """M2/M3 read the TWAP gap; M4a reads the available advantage."""
    linear = build_document(_sweep(LINEAR_ENCODING))["verdict"]
    power = build_document(_sweep(POWER_LAW_ENCODING))["verdict"]
    assert linear["tolerance_denominator"] == TWAP_GAP
    assert linear["graded_attribute"] == "gap_fraction"
    assert power["tolerance_denominator"] == AVAILABLE_ADVANTAGE
    assert power["graded_attribute"] == "advantage_fraction"
    # And the two denominators really are different quantities, by a wide margin.
    assert linear["denominator_bps"] > 30.0 * power["denominator_bps"]


def test_the_capture_fraction_summary_is_the_advantage_fractions_complement():
    """One summary, two views, and the quartiles the right way round.

    ``advantage_fraction`` is a cost (larger is worse) and ``capture_fraction``
    is a benefit, so their quartiles swap and their "worst" ends do too. Getting
    that backwards would be invisible in a passing run and wrong in a report.
    """
    summary = build_document(_sweep(POWER_LAW_ENCODING))["summary"]
    advantage, capture = summary["advantage_fraction"], summary["capture_fraction"]
    assert capture["median"] == pytest.approx(1.0 - advantage["median"])
    assert capture["q1"] == pytest.approx(1.0 - advantage["q3"])
    assert capture["q3"] == pytest.approx(1.0 - advantage["q1"])
    assert capture["worst"] == pytest.approx(1.0 - advantage["worst"])
    assert capture["worst"] <= capture["median"] <= capture["q3"]
    assert advantage["worst"] >= advantage["median"] >= advantage["q1"]


def test_the_linear_world_reports_no_capture_fraction():
    """There is nothing to capture where the closed form *is* the optimum.

    Reporting a capture fraction in Phase 1 would invite the reading M4a's whole
    framing exists to prevent — that the agent beat Almgren–Chriss inside AC's
    own assumptions, which §1.1 names a red flag rather than a result.
    """
    summary = build_document(_sweep(LINEAR_ENCODING))["summary"]
    assert "capture_fraction" not in summary
    assert "advantage_fraction" not in summary
    for record in build_document(_sweep(LINEAR_ENCODING))["seeds"]:
        assert record["grade"]["capture_fraction"] is None
        assert record["grade"]["advantage_fraction"] is None


def test_a_red_flagged_seed_fails_the_verdict():
    """Non-vacuity for `red_flags`: a seed below the optimum must stop the run."""
    import dataclasses

    sweep = _sweep(POWER_LAW_ENCODING)
    flagged = dataclasses.replace(sweep.grades[0], red_flag=True)
    sweep = dataclasses.replace(sweep, grades=(flagged,) + sweep.grades[1:])

    verdict = build_document(sweep)["verdict"]
    assert verdict["red_flags"] == ["seed0"]
    assert verdict["passed"] is False


# ---------------------------------------------------------------------------
# Subset runs: the seed's address is not its position in a list
# ---------------------------------------------------------------------------


def test_a_full_sweeps_seed_records_carry_their_own_ordinals():
    """The unchanged path, so the two tests below mean something."""
    document = build_document(_sweep(POWER_LAW_ENCODING))
    assert [record["ordinal"] for record in document["seeds"]] == list(range(10))


def test_a_subset_sweep_refuses_to_become_a_metrics_document():
    """`run_sweep(ordinals=...)` may train a subset; it may not report one.

    Every number in the document is a statement about the sweep — median, IQR,
    worst seed, the epsilon verdict — and none of them survives being computed
    over one seed. Invariant 4 asks for dispersion, and a median over one value
    is not a median.
    """
    subset = dataclasses.replace(
        _sweep(POWER_LAW_ENCODING, n_seeds=1), ordinals=(9,)
    )
    with pytest.raises(ValueError, match="cannot be written from a subset"):
        build_document(subset)


def test_a_seeds_address_survives_being_run_out_of_position():
    """The provenance trap the refusal above removes, asserted on the labelling.

    Before `SweepResult.ordinals` existed the seed records took their address
    from their *position*, which was right only because the list was always the
    full range. Run ordinals 9 and 4 and the old code would have written them as
    0 and 1, with seed 0's and seed 1's `env_stream_base` — a false provenance
    stamp in the file invariant 1 rests on. `addresses` is what makes the
    labelling follow the seed rather than the slot.
    """
    sweep = dataclasses.replace(_sweep(POWER_LAW_ENCODING, n_seeds=2), ordinals=(9, 4))
    assert sweep.addresses == (9, 4)

    experiment = sweep.experiment
    expected = [
        experiment.seeds.env_streams(ordinal, experiment.ppo.num_envs)[0]
        for ordinal in (9, 4)
    ]
    assert expected[0] != expected[1], "the two seeds must occupy different streams"

    # The document itself is refused, so the labelling is checked on the pieces
    # it would have used — which is where the falsehood would have been written.
    assert [
        experiment.seeds.env_streams(ordinal, experiment.ppo.num_envs)[0]
        for ordinal in sweep.addresses
    ] == expected


def test_a_full_range_sweep_is_not_mistaken_for_a_subset():
    """`ordinals` set to the whole range is still a sweep, not a subset."""
    sweep = dataclasses.replace(_sweep(POWER_LAW_ENCODING), ordinals=tuple(range(10)))
    document = build_document(sweep)
    assert [record["ordinal"] for record in document["seeds"]] == list(range(10))


# ---------------------------------------------------------------------------
# M4b — the liquidity document, on fabricated data, before the training run
# ---------------------------------------------------------------------------

M4B_CONFIG = REPO_ROOT / "configs" / "m4b_liquidity.yaml"

#: A small path count. Nothing here checks a *number*; it checks that every key a
#: reader, a test and a figure depend on exists in a document assembled by the
#: real writer from real grades. The house note's whole point is that this must
#: cost milliseconds, or it will be run after the evening instead of before it.
FABRICATED_PATHS = 64


def _liquidity_sweep(n_seeds: int = 10, *, shuffled: bool = True):
    """A liquidity sweep whose seeds are the DP's own greedy policy, nudged.

    Real grades through the real grader: the schedules come out of
    :meth:`~temper.oracle.adaptive.AdaptiveOptimum.greedy_weights` on real
    sampled paths, get perturbed per seed, and go through
    :func:`~temper.eval.conditional.grade_conditional` against a real
    :class:`~temper.eval.reference.LiquidityReferenceRow`. Only the *training* is
    fabricated, because a ``TrainResult``'s ``as_dict`` never touches a network.

    Deliberately a **coarse** reference — a 201-point grid and 64 paths — because
    this module is about the writer and not about the reference. The committed
    numbers come from ``tools/m4b_reference_table.py``; what is under test here is
    that ``build_document`` can turn grades into a file without a training run to
    find out.
    """
    import numpy as np

    from temper.eval.conditional import grade_conditional, trajectory_quantiles
    from temper.eval.reference import liquidity_reference_row
    from temper.eval.sweep import grade_liquidity_baselines

    experiment = load_experiment(M4B_CONFIG)
    market, order_size = experiment.case.market, experiment.case.order_size
    reference = liquidity_reference_row(
        market,
        order_size,
        experiment.lambda_risk,
        experiment.liquidity,
        root_seed=experiment.seeds.root_seed,
        paths=FABRICATED_PATHS,
        grid_points=201,
        quadrature_nodes=7,
    )

    rng = np.random.default_rng(4)
    multipliers = experiment.liquidity.draw(
        np.random.default_rng(11), (FABRICATED_PATHS, market.n_bins)
    )
    optimum = __import__(
        "temper.oracle", fromlist=["adaptive_optimum"]
    ).adaptive_optimum(
        market, order_size, experiment.lambda_risk, experiment.liquidity, points=201
    )
    weights = optimum.greedy_weights(multipliers)

    grades, shuffles, quantiles = [], [], []
    for ordinal in range(n_seeds):
        nudged = weights * (1.0 + 1e-3 * rng.normal(size=weights.shape))
        nudged = np.clip(nudged, 1e-9, None)
        nudged /= nudged.sum(axis=1, keepdims=True)
        trajectories = order_size * np.concatenate(
            (
                np.ones((nudged.shape[0], 1)),
                1.0 - np.cumsum(nudged, axis=1),
            ),
            axis=1,
        )
        trajectories[:, -1] = 0.0
        grades.append(
            grade_conditional(
                trajectories,
                multipliers,
                market,
                order_size,
                reference,
                name=f"seed{ordinal}",
            )
        )
        quantiles.append(trajectory_quantiles(trajectories))
        if shuffled:
            shuffles.append(
                grade_conditional(
                    np.tile(reference.static.trajectory, (FABRICATED_PATHS, 1)),
                    multipliers,
                    market,
                    order_size,
                    reference,
                    name=f"seed{ordinal}_shuffled",
                )
            )

    return SweepResult(
        experiment=experiment,
        baselines={},
        grades=(),
        training=tuple(
            _training_result(experiment.ppo, experiment.ppo.num_updates)
            for _ in grades
        ),
        seconds=9000.0,
        provenance=PROVENANCE,
        pairs=tuple(() for _ in grades),
        liquidity_reference=reference,
        liquidity_grades=tuple(grades),
        shuffled_grades=tuple(shuffles),
        liquidity_baselines=grade_liquidity_baselines(
            experiment, reference, multipliers
        ),
        schedule_quantiles=tuple(quantiles),
    )


def test_the_liquidity_document_assembles_without_a_training_run():
    """Every key M4b's readers depend on, from real grades, in milliseconds."""
    document = build_document(_liquidity_sweep())

    assert document["milestone"] == "M4b"
    assert document["config"]["cost_encoding"] == POWER_LAW_ENCODING
    assert document["liquidity"]["model"] == "lognormal"
    assert document["liquidity"]["invented"] is True, (
        "the results file must record that the liquidity process is Temper's own"
    )
    assert document["reference_kind"] == "converged and bracketed, not certified"
    assert document["reference"]["adaptive"]["certified"] is False
    assert len(document["seeds"]) == 10

    for record in document["seeds"]:
        grade = record["grade"]
        assert set(grade) >= {
            "objective_bps",
            "half_width_bps",
            "paired_sd_bps",
            "paths",
            "excess_bps",
            "advantage_fraction",
            "capture_fraction",
            "paths_below_clairvoyant",
            "red_flag",
            "soft_flag",
            "mean_trajectory",
        }
        assert set(record["schedule_quantiles"]) == {"q25", "q50", "q75"}
        assert "shuffled" in record, "the control travels with the seed it controls"


def test_the_liquidity_verdict_reports_the_level_shift_on_its_own_line():
    """The single number M4b is most at risk of quietly crediting to the agent.

    ``J_M4a - J_static*`` is a constant any static solver picks up for free by
    re-solving at the inflated coefficient. An agent measured against M4a's
    schedule appears to gain the *naive* gap; the denominator is the adaptive
    advantage, and both numbers are in the file so a reader can see the
    difference rather than take the fraction on trust.
    """
    document = build_document(_liquidity_sweep())
    verdict = document["verdict"]

    assert set(verdict) >= {
        "tolerance_denominator",
        "graded_attribute",
        "epsilon_met",
        "per_seed_met",
        "red_flags",
        "seeds_below_clairvoyant",
        "soft_flags",
        "shuffled_control_met",
        "shuffled_capture_bar",
        "denominator_bps",
        "median_excess_bps",
        "level_shift_bps",
        "level_shift_fraction_of_advantage",
        "naive_gap_bps",
        "passed",
    }
    assert verdict["tolerance_denominator"] == AVAILABLE_ADVANTAGE
    assert verdict["graded_attribute"] == "advantage_fraction"
    assert verdict["denominator_bps"] > 0.0
    assert verdict["naive_gap_bps"] > verdict["denominator_bps"], (
        "the naive gap must exceed the adaptive advantage by the level shift; if "
        "it does not, the two are being computed from the same rung"
    )
    assert verdict["level_shift_bps"] == pytest.approx(
        verdict["naive_gap_bps"] - verdict["denominator_bps"], rel=1e-9
    )
    assert 0.0 < verdict["level_shift_fraction_of_advantage"] < 0.10


def test_the_liquidity_summary_carries_the_headline_and_its_control():
    """Median, IQR, per-seed values — and the shuffled control beside them."""
    document = build_document(_liquidity_sweep())
    summary = document["summary"]

    assert set(summary) >= {
        "advantage_fraction",
        "objective",
        "excess",
        "capture_fraction",
    }
    capture = summary["capture_fraction"]
    advantage = summary["advantage_fraction"]
    assert len(capture["values"]) == 10
    assert capture["median"] == pytest.approx(1.0 - advantage["median"])
    # The worst seed is the *largest* excess, so the worst capture is the
    # smallest — the direction a summariser that treats every quantity as a cost
    # would get backwards exactly once, silently, in a verdict.
    assert capture["worst"] == pytest.approx(1.0 - advantage["worst"])
    assert capture["worst"] <= capture["median"]

    control = document["shuffled_control"]
    assert control is not None and len(control["values"]) == 10
    assert document["verdict"]["shuffled_control_met"] in {True, False}


def test_a_liquidity_sweep_without_a_reference_refuses_to_write():
    """The DP is what the grades are excesses *over*; a file without it is a lie."""
    sweep = dataclasses.replace(_liquidity_sweep(), liquidity_reference=None)
    with pytest.raises(ValueError, match="no reference row"):
        build_document(sweep)


def test_every_fixed_rung_is_graded_through_the_agent_s_own_route():
    """Invariant 4: the baselines are on the chart, and by the same arithmetic.

    The static optimum is the control variate, so its sampled grade must return
    its closed form *exactly* — that is the sharpest available check that the
    grading path is unbiased, and it costs nothing.
    """
    sweep = _liquidity_sweep()
    document = build_document(sweep)
    baselines = document["baselines"]
    assert set(baselines) == {"twap", "ac", "tangent", "m4a", "static"}
    assert baselines["static"]["objective_bps"] == pytest.approx(
        sweep.liquidity_reference.static.objective, rel=1e-12
    )
    assert baselines["static"]["capture_fraction"] == pytest.approx(0.0, abs=1e-12)
    for name in ("twap", "ac", "tangent", "m4a"):
        row = baselines[name]
        closed = sweep.liquidity_reference.schedules[name].objective
        assert abs(row["objective_bps"] - closed) <= 4.0 * row["half_width_bps"]


# ---------------------------------------------------------------------------
# The *driver's* reporting path, on fabricated data
# ---------------------------------------------------------------------------
#
# These exist because the house note's lesson arrived a second time, during M4b
# and before the run rather than after it — but only because the run was watched.
# ``tools/train.py`` reads a grade's fields to print one line per seed, and a
# ``LiquidityGrade`` has none of ``relative_excess``, ``gap_fraction`` or
# ``deviation``: the driver would have trained seed 0 for twenty minutes and then
# died on an ``AttributeError`` in a print statement. The sibling defect was
# quieter and worse — ``--dry-run`` printed the *deterministic* world's advantage
# as M4b's bar, understating it by 1.7x in the flattering direction.
#
# Both are pure functions of data. Neither had a test, for exactly the reason the
# note describes: the only way to reach them was to pay for the producer first.


def _driver():
    """``tools/train.py`` as a module. Imported inside the tests, not at collection.

    The driver pins the OpenMP pools from the config before torch is imported, so
    importing it at module scope would do that as a side effect of *collecting*
    this file.
    """
    import importlib
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return importlib.import_module("tools.train")


def test_the_per_seed_line_renders_in_both_grade_shapes():
    """The line the driver prints after every seed, in each world's grade type.

    A ``Grade`` has a trajectory, a deviation and a TWAP gap; a
    ``LiquidityGrade`` has a distribution of schedules and a confidence interval.
    Rendering the second through the first's fields is the ``AttributeError``
    that only a training run could reach.
    """
    driver = _driver()
    result = _training_result(load_experiment(CONFIGS[POWER_LAW_ENCODING]).ppo, 4)

    analytic = _sweep(POWER_LAW_ENCODING, n_seeds=5).grades[0]
    line = driver._seed_line(0, analytic, result)
    assert "seed 0" in line and "of the TWAP gap" in line and "capture" in line

    liquidity = _liquidity_sweep(n_seeds=5).liquidity_grades[0]
    line = driver._seed_line(3, liquidity, result)
    assert "seed 3" in line
    assert "±" in line, "a sampled grade must print its interval, not a bare level"
    assert "excess over J_DP" in line
    assert "paths" in line
    assert "TWAP gap" not in line, (
        "the liquidity line quoted a TWAP gap; that field does not exist on a "
        "LiquidityGrade and the number would be from the wrong world"
    )


def test_the_liquidity_line_shouts_when_a_path_beats_perfect_information():
    """The rigorous red flag has to be *visible* in the run's own output."""
    driver = _driver()
    grade = _liquidity_sweep(n_seeds=5).liquidity_grades[0]
    flagged = dataclasses.replace(grade, paths_below_clairvoyant=7, red_flag=True)
    line = driver._seed_line(1, flagged, _training_result(None, 4))
    assert "7 PATHS BELOW CLAIRVOYANT" in line
    assert "RED FLAG" in line


def test_the_denominator_refuses_the_wrong_world_rather_than_answering():
    """The quieter of the two defects, and the one that would have been believed.

    In the liquidity world the bar is a fraction of the *adaptive* advantage,
    0.0621 bps. The deterministic row's ``available_advantage`` is M4a's tangent
    advantage, 0.0367 — so a bar printed from it reads 1.7x tighter than the one
    the agent is actually held to, in the direction that flatters the result.
    """
    experiment = load_experiment(M4B_CONFIG)
    with pytest.raises(ValueError, match="adaptive advantage"):
        experiment.denominator_bps()

    reference = _liquidity_sweep(n_seeds=5).liquidity_reference
    assert experiment.denominator_bps(reference) == reference.adaptive_advantage
    assert reference.adaptive_advantage > experiment.reference().available_advantage


def test_the_figure_is_skipped_rather_than_half_drawn_without_the_curve(tmp_path, capsys):
    """No value-of-sight curve, no figure — the honest failure, not the flattering one.

    The curve against ``sigma_L`` is what stops a single invented parameter with a
    single number beside it reading as calibration. A figure drawn without it
    would be the more impressive picture and the less honest one, so its absence
    is a skip with a message rather than a panel quietly left out.
    """
    import dataclasses as dc

    driver = _driver()
    sweep = _liquidity_sweep(n_seeds=5)
    document = build_document(sweep)
    experiment = dc.replace(
        sweep.experiment, results_figure=tmp_path / "m4b_adaptivity"
    )

    real = driver.REPO_ROOT
    try:
        driver.REPO_ROOT = tmp_path  # no results/m4b_reference.json under here
        driver.write_figure(experiment, document)
    finally:
        driver.REPO_ROOT = real
    out = capsys.readouterr().out
    assert "skipped the figure" in out and "value-of-sight" in out
    assert not list(tmp_path.glob("m4b_adaptivity.*"))


def test_the_figure_draws_from_the_committed_table_when_it_is_there(tmp_path):
    """And when the curve exists, both panels are drawn from committed data only."""
    import dataclasses as dc

    driver = _driver()
    table = REPO_ROOT / "results" / "m4b_reference.json"
    if not table.exists():
        pytest.skip("task 0's table has not been generated in this tree")

    sweep = _liquidity_sweep(n_seeds=5)
    document = build_document(sweep)
    experiment = dc.replace(
        sweep.experiment, results_figure=tmp_path / "m4b_adaptivity"
    )
    driver.write_figure(experiment, document)
    written = sorted(tmp_path.glob("m4b_adaptivity.*"))
    assert written and written[0].stat().st_size > 10_000


def test_the_caption_never_omits_the_three_things_it_may_not(tmp_path):
    """Invented, denominator, bracket — every time this figure is drawn."""
    driver = _driver()
    table = REPO_ROOT / "results" / "m4b_reference.json"
    if not table.exists():
        pytest.skip("task 0's table has not been generated in this tree")

    from tools.m4b_adaptivity import build_rungs, caption

    sweep = _liquidity_sweep(n_seeds=5)
    document = build_document(sweep)
    text = caption(sweep.experiment, document, build_rungs(document))

    assert "INVENTED" in text
    assert "ADAPTIVE advantage" in text and "NOT J_M4a - J_DP" in text
    assert "level shift" in text
    assert "CONVERGED AND BRACKETED, not certified" in text
    assert "no price sampling" in text
    assert "shuffled control" in text.lower()

    # And it fits. matplotlib draws text straight past the figure edge without a
    # word of complaint, and the house note records a caption doing exactly that
    # on a committed artefact — so the width is bounded where the string is built
    # and the bound is checked here rather than noticed in a picture.
    from tools.m4b_adaptivity import CAPTION_WIDTH

    overlong = [line for line in text.splitlines() if len(line) > CAPTION_WIDTH]
    assert not overlong, (
        f"{len(overlong)} caption line(s) exceed {CAPTION_WIDTH} characters and "
        f"will run off the canvas: {overlong[0][:80]!r}..."
    )


def test_the_figure_tool_runs_end_to_end_on_a_fabricated_result(tmp_path):
    """``tools/m4b_adaptivity.py`` as the Makefile invokes it, with no training.

    ``write_figure`` was already covered; ``main`` was not, and the two are
    different code — ``main`` resolves paths, reads both artefacts off disk and
    reports where it wrote. That last part is where it failed the first time this
    ran: the figure had already been written and the process died on the
    ``relative_to`` in the line that says so. A driver that throws away a file's
    only mention while reporting it is the same shape as one that dies while
    printing a grade, which is the defect this module exists for.
    """
    import json
    import runpy
    import sys

    table = REPO_ROOT / "results" / "m4b_reference.json"
    if not table.exists():
        pytest.skip("task 0's table has not been generated in this tree")

    document = build_document(_liquidity_sweep(n_seeds=5))
    metrics = tmp_path / "m4b_liquidity.json"
    metrics.write_text(json.dumps(document), encoding="utf-8")

    config = (REPO_ROOT / "configs" / "m4b_liquidity.yaml").read_text(encoding="utf-8")
    config = config.replace(
        "metrics: results/m4b_liquidity.json", f"metrics: {metrics.as_posix()}"
    ).replace(
        "figure: results/m4b_adaptivity",
        f"figure: {(tmp_path / 'm4b_adaptivity').as_posix()}",
    )
    config_path = tmp_path / "m4b.yaml"
    config_path.write_text(config, encoding="utf-8")

    argv = sys.argv
    try:
        sys.argv = ["m4b_adaptivity.py", "--config", str(config_path)]
        with pytest.raises(SystemExit) as exit_info:
            runpy.run_path(
                str(REPO_ROOT / "tools" / "m4b_adaptivity.py"), run_name="__main__"
            )
    finally:
        sys.argv = argv

    assert exit_info.value.code == 0
    written = sorted(tmp_path.glob("m4b_adaptivity.*"))
    assert written and written[0].stat().st_size > 50_000


@pytest.mark.parametrize("encoding", sorted(CONFIGS))
def test_the_closing_summary_renders_in_the_deterministic_worlds(encoding, capsys):
    """``print_verdict`` on a fabricated sweep, in each world that has trajectories."""
    driver = _driver()
    document = build_document(_sweep(encoding))
    driver.print_verdict(load_experiment(CONFIGS[encoding]), document)
    out = capsys.readouterr().out

    assert "median excess" in out and "shares against a derived bound" in out
    assert "capture fraction" not in out or "certified optimum" in out
    assert "liquidity-shuffled control" not in out


def test_the_closing_summary_renders_in_the_liquidity_world(capsys):
    """The same function on M4b's document — the branch that cost a full sweep.

    This block was inline in ``main`` and reachable only by training ten seeds
    first. It read ``summary['relative_excess']``, which a liquidity summary does
    not have, and the run died on it **after** grading all ten seeds and writing
    both artefacts. Nothing was lost only because ``write_outputs`` runs before
    the printing. Extracting it and calling it here costs milliseconds.
    """
    driver = _driver()
    sweep = _liquidity_sweep()
    document = build_document(sweep)
    driver.print_verdict(sweep.experiment, document)
    out = capsys.readouterr().out

    # The four things a liquidity run must say, and the two it must not.
    assert "advantage_fraction" in out
    assert "level shift" in out and "not the agent's" in out
    assert "liquidity-shuffled control" in out
    assert "capture fraction" in out
    assert "converged, bracketed" in out, (
        "the closing summary called a dynamic program 'the certified optimum'; "
        "M4a earned that word and this is not the same word"
    )
    assert "certified optimum" not in out
    assert "shares against a derived bound" not in out, (
        "a liquidity-observing policy has a distribution of schedules, so there "
        "is no single trajectory deviation to quote"
    )


def test_the_closing_summary_shouts_the_rigorous_red_flag(capsys):
    """A seed below perfect information has to be visible in the run's own output."""
    import copy

    driver = _driver()
    sweep = _liquidity_sweep()
    document = copy.deepcopy(build_document(sweep))
    document["verdict"]["seeds_below_clairvoyant"] = ["seed3"]
    driver.print_verdict(sweep.experiment, document)
    out = capsys.readouterr().out
    assert "RED FLAG (rigorous)" in out and "seed3" in out
