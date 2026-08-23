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
``build_document`` over the pair in both encodings — asserting the shape a
committed result must have. The portable form of the rule is
``docs/house-notes.md``, *The artefact writer is tested on fabricated data, not on
the run*.

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

