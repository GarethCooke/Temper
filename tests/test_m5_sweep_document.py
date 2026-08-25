"""M5 task 6 — the reporting path, exercised before the run rather than by it.

``docs/house-notes.md``, *No code path may be reachable only at the end of a long
run*. M5's ROADMAP row carries it as a definition-of-done item rather than as
advice, and it names the whole path rather than the producer: the document
assembly, the three-number computation, the shuffled-control re-grade, the
red-flag evaluation, the figure and its caption, and every line that reports where
a file was written.

M4b obeyed that note where its brief named the function and was bitten **four more
times** in code the brief had not — twice after a full sweep had been graded. So
here the list is a list, each entry is asserted, and the one entry that does not
exist yet (the figure) is asserted to *not exist*, which is what turns the wrap-up
session's first move into a red test rather than a silent gap.

The driver runs :func:`tools.train.alpha_reporting_pass` before seed 0. This module
runs the same function, so what the suite checks and what the run checks are one
object.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
from dataclasses import replace

import numpy as np
import pytest

from temper.agents.ppo import TrainResult
from temper.eval.experiment import load_experiment
from temper.eval.sweep import (
    ALPHA_CAPTURE_BAR,
    BAR_SUFFIX,
    ALPHA_DERIVED,
    ALPHA_DIRECTIONS,
    ALPHA_HEADLINE,
    PREMIUM_RATIO_BAR,
    SHUFFLED_NET_CAPTURE_BAR,
    BudgetBound,
    SweepResult,
    build_alpha_document,
    format_alpha_headline,
    seal_verdict,
    refuse_if_budget_bound,
)

from .conftest import REPO_ROOT

CONFIG = REPO_ROOT / "configs" / "m5_alpha.yaml"

#: Paths for the fabricated grades. Enough that the rollout, the conditional cost
#: and the pairing all run for real; small enough that this is a test.
PASS_PATHS = 96


def _driver():
    """`tools/train.py` as a module. It is a script, and this is the one importer."""
    spec = importlib.util.spec_from_file_location(
        "temper_train_driver", REPO_ROOT / "tools" / "train.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def experiment():
    return load_experiment(CONFIG)


@pytest.fixture(scope="module")
def passed(experiment):
    """The document the pre-run pass produced — the same call the driver makes."""
    return _driver().alpha_reporting_pass(experiment, paths=PASS_PATHS).document


def test_the_whole_reporting_path_runs_on_fabricated_data(passed, experiment):
    """Every named stage completed, and the document it produced is well formed."""
    assert passed["milestone"] == "M5"
    assert passed["signal"]["rho"] == experiment.signal.correlation()
    assert len(passed["seeds"]) == experiment.seeds.n_seeds
    assert passed["reference_kind"]["execution_floor"]["certified"] is True
    assert passed["reference_kind"]["adaptive_optimum"]["certified"] is False
    for seed in passed["seeds"]:
        for key in ("training", "budget", "grade", "headline", "per_bin_alpha_bps",
                    "shuffled", "shuffled_headline"):
            assert seed[key] is not None, key


def test_the_three_numbers_are_reported_together_and_never_one(passed, rebuilt):
    """Item 4, as a property of the code rather than of everyone's discipline.

    At the optimum 45 % of the gross effect is paid back, so a net capture alone
    cannot distinguish a policy that trades the signal well from one that trades
    it badly and executes well. The formatter emits all three or it is not the
    formatter every call site uses.
    """
    line = format_alpha_headline(rebuilt["grades"][0])
    assert "alpha" in line and "premium" in line and "net" in line
    # Every fraction carries the bps it is a fraction of.
    assert line.count("bps") >= 3

    head = passed["verdict"]["headline"]
    for name in ALPHA_HEADLINE:
        assert f"{name}_median" in head, name
    for absolute in (
        "alpha_bps_median",
        "reference_alpha_bps",
        "execution_premium_bps_median",
        "reference_premium_bps",
        "median_excess_bps",
        "advantage_bps",
    ):
        assert absolute in head, absolute

    assert passed["verdict"]["alpha_capture_met"] is not None
    assert passed["verdict"]["premium_ratio_met"] is not None
    assert ALPHA_CAPTURE_BAR == 0.85 and PREMIUM_RATIO_BAR == 1.30


def test_the_shuffled_control_changed_the_answer(passed):
    """Item 2: the control gets its own line, and it has to *do* something.

    A fixed schedule ignores the observation, so shuffling it returns the
    identical number and the control's code path is exercised while the control is
    not. The pass uses a signal-reading policy for exactly that reason, and this
    asserts the difference rather than the execution.
    """
    seed = passed["seeds"][0]
    assert seed["shuffled"]["alpha_bps"] != seed["grade"]["alpha_bps"]
    assert seed["shuffled_headline"]["net_capture"] != seed["headline"]["net_capture"]
    control = passed["shuffled_control"]
    assert control["bar"] == SHUFFLED_NET_CAPTURE_BAR == -0.50
    assert control["net_capture"] is not None
    assert control["address"][0] == "m5/reference"


def test_the_alpha_term_is_decomposed_per_bin_and_closes(passed):
    """Item 3, and the first entry is the one worth looking at.

    Three separate defects in this milestone lived at the first bin — the alpha
    sum starting a bin early, the conditional variance losing the first bin's
    whole share, and the seam's own timing being one bin out. Entry 0 is exactly
    zero because ``xi_0`` is predicted by nothing, and it is kept in the list
    rather than trimmed so a shifted index shows up as a non-zero first entry.
    """
    for seed in passed["seeds"]:
        per_bin = seed["per_bin_alpha_bps"]
        assert len(per_bin) == 13
        assert per_bin[0] == 0.0, (
            "the first bin carries alpha; every schedule holds the whole order "
            "through xi_0 and nothing predicts it, so the sum starts a bin early"
        )
        assert sum(per_bin) == pytest.approx(-seed["grade"]["alpha_bps"], rel=1e-9)
        assert any(value != 0.0 for value in per_bin[1:])


def test_the_pre_run_pass_renders_the_figure_and_its_caption(experiment):
    """The successor to a test that was written to fail on purpose.

    Through task 6 this asserted that M5's figure did **not** exist, with the
    instruction that the only way to make it green was to wire the figure into
    `alpha_reporting_pass`. That is the trick worth keeping: a placeholder that
    goes red the moment the work it is holding a place for lands, so the wiring
    cannot be forgotten by anybody who runs the suite.

    The figure itself is drawn from COMMITTED artefacts by `make m5-figure` after
    the sweep, because at sweep time the document it reads is not yet committed.
    The caption is a different matter — it carries numbers, so it is a claim, and
    M4b lost a figure tool to its caption after a full sweep had rendered. So the
    pass renders both on the fabricated document, into `results/scratch/`.
    """
    scratch = REPO_ROOT / "results" / "scratch"
    drawn = scratch / "m5_pass_figure.png"
    if drawn.exists():
        drawn.unlink()

    _driver()._fabricated_alpha_figure(
        json.loads((scratch / "m5_pass.json").read_text(encoding="utf-8")), scratch
    )

    assert drawn.exists() and drawn.stat().st_size > 10_000
    source = (REPO_ROOT / "tools" / "train.py").read_text(encoding="utf-8")
    assert "make m5-figure" in source, (
        "the sweep's figure branch no longer says where the figure comes from"
    )


def test_a_budget_bound_sweep_cannot_pass(experiment, passed):
    """Item 0, at the document: a bound budget fails the verdict outright."""
    reference_doc = passed
    assert reference_doc["verdict"]["timed_out"] == []
    for seed in reference_doc["seeds"]:
        assert seed["budget"]["timed_out"] is False
        assert seed["budget"]["bound_at_update"] is None
        assert seed["budget"]["target_updates"] == experiment.ppo.num_updates

    with pytest.raises(BudgetBound, match="runaway guard"):
        refuse_if_budget_bound(
            [{"timed_out": True, "updates": 612, "target_updates": 751}],
            comparison="a cross-seed comparison",
        )
    refuse_if_budget_bound(
        [{"timed_out": False, "updates": 751, "target_updates": 751}],
        comparison="a clean one",
    )


def test_a_bound_budget_turns_the_verdict_red(experiment, rebuilt):
    """And the same at the sweep level, by rebuilding the document with one bound.

    Checked by construction rather than by inspection: a document whose seeds
    trained different numbers of updates is a median over different amounts of
    training, and `passed` must be False however good the numbers look.
    """
    driver = _driver()
    fabricated = driver._fabricated_training(experiment)
    bound = replace(fabricated[3], timed_out=True, updates=experiment.ppo.num_updates - 7)
    fabricated[3] = bound

    grade = rebuilt
    sweep = SweepResult(
        experiment=experiment,
        baselines={},
        grades=(),
        training=tuple(fabricated),
        seconds=1.0,
        provenance=experiment.provenance(REPO_ROOT),
        pairs=tuple(() for _ in fabricated),
        ordinals=tuple(range(experiment.seeds.n_seeds)),
        alpha_grades=grade["grades"],
        shuffled_alpha_grades=grade["shuffled"],
        alpha_detail=grade["detail"],
        alpha_reference_row=grade["reference"],
        alpha_baselines={},
    )
    document = build_alpha_document(sweep)
    assert document["verdict"]["timed_out"] == [3]
    assert document["verdict"]["passed"] is False
    assert document["verdict"]["budgets"][3]["bound_at_update"] == (
        experiment.ppo.num_updates - 7
    )


@pytest.fixture(scope="module")
def rebuilt(experiment):
    """Real grades for a signal-reading policy, reused by the tests that need one."""
    from temper.eval.sweep import alpha_reference, grade_alpha

    driver = _driver()
    reference = alpha_reference(experiment)
    policy = driver._TiltedSchedule(experiment.case.market, experiment.case.order_size)
    grade, detail = grade_alpha(
        experiment, policy, reference, name="fabricated", paths=PASS_PATHS
    )
    shuffled, _ = grade_alpha(
        experiment,
        policy,
        reference,
        name="fabricated_shuffled",
        paths=PASS_PATHS,
        shuffled=True,
    )
    n = experiment.seeds.n_seeds
    return {
        "reference": reference,
        "grades": tuple(replace(grade, name=f"seed{i}") for i in range(n)),
        "shuffled": tuple(replace(shuffled, name=f"seed{i}_s") for i in range(n)),
        "detail": tuple(dict(detail) for _ in range(n)),
    }


def test_the_training_env_and_the_graded_env_show_the_same_thing(experiment):
    """The check that would have saved twenty minutes, and could have saved more.

    M5 task 6's first sweep trained seed 0 on a two-coordinate observation —
    `train_seed` had never been handed the signal stream — and graded it on a
    three-coordinate one. The network is built at the training width, so grading
    died on ``mat1 and mat2 shapes cannot be multiplied (1x3 and 2x64)``.

    It was loud, and that is luck rather than design: had the widths happened to
    match, an agent trained blind and graded sighted would have produced a
    plausible capture fraction about a world it never traded in, with every
    identity, differential and guard in this repo still green — because each of
    them checks ONE env, and this is a statement about two.
    """
    driver = _driver()
    training_space, graded_space = driver.assert_training_and_grading_agree(experiment)
    assert training_space.shape == graded_space.shape == (3,), (
        "M5's observation is (time left, inventory left, s); if this is two wide "
        "the signal seam is not reaching the env the agent trains in"
    )


def test_training_and_evaluation_signals_come_from_disjoint_pools(experiment):
    """Invariant 5's out-of-sample claim, where both halves are in scope at once."""
    from temper.eval.sweep import evaluation_signal, training_signal

    assert training_signal(experiment).pool == "m5/signal-train"
    assert evaluation_signal(experiment).pool == "m5/signal-eval"
    assert training_signal(experiment).pool != evaluation_signal(experiment).pool


def test_the_worst_seed_is_the_worst_seed(experiment, rebuilt):
    """`summarise` calls `worst` the MAXIMUM, so only costs may be summarised.

    M5's first sweep shipped a reported worst net capture of 0.9559 — its *best*
    seed — because `net_capture` is a benefit and was summarised directly, then
    inverted into `advantage_fraction`. The verdict was unaffected (the true worst,
    0.1075, also clears the 0.25 bar) and the number was wrong, and a worse sweep
    is exactly where it would have mattered.

    So the direction is asserted rather than reasoned about: build a document from
    grades with *known, different* net captures and require the reported worst to
    be the one that actually did worst.
    """
    from dataclasses import replace as _replace

    base = rebuilt["grades"][0]
    n = experiment.seeds.n_seeds
    # Ten grades whose net capture is strictly increasing, by moving the objective:
    # net_capture = (J_M4a - J) / (J_M4a - J_DP), so a LOWER objective is better.
    spread = [
        _replace(base, objective=base.objective - 0.001 * i, name=f"seed{i}")
        for i in range(n)
    ]
    captures = [g.net_capture for g in spread]
    assert captures[0] < captures[-1], "the fixture is not actually spread"

    driver = _driver()
    sweep = SweepResult(
        experiment=experiment,
        baselines={},
        grades=(),
        training=tuple(driver._fabricated_training(experiment)),
        seconds=1.0,
        provenance=experiment.provenance(REPO_ROOT),
        pairs=tuple(() for _ in range(n)),
        ordinals=tuple(range(n)),
        alpha_grades=tuple(spread),
        shuffled_alpha_grades=rebuilt["shuffled"],
        alpha_detail=rebuilt["detail"],
        alpha_reference_row=rebuilt["reference"],
        alpha_baselines={},
    )
    summary = build_alpha_document(sweep)["summary"]

    assert summary["net_capture"]["worst"] == pytest.approx(min(captures)), (
        "the reported worst net capture is not the smallest one; `summarise` "
        "returns the MAXIMUM and net capture is a benefit, so it must be derived "
        "from the cost rather than summarised directly"
    )
    assert summary["advantage_fraction"]["worst"] == pytest.approx(
        max(1.0 - c for c in captures)
    )
    assert summary["advantage_fraction"]["median"] == pytest.approx(
        1.0 - summary["net_capture"]["median"]
    )


@pytest.fixture(scope="module")
def dominated(experiment, rebuilt):
    """Ten grades in which ONE seed is worse than every other by every measure.

    The blind spot that let the first sweep ship its best seed as its worst was
    fabricated data with no spread: repeat one grade ten times and `max` and `min`
    return the same number, so a direction error is invisible. Repeat it with a
    spread and the test can still be written to agree with whatever table the code
    happens to hold — which is a tautology, not a check.

    So the spread here is *dominated*: seed 9 has the highest objective, the least
    gross alpha and the highest execution cost, and therefore the worst value of
    every quantity the document reports, derived ones included. No table is
    consulted to know that. "The worst seed is seed 9" is the definition of worst,
    and every reported ``worst`` has to be seed 9's number or it is wrong.
    """
    base = rebuilt["grades"][0]
    n = experiment.seeds.n_seeds
    grades = tuple(
        replace(
            base,
            name=f"seed{i}",
            objective=base.objective + 0.0011 * i,
            alpha_bps=base.alpha_bps * (1.0 - 0.02 * i),
            execution_bps=base.execution_bps + 0.0009 * i,
        )
        for i in range(n)
    )
    sweep = SweepResult(
        experiment=experiment,
        baselines={},
        grades=(),
        training=tuple(_driver()._fabricated_training(experiment)),
        seconds=1.0,
        provenance=experiment.provenance(REPO_ROOT),
        pairs=tuple(() for _ in range(n)),
        ordinals=tuple(range(n)),
        alpha_grades=grades,
        shuffled_alpha_grades=rebuilt["shuffled"],
        alpha_detail=rebuilt["detail"],
        alpha_reference_row=rebuilt["reference"],
        alpha_baselines={},
    )
    return sweep, build_alpha_document(sweep), n - 1


def test_every_reported_worst_is_the_seed_that_is_worst_by_construction(dominated):
    """The general form of the defect, asked of every reported field at once.

    ``test_the_worst_seed_is_the_worst_seed`` fabricates a spread in one field and
    checks that field. This checks all of them, against a seed that is worst on
    every axis at once, so no direction table is trusted to define the answer.

    What it catches that the single-field fix did not: ``alpha_capture`` reported
    1.1099 — its BEST seed — as its worst, in the same artefact and for the same
    reason, and repairing ``net_capture`` by hand did nothing for it. A per-field
    repair fixes a field; this fixes the class.
    """
    _, document, worst_seed = dominated
    summary = document["summary"]
    for name, block in summary.items():
        values = block["values"]
        assert min(values) < max(values), f"{name} does not vary across seeds"
        assert block["worst"] == pytest.approx(values[worst_seed]), (
            f"{name} reports worst={block['worst']}, but the seed that is worse "
            f"than every other on every axis reads {values[worst_seed]}"
        )
    # And the declared direction has to match what the dominated seed showed, or
    # the table and the document are agreeing with each other about a wrong answer.
    for name, direction in ALPHA_DIRECTIONS.items():
        values = summary[name]["values"]
        observed = "cost" if values[worst_seed] == max(values) else "benefit"
        assert observed == direction, f"{name} is declared {direction}, reads {observed}"


def test_a_reported_field_with_no_declared_direction_is_refused(dominated, monkeypatch):
    """The table has to be consulted, not merely written down.

    A field added to the summary and to neither table is a field whose ``worst``
    nobody chose. Simulated by removing a declaration rather than adding a field,
    which is the same hole approached from the other side.
    """
    import temper.eval.sweep as sweep_module

    trimmed = dict(ALPHA_DIRECTIONS)
    trimmed.pop("alpha_capture")
    monkeypatch.setattr(sweep_module, "ALPHA_DIRECTIONS", trimmed)
    with pytest.raises((AssertionError, KeyError)):
        build_alpha_document(dominated[0])


def test_summarise_refuses_a_direction_it_does_not_understand():
    """A typo in the direction must not silently mean `cost`."""
    from temper.eval.grading import summarise

    assert summarise("c", [1.0, 3.0, 2.0]).worst == 3.0
    assert summarise("c", [1.0, 3.0, 2.0], direction="cost").worst == 3.0
    assert summarise("b", [1.0, 3.0, 2.0], direction="benefit").worst == 1.0
    with pytest.raises(ValueError, match="cost.*benefit"):
        summarise("x", [1.0], direction="higher_is_better")


#: Every bar M5's document records, written out here so a bar a future milestone
#: adds has to arrive with its own veto case rather than inheriting a green suite.
#: The test below requires this list to be exactly what the document carries.
M5_BARS = [
    "alpha_capture_met",
    "epsilon_met",
    "per_seed_met",
    "premium_ratio_met",
    "shuffled_control_met",
]


def _passing_verdict(**overrides):
    """A verdict that clears every bar, sealed. The base case for the vetoes below.

    Built from ``M5_BARS`` rather than from a graded document on purpose: whether a
    fabricated policy happens to clear five bars is not the subject, and a fixture
    that has to pass before it can be made to fail is a fixture that will one day
    fail for the other reason.
    """
    verdict = {bar: True for bar in M5_BARS}
    verdict.update({"red_flags": [], "timed_out": []})
    verdict.update(overrides)
    return seal_verdict(verdict)


def test_the_verdict_gates_on_every_bar_it_records(dominated):
    """Not on the two that existed when the line was first written.

    M5 recorded five bars and gated on two. The other three sat beside the answer
    looking like they meant something: `alpha_capture_met` and `premium_ratio_met`
    were computed *below* the `passed` line, and `gate_met` has been in the
    non-alpha document the same way since M3.
    """
    verdict = dominated[1]["verdict"]
    bars = sorted(k for k in verdict if k.endswith(BAR_SUFFIX))
    assert bars, "the document records no bars at all"
    assert verdict["gated_on"] == bars
    assert bars == M5_BARS, (
        "the document's bars have changed. Add the new one to M5_BARS so it gets "
        "its own veto case below; a bar with no negative test is a bar nobody has "
        "watched fail"
    )


@pytest.mark.parametrize("bar", M5_BARS)
def test_any_single_bar_can_veto_the_verdict_and_is_named_when_it_does(bar):
    """One negative case per bar, because "gates on all of them" is five claims.

    A rule written over the fields can still be wrong for one field — a bar read
    from the wrong place, or `None` where it should be `False` — and a single
    test on a single bar would not show it. Each is flipped in turn and must both
    veto and say so: a verdict that fails without naming which bar failed sends a
    reader back to recompute five numbers by hand.
    """
    assert _passing_verdict()["passed"] is True

    verdict = _passing_verdict(**{bar: False})

    assert verdict["passed"] is False
    assert verdict["failed_bars"] == [bar]


def test_a_bar_that_does_not_apply_is_recorded_rather_than_dropped():
    """`None` is "no control was run", not "the control passed" and not a failure.

    Absent and not-applicable look identical once a field is simply missing, which
    is how a bar goes quiet. It is carried in `bars_not_applicable` instead.
    """
    verdict = _passing_verdict(shuffled_control_met=None)
    assert verdict["passed"] is True
    assert verdict["bars_not_applicable"] == ["shuffled_control_met"]
    assert verdict["failed_bars"] == []


def test_the_refusals_that_are_not_bars_still_refuse():
    """Red flags and a bound budget are not tolerances, and still veto.

    They are deliberately outside the `_met` contract: a red flag is a defect with
    a proof, and a budget that bound early means the seeds were not trained the
    same amount, so neither is a threshold anyone chose. They keep their own
    clauses in the seal.
    """
    assert _passing_verdict(red_flags=["seed3"])["passed"] is False
    assert _passing_verdict(timed_out=[7])["passed"] is False
    # ...and neither is reported as a failed bar, because neither is one.
    assert _passing_verdict(red_flags=["seed3"])["failed_bars"] == []


def test_a_verdict_with_no_bars_is_refused():
    """Losing the tolerances must not read as meeting them."""
    with pytest.raises(AssertionError, match="no bars is not a verdict"):
        seal_verdict({"red_flags": [], "timed_out": []})

@pytest.fixture(scope="module")
def rehearsal():
    """One end-to-end invocation of the driver, argv and exit code included.

    Module-scoped because it costs seconds rather than milliseconds, and because
    the two claims below are about the same single run: what it printed, and what
    it left on disk.
    """
    committed = REPO_ROOT / "results" / "m5_alpha.json"
    before = committed.read_bytes() if committed.exists() else None

    driver = _driver()
    argv = sys.argv
    sys.argv = ["train.py", "--config", str(CONFIG), "--rehearse"]
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            status = driver.main()
    finally:
        sys.argv = argv

    after = committed.read_bytes() if committed.exists() else None
    return {
        "status": status,
        "out": buffer.getvalue(),
        "artefact_before": before,
        "artefact_after": after,
    }


def test_the_driver_runs_end_to_end_from_its_own_command_line(rehearsal):
    """The clause the house note just gained, applied to the entry point.

    `alpha_reporting_pass` is a function that calls functions, and `main` is not
    one of them. Two of M5's three post-run defects were down there — `main`'s
    `--expect` check reading a name that had been unbound for two milestones, and
    the baselines line reading the empty dict of the wrong world — so neither was
    reachable by any amount of coverage underneath, and both fired only after ten
    seeds had trained.

    This invokes the driver the way a user does. It trains nothing.
    """
    assert rehearsal["status"] == 0
    out = rehearsal["out"]
    # Both exit branches, not only the one a passing fabrication would reach.
    assert "--expect any exited 0 (want 0)" in out
    assert "exited 1 (want 1)" in out
    # And the two lines that were wrong. The baselines line is the one that read
    # `sweep.baselines` and so printed nothing at all.
    assert "baselines graded through the same rollout (bps): twap" in out
    assert "THE THREE NUMBERS" in out
    assert "verdict:" in out


def test_the_rehearsal_writes_nothing_a_reader_would_quote(rehearsal):
    """A dry run that can touch a committed artefact is not a dry run.

    `results/m5_alpha.json` is the file every other document quotes. The rehearsal
    runs the same `write_outputs` the real driver does, so the only thing between a
    rehearsal and the committed result is one `write=False` argument — which makes
    it worth an assertion rather than a reading of the code.
    """
    assert rehearsal["artefact_after"] == rehearsal["artefact_before"]
