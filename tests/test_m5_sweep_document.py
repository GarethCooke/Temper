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

import importlib.util
import json
from dataclasses import replace

import numpy as np
import pytest

from temper.agents.ppo import TrainResult
from temper.eval.experiment import load_experiment
from temper.eval.sweep import (
    ALPHA_CAPTURE_BAR,
    ALPHA_HEADLINE,
    PREMIUM_RATIO_BAR,
    SHUFFLED_NET_CAPTURE_BAR,
    BudgetBound,
    SweepResult,
    build_alpha_document,
    format_alpha_headline,
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
    return _driver().alpha_reporting_pass(experiment, paths=PASS_PATHS)


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


def test_the_figure_does_not_exist_yet_and_the_pass_says_so(experiment):
    """The one entry on the list that is not written, asserted as absent.

    The wrap-up session writes M5's figure. When it does, this test goes red and
    the only way to make it green is to wire the figure into
    ``alpha_reporting_pass`` — which is the point: a figure that has never been
    drawn on fabricated data is precisely the code path the house note is about,
    and M4b was bitten by its caption after a full sweep.
    """
    assert not (REPO_ROOT / "results" / "m5_alpha.png").exists(), (
        "M5's figure now exists. Add it to tools/train.py's alpha_reporting_pass "
        "— the figure AND its caption — and then delete this test."
    )
    source = (REPO_ROOT / "tools" / "train.py").read_text(encoding="utf-8")
    assert "no figure: M5's is the wrap-up session's" in source


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
