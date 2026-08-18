"""M4a task 5 — the training point: the project's first *earned* advantage, checked.

The run takes about three hours and lives behind ``make m4a``. What runs on every
commit is this: ``results/m4a_power_law.json`` is read back and checked against
the config that claims to have produced it (invariant 1, by digest), against the
pre-stated bars (invariant 3), and against the milestone's one hard failure —
a seed scoring below the *certified* optimum, which is a defect and never a win.

The claim being checked, stated once
------------------------------------
Not "the agent beats Almgren–Chriss". The sentence is: **the agent finds the
optimum of a world whose closed form is derived at a tangent, and the tangent
costs 1.54 %**. §1.1 names an agent that beats AC *inside AC's assumptions* a red
flag; this is outside them, and the difference is the milestone.

Both numbers travel together everywhere
---------------------------------------
The capture fraction leads and the absolute excess in bps sits beside it, in the
results file and in every assertion below. ``ARCHITECTURE.md`` §9 records why a
fraction alone is dangerous — at low lambda it made a healthy agent look like a
degrading one — and the same trap runs the other way here: a capture fraction
near 1 on an advantage of 0.037 bps is a *small absolute claim* and has to read
as one.

There is no Monte-Carlo interval anywhere in this module. Grading is analytic, so
"dispersion" means the spread across training seeds and nothing else; sampling
intervals arrive with M4b's liquidity noise.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from temper.agents import PPOPolicy
from temper.eval.experiment import AVAILABLE_ADVANTAGE, load_experiment
from temper.eval.grading import grade_policy
from temper.eval.metrics import LINEAR, POWER_LAW, WorldMismatch, check_grades_world
from temper.eval.provenance import config_digest
from temper.eval.reference import trajectory_deviation
from temper.eval.sweep import train_seed

from .conftest import REPO_ROOT

M4A = load_experiment(REPO_ROOT / "configs" / "m4a_power_law.yaml")
M3_POINT = load_experiment(
    REPO_ROOT / "configs" / "m3_frontier" / "lambda_1e-3.5.yaml"
)


def _document(experiment) -> dict:
    """Load a committed result, or skip the module if it has not been generated."""
    path = experiment.results_metrics
    if not path.exists():
        pytest.skip(
            f"{path.relative_to(REPO_ROOT)} has not been generated in this tree. "
            "Run `make m4a` (about three hours, unattended) from a committed tree.",
            allow_module_level=True,
        )
    return json.loads(path.read_text(encoding="utf-8"))


DOCUMENT = _document(M4A)
M3_DOCUMENT = json.loads(M3_POINT.results_metrics.read_text(encoding="utf-8"))
REFERENCE = M4A.reference()


# ---------------------------------------------------------------------------
# Invariant 1 — the result and the config that produced it
# ---------------------------------------------------------------------------


def test_the_result_was_produced_by_the_committed_config():
    provenance = DOCUMENT["provenance"]
    assert provenance["config"] == M4A.path.name
    assert provenance["config_sha256"] == config_digest(M4A.path)
    assert provenance["git_dirty"] is False, (
        "the source tree was dirty when this run started, so its recorded "
        "revision does not contain the code that produced it"
    )
    assert DOCUMENT["milestone"] == "M4a"


def test_the_result_records_the_world_it_was_produced_in():
    """The world is a committed field, not an inference from the numbers."""
    assert DOCUMENT["config"]["cost_encoding"] == POWER_LAW
    assert DOCUMENT["reference"]["encoding"] == POWER_LAW
    for record in DOCUMENT["seeds"]:
        assert record["grade"]["encoding"] == POWER_LAW
    for grade in DOCUMENT["baselines"].values():
        assert grade["encoding"] == POWER_LAW


def test_the_claim_travels_with_the_result():
    """The sentence the result is allowed to make, copied verbatim from the config.

    Three things it must say, because each is a way the claim could be overstated
    on the way to a README: that the reference is the *certified* optimum of the
    power-law world; that the grading is analytic; and that none of this is a
    statement about real fills.
    """
    claim = DOCUMENT["claim"]
    assert claim == M4A.estimator.claim
    assert "certified" in claim
    assert "analytically" in claim
    assert "NOT a claim about real fills" in claim


# ---------------------------------------------------------------------------
# Invariant 3 — the bars were pre-stated, and in the right units
# ---------------------------------------------------------------------------


def test_the_tolerance_is_a_fraction_of_the_available_advantage():
    """The denominator is the milestone. Checked, not trusted.

    At this lambda 5 % of the TWAP gap is 1.8–2.0x the *entire* available
    advantage, so a result graded to M2's and M3's denominator would say nothing
    about the mis-specification. This asserts the committed config did not do
    that, and asserts the arithmetic that makes it matter.
    """
    tolerances = DOCUMENT["config"]["tolerances"]
    assert tolerances["denominator"] == AVAILABLE_ADVANTAGE
    assert tolerances["graded_attribute"] == "advantage_fraction"
    assert DOCUMENT["verdict"]["tolerance_denominator"] == AVAILABLE_ADVANTAGE

    advantage = REFERENCE.available_advantage
    assert DOCUMENT["verdict"]["denominator_bps"] == pytest.approx(
        advantage, rel=1e-12
    )
    twap_gap_bps = REFERENCE.twap.objective - REFERENCE.optimal.objective
    assert 0.05 * twap_gap_bps > 1.8 * advantage, (
        "5 % of the TWAP gap is no longer larger than the whole available "
        "advantage; the reason this milestone changed denominator has moved"
    )


def test_the_lambda_is_the_one_both_worlds_rules_select():
    """Gate 1 of task 0, re-derived from the oracle on every commit."""
    selections = M4A.verify_lambda_rule_agrees_across_worlds()
    assert len(set(selections.values())) == 1
    assert M4A.lambda_risk == selections[POWER_LAW]
    assert M4A.lambda_risk == M3_POINT.lambda_risk, (
        "M4a is no longer at the lambda M2 and M3 committed, so its capture "
        "fraction cannot be put beside their gap fractions"
    )


def test_the_available_advantage_clears_the_one_percent_gate():
    """Gate 2 of task 0. Below 1 % the milestone leads with M4b instead."""
    assert REFERENCE.advantage_fraction >= 0.01
    assert REFERENCE.available_advantage == pytest.approx(0.03674, abs=1e-5)


def test_the_band_discriminates_in_trajectory_space():
    """Gate 3 of task 0: the band is well inside the AC separation."""
    band = M4A.band()
    assert band.local is True, "the power-law band must be labelled local"
    separation = trajectory_deviation(
        REFERENCE.tangent.trajectory, REFERENCE.optimal.trajectory
    )
    assert separation / band.bound_shares >= 2.0
    assert band.bound_shares == pytest.approx(4739.0, abs=2.0)
    assert separation == pytest.approx(16878.0, abs=2.0)


# ---------------------------------------------------------------------------
# The verdict, and the number the milestone leads with
# ---------------------------------------------------------------------------


def test_the_median_capture_fraction_meets_the_pre_stated_bar():
    """`c >= 0.95` — and the absolute excess in bps beside it, every time."""
    capture = DOCUMENT["summary"]["capture_fraction"]
    advantage = DOCUMENT["summary"]["advantage_fraction"]
    bar = M4A.tolerances.epsilon_fraction
    assert capture["median"] >= 1.0 - bar, (
        f"median capture {capture['median']:.4f} is below the pre-stated "
        f"{1 - bar:.2f}; median excess "
        f"{DOCUMENT['verdict']['median_excess_bps']:+.5f} bps against a bar of "
        f"{bar * REFERENCE.available_advantage:.5f} bps"
    )
    assert advantage["median"] == pytest.approx(1.0 - capture["median"], rel=1e-12)
    assert DOCUMENT["verdict"]["epsilon_met"] is True
    # The absolute claim, stated: it is small, and it should read as small.
    assert abs(DOCUMENT["verdict"]["median_excess_bps"]) < 0.01


def test_no_seed_is_worse_than_the_per_seed_floor():
    capture = DOCUMENT["summary"]["capture_fraction"]
    floor = 1.0 - M4A.tolerances.per_seed_fraction
    assert capture["worst"] >= floor, (
        f"the worst seed captured {capture['worst']:.4f}, below the per-seed "
        f"floor of {floor:.2f}"
    )
    assert DOCUMENT["verdict"]["per_seed_met"] is True
    assert DOCUMENT["verdict"]["passed"] is True


def test_no_seed_scored_below_the_certified_optimum():
    """The red flag. A strictly lower objective is a defect, never a win.

    The optimum is certified in task 1 — Cholesky PD, a relative KKT residual of
    1.2e-15 against a 1e-12 bar, 3 600 perturbations uphill, and an independent
    solver agreeing to 3.1e-15 of X. So there is nothing below it to find, and a
    seed that appeared to find something would be reporting a defect in the
    metric, the env or the grading path (``ARCHITECTURE.md`` §1.1).
    """
    assert DOCUMENT["verdict"]["red_flags"] == []
    for record in DOCUMENT["seeds"]:
        grade = record["grade"]
        assert grade["red_flag"] is False
        assert grade["excess_bps"] >= -M4A.tolerances.red_flag_rtol * abs(
            REFERENCE.optimal.objective
        )


def test_the_agent_actually_beat_the_closed_form():
    """The milestone's substance: capture is positive, on the median and per seed.

    A capture fraction of zero is the Almgren–Chriss schedule exactly. Anything
    at or below it would mean the agent learned nothing the closed form did not
    already know, and the milestone would have measured a mis-specification
    without recovering any of it.
    """
    for record in DOCUMENT["seeds"]:
        capture = record["grade"]["capture_fraction"]
        assert capture > 0.0, (
            f"{record['grade']['name']} captured {capture:.4f} — at or below the "
            "tangent-derived schedule the vendored library would have run"
        )


# ---------------------------------------------------------------------------
# Invariant 4 — ten seeds, dispersion, the baselines through the same grader
# ---------------------------------------------------------------------------


def test_ten_seeds_with_dispersion_reported():
    assert len(DOCUMENT["seeds"]) == 10 == M4A.seeds.n_seeds
    for name in ("gap_fraction", "advantage_fraction", "objective", "deviation"):
        summary = DOCUMENT["summary"][name]
        assert len(summary["values"]) == 10
        assert summary["q1"] <= summary["median"] <= summary["q3"]
        assert summary["worst"] >= summary["median"]
    capture = DOCUMENT["summary"]["capture_fraction"]
    # Capture is a *benefit*, so its "worst" is the smallest, which is the
    # complement of advantage_fraction's largest.
    assert capture["worst"] <= capture["median"] <= capture["q3"]


def test_the_dispersion_is_across_seeds_and_says_so():
    """There is no Monte-Carlo interval in M4a, and the result must not imply one.

    Grading is analytic — one deterministic rollout, then a closed form — so
    every spread reported here is across training seeds. A results file that
    called it a confidence interval would be claiming a sampling statement it
    never computed.
    """
    assert "confidence" not in json.dumps(DOCUMENT["summary"]).lower()
    for name in ("capture_fraction", "advantage_fraction"):
        assert set(DOCUMENT["summary"][name]) >= {"median", "q1", "q3", "iqr", "worst"}


def test_the_four_baselines_are_on_the_table_with_the_oracles_numbers():
    """TWAP, AC, the tangent sinh and the certified optimum, through the grader."""
    baselines = DOCUMENT["baselines"]
    assert set(baselines) == {"twap", "ac", "tangent", "optimal"}
    assert baselines["optimal"]["excess_bps"] == pytest.approx(0.0, abs=1e-12)
    assert baselines["optimal"]["capture_fraction"] == pytest.approx(1.0, abs=1e-9)
    # The tangent captures exactly none of the advantage: it *is* the closed
    # form's answer, and the advantage is defined as its excess.
    assert baselines["tangent"]["capture_fraction"] == pytest.approx(0.0, abs=1e-9)
    assert baselines["twap"]["gap_fraction"] == pytest.approx(1.0, rel=1e-12)
    for name, grade in baselines.items():
        assert grade["objective_bps"] == pytest.approx(
            REFERENCE.schedules[name].objective, rel=1e-12
        )
        assert not grade["red_flag"]


def test_the_agent_sits_between_the_tangent_and_the_optimum():
    """Where the milestone's whole claim lives, as an ordering on one number."""
    baselines = DOCUMENT["baselines"]
    median_objective = DOCUMENT["summary"]["objective"]["median"]
    assert (
        baselines["optimal"]["objective_bps"]
        <= median_objective
        < baselines["tangent"]["objective_bps"]
        < baselines["ac"]["objective_bps"]
        < baselines["twap"]["objective_bps"]
    )


# ---------------------------------------------------------------------------
# The schedules, the band, and the wall clock
# ---------------------------------------------------------------------------


def test_every_seed_realised_a_monotone_schedule_that_fully_liquidated():
    for record in DOCUMENT["seeds"]:
        trajectory = np.asarray(record["grade"]["trajectory"], dtype=float)
        assert trajectory[0] == pytest.approx(M4A.case.order_size, rel=1e-12)
        assert trajectory[-1] == 0.0
        assert np.all(np.diff(trajectory) <= 1e-9)


def test_the_reported_objective_is_what_the_reported_trajectory_costs():
    """Regrade every committed trajectory through the registry, from scratch."""
    from temper.eval.grading import grade_trajectory

    for record in DOCUMENT["seeds"]:
        grade = record["grade"]
        regraded = grade_trajectory(
            np.asarray(grade["trajectory"], dtype=float),
            M4A.case.market,
            M4A.case.order_size,
            REFERENCE,
            name=grade["name"],
        )
        assert regraded.objective == pytest.approx(grade["objective_bps"], rel=1e-12)
        assert regraded.capture_fraction == pytest.approx(
            grade["capture_fraction"], rel=1e-9
        )


def test_the_band_is_reported_local_and_the_median_sits_inside_it():
    band = DOCUMENT["bands"]["epsilon"]
    assert band["local"] is True
    assert band["encoding"] == POWER_LAW
    assert band["bound_shares"] == pytest.approx(M4A.band().bound_shares, rel=1e-12)
    assert DOCUMENT["summary"]["deviation"]["median"] <= band["bound_shares"], (
        f"median deviation {DOCUMENT['summary']['deviation']['median']:,.0f} "
        f"shares is outside the derived band of {band['bound_shares']:,.0f}"
    )


def test_the_agent_is_closer_to_the_optimum_than_the_closed_form_is():
    """The claim in trajectory space, which is where it is largest.

    1.54 % of expected cost is a small number; 16 878 shares is 16.9 % of the
    parent order. The second is the one a chart shows, and it is the same fact.
    """
    separation = trajectory_deviation(
        REFERENCE.tangent.trajectory, REFERENCE.optimal.trajectory
    )
    assert DOCUMENT["summary"]["deviation"]["median"] < 0.5 * separation


def test_the_run_recorded_its_wall_clock_and_stayed_inside_the_stated_bounds():
    verdict = DOCUMENT["verdict"]
    assert verdict["sweep_seconds"] <= M4A.runtime.sweep_seconds
    assert not verdict["timed_out"]
    for record in DOCUMENT["seeds"]:
        seconds = record["training"]["seconds"]
        assert 0.0 < seconds <= M4A.runtime.seconds_per_seed
        assert record["training"]["updates"] == M4A.ppo.num_updates


def test_the_figure_was_written_beside_the_metrics():
    for suffix in M4A.figure_formats:
        figure = Path(f"{M4A.results_figure}.{suffix}")
        assert figure.exists(), f"{figure.name} is missing; re-run `make m4a`"
        assert figure.stat().st_size > 10_000


def test_the_degradation_figure_is_committed_beside_the_result():
    """Task 6's figure is a *view* of this result: no training on its path.

    Byte-identity on redraw is checked by regenerating it here from the committed
    JSON and the oracle, exactly as ``make m4a-figure`` does, and comparing bytes.
    That is what makes it reproducible from a clean clone rather than an artefact
    somebody happened to have.
    """
    import importlib.util
    import tempfile

    stem = REPO_ROOT / "results" / "m4a_degradation"
    for suffix in M4A.figure_formats:
        committed = Path(f"{stem}.{suffix}")
        assert committed.exists(), (
            f"{committed.name} is missing; run `make m4a-figure`"
        )
        assert committed.stat().st_size > 10_000

    spec = importlib.util.spec_from_file_location(
        "m4a_degradation", REPO_ROOT / "tools" / "m4a_degradation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from temper.eval.figures import degradation_figure
    from temper.eval.provenance import Provenance

    curves = module.build_curves(M4A, DOCUMENT)
    with tempfile.TemporaryDirectory() as directory:
        written = degradation_figure(
            Path(directory) / "m4a_degradation",
            curves=curves,
            provenance=Provenance(**DOCUMENT["provenance"]),
            caption=module.caption(M4A, DOCUMENT, curves),
            formats=M4A.figure_formats,
        )
        for path in written:
            committed = Path(f"{stem}.{path.suffix.lstrip('.')}")
            assert path.read_bytes() == committed.read_bytes(), (
                f"{committed.name} does not redraw byte-identically from the "
                "committed result"
            )


# ---------------------------------------------------------------------------
# Everything except the world is M3's
# ---------------------------------------------------------------------------


def test_only_the_world_and_the_graded_encoding_differ_from_m3s_point():
    """"Everything else identical to M3's committed configuration" — asserted.

    Compared on the two results files' config blocks, so the statement is about
    what actually ran rather than about what the YAML says today. Out of scope
    for this milestone was hyperparameter search; this is what makes that a fact.
    """
    mine = dict(DOCUMENT["config"])
    theirs = dict(M3_DOCUMENT["config"])
    allowed = {
        "path",
        "milestone",
        "cost_encoding",
        "frontier_grid",
        "tolerances",
        "estimator",
        "runtime",
        "gate",
    }
    differing = sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k))
    assert set(differing) <= allowed, f"unexpected differences: {differing}"
    assert mine["ppo"] == theirs["ppo"]
    assert mine["seeding"] == theirs["seeding"]
    assert mine["reward_scale"] == theirs["reward_scale"]
    assert mine["case"] == theirs["case"]
    assert mine["lambda_risk"] == theirs["lambda_risk"]
    assert mine["estimator"]["regime"] == theirs["estimator"]["regime"] == "antithetic"


def test_the_reward_scale_carries_at_the_episode_level():
    """The sanity check the brief asked to be recorded rather than assumed.

    The committed scale was set against a Phase-1 objective of 2.3546 bps; the
    power-law objective at this lambda is 2.3832 bps. 1.2 % apart, so the scale
    that put a typical per-step reward near unit scale still does.
    """
    linear = M3_DOCUMENT["reference"]["schedules"]["optimal"]["objective_bps"]
    power = DOCUMENT["reference"]["schedules"]["optimal"]["objective_bps"]
    assert abs(power - linear) / linear < 0.02
    assert DOCUMENT["config"]["reward_scale"] == 0.02


def test_the_grader_could_not_have_used_the_other_worlds_metrics():
    """The registry rule, exercised against this milestone's own reference."""
    from temper.eval.metrics import GRADED

    check_grades_world(REFERENCE.encoding, GRADED[POWER_LAW])
    with pytest.raises(WorldMismatch):
        check_grades_world(REFERENCE.encoding, GRADED[LINEAR])


# ---------------------------------------------------------------------------
# Regeneration (marked): one seed, reproduced
# ---------------------------------------------------------------------------


@pytest.mark.training
def test_one_seed_retrains_to_the_same_verdict():
    """Invariant 1 at the granularity training supports — see M2's and M3's twins."""
    committed = DOCUMENT["seeds"][0]["grade"]
    _, policy = train_seed(M4A, 0)
    assert isinstance(policy, PPOPolicy)
    kwargs = dict(
        root_seed=M4A.seeds.root_seed,
        pool=M4A.seeds.eval_pool,
        streams=M4A.seeds.eval_streams,
        name="seed0",
    )
    regraded = grade_policy(
        policy, M4A.case.market, M4A.case.order_size, REFERENCE, **kwargs
    )
    assert regraded.capture_fraction >= 1.0 - M4A.tolerances.per_seed_fraction, (
        f"retrained seed 0 captured {regraded.capture_fraction:.5f}, committed "
        f"{committed['capture_fraction']:.5f}"
    )
    assert not regraded.red_flag
    again = grade_policy(
        policy, M4A.case.market, M4A.case.order_size, REFERENCE, **kwargs
    )
    assert regraded.objective == again.objective
