"""M3 task 1 — the committed antithetic validation meets the gate it was pre-stated against.

The run itself takes a night and lives behind ``make validate``. What runs on
every commit is this: ``results/m3_antithetic_validation.json`` is read back and
checked against the config that claims to have produced it (invariant 1, by
digest), against M2's committed control-variate result (the gate is stated
*against that*, not against ε), and for the reward-variance evidence the brief
asks for — measured per update inside the run, not inferred from the outcome.

The one marked test regenerates a single seed and checks it reproduces its
committed verdict, exactly as ``tests/test_m2_rediscovery.py`` does for M2 and
for the same reason: reproducibility is a property of the pipeline, not of the
sample size.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from temper.agents import PPOPolicy
from temper.eval.experiment import load_experiment
from temper.eval.grading import grade_policy
from temper.eval.provenance import config_digest
from temper.eval.reference import trajectory_deviation
from temper.eval.sweep import train_seed
from temper.oracle import optimal_trajectory, schedule_moments

from .conftest import REPO_ROOT, m2_experiment

M2 = m2_experiment()
M3 = load_experiment(REPO_ROOT / "configs" / "m3_antithetic_validation.yaml")


def _document(experiment) -> dict:
    """Load a committed result, or skip the module if it has not been generated.

    A skip, not a failure: "the run has not happened in this tree yet" is the
    state right after the code lands and before the night it takes — the only
    order that produces an artefact whose recorded revision contains the code
    that made it. Once committed, nothing here skips again.
    """
    path = experiment.results_metrics
    if not path.exists():
        pytest.skip(
            f"{path.relative_to(REPO_ROOT)} has not been generated in this tree. "
            "Run `make validate` (a night, unattended) from a committed tree.",
            allow_module_level=True,
        )
    return json.loads(path.read_text(encoding="utf-8"))


DOCUMENT = _document(M3)
M2_DOCUMENT = json.loads(M2.results_metrics.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Invariant 1 — the result and the config that produced it
# ---------------------------------------------------------------------------


def test_the_result_was_produced_by_the_committed_config():
    provenance = DOCUMENT["provenance"]
    assert provenance["config"] == M3.path.name
    assert provenance["config_sha256"] == config_digest(M3.path), (
        f"{M3.results_metrics.name} was produced by a different version of "
        f"{M3.path.name}; re-run the validation or restore the config"
    )


def test_the_result_carries_a_git_revision_that_contains_its_code():
    provenance = DOCUMENT["provenance"]
    assert provenance["git_rev"] != "unknown"
    assert len(provenance["git_rev"]) == 40
    assert provenance["git_dirty"] is False, (
        "this result was produced from a modified working tree; its recorded "
        "revision does not contain the code that made it. Commit, then re-run."
    )


def test_the_result_names_its_milestone_regime_and_claim():
    assert DOCUMENT["milestone"] == "M3"
    assert DOCUMENT["config"]["estimator"]["regime"] == "antithetic"
    assert DOCUMENT["config"]["estimator"]["control_variate"] is False
    assert DOCUMENT["claim"] == M3.estimator.claim
    assert "antithetic" in DOCUMENT["claim"].lower()
    assert DOCUMENT["config"]["ppo"]["torch_threads"] == 8


def test_the_result_was_graded_at_the_rule_selected_lambda():
    selected = M3.verify_lambda_rule()
    assert DOCUMENT["config"]["lambda_risk"] == M3.lambda_risk == M2.lambda_risk
    assert DOCUMENT["reference"]["lambda"] == selected.lambda_risk


def test_the_run_differs_from_m2s_control_variate_run_only_where_the_brief_says():
    """"Everything else identical to M2's committed configuration" — asserted.

    Compared on the two results files' config blocks, so the statement is about
    what actually ran, not about what the YAML says today.
    """
    mine = dict(DOCUMENT["config"])
    theirs = dict(M2_DOCUMENT["config"])
    allowed = {"path", "milestone", "estimator", "runtime", "gate", "seeding"}
    differing = sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k))
    assert set(differing) <= allowed, f"unexpected differences: {differing}"
    mine_seeding, theirs_seeding = dict(mine["seeding"]), dict(theirs["seeding"])
    assert mine_seeding.pop("n_seeds") == 10
    assert theirs_seeding.pop("n_seeds") == 5
    assert mine_seeding == theirs_seeding, "seed addressing must be M2's"
    assert mine["ppo"] == theirs["ppo"]
    assert mine["reward_scale"] == theirs["reward_scale"]
    assert mine["tolerances"] == theirs["tolerances"]


def test_the_trace_budget_is_recorded_and_honoured():
    budget = DOCUMENT["trace_points"]
    assert budget == M3.trace_points
    for record in DOCUMENT["seeds"]:
        training = record["training"]
        for name in ("train_returns", "train_return_variance", "approx_kl", "entropy", "value_loss"):
            trace = training[name]
            assert trace, f"{name} is empty"
            if budget is None:
                assert len(trace) == training["updates"]
            else:
                assert len(trace) <= budget
        pair = record["pair"]
        assert pair["updates"] == training["updates"]
        for name, trace in pair["traces"].items():
            if budget is None:
                assert len(trace) == training["updates"], name
            else:
                assert len(trace) <= budget


# ---------------------------------------------------------------------------
# Invariant 4 — ten seeds, dispersion, the baselines through the same grader
# ---------------------------------------------------------------------------


def test_ten_seeds_with_dispersion_reported():
    assert len(DOCUMENT["seeds"]) == 10 == M3.seeds.n_seeds
    for name in ("gap_fraction", "relative_excess", "objective", "deviation"):
        summary = DOCUMENT["summary"][name]
        assert len(summary["values"]) == 10
        assert summary["q1"] <= summary["median"] <= summary["q3"]
        assert summary["worst"] >= summary["median"]


def test_the_three_baselines_are_on_the_table_with_the_oracles_numbers():
    reference = M3.reference()
    baselines = DOCUMENT["baselines"]
    assert set(baselines) == {"twap", "ac", "optimal"}
    assert baselines["optimal"]["gap_fraction"] == pytest.approx(0.0, abs=1e-12)
    assert baselines["twap"]["gap_fraction"] == pytest.approx(1.0, rel=1e-12)
    for name, grade in baselines.items():
        assert grade["objective_bps"] == pytest.approx(
            reference.schedules[name].objective, rel=1e-12
        )
        assert not grade["red_flag"]


def test_every_seed_realised_a_monotone_schedule_that_fully_liquidated():
    for record in DOCUMENT["seeds"]:
        trajectory = np.asarray(record["grade"]["trajectory"], dtype=float)
        assert trajectory[0] == pytest.approx(M3.case.order_size, rel=1e-12)
        assert trajectory[-1] == 0.0
        assert np.all(np.diff(trajectory) <= 1e-9)


def test_the_reported_objective_is_what_the_reported_trajectory_costs():
    market, order_size = M3.case.market, M3.case.order_size
    optimum = optimal_trajectory(market, order_size, M3.lambda_risk)
    j_optimal = DOCUMENT["reference"]["schedules"]["optimal"]["objective_bps"]
    for record in DOCUMENT["seeds"]:
        grade = record["grade"]
        trajectory = np.asarray(grade["trajectory"], dtype=float)
        objective = schedule_moments(trajectory, market, order_size=order_size).objective(
            M3.lambda_risk
        )
        assert grade["objective_bps"] == pytest.approx(objective, rel=1e-12)
        assert grade["excess_bps"] == pytest.approx(objective - j_optimal, rel=1e-9)
        assert grade["deviation_shares"] == pytest.approx(
            trajectory_deviation(trajectory, optimum), rel=1e-9
        )


# ---------------------------------------------------------------------------
# The red flag — a hard failure, always
# ---------------------------------------------------------------------------


def test_no_seed_scored_below_the_certified_optimum():
    assert not DOCUMENT["verdict"]["red_flags"]
    j_optimal = DOCUMENT["reference"]["schedules"]["optimal"]["objective_bps"]
    for record in DOCUMENT["seeds"]:
        assert not record["grade"]["red_flag"]
        assert record["grade"]["excess_bps"] >= -1e-9 * abs(j_optimal)


# ---------------------------------------------------------------------------
# The gate — against M2's control variate, not against epsilon
# ---------------------------------------------------------------------------


def test_the_gate_is_stated_against_the_committed_control_variate_result():
    gate = DOCUMENT["gate"]
    assert gate is not None
    assert gate["median_gap_fraction_max"] == M3.gate.median_gap_fraction == 0.002
    assert gate["reference"] == M2.results_metrics.name
    assert gate["reference_regime"] == "control_variate"
    assert gate["reference_median_gap_fraction"] == pytest.approx(
        M2_DOCUMENT["summary"]["gap_fraction"]["median"], rel=0.0, abs=0.0
    )
    assert gate["median_gap_fraction"] == DOCUMENT["summary"]["gap_fraction"]["median"]


def test_the_median_gap_is_within_the_gate():
    """Task 1's acceptance: within an order of magnitude of the variate's 0.0002.

    A median in the sampled regime's neighbourhood (~0.098) would mean the
    pairing is not cancelling and the milestone's argument is wrong.
    """
    median = DOCUMENT["summary"]["gap_fraction"]["median"]
    assert median <= M3.gate.median_gap_fraction, (
        f"median gap fraction {median:.5f} misses the gate "
        f"{M3.gate.median_gap_fraction}; the pairing is not reproducing the "
        "control variate's answer"
    )
    assert DOCUMENT["gate"]["met"] is True
    assert DOCUMENT["verdict"]["gate_met"] is True


def test_the_epsilon_verdict_is_reported_too_and_met():
    verdict = DOCUMENT["verdict"]
    assert verdict["epsilon_met"] and verdict["per_seed_met"] and verdict["passed"]
    assert DOCUMENT["summary"]["gap_fraction"]["worst"] <= M3.tolerances.per_seed_fraction


def test_the_variance_reduced_populations_do_not_overlap_the_sampled_one():
    """Every antithetic seed beats M2's best sampled-reward seed."""
    sampled = json.loads(
        (REPO_ROOT / "results" / "m2_rediscovery_sampled.json").read_text(encoding="utf-8")
    )
    best_sampled = min(sampled["summary"]["gap_fraction"]["values"])
    worst_here = DOCUMENT["summary"]["gap_fraction"]["worst"]
    assert worst_here < best_sampled, (
        f"the worst antithetic seed ({worst_here:.4f}) does not beat the sampled "
        f"regime's best ({best_sampled:.4f}); the estimator change is inside seed noise"
    )


# ---------------------------------------------------------------------------
# The reward-variance evidence — cancellation measured, not inferred
# ---------------------------------------------------------------------------


def test_the_reward_variance_was_measured_per_update_and_the_noise_is_gone():
    """The primary half's per-update return variance is the sampled regime's;
    the averaged return's variance is what the agent trained on. Same run, same
    actions. The ratio is the direct evidence that the pairing cancelled.
    """
    summary = DOCUMENT["summary"]["reward_variance"]
    # The sampled half carries the full Phase-1 noise: ~95 bps per-episode SD on
    # this case, i.e. thousands of bps^2 — the 1:70 ratio M2 measured.
    assert summary["sampled_median"] > 1_000.0
    assert summary["averaged_median"] < 0.01 * summary["sampled_median"]
    assert summary["variance_ratio_median"] < 0.01
    for record in DOCUMENT["seeds"]:
        pair = record["pair"]
        assert pair["episodes_per_update"] == M3.ppo.num_envs
        medians = pair["median"]
        assert medians["sampled_variance"] > 1_000.0
        assert medians["averaged_variance"] < 0.01 * medians["sampled_variance"]
        # The mirror half is the same noise, mirrored: same variance to rounding.
        assert medians["mirror_variance"] == pytest.approx(
            medians["sampled_variance"], rel=0.05
        )
        sampled = np.asarray(pair["traces"]["sampled_variance"], dtype=float)
        averaged = np.asarray(pair["traces"]["averaged_variance"], dtype=float)
        assert np.all(averaged <= sampled), "an update where averaging added variance"
        # And what the loop itself saw the agent train on, in scaled units, is
        # the averaged variance times the scale squared.
        trained = np.asarray(record["training"]["train_return_variance"], dtype=float)
        scale = DOCUMENT["config"]["reward_scale"]
        assert np.allclose(trained, averaged * scale**2, rtol=1e-6, atol=1e-12)


# ---------------------------------------------------------------------------
# The derived band, the runtime, and the figure
# ---------------------------------------------------------------------------


def test_the_band_is_reported_and_the_median_sits_inside_it():
    band = DOCUMENT["bands"]["epsilon"]
    assert band["bound_shares"] == pytest.approx(M3.band().bound_shares, rel=1e-12)
    assert DOCUMENT["summary"]["deviation"]["median"] <= band["bound_shares"]


def test_the_run_recorded_its_wall_clock_and_stayed_inside_the_stated_bounds():
    """Measured per-seed wall-clock is what task 3 sizes the sweep from."""
    verdict = DOCUMENT["verdict"]
    assert verdict["sweep_seconds"] <= M3.runtime.sweep_seconds
    assert not verdict["timed_out"]
    for record in DOCUMENT["seeds"]:
        seconds = record["training"]["seconds"]
        assert 0.0 < seconds <= M3.runtime.seconds_per_seed
        assert record["training"]["updates"] == M3.ppo.num_updates


def test_the_figure_was_written_beside_the_metrics():
    for suffix in M3.figure_formats:
        figure = Path(f"{M3.results_figure}.{suffix}")
        assert figure.exists(), f"{figure.name} is missing; re-run the validation"
        assert figure.stat().st_size > 10_000


# ---------------------------------------------------------------------------
# Regeneration (marked): one seed, reproduced
# ---------------------------------------------------------------------------


@pytest.mark.training
def test_one_seed_retrains_to_the_same_verdict():
    """Invariant 1 at the granularity training supports — see M2's twin test.

    Bitwise on one host at one thread count; the *verdict* across hosts. The
    seed must land inside the per-seed floor, raise no red flag, and grade
    identically twice.
    """
    committed = DOCUMENT["seeds"][0]["grade"]
    _, policy = train_seed(M3, 0)
    assert isinstance(policy, PPOPolicy)
    kwargs = dict(
        root_seed=M3.seeds.root_seed, pool=M3.seeds.eval_pool,
        streams=M3.seeds.eval_streams, name="seed0",
    )
    regraded = grade_policy(policy, M3.case.market, M3.case.order_size, M3.reference(), **kwargs)
    assert regraded.gap_fraction <= M3.tolerances.per_seed_fraction, (
        f"retrained seed 0 scored {regraded.gap_fraction:.5f}, committed "
        f"{committed['gap_fraction']:.5f}"
    )
    assert not regraded.red_flag
    again = grade_policy(policy, M3.case.market, M3.case.order_size, M3.reference(), **kwargs)
    assert regraded.objective == again.objective
