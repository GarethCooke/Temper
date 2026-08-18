"""M3 tasks 3–5 — the frontier sweep's configs, aggregate and figure.

Two halves. The first runs on every commit against the *committed configs*:
the manifest loads, the nine point configs are byte-identical to what the
generator writes from it, every point differs from the template only where the
brief says a point may, the grid contains M2's rule-selected lambda, and the
aggregate + figure path renders headless and byte-identically on fabricated
points (so a plotting break is a fast red test rather than a surprise after a
day of training). The second half reads the committed sweep results back and
checks each point and the aggregate — it skips until the sweep has run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from temper.eval.experiment import FRONTIER_GRIDS, load_experiment
from temper.eval.frontier import (
    CLAIM_MECHANISM_END,
    POINT_MAY_CHANGE,
    aggregate,
    load_manifest,
    point_experiments,
    point_name,
    stale_point_configs,
)
from temper.eval.grading import grade_trajectory, summarise
from temper.oracle import VENDOR_LAMBDA_GRID, optimal_trajectory

from .conftest import REPO_ROOT

MANIFEST = load_manifest(REPO_ROOT / "configs" / "m3_frontier.yaml")
TEMPLATE = load_experiment(MANIFEST.template)
POINTS = point_experiments(MANIFEST)


# ---------------------------------------------------------------------------
# The grid and the manifest
# ---------------------------------------------------------------------------


def test_the_frontier_grid_is_a_vendor_subgrid_containing_m2s_lambda():
    grid = FRONTIER_GRIDS["m3"]
    assert len(grid) == 9
    assert set(grid) <= set(VENDOR_LAMBDA_GRID)
    assert grid == tuple(sorted(grid))
    assert TEMPLATE.lambda_risk in grid  # M2's rule-selected 10^-3.5
    assert grid[0] == 1e-05 and grid[-1] == 0.1
    # Half-decade spacing, exactly the vendor grid's floats.
    assert all(a is b or a == b for a, b in zip(grid, VENDOR_LAMBDA_GRID[8:17]))
    assert MANIFEST.grid == "m3" and MANIFEST.lambdas == grid


def test_point_names_are_readable_and_unique():
    names = [point_name(lam) for lam in MANIFEST.lambdas]
    assert names[0] == "lambda_1e-5.0" and names[-1] == "lambda_1e-1.0"
    assert names[3] == "lambda_1e-3.5"
    assert len(set(names)) == len(names)


def test_the_committed_point_configs_are_what_the_generator_writes():
    """A sweep point cannot drift from the template: bytes, not intent."""
    stale = stale_point_configs(MANIFEST, REPO_ROOT)
    assert not stale, (
        f"stale point configs: {[p.name for p in stale]}; regenerate with "
        "`python tools/m3_frontier.py configs` and commit"
    )
    assert len(POINTS) == 9


@pytest.mark.parametrize("experiment", POINTS, ids=lambda e: e.path.stem)
def test_each_point_differs_from_the_template_only_where_a_point_may(experiment):
    mine, theirs = experiment.as_dict(), TEMPLATE.as_dict()
    differing = {k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k)}
    # `path` differs trivially, `frontier_grid` and `lambda_risk` are the point.
    differing -= {"path", "frontier_grid", "lambda_risk"}
    assert differing <= POINT_MAY_CHANGE, f"{experiment.path.name} differs in {differing}"
    assert experiment.milestone == "M3"
    assert experiment.frontier_grid == "m3"
    assert experiment.gate is None
    assert experiment.case == TEMPLATE.case
    assert experiment.tolerances == TEMPLATE.tolerances
    assert experiment.seeds == TEMPLATE.seeds
    assert experiment.reward_scale == TEMPLATE.reward_scale
    assert experiment.estimator.regime == TEMPLATE.estimator.regime == "antithetic"
    # The mechanism paragraph of the claim is the template's, word for word.
    cut = TEMPLATE.estimator.claim.index(CLAIM_MECHANISM_END)
    assert experiment.estimator.claim[:cut] == TEMPLATE.estimator.claim[:cut]
    assert "one point of the M3 frontier sweep" in experiment.estimator.claim
    # PPO: identical apart from the update budget the amendment fixed.
    mine_ppo, theirs_ppo = dict(mine["ppo"]), dict(theirs["ppo"])
    assert mine_ppo.pop("total_timesteps") == MANIFEST.overrides["ppo"]["total_timesteps"]
    theirs_ppo.pop("total_timesteps")
    assert mine_ppo == theirs_ppo
    assert experiment.trace_points == 128
    # It verifies as a frontier point, not as a rule selection.
    row = experiment.verify_lambda_rule()
    assert row.lambda_risk == experiment.lambda_risk
    assert experiment.results_metrics.parent == MANIFEST.results_points_dir


def test_a_point_off_the_grid_or_a_grid_without_m2s_lambda_is_refused():
    from dataclasses import replace

    with pytest.raises(ValueError):
        replace(POINTS[0], lambda_risk=2e-4).verify_lambda_rule()
    with pytest.raises(ValueError):
        replace(POINTS[0], frontier_grid="nope").verify_lambda_rule()
    # A frontier grid that lacks the rule-selected point would be a set chosen
    # after the fact; simulate one by naming a grid whose members exclude it.
    FRONTIER_GRIDS["_probe"] = tuple(VENDOR_LAMBDA_GRID[8:11])
    try:
        with pytest.raises(ValueError, match="rule-selected"):
            replace(POINTS[0], frontier_grid="_probe").verify_lambda_rule()
    finally:
        del FRONTIER_GRIDS["_probe"]


def test_the_manifest_states_the_budget_the_amendment_fixed():
    assert MANIFEST.overrides["ppo"]["total_timesteps"] == 5_000_000
    assert POINTS[0].ppo.num_updates == 751
    assert MANIFEST.overrides["results"]["trace_points"] == 128
    assert MANIFEST.full_budget_point == TEMPLATE.results_metrics
    assert MANIFEST.milestone == "M3"


# ---------------------------------------------------------------------------
# The aggregate and the figure, exercised on fabricated points
# ---------------------------------------------------------------------------


def _fake_point_document(experiment, rng: np.random.Generator) -> dict:
    """A results file's worth of structure, from the oracle and a perturbation.

    Ten "seeds" that are the optimum with a small monotone perturbation, graded
    exactly as a real seed would be. Enough for the aggregate's arithmetic and
    the figure's rendering to be exercised in seconds.
    """
    market, order_size = experiment.case.market, experiment.case.order_size
    reference = experiment.reference()
    optimum = optimal_trajectory(market, order_size, experiment.lambda_risk)
    seeds = []
    for ordinal in range(10):
        trades = -np.diff(optimum)
        jitter = np.exp(rng.normal(0.0, 0.03, size=trades.size))
        trades = trades * jitter
        trades = trades / trades.sum() * order_size
        trajectory = np.concatenate([[order_size], order_size - np.cumsum(trades)])
        trajectory[-1] = 0.0
        grade = grade_trajectory(trajectory, market, order_size, reference, name=f"seed{ordinal}")
        seeds.append(
            {
                "ordinal": ordinal,
                "env_stream_base": ordinal * 4096,
                "training": {"seconds": 600.0 + ordinal, "updates": experiment.ppo.num_updates, "timed_out": False},
                "grade": grade.as_dict(),
            }
        )
    # Graded through the pure trajectory grader, not through the env: this is a
    # fabrication for the aggregate's arithmetic and must not spend `eval` streams.
    baselines = {
        name: grade_trajectory(
            reference.schedules[name].trajectory, market, order_size, reference, name=name
        ).as_dict()
        for name in ("twap", "ac", "optimal")
    }
    gaps = [s["grade"]["gap_fraction"] for s in seeds]
    summary = {"gap_fraction": summarise("gap_fraction", gaps).as_dict()}
    tolerances = experiment.tolerances
    return {
        "milestone": "M3",
        "provenance": {"config": experiment.path.name, "config_sha256": "0" * 64,
                       "git_rev": "f" * 40, "git_dirty": False, "python": "3.12"},
        "config": experiment.as_dict(),
        "reference": reference.as_dict(),
        "bands": {"epsilon": experiment.band().as_dict()},
        "baselines": baselines,
        "seeds": seeds,
        "summary": summary,
        "verdict": {
            "epsilon_met": summary["gap_fraction"]["median"] <= tolerances.epsilon_gap_fraction,
            "per_seed_met": summary["gap_fraction"]["worst"] <= tolerances.per_seed_gap_fraction,
            "red_flags": [], "timed_out": [], "sweep_seconds": 6100.0,
        },
    }


@pytest.fixture(scope="module")
def fabricated(tmp_path_factory):
    """A manifest whose points have fabricated results, in a temp results dir."""
    from dataclasses import replace

    root = tmp_path_factory.mktemp("frontier")
    points_dir = root / "results" / "m3_frontier"
    points_dir.mkdir(parents=True)
    manifest = replace(
        MANIFEST,
        results_points_dir=points_dir,
        results_metrics=root / "results" / "m3_frontier.json",
        results_figure=root / "results" / "m3_frontier",
        full_budget_point=None,
    )
    rng = np.random.default_rng(3)
    for experiment in POINTS:
        document = _fake_point_document(experiment, rng)
        # The aggregate refuses a results file whose digest is not the config's.
        from temper.eval.provenance import config_digest

        document["provenance"]["config_sha256"] = config_digest(experiment.path)
        (points_dir / f"{point_name(experiment.lambda_risk)}.json").write_text(
            json.dumps(document), encoding="utf-8"
        )
    return manifest


def test_the_aggregate_reads_every_point_and_keeps_its_arithmetic_straight(fabricated):
    document = aggregate(fabricated, REPO_ROOT)
    assert document["milestone"] == "M3"
    assert [p["lambda"] for p in document["points"]] == list(MANIFEST.lambdas)
    assert document["verdict"]["complete"] and document["verdict"]["points"] == 9
    floor = document["variance_floor_bps2"]
    for point in document["points"]:
        assert point["n_seeds"] == 10
        assert point["variance_floor_bps2"] == floor
        for seed in point["seeds"]:
            assert seed["excess_variance_bps2"] == pytest.approx(seed["variance_bps2"] - floor)
            assert seed["objective_bps"] == pytest.approx(
                seed["expected_bps"] + point["lambda"] * seed["variance_bps2"], rel=1e-9
            )
        assert point["baselines"]["twap"]["gap_fraction"] == pytest.approx(1.0)
        assert point["baselines"]["optimal"]["gap_fraction"] == pytest.approx(0.0, abs=1e-12)
        s = point["summary"]["gap_fraction"]
        assert s["q1"] <= s["median"] <= s["q3"] <= s["worst"]
    curves = document["curves"]
    assert set(curves["lambda"]) >= set(MANIFEST.lambdas)
    # The dense optimal curve is monotone: more risk aversion, more expected cost, less variance.
    e = [c["expected_bps"] for c in curves["optimal"]]
    v = [c["excess_variance_bps2"] for c in curves["optimal"]]
    assert all(a <= b + 1e-12 for a, b in zip(e, e[1:]))
    assert all(a >= b - 1e-9 for a, b in zip(v, v[1:]))
    assert all(x > 0.0 for x in v)
    # TWAP is one point.
    twap = {(round(p["baselines"]["twap"]["expected_bps"], 9), round(p["baselines"]["twap"]["excess_variance_bps2"], 6)) for p in document["points"]}
    assert len(twap) == 1


def test_the_aggregate_refuses_a_missing_point_unless_told_partial(fabricated):
    from dataclasses import replace

    (fabricated.results_points_dir / f"{point_name(MANIFEST.lambdas[4])}.json").rename(
        fabricated.results_points_dir / "hidden.json"
    )
    try:
        with pytest.raises(FileNotFoundError):
            aggregate(fabricated, REPO_ROOT)
        partial = aggregate(fabricated, REPO_ROOT, require_complete=False)
        assert not partial["verdict"]["complete"] and partial["verdict"]["points"] == 8
        assert partial["missing_points"] == [f"{point_name(MANIFEST.lambdas[4])}.json"]
    finally:
        (fabricated.results_points_dir / "hidden.json").rename(
            fabricated.results_points_dir / f"{point_name(MANIFEST.lambdas[4])}.json"
        )
    _ = replace  # keep the import honest


def test_the_frontier_figure_renders_headless_and_byte_identically(fabricated, tmp_path):
    from temper.eval.figures import frontier_figure
    from temper.eval.provenance import Provenance

    document = aggregate(fabricated, REPO_ROOT)
    provenance = Provenance(**document["provenance"])
    first = frontier_figure(tmp_path / "frontier", aggregate=document, provenance=provenance, caption="render check\nline two\nline three\nline four")
    assert len(first) == 1 and first[0].stat().st_size > 10_000
    again = frontier_figure(tmp_path / "frontier", aggregate=document, provenance=provenance, caption="render check\nline two\nline three\nline four")
    assert again[0].read_bytes() == first[0].read_bytes()
    with pytest.raises(ValueError):
        frontier_figure(tmp_path / "empty", aggregate={**document, "points": []}, provenance=provenance, caption="none")


# ---------------------------------------------------------------------------
# The committed sweep (skips until it has run)
# ---------------------------------------------------------------------------


def _committed() -> dict | None:
    path = MANIFEST.results_metrics
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


COMMITTED = _committed()
needs_sweep = pytest.mark.skipif(
    COMMITTED is None,
    reason="results/m3_frontier.json has not been generated in this tree; run the sweep",
)


@needs_sweep
def test_the_committed_aggregate_is_what_the_points_aggregate_to():
    fresh = aggregate(MANIFEST, REPO_ROOT)
    for key in ("grid", "points", "variance_floor_bps2", "curves", "full_budget_point"):
        assert fresh[key] == COMMITTED[key], f"{key} differs from a re-aggregation"


@needs_sweep
def test_every_committed_point_was_produced_from_a_committed_tree():
    from temper.eval.provenance import config_digest

    for experiment, point in zip(POINTS, COMMITTED["points"]):
        provenance = point["provenance"]
        assert provenance["config_sha256"] == config_digest(experiment.path)
        assert provenance["git_dirty"] is False
        assert len(provenance["git_rev"]) == 40
    assert COMMITTED["provenance"]["git_dirty"] is False


@needs_sweep
def test_no_seed_at_any_lambda_scored_below_the_certified_optimum():
    """The red flag is a hard failure on every seed at every lambda."""
    assert COMMITTED["verdict"]["red_flag_free"]
    for point in COMMITTED["points"]:
        assert not point["red_flags"]
        for seed in point["seeds"]:
            assert not seed["red_flag"]


@needs_sweep
def test_every_lambda_reports_ten_seeds_three_baselines_and_dispersion():
    for point in COMMITTED["points"]:
        assert point["n_seeds"] == 10
        assert set(point["baselines"]) == {"twap", "ac", "optimal"}
        assert point["baselines"]["twap"]["gap_fraction"] == pytest.approx(1.0, rel=1e-12)
        assert point["baselines"]["optimal"]["gap_fraction"] == pytest.approx(0.0, abs=1e-12)
        s = point["summary"]["gap_fraction"]
        assert len(s["values"]) == 10 and s["q1"] <= s["median"] <= s["q3"]
        assert point["band"]["bound_shares"] > 0.0
        assert not point["timed_out"]


@needs_sweep
def test_the_sweep_includes_m2s_lambda_and_it_still_meets_epsilon():
    at_m2 = [p for p in COMMITTED["points"] if p["lambda"] == TEMPLATE.lambda_risk]
    assert len(at_m2) == 1
    assert at_m2[0]["epsilon_met"] and at_m2[0]["per_seed_met"]


@needs_sweep
def test_the_committed_frontier_figure_exists_and_redraws_byte_identically(tmp_path):
    from temper.eval.figures import frontier_figure
    from temper.eval.provenance import Provenance

    figure = Path(f"{MANIFEST.results_figure}.png")
    assert figure.exists() and figure.stat().st_size > 10_000
    import sys

    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from m3_frontier import _caption

    redrawn = frontier_figure(
        tmp_path / "frontier",
        aggregate=COMMITTED,
        provenance=Provenance(**COMMITTED["provenance"]),
        caption=_caption(COMMITTED),
    )
    assert redrawn[0].read_bytes() == figure.read_bytes()


# ---------------------------------------------------------------------------
# The provenance parser, against the shapes git actually emits
# ---------------------------------------------------------------------------


def test_a_regenerated_results_file_does_not_make_the_tree_read_dirty():
    """The M3 sweep's own output must not flip the flag its artefacts are gated on.

    This is a regression test with a scar. `_git` used to `.strip()` git's
    stdout, which removed the leading space of the *first* porcelain line only —
    and an unstaged modification is encoded as exactly that leading space. The
    first line's path was then read one character short (`sults/...`), failed
    the `results/` prefix test, and a tree whose only change was a regenerated
    figure reported dirty. It cost an aggregate and a figure, twice, and the
    first time it was misattributed to an editor.

    Order matters in the cases below: the defect only ever bit the first line.
    """
    from temper.eval.provenance import _source_is_dirty

    ignored = [
        " M results/m3_frontier.json",
        "?? results/m3_frontier/lambda_1e-3.5.png",
        " D results/m3_frontier/lambda_1e-3.png",
        "A  results/m3_frontier.png",
        "?? results/scratch/run_sweep.cmd",
        "?? results/m3_frontier/",
    ]
    for line in ignored:
        assert not _source_is_dirty(line), f"{line!r} should not make the tree dirty"
    # ...in any order, and all together.
    assert not _source_is_dirty("\n".join(ignored))
    for index in range(len(ignored)):
        rotated = ignored[index:] + ignored[:index]
        assert not _source_is_dirty("\n".join(rotated))

    # And the flag is not vacuous: source changes are dirty wherever they sit.
    for source in (" M temper/eval/figures.py", "?? tools/new_tool.py", "M  configs/x.yaml"):
        assert _source_is_dirty(source)
        assert _source_is_dirty("\n".join([*ignored, source]))
        assert _source_is_dirty("\n".join([source, *ignored]))
    # A source file *renamed into* results/ is still a source change.
    assert _source_is_dirty('R  temper/eval/x.py -> results/x.py')


def test_the_git_helper_returns_git_output_verbatim():
    """`_git` must not strip: the porcelain format encodes state in column 1."""
    from pathlib import Path

    from temper.eval.provenance import _git

    status = _git(Path(REPO_ROOT), "status", "--porcelain")
    assert status is not None
    for line in status.splitlines():
        if line.strip():
            assert line[:2] == line[:2].upper() or line[0] == " ", line
            assert len(line) > 3 and line[2] == " ", (
                f"porcelain line {line!r} lost its status columns; _git stripped it"
            )
