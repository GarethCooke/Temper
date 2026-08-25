"""M5 task 6 — the committed sweep result, read back and checked against its config.

The same shape as `tests/test_m2_rediscovery.py`: the four-and-a-half-hour sweep
lives behind the ``training`` marker, and what runs on every commit is this. The
JSON in ``results/`` is read back and checked against the config that claims to
have produced it, so a session that loosened a bar would go red here rather than
green everywhere.

M5 adds a second job for this module. Its artefact is the first in the repo that
has been *touched after its run*: four summary aggregates were taken in the wrong
direction, and rather than spend the run again they were recomputed from the
file's own per-seed values by ``tools/rebuild_m5_summary.py``. A file that has
been corrected has to carry that fact where a reader looks, and the correction
block has to be checkable rather than merely present — so the tests below rebuild
the summary independently, and require the recorded before-and-after to be the
numbers a reader would get by reproducing both.
"""

from __future__ import annotations

import json

import pytest

from temper.eval.grading import summarise
from temper.eval.provenance import config_digest
from temper.eval.sweep import (
    ALPHA_CAPTURE_BAR,
    ALPHA_DERIVED,
    ALPHA_DIRECTIONS,
    PREMIUM_RATIO_BAR,
    SHUFFLED_NET_CAPTURE_BAR,
    BAR_SUFFIX,
    invert_summary,
    seal_verdict,
)

from .conftest import REPO_ROOT

ARTEFACT = REPO_ROOT / "results" / "m5_alpha.json"
CONFIG = REPO_ROOT / "configs" / "m5_alpha.yaml"


@pytest.fixture(scope="module")
def result():
    if not ARTEFACT.exists():
        pytest.skip("results/m5_alpha.json has not been produced in this tree")
    return json.loads(ARTEFACT.read_text(encoding="utf-8"))


def test_the_result_names_the_config_it_was_measured_against(result):
    """Invariant 1 with teeth: the digest pins the thresholds, not just the path."""
    assert result["provenance"]["config"] == CONFIG.name
    assert result["provenance"]["config_sha256"] == config_digest(CONFIG)
    assert result["provenance"]["git_dirty"] is False


def test_the_three_numbers_clear_the_bars_the_brief_pre_stated(result):
    """Reported together, with the bps beside each fraction, exactly as they are read.

    Never the net capture alone. M5's own methodological finding is that a single
    capture fraction cannot tell a policy that trades the signal well from one that
    trades it badly and executes well, so the assertion is written the way the
    result must always be quoted.
    """
    headline = result["verdict"]["headline"]
    tolerances = result["config"]["tolerances"]
    summary = result["summary"]

    assert headline["alpha_capture_median"] >= ALPHA_CAPTURE_BAR
    assert headline["alpha_bps_median"] > 0.0
    assert headline["premium_ratio_median"] <= PREMIUM_RATIO_BAR
    assert headline["execution_premium_bps_median"] > 0.0
    assert summary["advantage_fraction"]["median"] <= tolerances["epsilon_fraction"]
    assert summary["advantage_fraction"]["worst"] <= tolerances["per_seed_fraction"]

    # The control, which is the reason the three numbers mean anything at all.
    shuffled = result["shuffled_control"]["net_capture"]
    assert shuffled["median"] <= SHUFFLED_NET_CAPTURE_BAR
    assert result["verdict"]["red_flags"] == []
    assert result["verdict"]["timed_out"] == []
    assert result["verdict"]["passed"] is True


def test_no_seed_hit_the_runaway_guard(result):
    """`max_seconds` is a runaway guard, not the budget, and now it is read.

    A seed that bound early trained fewer updates than its config named, so the
    sweep's median and its worst seed would be summaries over different amounts of
    training. Recorded per seed with the update it fired at, so this is a fact
    about the run rather than a promise about the configuration.
    """
    budgets = result["verdict"]["budgets"]
    assert len(budgets) == len(result["seeds"])
    assert len(budgets) == result["config"]["seeding"]["n_seeds"]
    for budget in budgets:
        assert budget["timed_out"] is False
        assert budget["bound_at_update"] is None
        assert budget["updates"] == budget["target_updates"]


def _rebuild(summary: dict) -> dict:
    """The summary the current code produces from the file's own per-seed values."""
    rebuilt = {
        name: summarise(name, summary[name]["values"], direction=direction).as_dict()
        for name, direction in ALPHA_DIRECTIONS.items()
    }
    for derived, source in ALPHA_DERIVED.items():
        rebuilt[derived] = invert_summary(derived, rebuilt[source])
    return rebuilt


def test_every_summary_agrees_with_the_files_own_per_seed_values(result):
    """The correction, checked by redoing it rather than by trusting it.

    Bitwise, because the recomputation runs the same operations on the same inputs.
    If this ever goes red the file and the code disagree about what the run found,
    and the file is the thing other documents quote.
    """
    summary = result["summary"]
    rebuilt = _rebuild(summary)
    assert set(summary) == set(rebuilt)
    for name in sorted(summary):
        assert list(summary[name]["values"]) == list(rebuilt[name]["values"])
        for key in ("median", "q1", "q3", "iqr", "worst"):
            assert summary[name][key] == rebuilt[name][key], f"{name}.{key}"


def test_no_capture_fraction_exceeds_what_was_available_to_capture(result):
    """The symptom that made the defect visible from outside, kept visible.

    ``alpha_capture.worst = 1.109916`` said one seed monetised more alpha than the
    converged optimum has to give. It was the sweep's BEST seed reported as its
    worst. A capture *median* above 1.0 would be a real finding worth chasing; a
    worst above 1.0 is arithmetic, and arithmetic is checkable here.
    """
    summary = result["summary"]
    assert summary["alpha_capture"]["worst"] <= summary["alpha_capture"]["median"]
    assert summary["net_capture"]["worst"] <= summary["net_capture"]["median"]
    assert summary["alpha_capture"]["worst"] <= 1.0
    assert summary["alpha_capture"]["worst"] == min(
        summary["alpha_capture"]["values"]
    )
    assert summary["net_capture"]["worst"] == min(summary["net_capture"]["values"])


def test_the_file_says_it_was_corrected_and_says_what_the_correction_was(result):
    """A file touched after its run must say so where a reader looks.

    Stronger than an erratum in a commit message, not weaker: a reader holding
    this JSON never sees the commit, and the number they would otherwise quote is
    the one that was wrong.
    """
    provenance = result["provenance"]
    assert provenance["unmodified_run_output"] is False
    assert provenance["see"] == "summary_correction"

    correction = result["summary_correction"]
    assert correction["produced_at_rev"] == provenance["git_rev"]
    assert correction["recomputed_at_rev"] != provenance["git_rev"]
    assert correction["per_seed_values_changed"] == 0
    assert correction["verdict_changed"] is False
    assert correction["tool"] == "tools/rebuild_m5_summary.py"
    assert set(correction["fields"]) == {
        "alpha_bps",
        "alpha_capture",
        "advantage_fraction",
        "net_capture",
    }


def test_the_recorded_before_values_are_the_ones_the_defect_produced(result):
    """Both halves of the correction reproduce, not just the half that survived.

    ``now`` is checked against the file's current summary by the test above. This
    checks ``was``: re-run the DEFECT - summarise every field as a cost, invert
    ``advantage_fraction`` out of ``net_capture`` the way the original code did -
    and require the recorded before-values to be what comes back. A correction
    block that cannot reproduce what it replaced is a claim, not a record.
    """
    summary = result["summary"]
    as_costs = {
        name: summarise(name, summary[name]["values"]).as_dict()
        for name in ALPHA_DIRECTIONS
        if name != "advantage_fraction"
    }
    # The original derived advantage_fraction from a directly summarised
    # net_capture, which is the inversion that put the best seed in `worst`.
    as_costs["net_capture"] = summarise(
        "net_capture", summary["net_capture"]["values"]
    ).as_dict()
    as_costs["advantage_fraction"] = invert_summary(
        "advantage_fraction", as_costs["net_capture"]
    )

    for name, changed in result["summary_correction"]["fields"].items():
        for key, record in changed.items():
            assert record["was"] == as_costs[name][key], f"{name}.{key} was"
            assert record["now"] == summary[name][key], f"{name}.{key} now"


def test_the_committed_verdict_still_passes_under_the_gate_that_reads_all_five_bars(
    result,
):
    """The gating change is not retroactive, and that is a measurement.

    When this sweep ran, `passed` was the AND over two bars; `alpha_capture_met`
    and `premium_ratio_met` were computed below the line and gated nothing. Both
    are true, so re-sealing the committed verdict under the rule that gates on
    every recorded bar returns the same answer — which is worth asserting rather
    than assuming, because the alternative is a milestone whose recorded PASS was
    only a pass under a rule the repo has since stopped using.
    """
    verdict = dict(result["verdict"])
    recorded = verdict["passed"]
    bars = sorted(k for k in verdict if k.endswith(BAR_SUFFIX))
    assert len(bars) == 5, bars

    seal_verdict(verdict)

    assert verdict["passed"] is recorded is True
    assert verdict["failed_bars"] == []
    assert verdict["bars_not_applicable"] == []
    assert verdict["gated_on"] == bars


# ---------------------------------------------------------------------------
# The figure. A view of the two artefacts above, and nothing computed in it.
# ---------------------------------------------------------------------------

FIGURE = REPO_ROOT / "results" / "m5_alpha.png"
REFERENCE_TABLE = REPO_ROOT / "results" / "m5_reference.json"


def _figure_tool():
    """`tools/m5_alpha_figure.py` as a module. It is a script; this is its importer."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "temper_m5_figure_tool", REPO_ROOT / "tools" / "m5_alpha_figure.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def table():
    if not REFERENCE_TABLE.exists():
        pytest.skip("results/m5_reference.json has not been produced in this tree")
    return json.loads(REFERENCE_TABLE.read_text(encoding="utf-8"))


def test_net_capture_is_linear_on_the_plane_the_figure_draws(result):
    """The contours are only honest if the surface under them is the real one.

    The left panel draws straight iso-net-capture lines, which is a CLAIM: that
    net capture is exactly linear in (alpha capture, premium ratio) with slopes
    A/D and -P/D. If it were only approximately linear, a reader placing a seed
    against the 0.90 line would be reading a number that is not the one in the
    artefact.

    So it is checked against all twenty-four graded policies in the file - ten
    seeds, ten shuffled controls, four baselines - rather than argued from the
    algebra. Everything the figure needs is in `build_plane`, so this exercises
    that function rather than a copy of it.
    """
    plane = _figure_tool().build_plane(result)
    A = plane["alpha_available_bps"]
    P = plane["premium_bps"]
    Dnom = plane["advantage_bps"]
    intercept = plane["net_intercept"]

    graded = (
        [r["grade"] for r in result["seeds"]]
        + [r["shuffled"] for r in result["seeds"]]
        + list(result["baselines"].values())
    )
    assert len(graded) == 24
    worst = 0.0
    for g in graded:
        predicted = intercept + (
            A * g["alpha_capture"] - P * g["premium_ratio"]
        ) / Dnom
        worst = max(worst, abs(predicted - g["net_capture"]))
    assert worst < 1e-12, (
        f"net capture is not linear on the plane the figure draws: worst "
        f"disagreement {worst:.3e} over {len(graded)} graded policies"
    )


def test_the_denominator_is_the_gap_and_not_the_gross_alpha(result):
    """D = A - P, asserted, because the caption says so in every drawing.

    The one way this figure could mislead is by quoting a capture fraction over a
    denominator other than the one the brief pre-stated. A capture over the GROSS
    alpha would read 1.83x larger and would be the flattering direction.
    """
    plane = _figure_tool().build_plane(result)
    assert plane["advantage_bps"] == pytest.approx(
        plane["alpha_available_bps"] - plane["premium_bps"], abs=1e-15
    )
    assert plane["advantage_bps"] < plane["alpha_available_bps"]


def test_the_offset_the_figure_annotates_is_the_one_in_the_artefact(result):
    """The number in the annotation comes off the file, not out of a docstring."""
    plane = _figure_tool().build_plane(result)
    anchor = plane["baselines"]["optimal"]
    # A schedule that cannot monetise alpha, reading a non-zero alpha capture.
    assert anchor["premium_ratio"] == pytest.approx(0.0, abs=1e-12)
    assert anchor["net_capture"] == pytest.approx(0.0, abs=1e-12)
    assert anchor["alpha_capture"] != 0.0
    assert abs(anchor["alpha_capture"]) < 0.01, (
        "the M4a schedule's spurious alpha has grown beyond the 1/sqrt(M) scale "
        "the caption attributes it to"
    )
    # And it is exactly why the DP is off its own 1.00 line.
    assert plane["net_intercept"] == pytest.approx(-anchor["alpha_capture"] * (
        plane["alpha_available_bps"] / plane["advantage_bps"]
    ), rel=1e-9)


def test_the_caption_names_the_four_things_it_may_never_omit(result, table):
    """A caption with numbers in it is a claim, so its claims are asserted."""
    tool = _figure_tool()
    plane = tool.build_plane(result)
    curve = tool.build_curve(table, result)
    text = tool.caption(result, plane, curve)
    # Wrapped for the canvas, so a phrase can straddle a line break. The claims
    # are read off the unwrapped sentence and the width off the wrapped one.
    flowed = " ".join(text.split())

    assert "DENOMINATOR" in flowed
    assert "INVENTED" in flowed
    assert "CONVERGED, NOT CERTIFIED" in flowed
    assert "THE OFFSET" in flowed
    # Never the net capture alone.
    assert "alpha capture" in flowed and "execution premium" in flowed
    assert "net capture" in flowed
    assert "shuffled control" in flowed
    for line in text.splitlines():
        assert len(line) <= tool.CAPTION_WIDTH, (
            f"a caption line is {len(line)} characters against a canvas measured "
            f"at {tool.CAPTION_WIDTH}; matplotlib will not tell you it overflowed"
        )


def test_the_figure_redraws_byte_identically_from_the_committed_artefacts(tmp_path):
    """A view of results, not a second route to them.

    Nothing in the figure path computes a cost, so a clean clone reproduces the
    committed PNG exactly without a training run. If this goes red, either the
    figure has started computing something or the artefacts under it have moved.
    """
    if not FIGURE.exists():
        pytest.skip("results/m5_alpha.png has not been drawn in this tree")
    tool = _figure_tool()
    status = tool.main(["--out", str(tmp_path / "m5_alpha")])
    assert status == 0
    redrawn = tmp_path / "m5_alpha.png"
    assert redrawn.read_bytes() == FIGURE.read_bytes(), (
        "results/m5_alpha.png does not redraw byte-identically from "
        "results/m5_alpha.json and results/m5_reference.json"
    )


def test_a_missing_input_skips_the_figure_rather_than_half_drawing_it(tmp_path):
    """The failure mode that must stay visible: no file is better than a wrong one."""
    tool = _figure_tool()
    status = tool.main(
        ["--sweep", str(tmp_path / "absent.json"), "--out", str(tmp_path / "m5")]
    )
    assert status == 1
    assert not (tmp_path / "m5.png").exists()
