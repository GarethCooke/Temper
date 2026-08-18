"""M1 task 1 and M4a task 2 — the invariant-7 resolution, and what enforces it now.

The question M1's brief posed: do FrontierView's power-law moments and the
linearised moments the closed form solves reduce to each other on the Phase-1
parameter sets? If they did, ``cost_moments`` would be the canonical encoding and
there would be nothing to resolve.

They do not, and the first half of this module is where that is measured rather
than asserted. The gap is 12 % to 54 % of expected cost across the vendored
cases — a 0.6-power law against its own tangent, which is not a rounding
difference. So Phase 1 is the linearised world end-to-end (``ARCHITECTURE.md``
§9, *Phase 1 is the linearised world end-to-end*).

What changed at M4a is *why* that is safe. The old rule was a flat refusal:
``GRADEABLE_ENCODINGS = {LINEAR}``, which could not survive the power law
becoming a world rather than a note — and a bypassed check is not a check. The
rule now is **a metric grades the world that charges it**, the registries are
keyed by encoding, and the second half of this module checks it four ways:

* every world's graded set is complete and carries its own encoding;
* :func:`~temper.eval.metrics.metrics_for` refuses a world nothing scores, and
  :func:`~temper.eval.metrics.check_grades_world` refuses a hand-assembled
  mapping whose metrics charge the other one;
* every graded metric is evaluated against *both* encodings on every golden case
  and must equal its own — so the label is true rather than merely honest;
* no linear metric may name ``cost_moments`` and no power-law metric may reach
  for ``schedule_moments``, statically.

The last of those is the one the flat rule could never have caught, because
linear was permitted to everything equally: a *linear* metric grading a
power-law env. That is the live failure mode from M4a onward.
"""

from __future__ import annotations

import inspect
import re

import numpy as np
import pytest

from temper.env import ExecutionEnv, power_law_temporary
from temper.eval import metrics as metrics_module
from temper.eval.metrics import (
    CONTEXT,
    CORE_METRICS,
    GRADED,
    LINEAR,
    POWER_LAW,
    Metric,
    WorldMismatch,
    check_grades_world,
    metrics_for,
    register_context,
    register_graded,
)
from temper.oracle import (
    ENCODINGS,
    cost_moments,
    linear_cost_moments,
    linearised_eta,
    schedule_moments,
)

from .conftest import M1_CONFIG, REPO_ROOT

ENCODING = M1_CONFIG["objective_encoding"]
REDUCTION_RTOL = float(ENCODING["reduction_rtol"])
EXPECT_REDUCTION = bool(ENCODING["expect_reduction"])


def _schedules(case):
    """The two schedules the goldens pin, as arrays."""
    return {
        "ac": np.asarray(case.ac["trajectory"], dtype=float),
        "twap": np.asarray(case.twap["trajectory"], dtype=float),
    }


def _moments(encoding, trajectory, case):
    """What `encoding` charges this schedule, straight off the oracle."""
    if encoding == LINEAR:
        return linear_cost_moments(
            trajectory, case.market, linearised_eta(case.market, case.order_size)
        )
    return cost_moments(trajectory, case.market)


# ---------------------------------------------------------------------------
# The measurement the resolution rests on
# ---------------------------------------------------------------------------


def test_the_two_encodings_do_not_reduce_to_each_other(golden_case):
    """The branch point of M1 task 1, evaluated rather than assumed.

    Were the gap below `reduction_rtol` on every case, the honest move would be to
    pin an equality and call `linear_cost_moments` a redundant spelling of
    `cost_moments`. It is not: a power law and its tangent agree only where they
    touch. Flip `expect_reduction` in the config and this test will tell you
    which way the world actually is.

    It is also what makes M4a a milestone rather than a refactor. If the two
    encodings agreed, the Almgren–Chriss schedule would already solve the
    power-law world and there would be no advantage to earn.
    """
    market = golden_case.market
    eta_tilde = linearised_eta(market, golden_case.order_size)
    for name, trajectory in _schedules(golden_case).items():
        power = cost_moments(trajectory, market).expected
        linear = linear_cost_moments(trajectory, market, eta_tilde).expected
        relative = abs(power - linear) / abs(power)
        reduces = relative <= REDUCTION_RTOL
        assert reduces == EXPECT_REDUCTION, (
            f"{golden_case.case_id} ({name}): power law {power:.6f} bps vs linear "
            f"{linear:.6f} bps, relative gap {relative:.3e} — the config says "
            f"expect_reduction = {EXPECT_REDUCTION}"
        )


def test_only_the_temporary_term_differs_between_the_encodings(golden_case):
    """Permanent, spread and variance are shared; the fork is one term wide.

    Worth pinning because it bounds the blast radius of the quarantine: the
    variance a result reports is not a linearisation of anything, and
    ``shortfall_variance`` can be graded without qualification.

    It is also the reason M4a's differential needs a new analytic reference for
    only *one* of the two moments. ``shortfall_variance_bps2`` does not depend on
    the impact model at all, so the variance side of invariant 6's expectation
    test carries over from M1 unchanged — asserted here rather than assumed.
    """
    market = golden_case.market
    eta_tilde = linearised_eta(market, golden_case.order_size)
    for trajectory in _schedules(golden_case).values():
        power = cost_moments(trajectory, market)
        linear = linear_cost_moments(trajectory, market, eta_tilde)
        assert power.permanent == pytest.approx(linear.permanent, rel=1e-15)
        assert power.spread == pytest.approx(linear.spread, rel=1e-15)
        assert power.variance == pytest.approx(linear.variance, rel=1e-15)
        assert power.temporary != pytest.approx(linear.temporary, rel=REDUCTION_RTOL)


def test_schedule_moments_is_the_linear_encoding(golden_case):
    """The Phase-1 env, reward and oracle all speak this one function."""
    market = golden_case.market
    eta_tilde = linearised_eta(market, golden_case.order_size)
    for trajectory in _schedules(golden_case).values():
        expected = linear_cost_moments(trajectory, market, eta_tilde)
        assert schedule_moments(trajectory, market) == expected


# ---------------------------------------------------------------------------
# The registries are keyed by world, and both worlds are complete
# ---------------------------------------------------------------------------


def test_the_registries_are_populated_and_disjoint_per_world():
    """A rule over an empty registry would pass and mean nothing."""
    assert set(GRADED) == set(ENCODINGS)
    assert set(CONTEXT) == set(ENCODINGS)
    for encoding in ENCODINGS:
        assert GRADED[encoding], f"no graded metrics for the {encoding} world"
        assert not set(GRADED[encoding]) & set(CONTEXT[encoding])


def test_every_world_carries_the_metrics_the_grading_path_looks_up():
    """The grade is assembled from three names; a world missing one fails late.

    Late meaning "after a night of training", when the sweep tries to score the
    policy it just spent four hours producing. Checked at import time instead.
    """
    for encoding in ENCODINGS:
        assert set(GRADED[encoding]) == set(CORE_METRICS), (
            f"the {encoding} world's graded set is {sorted(GRADED[encoding])}, "
            f"not {sorted(CORE_METRICS)}"
        )


def test_every_graded_metric_declares_the_world_it_is_filed_under():
    """The key and the label agree, which is what makes the key trustworthy."""
    for encoding, world in GRADED.items():
        for name, metric in world.items():
            assert metric.encoding == encoding, (
                f"{name!r} is filed under {encoding!r} but declares "
                f"{metric.encoding!r}"
            )


def test_the_vendored_power_law_still_has_a_context_home():
    """M4a gave the power law a world; it did not empty the quarantine.

    The two ``power_law_*`` names are how M1, M2 and M3 report the vendored
    charge *beside* a Phase-1 claim. They stay in ``CONTEXT``, so a Phase-1
    result can still quote the number and the grading path still has no route to
    it — which is the half of the old rule that was always right.
    """
    assert set(CONTEXT[POWER_LAW]) == {
        "power_law_expected_cost",
        "power_law_objective",
    }
    assert all(metric.encoding == POWER_LAW for metric in CONTEXT[POWER_LAW].values())
    assert not CONTEXT[LINEAR], "nothing needs a linear context metric yet"


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------


def test_a_linear_metric_is_refused_against_a_power_law_env():
    """The failure the flat rule could never have caught.

    ``GRADEABLE_ENCODINGS = {LINEAR}`` permitted linear metrics to everything
    equally, so a Phase-1 objective scoring an M4a env would have passed every
    check the repo had. It is now the thing the grading path asserts against
    first.
    """
    with pytest.raises(WorldMismatch, match="grades the world that charges it"):
        check_grades_world(POWER_LAW, GRADED[LINEAR])
    with pytest.raises(WorldMismatch, match="grades the world that charges it"):
        check_grades_world(LINEAR, GRADED[POWER_LAW])

    # And the matched pairs are accepted, or the test above proves nothing.
    for encoding in ENCODINGS:
        check_grades_world(encoding, GRADED[encoding])


def test_the_refusal_names_the_encoding_an_env_actually_reports():
    """Not a string the test invented: the env is the source of the world.

    ``cost_encoding`` is read off a live :class:`ExecutionEnv` here so that a
    future model whose encoding disagreed with what the registry expects fails
    at this line rather than at grade time.
    """
    case = M1_CONFIG["variational"]["cases"][0]
    from .conftest import case_by_id

    golden = case_by_id(case)
    env = ExecutionEnv(
        golden.market,
        golden.order_size,
        golden.lambda_risk,
        temporary_impact=power_law_temporary(golden.market),
        root_seed=1,
    )
    assert env.cost_encoding == POWER_LAW
    check_grades_world(env.cost_encoding, metrics_for(env.cost_encoding))
    with pytest.raises(WorldMismatch):
        check_grades_world(env.cost_encoding, GRADED[LINEAR])


def test_metrics_for_refuses_a_world_that_does_not_exist():
    with pytest.raises(WorldMismatch, match="known worlds are"):
        metrics_for("vibes")


def test_metrics_for_refuses_a_world_nothing_scores():
    """An empty world is a mismatch, not an empty answer.

    Returning ``{}`` would let a caller loop over no metrics and report a grade
    of nothing at all, which is the shape of every silently-passing test.
    """
    GRADED["temporarily_empty"] = {}
    try:
        with pytest.raises(WorldMismatch, match="no graded metrics registered"):
            metrics_for("temporarily_empty")
    finally:
        GRADED.pop("temporarily_empty")


def test_an_unknown_encoding_cannot_be_registered_at_all():
    with pytest.raises(ValueError, match="unknown cost encoding"):
        Metric(name="mystery", encoding="vibes", summary="", fn=lambda *_: 0.0)


def test_a_name_cannot_be_registered_twice_in_one_world():
    """Two metrics under one name is a silent substitution waiting to happen."""
    duplicate = Metric(
        name="objective",
        encoding=LINEAR,
        summary="an impostor",
        fn=lambda trajectory, market, lambda_risk: 0.0,
    )
    with pytest.raises(ValueError, match="already registered"):
        register_graded(duplicate)


def test_context_still_refuses_to_be_reached_from_the_graded_path():
    """A context metric is registered, retrievable, and not in any graded world."""
    offender = Metric(
        name="sneaky_power_law",
        encoding=POWER_LAW,
        summary="a context metric trying to be a grade",
        fn=lambda trajectory, market, lambda_risk: 0.0,
    )
    try:
        register_context(offender)
        assert CONTEXT[POWER_LAW]["sneaky_power_law"] is offender
        assert "sneaky_power_law" not in metrics_for(POWER_LAW)
    finally:
        CONTEXT[POWER_LAW].pop("sneaky_power_law", None)


# ---------------------------------------------------------------------------
# The labels are true, not merely honest — in both worlds
# ---------------------------------------------------------------------------


def test_no_metric_reaches_for_the_other_world_by_hand():
    """The static half, now symmetric.

    A linear metric may not name ``cost_moments`` and a power-law metric may not
    name ``schedule_moments`` or ``linear_cost_moments``. Before M4a only the
    first direction existed, because only one direction was a mistake.
    """
    forbidden = {
        LINEAR: ("cost_moments",),
        POWER_LAW: ("schedule_moments", "linear_cost_moments"),
    }
    for encoding, world in GRADED.items():
        for name, metric in world.items():
            source = inspect.getsource(metric.fn)
            for banned in forbidden[encoding]:
                # Word boundaries, not substrings: `linear_cost_moments` contains
                # `cost_moments`, and a naive `in` would convict the linear
                # metrics of exactly the thing they are the correct spelling of.
                assert not re.search(rf"{banned}", source), (
                    f"graded metric {name!r} in the {encoding} world references "
                    f"{banned}"
                )


def test_graded_metrics_really_compute_what_they_claim(golden_case):
    """The behavioural half, run in both worlds on every vendored case.

    A metric could declare its encoding and quietly charge the other one. So
    every graded metric is evaluated on every case and must land on *its own*
    world's value — and, for the two that depend on temporary impact, must land
    clear of the other world's by more than the reduction tolerance. The
    variance is exempt from the second half because the two encodings share it
    by construction, which the test above measures.
    """
    case = golden_case
    market, lam = case.market, case.lambda_risk
    for encoding, world in GRADED.items():
        other = POWER_LAW if encoding == LINEAR else LINEAR
        for trajectory in _schedules(case).values():
            mine = _moments(encoding, trajectory, case)
            theirs = _moments(other, trajectory, case)
            for name, mine_value, their_value in (
                ("objective", mine.objective(lam), theirs.objective(lam)),
                ("expected_cost", mine.expected, theirs.expected),
                ("shortfall_variance", mine.variance, theirs.variance),
            ):
                value = world[name](trajectory, market, lam)
                assert value == pytest.approx(mine_value, rel=1e-15), (
                    f"graded metric {name!r} does not match the {encoding} encoding"
                )
                if name != "shortfall_variance":
                    assert abs(value - their_value) > REDUCTION_RTOL * abs(
                        their_value
                    ), (
                        f"graded metric {name!r} in the {encoding} world is "
                        f"indistinguishable from the {other} one on this case; the "
                        "behavioural check has lost its teeth"
                    )


def test_the_two_worlds_agree_on_the_variance_and_only_the_variance(golden_case):
    """M4a's inheritance, stated as arithmetic.

    ``shortfall_variance`` is bit-for-bit the same number in both worlds because
    the shock model is untouched; the other two are not. That is why M4a could
    swap the analytic reference for E[cost] alone and keep M1's variance tiers
    verbatim.
    """
    case = golden_case
    market, lam = case.market, case.lambda_risk
    for trajectory in _schedules(case).values():
        linear = GRADED[LINEAR]["shortfall_variance"](trajectory, market, lam)
        power = GRADED[POWER_LAW]["shortfall_variance"](trajectory, market, lam)
        assert linear == power


def test_context_metrics_really_are_the_power_law(golden_case):
    """The mirror check: context metrics must not have drifted linear."""
    case = golden_case
    market, lam = case.market, case.lambda_risk
    for trajectory in _schedules(case).values():
        power = cost_moments(trajectory, market)
        assert CONTEXT[POWER_LAW]["power_law_expected_cost"](
            trajectory, market, lam
        ) == pytest.approx(power.expected, rel=1e-15)
        assert CONTEXT[POWER_LAW]["power_law_objective"](
            trajectory, market, lam
        ) == pytest.approx(power.objective(lam), rel=1e-15)


def test_the_resolution_is_recorded_in_the_constitution():
    """§9 is where a decision this size lives; code comments are not.

    Constitution: "any change is a §9 amendment". A future session reading
    ``metrics.py`` alone would find a design; reading §9 it finds a decision with
    its reason, which is the difference between a convention and a rule. Both
    entries have to be there — M1's quarantine and M4a's generalisation of it.
    """
    architecture = (REPO_ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    amendments = architecture.split("## 9. Amendment log", 1)[-1]
    assert "cost_moments" in amendments
    assert "reporting context" in amendments, (
        "the invariant-7 resolution is not recorded in ARCHITECTURE.md §9"
    )
    assert "grades the world that charges it" in amendments, (
        "M4a's replacement of the flat quarantine is not recorded in §9"
    )
    assert "§9" in (metrics_module.__doc__ or ""), (
        "temper/eval/metrics.py does not point back at the amendment"
    )
