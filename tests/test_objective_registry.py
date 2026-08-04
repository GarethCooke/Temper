"""Task 1 — the invariant-7 resolution, and the quarantine that enforces it.

The question the brief poses: do FrontierView's power-law moments and the
linearised moments the closed form solves reduce to each other on the Phase-1
parameter sets? If they did, ``cost_moments`` would be the canonical encoding and
there would be nothing to resolve.

They do not, and this module is where that is measured rather than asserted. The
gap is 12 % to 54 % of expected cost across the vendored cases — a 0.6-power law
against its own tangent, which is not a rounding difference. So Phase 1 is the
linearised world end-to-end (``ARCHITECTURE.md`` §9, 2026-08-04) and
``cost_moments`` is quarantined to :data:`temper.eval.metrics.CONTEXT`.

The quarantine is checked three ways, because a label is only worth what enforces
it: the registry *refuses* a power-law metric under ``GRADED``; every graded
metric is evaluated against both encodings on every golden case and must equal
the linear one; and no graded metric's source may name ``cost_moments``.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from temper.eval import metrics as metrics_module
from temper.eval.metrics import (
    CONTEXT,
    GRADED,
    LINEAR,
    POWER_LAW,
    Metric,
    register_context,
    register_graded,
)
from temper.oracle import cost_moments, linear_cost_moments, linearised_eta, schedule_moments

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


# ---------------------------------------------------------------------------
# The measurement the resolution rests on
# ---------------------------------------------------------------------------


def test_the_two_encodings_do_not_reduce_to_each_other(golden_case):
    """The branch point of task 1, evaluated rather than assumed.

    Were the gap below `reduction_rtol` on every case, the honest move would be to
    pin an equality and call `linear_cost_moments` a redundant spelling of
    `cost_moments`. It is not: a power law and its tangent agree only where they
    touch. Flip `expect_reduction` in the config and this test will tell you
    which way the world actually is.
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
    variance a Phase-1 result reports is not a linearisation of anything, and
    ``shortfall_variance`` can be graded without qualification.
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
    """The env, the reward and the oracle all speak this one function."""
    market = golden_case.market
    eta_tilde = linearised_eta(market, golden_case.order_size)
    for trajectory in _schedules(golden_case).values():
        expected = linear_cost_moments(trajectory, market, eta_tilde)
        assert schedule_moments(trajectory, market) == expected


# ---------------------------------------------------------------------------
# The quarantine
# ---------------------------------------------------------------------------


def test_the_registries_are_populated_and_disjoint():
    """A quarantine over an empty registry would pass and mean nothing."""
    assert GRADED, "no graded metrics registered"
    assert CONTEXT, "no context metrics registered"
    assert not set(GRADED) & set(CONTEXT)


def test_every_graded_metric_uses_the_linear_encoding():
    assert {metric.encoding for metric in GRADED.values()} == {LINEAR}


def test_the_power_law_metrics_live_in_context():
    assert POWER_LAW in {metric.encoding for metric in CONTEXT.values()}
    assert all(metric.encoding == POWER_LAW for metric in CONTEXT.values())


def test_registering_a_power_law_metric_as_graded_is_refused():
    """The mechanical half of the quarantine (constitution invariant 7)."""
    offender = Metric(
        name="sneaky_power_law",
        encoding=POWER_LAW,
        summary="a power-law metric trying to become a grade",
        fn=lambda trajectory, market, lambda_risk: 0.0,
    )
    with pytest.raises(ValueError, match="cannot be graded in Phase 1"):
        register_graded(offender)
    assert "sneaky_power_law" not in GRADED

    # It is welcome under CONTEXT, which is the point of having two registries.
    try:
        register_context(offender)
        assert CONTEXT["sneaky_power_law"] is offender
    finally:
        CONTEXT.pop("sneaky_power_law", None)


def test_an_unknown_encoding_cannot_be_registered_at_all():
    with pytest.raises(ValueError, match="unknown cost encoding"):
        Metric(name="mystery", encoding="vibes", summary="", fn=lambda *_: 0.0)


def test_no_graded_metric_names_cost_moments():
    """The static half: a graded metric may not reach for the power law by hand."""
    for name, metric in GRADED.items():
        source = inspect.getsource(metric.fn)
        assert "cost_moments" not in source.replace("schedule_moments", "").replace(
            "linear_cost_moments", ""
        ), f"graded metric {name!r} references cost_moments"


def test_graded_metrics_really_compute_what_they_claim(golden_case):
    """The behavioural half: labels checked against arithmetic, not trusted.

    A metric could declare ``LINEAR`` and quietly charge the power law. So every
    graded metric is evaluated on every vendored case and must land on the linear
    value — and, for the two that depend on temporary impact, must land clear of
    the power-law value by more than the reduction tolerance.
    """
    case = golden_case
    market, lam = case.market, case.lambda_risk
    eta_tilde = linearised_eta(market, case.order_size)
    for trajectory in _schedules(case).values():
        linear = linear_cost_moments(trajectory, market, eta_tilde)
        power = cost_moments(trajectory, market)
        for name, linear_value, power_value in (
            ("objective", linear.objective(lam), power.objective(lam)),
            ("expected_cost", linear.expected, power.expected),
            ("shortfall_variance", linear.variance, power.variance),
        ):
            value = GRADED[name](trajectory, market, lam)
            assert value == pytest.approx(linear_value, rel=1e-15), (
                f"graded metric {name!r} does not match the linear encoding"
            )
            if name != "shortfall_variance":  # variance is shared by construction
                assert abs(value - power_value) > REDUCTION_RTOL * abs(power_value), (
                    f"graded metric {name!r} is indistinguishable from the power law "
                    "on this case; the behavioural check has lost its teeth"
                )


def test_context_metrics_really_are_the_power_law(golden_case):
    """The mirror check: context metrics must not have drifted linear."""
    case = golden_case
    market, lam = case.market, case.lambda_risk
    for trajectory in _schedules(case).values():
        power = cost_moments(trajectory, market)
        assert CONTEXT["power_law_expected_cost"](trajectory, market, lam) == pytest.approx(
            power.expected, rel=1e-15
        )
        assert CONTEXT["power_law_objective"](trajectory, market, lam) == pytest.approx(
            power.objective(lam), rel=1e-15
        )


def test_the_resolution_is_recorded_in_the_constitution():
    """§9 is where a decision this size lives; code comments are not.

    Constitution: "any change is a §9 amendment". A future session reading
    ``metrics.py`` alone would find a design; reading §9 it finds a decision with
    its reason, which is the difference between a convention and a rule.
    """
    architecture = (REPO_ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    amendments = architecture.split("## 9. Amendment log", 1)[-1]
    assert "cost_moments" in amendments
    assert "reporting context" in amendments, (
        "the invariant-7 resolution is not recorded in ARCHITECTURE.md §9"
    )
    assert "§9" in (metrics_module.__doc__ or ""), (
        "temper/eval/metrics.py does not point back at the amendment"
    )
