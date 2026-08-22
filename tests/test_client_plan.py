"""The prediction the measured run is accepted against, checked without a venue.

M6's acceptance criterion is *realised matches predicted*, not *a number was
produced*. That makes the prediction a first-class object: it is computed from
the committed ladder and the committed policy before anything is sent, and if it
were wrong the run would either fail for the wrong reason or pass for one.

So the prediction is checked here in three ways that need no server:

* the **decision** — a fraction of remaining inventory rounded onto whole shares,
  with the final bin forced to the remainder, matching `ExecutionEnv`'s terminal
  condition;
* the **closed loop** — bin `k + 1`'s inventory is what bin `k` actually left, so
  a thin ladder that cannot fill bin one changes every decision after it;
* the **arithmetic** — the reference ladder's predicted schedule and slippage,
  pinned as numbers, so a change to the pricing or the walk moves a test rather
  than only an artefact.

The committed config's own shapes are used, because the prediction is a statement
about *those bytes* and a fixture invented here would be a statement about
nothing.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from client.book import slippage_bps
from client.inference import load_policy
from client.ladder import Ladder, ladder_from_mapping
from client.plan import predict, requested_quantity

from .conftest import REPO_ROOT

CONFIG_PATH = REPO_ROOT / "configs" / "m6_anvil.yaml"
CHECKPOINT = REPO_ROOT / "results" / "m4a_power_law_policy.npz"
TICKER = 101
PARENT = 1000

if not CHECKPOINT.exists():  # pragma: no cover - the artefact is committed
    pytest.skip(
        f"{CHECKPOINT.relative_to(REPO_ROOT)} has not been exported in this tree",
        allow_module_level=True,
    )

CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
SHAPES = CONFIG["ladder"]["shapes"]
POLICY = load_policy(CHECKPOINT)


def shape(name: str) -> Ladder:
    return ladder_from_mapping(name, SHAPES[name])


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fraction", "inventory", "expected"),
    [
        (0.42052, 1000, 421),
        (0.5, 3, 2),          # half-up on a non-negative product, stated not implied
        (0.0, 500, 0),
        (1.0, 500, 500),
        (2.0, 500, 500),      # never more than is left
        (0.3, 0, 0),
    ],
)
def test_the_requested_quantity_rounds_the_way_the_prediction_does(
    fraction, inventory, expected
):
    assert requested_quantity(fraction, inventory, final=False) == expected


def test_the_final_bin_takes_the_whole_remainder():
    """`ExecutionEnv` force-liquidates the last bin; so does the client.

    The trained policy was shaped by that boundary — its open-loop schedule
    leaves ~0.9 % of the parent order for the terminal step and its last-bin
    fraction is only ~0.30, so a client that honoured the fraction instead of
    the boundary would end every run holding inventory it never decided to keep.
    """
    assert requested_quantity(0.30, 931, final=True) == 931
    assert requested_quantity(0.0, 931, final=True) == 931


# ---------------------------------------------------------------------------
# The reference ladder's committed prediction
# ---------------------------------------------------------------------------


def test_the_reference_ladder_fills_the_whole_parent_order():
    execution = predict(POLICY, shape("reference"), TICKER, PARENT)
    assert execution.filled == PARENT
    assert execution.complete
    assert len(execution.bins) == POLICY.n_bins


def test_the_largest_bin_is_a_plausible_fraction_of_posted_depth():
    """Task 0's gate, as an assertion rather than a note in a brief.

    Being unable to fill is a venue fact, not an agent result. The gate fixed the
    ladder and the parent order *together* so that the largest bin walks more
    than one level without sweeping the book — and a later edit to either number
    that broke that would otherwise only show up as a strange-looking artefact.
    """
    ladder = shape("reference")
    execution = predict(POLICY, ladder, TICKER, PARENT)
    largest = max(plan.requested for plan in execution.bins)
    assert largest == 421
    assert largest / PARENT == pytest.approx(0.421, abs=0.001)
    fraction_of_depth = largest / ladder.depth("B")
    assert 0.2 <= fraction_of_depth <= 0.35, (
        f"the largest bin is {fraction_of_depth:.1%} of posted depth; the gate "
        "fixed it at 29.9 %, large enough to walk more than one level and small "
        "enough not to sweep the book"
    )
    assert len(execution.bins[0].fills) == 2, "bin one crosses two levels"


def test_the_reference_ladders_predicted_slippage_is_the_committed_number():
    execution = predict(POLICY, shape("reference"), TICKER, PARENT)
    assert slippage_bps(
        execution.arrival_mid, execution.vwap, "S"
    ) == pytest.approx(11.21, abs=5e-4)


def test_the_arrival_mid_is_the_ladders_centre_by_symmetry():
    """One fewer thing a reader has to trust: `(bestBid + bestAsk) / 2 = centre`."""
    for name in SHAPES:
        ladder = shape(name)
        book = ladder.as_book(TICKER)
        assert book.mid_ticks() == ladder.mid == float(ladder.centre)


# ---------------------------------------------------------------------------
# The closed loop
# ---------------------------------------------------------------------------


def test_a_thin_ladder_cannot_fill_bin_one_and_the_policy_sees_it():
    """The state `ExecutionEnv` has never produced.

    `ExecutionEnv.step` clips to `[0, inventory]` and has always filled exactly
    what was asked, so a partial fill puts the policy off-distribution in
    inventory-remaining. This is the shape that forces it deterministically, and
    the assertion is that the *next* bin's decision reflects the shortfall rather
    than the schedule the policy would have run.
    """
    thin = predict(POLICY, shape("thin"), TICKER, PARENT)
    first = thin.bins[0]
    assert first.requested == 421
    assert first.filled == thin.bins[0].requested - 46 == 375
    assert first.unfilled == 46
    assert len(first.fills) == 8, "it sweeps every level of the thin ladder"

    reference = predict(POLICY, shape("reference"), TICKER, PARENT)
    assert thin.bins[1].inventory_before == 625
    assert reference.bins[1].inventory_before == 579
    assert thin.bins[1].requested > reference.bins[1].requested, (
        "carrying 46 unfilled shares into bin two must change bin two's decision"
    )
    assert thin.filled == PARENT, "the parent order still completes"


def test_inventory_is_conserved_bin_by_bin():
    for name in SHAPES:
        execution = predict(POLICY, shape(name), TICKER, PARENT)
        inventory = PARENT
        for plan in execution.bins:
            assert plan.inventory_before == inventory
            assert plan.filled <= plan.requested
            inventory = plan.inventory_after
        assert inventory == 0


def test_a_wider_spread_costs_more_and_nothing_else_changes():
    """Varying the ladder moves the price and leaves the decisions alone.

    Which is why the brief varies the ladder rather than the policy: the schedule
    is a function of inventory, and inventory only moves when a fill is short.
    Same quantities, different prices, strictly worse slippage.
    """
    reference = predict(POLICY, shape("reference"), TICKER, PARENT)
    wide = predict(POLICY, shape("wide"), TICKER, PARENT)
    assert [plan.requested for plan in wide.bins] == [
        plan.requested for plan in reference.bins
    ]
    assert slippage_bps(wide.arrival_mid, wide.vwap, "S") > slippage_bps(
        reference.arrival_mid, reference.vwap, "S"
    )


# ---------------------------------------------------------------------------
# Portability
# ---------------------------------------------------------------------------


def test_the_schedule_is_the_same_shape_at_a_different_parent_size():
    """No market parameter enters at inference, so the parent size is free.

    Stated in the units the claim is actually about: the fraction of the parent
    order worked in each bin, which must not depend on how big the parent order
    is. It cannot be exact, and the reason is worth stating rather than hiding
    behind a loose tolerance — Anvil takes whole shares, so a 1,000-share order
    quantises its schedule 100x more coarsely than a 100,000-share one. The bound
    is therefore *one share of the smaller order*, which is what quantisation can
    cost; measured, it comes in at 0.57 of one.

    Checked on a ladder deep enough for both, so this compares decisions rather
    than what the book could supply.
    """
    deep = Ladder(
        name="deep",
        centre=100_000,
        half_spread=100,
        spacing=100,
        quantities=(100_000,) * 8,
    )
    small, large = 1_000, 100_000
    thin_grid = predict(POLICY, deep, TICKER, small)
    fine_grid = predict(POLICY, deep, TICKER, large)

    # Bin one is asked the identical question at both sizes — the observation is
    # (1.0, 1.0) either way — so its fraction is not merely close, it is equal.
    assert thin_grid.bins[0].fraction == fine_grid.bins[0].fraction

    one_share = 1.0 / small
    worked_thin = np.array([plan.requested / small for plan in thin_grid.bins])
    worked_fine = np.array([plan.requested / large for plan in fine_grid.bins])
    assert np.max(np.abs(worked_thin - worked_fine)) <= one_share
    assert np.max(np.abs(np.cumsum(worked_thin) - np.cumsum(worked_fine))) <= one_share
    assert thin_grid.filled == small and fine_grid.filled == large
