"""Void is a result, and this is what makes that sentence true rather than brave.

The M6 brief's void condition has no escape hatch by design: if the quantity
attributed to this client's own order ids does not equal the parent order, or if
somebody else traded during a measured run, the measurement is void and is
reported as void. There is no reconciliation path and none may be added — the
moment one exists, every number the milestone reports becomes an estimate.

A refusal nobody has ever seen fire is a refusal nobody knows works. So:

* the three void conditions are exercised deterministically, with no server;
* and one **whole live run** is made to void on purpose, by dropping a `trade`
  frame — which is exactly the hazard the wire cannot signal. Vendored §4: on
  ring overflow the fill is dropped, a server-side latch is set, the broadcaster
  clears it without emitting anything, and the resulting `seq` gap is not
  client-detectable. There is no error frame and no way to notice. The only
  defence available is the end-of-run reconciliation, and this is it firing.

The frame is dropped in the **final** bin, deliberately. Drop one earlier and the
client believes it still holds shares it has already sold, sells them again in a
later bin, and the totals can coincidentally reconcile — a good illustration of
why the check has to be on attributed quantity against the parent order rather
than on anything the client believes about itself. In the last bin there is no
later bin to hide it.
"""

from __future__ import annotations

import pytest
import yaml

from client.book import TradeTape
from client.plan import BinPlan, Execution
from client.run import RunConfig, Session, load_config, reconcile
from client.wire import TransportFault

from .conftest import REPO_ROOT

CONFIG_PATH = REPO_ROOT / "configs" / "m6_anvil.yaml"
CHECKPOINT = REPO_ROOT / "results" / "m4a_power_law_policy.npz"


def _config(run: str = "ladder") -> RunConfig:
    return load_config(CONFIG_PATH, run)


def _execution(order_ids, fills) -> Execution:
    """A finished run, assembled from `(order_id, ((price, qty), ...))` pairs."""
    execution = Execution(parent=1000, side="S", arrival_mid=100_000.0)
    inventory = 1000
    for index, (order_id, bin_fills) in enumerate(zip(order_ids, fills)):
        plan = BinPlan(
            index=index,
            time_remaining=1.0 - index / 13,
            inventory_before=inventory,
            fraction=0.4,
            requested=sum(qty for _, qty in bin_fills),
            limit=99_200,
            fills=bin_fills,
            order_id=order_id,
        )
        execution.bins.append(plan)
        inventory = plan.inventory_after
    return execution


def _tape(order_ids, fills, **counts) -> TradeTape:
    tape = TradeTape(**counts)
    for order_id, bin_fills in zip(order_ids, fills):
        tape.own(order_id)
        for price, qty in bin_fills:
            tape.fills.append(
                {"seq": len(tape.fills), "price": price, "qty": qty,
                 "aggr": "S", "takerId": order_id, "makerId": "f1", "ts": 0}
            )
    return tape


# ---------------------------------------------------------------------------
# The three conditions
# ---------------------------------------------------------------------------


def test_a_complete_run_does_not_void():
    """The control. Without it the tests below prove only that `reconcile` says no."""
    ids, fills = ("o1", "o2"), (((99_900, 400),), ((99_900, 600),))
    result = reconcile(_execution(ids, fills), _tape(ids, fills), _config())
    assert result["void"] is False
    assert result["attributed"] == 1000
    assert result["reasons"] == []


def test_a_missing_fill_voids_the_measurement():
    """A dropped `trade` frame, which the wire cannot tell you about."""
    ids = ("o1", "o2")
    realised = (((99_900, 400),), ((99_900, 600),))
    observed = (((99_900, 400),), ((99_900, 591),))  # nine shares never seen
    result = reconcile(_execution(ids, realised), _tape(ids, observed), _config())
    assert result["void"] is True
    assert result["attributed"] == 991
    assert "attributed quantity 991 != parent order 1000" in result["reasons"]


def test_a_third_party_fill_voids_a_measured_run():
    """Somebody else traded on the ticker while a measured run was under way.

    The ladder run's whole claim is that the book is the committed ladder and
    nothing else. A stranger's fill breaks that whether or not it touched this
    client's orders, because the book the prediction was computed from is no
    longer the book that traded.
    """
    ids, fills = ("o1", "o2"), (((99_900, 400),), ((99_900, 600),))
    result = reconcile(
        _execution(ids, fills), _tape(ids, fills, third_party=3), _config()
    )
    assert result["void"] is True
    assert result["third_party_fills"] == 3
    assert any("third-party" in reason for reason in result["reasons"])


def test_a_fill_taken_against_our_own_ladder_voids_a_measured_run():
    ids, fills = ("o1", "o2"), (((99_900, 400),), ((99_900, 600),))
    result = reconcile(
        _execution(ids, fills), _tape(ids, fills, against_us=1), _config()
    )
    assert result["void"] is True
    assert result["fills_against_our_ladder"] == 1
    assert any("against this client's ladder" in reason for reason in result["reasons"])


def test_the_demonstration_run_is_not_voided_by_other_participants():
    """The feeder run is *made* of other participants' flow.

    Its void condition is the attribution one alone — it never claimed the book
    was its own, so a stranger trading is the point rather than a defect. The
    committed feeder artefact records 641 third-party fills and is not void.
    """
    ids, fills = ("o1", "o2"), (((99_900, 400),), ((99_900, 600),))
    result = reconcile(
        _execution(ids, fills),
        _tape(ids, fills, third_party=641, against_us=2),
        _config("feeder"),
    )
    assert result["void"] is False
    assert result["third_party_fills"] == 641


def test_a_void_measurement_is_reported_and_the_number_is_not():
    """The number still exists and must not be presented as the reported one.

    Hiding it would be a different kind of dishonesty, so it is written down —
    under a name nothing could mistake for the reported figure.
    """
    from client.run import _measurement

    ids = ("o1", "o2")
    realised = (((99_900, 400),), ((99_900, 600),))
    observed = (((99_900, 400),), ((99_900, 591),))
    execution = _execution(ids, realised)
    document = {"reconciliation": reconcile(execution, _tape(ids, observed), _config())}
    measurement = _measurement(None, execution, document)
    assert measurement["void"] is True
    assert measurement["realised_slippage_bps"] is None
    assert measurement["unreported_bps"] is not None
    assert measurement["reasons"]


# ---------------------------------------------------------------------------
# One whole run, made to void on purpose
# ---------------------------------------------------------------------------


class _DroppingSession(Session):
    """A client that loses one `trade` frame in the final bin, silently.

    Not a simulation of a bug — a simulation of the *venue*, which drops a fill
    on ring overflow and tells nobody (vendored §4). The client cannot detect it,
    which is the point: the only thing standing between this and a confidently
    wrong number is the reconciliation at the end.
    """

    def __init__(self, config, policy) -> None:
        super().__init__(config, policy)
        self.dropping = False
        self.dropped = 0

    def absorb(self, frames) -> None:
        kept = []
        for frame in frames:
            if (
                self.dropping
                and self.dropped == 0
                and frame.get("type") == "trade"
                and str(frame.get("takerId", "")) in self.tape.own_ids
            ):
                self.dropped += 1
                self.note(event="frame_dropped_on_purpose", seq=frame.get("seq"))
                continue
            kept.append(frame)
        super().absorb(kept)

    def work_bin(self, index: int, inventory: int) -> BinPlan:
        self.dropping = index == self.config.n_bins - 1
        return super().work_bin(index, inventory)


@pytest.mark.anvil
def test_a_dropped_fill_voids_a_real_run_end_to_end():
    """The induced mismatch, on the wire, with the whole loop running.

    Needs a **fresh** `anvil_server` with the feeder off, exactly as a measured
    run does — the client refuses to build its ladder on a book it did not build.

    What this asserts is the shape of a good failure: the client worked the order
    end to end, the venue did what it was told, and the *measurement* is void with
    the reason recorded. By `ARCHITECTURE.md` §7's terms that is still a
    successful M6, because §7 asks for plumbing evidence — and reaching for a
    reconciliation to rescue the number would convert a good milestone into a bad
    one.
    """
    from client.inference import load_policy
    from client.run import build_document

    if not CHECKPOINT.exists():  # pragma: no cover - the artefact is committed
        pytest.skip("no committed policy checkpoint in this tree")

    config = _config("ladder")
    policy = load_policy(CHECKPOINT)
    session = _DroppingSession(config, policy)
    try:
        realised = session.execute()
    except TransportFault as error:
        pytest.skip(f"needs a fresh anvil_server with an empty book: {error}")
    finally:
        if session.stream is not None:
            session.teardown()

    assert session.dropped == 1, "the frame this test exists to lose was not lost"
    document = build_document(config, policy, None, realised, session.tape, session)
    measurement = document["measurement"]
    reconciliation = document["reconciliation"]

    assert reconciliation["void"] is True
    assert reconciliation["attributed"] < reconciliation["parent"]
    assert any("attributed quantity" in reason for reason in reconciliation["reasons"])
    assert measurement["realised_slippage_bps"] is None
    assert measurement["unreported_bps"] is not None, (
        "the number is still computed and written down; it is simply not the "
        "reported one"
    )
    # The run itself reached the end. That is the half worth keeping.
    assert len(realised.bins) == config.n_bins
    assert reconciliation["third_party_fills"] == 0
