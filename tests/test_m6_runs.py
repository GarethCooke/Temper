"""M6's committed runs, read back and checked against the config that made them.

The same shape as `tests/test_m2_rediscovery.py`: the artefact in `results/` is
read on every commit and compared with the committed bytes it claims to have come
from. Nothing here opens a socket — that would make the per-commit gate depend on
a running Anvil, which it must not.

What "regenerable" means here, precisely, because the word is doing work. The
*file* is not byte-reproducible: it carries wall-clock event stamps and ping
round-trips, which are properties of the evening it ran. The *measurement* is,
and in the strongest available sense — the predicted schedule is recomputed here
from `configs/m6_anvil.yaml` and the committed policy, and the realised fills the
venue produced have to equal it, level for level, bin for bin. A committed run
whose prediction no longer recomputes is a run whose ladder, policy or pricing
has moved underneath it.

The feeder run is deliberately held to less, and says so: it does not build its
own book, so it has no prediction and cannot have one. What is checked there is
that it completed, reconciled, and reported its own weak reproducibility.
"""

from __future__ import annotations

import json

import pytest
import yaml

from client.book import slippage_bps
from client.inference import load_policy
from client.ladder import ladder_from_mapping
from client.plan import predict

from .conftest import REPO_ROOT

CONFIG_PATH = REPO_ROOT / "configs" / "m6_anvil.yaml"
CHECKPOINT = REPO_ROOT / "results" / "m4a_power_law_policy.npz"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

#: The measured runs — the three that build their own book and are therefore
#: predictable in closed form. The feeder run is checked separately and to a
#: different standard.
LADDER_RUNS = ("ladder", "thin", "wide")

#: The run the milestone reports.
HEADLINE = "ladder"


def _artefact(run: str) -> dict | None:
    path = REPO_ROOT / CONFIG["runs"][run]["metrics"]
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


ARTEFACTS = {run: _artefact(run) for run in (*LADDER_RUNS, "feeder")}

if ARTEFACTS[HEADLINE] is None:  # pragma: no cover - the artefact is committed
    pytest.skip(
        "results/m6_anvil_ladder.json has not been produced in this tree. It "
        "needs a live anvil_server: `make m6` (see the Makefile's M6 block).",
        allow_module_level=True,
    )


@pytest.fixture(params=LADDER_RUNS, ids=LADDER_RUNS)
def ladder_run(request):
    document = ARTEFACTS[request.param]
    if document is None:
        pytest.skip(f"the {request.param} run has not been produced in this tree")
    return request.param, document


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_every_run_names_the_config_that_produced_it(ladder_run):
    """Invariant 1, by digest. A loosened config cannot keep an old result."""
    from temper.eval.provenance import config_digest

    run, document = ladder_run
    assert document["run"] == run
    assert document["provenance"]["config"] == "m6_anvil.yaml"
    assert document["provenance"]["config_sha256"] == config_digest(CONFIG_PATH)
    assert document["provenance"]["git_dirty"] is False
    assert len(document["provenance"]["git_rev"]) == 40


def test_every_run_names_the_policy_it_worked_the_order_with(ladder_run):
    """One artefact chain: sweep -> checkpoint -> run, each by content hash."""
    _, document = ladder_run
    policy = document["policy"]
    assert policy["checkpoint"] == "results/m4a_power_law_policy.npz"
    assert policy["n_bins"] == 13
    assert policy["selection"]["ordinal"] == 9
    assert policy["source_result"]["path"] == "results/m4a_power_law.json"


def test_the_run_states_it_is_a_demo(ladder_run):
    """`ARCHITECTURE.md` §7, in the artefact rather than only in a README.

    The number and the caveat travel together or they get separated, and the one
    that travels alone is always the number.
    """
    _, document = ladder_run
    assert "demo" in document["measurement"]["label"]
    assert "not execution-quality evidence" in document["claim"]
    for forbidden in ("epsilon", "capture_fraction", "red_flag", "gap_fraction"):
        assert forbidden not in document["measurement"], (
            f"{forbidden} is undefined on this venue and must not appear beside "
            "an Anvil number"
        )


# ---------------------------------------------------------------------------
# The prediction, recomputed
# ---------------------------------------------------------------------------


def test_the_committed_prediction_recomputes_from_the_committed_bytes(ladder_run):
    """The artefact's `predicted` block is derived, not remembered."""
    run, document = ladder_run
    policy = load_policy(CHECKPOINT)
    ladder = ladder_from_mapping(
        CONFIG["runs"][run]["shape"],
        CONFIG["ladder"]["shapes"][CONFIG["runs"][run]["shape"]],
    )
    recomputed = predict(
        policy,
        ladder,
        int(CONFIG["venue"]["ticker"]),
        int(CONFIG["order"]["parent"]),
        str(CONFIG["order"]["side"]),
    )
    assert recomputed.as_dict() == document["predicted"]


def test_the_venue_did_what_the_prediction_said_it_would(ladder_run):
    """M6's acceptance criterion: realised matches predicted, level for level.

    A dozen lines of Python on one side, Anvil's C++ matching engine over a wire
    on the other. That is M1's differential-oracle pattern applied to the wire
    leg — and it is the only thing an arrival-slippage number from a demo book
    *can* certify, which is why the milestone is accepted on it rather than on
    the number.
    """
    _, document = ladder_run
    comparison = document["comparison"]
    assert comparison["matched"], "the realised fills departed from the prediction"
    assert comparison["predicted_filled"] == comparison["realised_filled"]
    assert comparison["predicted_vwap_ticks"] == comparison["realised_vwap_ticks"]
    for row in comparison["bins"]:
        assert row["levels"][0] == row["levels"][1]


def test_the_whole_parent_order_reconciles(ladder_run):
    """The void condition, checked the way the brief states it.

    Attributed quantity equals the parent order, no third-party fill, nothing
    taken against the client's own ladder. A mismatch would void the measurement
    and there is no reconciliation path — the moment one exists, every number
    here is an estimate.
    """
    _, document = ladder_run
    reconciliation = document["reconciliation"]
    assert reconciliation["attributed"] == reconciliation["parent"] == 1000
    assert reconciliation["third_party_fills"] == 0
    assert reconciliation["fills_against_our_ladder"] == 0
    assert reconciliation["void"] is False
    assert reconciliation["reasons"] == []
    assert document["realised"]["complete"] is True


def test_the_arrival_price_came_off_the_book_and_is_the_ladders_centre(ladder_run):
    """Never `summary.last`. The mid is exactly the ladder's centre by symmetry."""
    run, document = ladder_run
    centre = float(CONFIG["ladder"]["shapes"][CONFIG["runs"][run]["shape"]]["centre_ticks"])
    assert document["realised"]["arrival_mid_ticks"] == centre
    arrival = next(event for event in document["events"] if event["event"] == "arrival")
    assert arrival["mid_ticks"] == centre
    assert (arrival["best_bid"] + arrival["best_ask"]) / 2 == centre


# ---------------------------------------------------------------------------
# The number
# ---------------------------------------------------------------------------


def test_the_reported_slippage_is_the_one_the_fills_imply(ladder_run):
    """Re-derived from the fills rather than trusted from the summary line."""
    _, document = ladder_run
    realised = document["realised"]
    recomputed = slippage_bps(
        realised["arrival_mid_ticks"], realised["vwap_ticks"], realised["side"]
    )
    assert document["measurement"]["realised_slippage_bps"] == pytest.approx(
        recomputed, rel=0, abs=1e-12
    )
    assert document["measurement"]["predicted_slippage_bps"] == pytest.approx(
        recomputed, rel=0, abs=1e-9
    )


def test_the_headline_run_reports_the_committed_number():
    """11.21 bps against a $10.0000 arrival mid, on the reference ladder."""
    document = ARTEFACTS[HEADLINE]
    assert document["measurement"]["realised_slippage_bps"] == pytest.approx(
        11.21, abs=5e-4
    )
    assert document["measurement"]["void"] is False
    assert document["measurement"]["unreported_bps"] is None


def test_a_thinner_book_costs_more_and_a_wider_spread_more_still():
    """The three shapes order the way the arithmetic says they must.

    Not a discovery — it is the check that the three runs are three runs and not
    one run written down three times.
    """
    reported = {
        run: ARTEFACTS[run]["measurement"]["realised_slippage_bps"]
        for run in LADDER_RUNS
        if ARTEFACTS[run] is not None
    }
    if len(reported) < 3:
        pytest.skip("not every ladder shape has been run in this tree")
    assert reported["ladder"] < reported["thin"] < reported["wide"]


def test_the_thin_ladder_actually_exercised_a_partial_fill():
    """The state `ExecutionEnv` has never produced, reached on a real venue.

    The simulator clips to `[0, inventory]` and has always filled exactly what
    was asked. Bin one of the thin run asked for 421 against 375 of depth, was
    filled short, had its resting remainder cancelled, and carried the shortfall
    into bin two's observation. That is the closed loop being *exercised* rather
    than merely present.
    """
    document = ARTEFACTS["thin"]
    if document is None:
        pytest.skip("the thin run has not been produced in this tree")
    first = document["realised"]["bins"][0]
    assert first["requested"] == 421
    assert first["unfilled"] == 46
    assert first["trade_frames"] == 8, "it swept every level of the thin ladder"
    cancels = [
        event
        for event in document["events"]
        if event["event"] == "cancelled_remainder"
    ]
    assert cancels and cancels[0]["accepted"] is True
    assert cancels[0]["remainder"] == 46
    assert document["realised"]["bins"][1]["inventory_before"] == 625


def test_the_stream_was_consumed_live_rather_than_polled():
    """Book replaces at roughly the publish cadence, and pings that measure.

    Vendored §4: the server queues rather than sheds, so a client that consumed
    slower than the cadence would get every frame *late*, with lag growing
    linearly. A run that priced against a stale book would still produce a
    number. These two are the evidence that it did not.
    """
    document = ARTEFACTS[HEADLINE]
    stream = document["stream"]
    seconds = CONFIG["grid"]["n_bins"] * CONFIG["grid"]["bin_seconds"]
    assert stream["book_replaces"] > 5 * seconds, (
        "far fewer full replaces than the ~14 Hz tick would deliver over the "
        "run: the client was not draining the stream"
    )
    pings = [value for value in stream["ping_seconds"] if value == value]
    assert len(pings) == CONFIG["grid"]["n_bins"]
    assert max(pings) < 1.0, (
        "a ping round-trip is queued behind whatever is already waiting for the "
        "socket, so a slow one means the client was behind when it priced"
    )


# ---------------------------------------------------------------------------
# The demonstration
# ---------------------------------------------------------------------------


def test_the_feeder_run_is_reported_separately_and_claims_less():
    """A demonstration, not a measurement, and the artefact says which.

    It is the only condition that shows the client surviving a book that moves
    between its bins because of somebody else's flow — and it is not
    byte-regenerable, because book state is wall-clock dependent. Both halves are
    reported.
    """
    document = ARTEFACTS["feeder"]
    if document is None:
        pytest.skip("the feeder run has not been produced in this tree")
    assert document["predicted"] is None
    assert "comparison" not in document
    assert "weak" in document["reproducibility"]
    assert document["ladder"]["built_by_client"] is False
    assert document["realised"]["filled"] > 0
    assert document["reconciliation"]["attributed"] == document["realised"]["filled"]
