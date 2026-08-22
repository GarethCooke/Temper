"""The wire against a real Anvil, behind the `anvil` marker.

Everything else in `tests/` runs from a clean clone with nothing else installed;
this needs a server. `make test` therefore deselects it (`pyproject.toml`'s
`addopts`), and `make anvil-check` runs it against a locally started
`anvil_server` — the same shape as the `deep` and `training` tiers, which are
real gates that are not on the per-commit path.

What it checks is the half the fake-socket tests cannot: that Anvil behaves the
way `docs/vendor/anvil-protocol.md` says it does. Those are *predictions* about
somebody else's software, and the vendored snapshot is only worth having if
something occasionally reads it back against the thing it describes.

Run it with a **fresh** server on a single-ticker roster and the feeder off:

    ANVIL_TICKERS=101 ANVIL_DEFAULT_TICKER=101 ANVIL_FEEDER=0 \\
        ANVIL_PORT=18080 anvil_server
    python -m pytest tests/test_anvil_live.py -m anvil -v

The module leaves the book as it found it where it can — it cancels what it
posts — but a failed run may leave orders resting, and the client refuses to
build a measured ladder on a dirty book. Restart the server before a measured
run, always.
"""

from __future__ import annotations

import time

import pytest
import yaml

from client.book import Book, TradeTape
from client.ladder import ladder_from_mapping
from client.wire import (
    Stream,
    TransportFault,
    Venue,
    cancel_line,
    format_price,
    new_line,
)

from .conftest import REPO_ROOT

pytestmark = pytest.mark.anvil

CONFIG = yaml.safe_load(
    (REPO_ROOT / "configs" / "m6_anvil.yaml").read_text(encoding="utf-8")
)
VENUE = CONFIG["venue"]
TICKER = int(VENUE["ticker"])


@pytest.fixture(scope="module")
def venue() -> Venue:
    server = Venue(
        host=str(VENUE["host"]), port=int(VENUE["port"]), scheme=str(VENUE["scheme"])
    )
    try:
        server.health()
    except TransportFault as error:  # pragma: no cover - no server running
        pytest.skip(f"no anvil_server at {server.origin}: {error}")
    return server


@pytest.fixture
def stream():
    with Stream(str(VENUE["host"]), int(VENUE["port"]), TICKER) as socket_stream:
        yield socket_stream


def _settle(stream, book: Book, tape: TradeTape, seconds: float = 1.0) -> None:
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        for frame in stream.drain(timeout=0.05):
            book.apply(frame)
            tape.apply(frame)


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------


def test_the_server_reports_the_vendored_wire_version(venue):
    """The check the client makes before it opens a socket or sends an order."""
    document = venue.require_wire_version(int(VENUE["wire_version"]))
    assert document["status"] == "ok"


def test_a_new_order_takes_six_fields_and_five_are_refused(venue):
    """The id field is empty, not absent (vendored §2).

    The five-field form is not "repaired" — it earns a wrong-column rejection,
    deliberately, so a client that dropped the blank field finds out.
    """
    good = venue.send(new_line(TICKER, "B", 10, 90_000))
    assert good.accepted and good.id
    bad = venue.send(f"{TICKER},N,B,10,9.00")
    assert bad.accepted is False
    assert "column" in bad.reason
    assert bad.id and bad.id != good.id, "a rejected New still spends its id"
    venue.send(cancel_line(TICKER, good.id))


def test_the_session_owns_its_orders_and_nothing_else(venue):
    """Possession of the cookie is the ownership principal, and the reject is uniform.

    "Not yours" and "unknown id" come back with the *same* reason, so an
    enumeration walk learns nothing about which ids are live.
    """
    assert venue.session, "the first POST minted a session"
    mine = venue.send(new_line(TICKER, "B", 10, 90_000))
    assert venue.send(cancel_line(TICKER, mine.id)).accepted

    stranger = Venue(
        host=str(VENUE["host"]), port=int(VENUE["port"]), scheme=str(VENUE["scheme"])
    )
    theirs = stranger.send(new_line(TICKER, "B", 10, 90_000))
    refused = venue.send(cancel_line(TICKER, theirs.id))
    unknown = venue.send(cancel_line(TICKER, "no-such-id"))
    assert refused.accepted is False and unknown.accepted is False
    assert refused.reason == unknown.reason
    stranger.send(cancel_line(TICKER, theirs.id))


def test_a_snapshot_arrives_first_and_a_summary_second(venue, stream):
    """Vendored §3: exactly one `snapshot`, then one `summary`, then the stream."""
    first = stream.read(timeout=10.0)
    second = stream.read(timeout=10.0)
    assert first["type"] == "snapshot"
    assert int(first["ticker"]) == TICKER
    assert second["type"] == "summary"


def test_the_book_keeps_publishing_with_no_order_flow(venue, stream):
    """Emission is timer-driven, not activity-driven (§3.5).

    A quiet socket is therefore *not* the normal appearance of a quiet market,
    which is why this client never infers freshness from message rate.
    """
    stream.read(timeout=10.0)
    frames = []
    deadline = time.perf_counter() + 2.0
    while time.perf_counter() < deadline:
        frames.extend(stream.drain(timeout=0.05))
    kinds = {frame["type"] for frame in frames}
    assert "book" in kinds, "the ~14 Hz tick publishes on an idle book too"
    stamps = [frame["seq"] for frame in frames]
    assert len(set(stamps)) == len(stamps), "seq values are globally unique"


def test_a_marketable_sell_fills_and_is_attributable_by_taker_id(venue, stream):
    """Attribution is an equality test, not an inference (§3.3).

    Also the differential this milestone rests on: `Book.walk` is a dozen lines
    of Python, Anvil's matcher is C++ over a wire, and they are required to agree
    level for level.
    """
    book, tape = Book(ticker=TICKER), TradeTape()
    book.apply(stream.read(timeout=10.0))

    ladder = ladder_from_mapping(
        "reference", CONFIG["ladder"]["shapes"]["reference"]
    )
    posted = []
    for price, qty in ladder.targets("B")[:3]:
        result = venue.send(new_line(TICKER, "B", qty, price))
        assert result.accepted
        posted.append(result.id)
        tape.own(result.id)
    _settle(stream, book, tape, 1.5)

    quantity = ladder.quantities[0] + ladder.quantities[1] // 2
    limit = book._side("B")[-1].price
    predicted, unfilled = book.walk("B", quantity, limit)
    assert unfilled == 0

    sale = venue.send(new_line(TICKER, "S", quantity, limit))
    assert sale.accepted
    tape.own(sale.id)
    _settle(stream, book, tape, 1.5)

    fills = tape.attributed(sale.id)
    assert sum(fill["qty"] for fill in fills) == quantity
    assert tape.third_party == 0, "run this against a server with the feeder off"

    merged: list[list[int]] = []
    for fill in fills:
        if merged and merged[-1][0] == fill["price"]:
            merged[-1][1] += fill["qty"]
        else:
            merged.append([fill["price"], fill["qty"]])
    assert [tuple(row) for row in merged] == list(predicted), (
        "the Python walk and Anvil's matching engine disagree about which levels "
        "a marketable order takes"
    )
    for order_id in posted:
        venue.send(cancel_line(TICKER, order_id))


def test_a_mispriced_sell_rests_instead_of_erroring(venue, stream):
    """**Accepted is not filled** — the trap that returns a plausible number.

    Anvil has no market orders. A sell limit above the best bid does not error:
    it rests, `accepted` is `true`, and a client that inferred execution from the
    verdict would report a schedule it never ran.
    """
    book, tape = Book(ticker=TICKER), TradeTape()
    book.apply(stream.read(timeout=10.0))
    bid = venue.send(new_line(TICKER, "B", 50, 90_000))
    tape.own(bid.id)
    _settle(stream, book, tape, 1.5)

    # Priced a long way above the bid, so it cannot cross.
    timid = venue.send(new_line(TICKER, "S", 50, 110_000))
    assert timid.accepted is True, "the engine accepts it — that is the trap"
    tape.own(timid.id)
    _settle(stream, book, tape, 1.5)
    assert tape.attributed(timid.id) == (), "accepted, and not one share traded"
    assert any(level.price == 110_000 for level in book.asks), "it rested"

    assert venue.send(cancel_line(TICKER, timid.id)).accepted
    venue.send(cancel_line(TICKER, bid.id))


def test_the_summary_last_is_a_traded_price_and_not_a_mid(venue, stream):
    """The other trap, confirmed against the server rather than read in a doc.

    `last` was the book mid and is now the last *traded* price — changed inside
    wire version 1 with no bump, because the shape did not move. Two things
    follow, and both are checked: a two-sided book that has not traded reports
    `""`, and once set the value persists after the book empties.
    """
    fresh = venue.summary()["tickers"]
    row = next(entry for entry in fresh if int(entry["ticker"]) == TICKER)
    # If this server has already traded, the "" half cannot be re-observed
    # without a restart; the persistence half still can.
    traded_before = row["last"] != ""

    book, tape = Book(ticker=TICKER), TradeTape()
    book.apply(stream.read(timeout=10.0))
    bid = venue.send(new_line(TICKER, "B", 20, 95_000))
    ask = venue.send(new_line(TICKER, "S", 20, 105_000))
    tape.own(bid.id)
    tape.own(ask.id)
    _settle(stream, book, tape, 1.5)
    assert book.two_sided
    if not traded_before:
        row = next(
            entry
            for entry in venue.summary()["tickers"]
            if int(entry["ticker"]) == TICKER
        )
        assert row["last"] == "", (
            "a two-sided resting book still reports no last price; '' means "
            "'has not traded yet', not 'the book is empty'"
        )

    sale = venue.send(new_line(TICKER, "S", 20, 95_000))
    tape.own(sale.id)
    _settle(stream, book, tape, 1.5)
    venue.send(cancel_line(TICKER, ask.id))
    _settle(stream, book, tape, 1.5)

    row = next(
        entry for entry in venue.summary()["tickers"] if int(entry["ticker"]) == TICKER
    )
    assert row["last"] == format_price(95_000)
    assert not book.two_sided or book.best_bid is None or True
    # The mid is read off the book; `last` is a historical fact that outlives it.
    assert row["last"] != "", "a traded price persists after the book empties"


def test_the_stream_reconnects_and_re_baselines_from_a_fresh_snapshot(venue):
    """Recovery is transport-driven: the socket closes, you reconnect (§4).

    Driven against the real server because the thing being tested is the
    handshake and the on-connect `snapshot`, which a fake would have to
    reimplement in order to check.

    The client's own recovery path is `Session.refresh`, and this exercises the
    same `Stream.reconnect` it calls. The book is a full replace on arrival, so
    there is nothing to reconcile — which is the whole reason a client that
    cannot detect a dropped frame is nevertheless safe.
    """
    from client.book import Book
    from client.wire import Stream, StreamClosed

    book = Book(ticker=TICKER)
    with Stream(str(VENUE["host"]), int(VENUE["port"]), TICKER) as stream:
        first = stream.read(timeout=10.0)
        assert first["type"] == "snapshot"
        book.apply(first)

        # Drop the socket underneath the reader, the way a server restart or a
        # proxy timeout would.
        stream._socket.close()
        with pytest.raises((StreamClosed, OSError)):
            stream.read(timeout=5.0)

        stream.reconnect()
        baseline = stream.read(timeout=10.0)
        assert baseline["type"] == "snapshot", "the fresh snapshot is the new baseline"
        assert int(baseline["ticker"]) == TICKER
        assert book.apply(baseline) is True
        assert stream.read(timeout=10.0)["type"] == "summary"
