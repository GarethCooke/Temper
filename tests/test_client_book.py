"""The client's book, its attribution, and the three traps that return a number.

M6's brief names three routes to a confident wrong answer, all of which return
plausible basis points rather than crashing: a limit that rests instead of
crossing, a silently dropped trade frame, and an arrival price read from
`summary.last`. Each has a defence, and a defence that is a convention rather
than an assertion is not a defence — so each is asserted here.

The fourth property is idempotence. `snapshot` and `book` are full replaces
(vendored §3.1, §3.2, §4), which is the only reason a client that cannot detect
a dropped frame is nevertheless safe: it heals from the next baseline without
knowing it was ever behind.
"""

from __future__ import annotations

import pytest

from client.book import (
    Book,
    Level,
    TradeTape,
    notional,
    slippage_bps,
    vwap_ticks,
)

TICKER = 101


def frame(kind: str, seq: int, bids, asks, ticker: int = TICKER) -> dict:
    return {
        "type": kind,
        "seq": seq,
        "ticker": ticker,
        "bids": [{"price": price, "qty": qty, "orders": 1} for price, qty in bids],
        "asks": [{"price": price, "qty": qty, "orders": 1} for price, qty in asks],
    }


LADDER = frame(
    "snapshot",
    1,
    [("9.99", 300), ("9.98", 260), ("9.97", 220)],
    [("10.01", 300), ("10.02", 260)],
)


@pytest.fixture
def book() -> Book:
    book = Book(ticker=TICKER)
    book.apply(LADDER)
    return book


# ---------------------------------------------------------------------------
# Full replaces
# ---------------------------------------------------------------------------


def test_applying_the_same_frame_twice_changes_nothing(book):
    """Idempotence, which is what makes an undetectable gap survivable."""
    before = (book.bids, book.asks)
    book.apply(LADDER)
    assert (book.bids, book.asks) == before
    assert book.replaces == 2, "the count still moves; the state does not"


def test_a_book_frame_replaces_rather_than_merges(book):
    """A `book` frame carries the latest full top-N, not a delta (§3.2)."""
    book.apply(frame("book", 2, [("9.99", 50)], [("10.01", 10)]))
    assert book.bids == (Level(99_900, 50, 1),)
    assert book.depth("B") == 50, "the levels the frame omitted are gone, not stale"


def test_a_frame_for_another_ticker_is_ignored_not_rejected(book):
    """A socket is single-ticker, but `summary` reaches every socket (§1)."""
    assert book.apply(frame("book", 3, [("1.00", 1)], [], ticker=999)) is False
    assert book.depth("B") == 780


def test_a_summary_frame_is_not_a_book(book):
    assert book.apply({"type": "summary", "seq": 4, "tickers": []}) is False


def test_seq_is_kept_as_a_watermark_and_never_compared(book):
    """It can step backwards, so applying an older-stamped frame must still work.

    Vendored §1: one global engine-thread stamp, delivered from two unsorted
    sources. A client that discarded a frame for having a lower `seq` would drop
    real book updates on a live socket.
    """
    book.apply(frame("book", 999, [("9.99", 10)], [("10.01", 10)]))
    book.apply(frame("book", 5, [("9.99", 20)], [("10.01", 10)]))
    assert book.depth("B") == 20, "the later-arriving frame won, as it must"
    assert book.seq == 5


# ---------------------------------------------------------------------------
# The arrival price
# ---------------------------------------------------------------------------


def test_the_mid_comes_from_the_book(book):
    assert book.mid_ticks() == 100_000.0


def test_a_one_sided_book_has_no_mid_and_says_so():
    """Not a zero, not the one side that exists — an error.

    A run whose arrival price was quietly taken from half a book would report a
    number that looks like every other number in the artefact.
    """
    one_sided = Book(ticker=TICKER)
    one_sided.apply(frame("snapshot", 1, [("9.99", 100)], []))
    with pytest.raises(ValueError, match="one-sided"):
        one_sided.mid_ticks()


def test_the_mid_survives_an_odd_spread():
    """Half-ticks are real, so the mid is a float and is not rounded away."""
    odd = Book(ticker=TICKER)
    odd.apply(frame("snapshot", 1, [("9.9999", 10)], [("10.0002", 10)]))
    assert odd.mid_ticks() == 100_000.5


# ---------------------------------------------------------------------------
# Walking the book
# ---------------------------------------------------------------------------


def test_a_walk_takes_levels_best_first(book):
    fills, unfilled = book.walk("B", 421, 99_700)
    assert fills == ((99_900, 300), (99_800, 121))
    assert unfilled == 0


def test_a_walk_stops_at_the_limit(book):
    """The half of *accepted is not filled* that a prediction can see.

    A sell priced above a level does not reach it. Anvil does not error on that;
    the order simply rests, and the REST verdict still says accepted.
    """
    fills, unfilled = book.walk("B", 421, 99_900)
    assert fills == ((99_900, 300),)
    assert unfilled == 121


def test_a_walk_larger_than_the_book_reports_the_remainder(book):
    fills, unfilled = book.walk("B", 1_000, 99_700)
    assert sum(qty for _, qty in fills) == 780
    assert unfilled == 220


def test_the_sweep_price_is_the_last_level_needed(book):
    assert book.sweep_price("B", 300) == 99_900
    assert book.sweep_price("B", 301) == 99_800
    assert book.sweep_price("B", 781) is None, "the book does not hold it at any price"


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


def trade(seq: int, taker: str, maker: str, price="9.99", qty=100) -> dict:
    return {
        "type": "trade",
        "seq": seq,
        "ticker": TICKER,
        "price": price,
        "qty": qty,
        "aggr": "S",
        "takerId": taker,
        "makerId": maker,
        "ts": 1_718_480_000_000,
    }


def test_a_fill_is_ours_when_we_were_the_taker():
    """`takerId` against the ids `POST /api/order` returned. No inference (§3.3)."""
    tape = TradeTape()
    tape.own("o9")
    assert tape.apply(trade(1, "o9", "o1")) is not None
    assert tape.total_qty == 100
    assert tape.attributed("o9")[0]["price"] == 99_900


def test_a_fill_against_our_resting_ladder_is_not_part_of_the_parent_order():
    """We were the maker, so those shares are not shares the policy asked for.

    Counting them would inflate the executed quantity with somebody else's
    decision — and in a measured run it is a void condition, because the ladder
    the prediction was computed from is no longer the book that traded.
    """
    tape = TradeTape()
    tape.own("o1")
    assert tape.apply(trade(1, "stranger", "o1")) is None
    assert tape.total_qty == 0
    assert tape.against_us == 1
    assert tape.third_party == 0


def test_a_fill_between_two_strangers_is_third_party():
    tape = TradeTape()
    tape.own("o9")
    assert tape.apply(trade(1, "x", "y")) is None
    assert tape.third_party == 1
    assert tape.against_us == 0


def test_duplicate_frames_are_deduped_by_seq():
    """Legitimate because `seq` values are globally *unique* (§4).

    A set membership test, never a comparison — which is the only use of `seq`
    in this client beyond the reconnect watermark.
    """
    tape = TradeTape()
    tape.own("o9")
    tape.apply(trade(7, "o9", "o1"))
    tape.apply(trade(7, "o9", "o1"))
    assert tape.total_qty == 100


def test_out_of_order_seq_does_not_confuse_the_tape():
    tape = TradeTape()
    tape.own("o9")
    tape.apply(trade(90, "o9", "o1", qty=10))
    tape.apply(trade(12, "o9", "o2", qty=20))
    assert tape.total_qty == 30


# ---------------------------------------------------------------------------
# The number
# ---------------------------------------------------------------------------


def test_vwap_and_notional_are_exact_integer_arithmetic():
    fills = ((99_900, 300), (99_800, 121))
    assert notional(fills) == 99_900 * 300 + 99_800 * 121
    assert vwap_ticks(fills) == notional(fills) / 421


def test_selling_below_the_arrival_mid_is_a_positive_cost():
    """Sign convention, stated once and asserted rather than remembered."""
    assert slippage_bps(100_000.0, 99_900.0, "S") == pytest.approx(10.0)
    assert slippage_bps(100_000.0, 100_100.0, "S") == pytest.approx(-10.0)


def test_buying_above_the_arrival_mid_is_a_positive_cost():
    assert slippage_bps(100_000.0, 100_100.0, "B") == pytest.approx(10.0)


def test_the_reference_ladders_arrival_slippage_is_the_committed_number():
    """The measured run's headline, computed from the committed ladder alone.

    Not a golden pulled from a run: 1,000 shares against the reference ladder
    fills 300 at 9.99 in bin one, 121 at 9.98, and the rest at the touch after
    each replenishment. This is the arithmetic that produces 11.21 bps, checked
    here so a change to the price handling moves a number in a test rather than
    only in an artefact.
    """
    fills = ((99_900, 300), (99_800, 121), (99_900, 579))
    assert sum(qty for _, qty in fills) == 1000
    assert slippage_bps(100_000.0, vwap_ticks(fills), "S") == pytest.approx(11.21)
