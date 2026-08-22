"""The client's view of one Anvil book, and the arrival price taken from it.

A `snapshot` and a `book` frame carry the same payload and are both **full
replaces** of the ticker's visible book (vendored §3.1, §3.2, §4). So this class
has no delta path, no merge and no reordering: applying a frame overwrites, and
applying the same frame twice is the same as applying it once. That is not
laziness — it is the property the whole recovery story rests on. Vendored §4:
ring-overflow loss is not signalled on the wire and a per-ticker `seq` gap is not
detectable, so the client cannot know it missed a frame; what it can do is heal
completely from the next full replace, which idempotence makes automatic.

`seq` is kept only as a reconnect watermark and a dedupe key. Nothing here
compares two of them.

Prices are integer ticks (see the package docstring). The wire carries them as
shortest-decimal strings and `client.wire.parse_price` converts by string
arithmetic, so a level's price survives the round trip exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from client.wire import parse_price

#: Frame types this module knows how to apply. Both are full replaces; the
#: difference is only which one establishes the baseline on connect.
SNAPSHOT, BOOK = "snapshot", "book"


@dataclass(frozen=True)
class Level:
    """One price level: the aggregate the read-side helper published (§3.1)."""

    price: int          # ticks
    qty: int            # summed resting quantity
    orders: int         # count of resting orders


@dataclass
class Book:
    """One ticker's visible book, as full replaces leave it."""

    ticker: int
    bids: tuple[Level, ...] = ()
    asks: tuple[Level, ...] = ()
    #: The stamp of the last frame applied — a watermark, never an ordering.
    seq: int | None = None
    #: How many full replaces have landed. Reported, because "the book was
    #: refreshed n times between these two bins" is the honest way to describe a
    #: coalesced feed.
    replaces: int = 0

    def apply(self, frame: dict) -> bool:
        """Apply a `snapshot` or `book` frame; return whether it was one.

        Frames for another ticker are ignored rather than rejected: a socket is
        single-ticker, but `summary` is cross-ticker and reaches every socket, so
        a strict handler here would turn a documented behaviour into an error.
        """
        if frame.get("type") not in (SNAPSHOT, BOOK):
            return False
        if int(frame.get("ticker", self.ticker)) != self.ticker:
            return False
        self.bids = _levels(frame.get("bids", ()))
        self.asks = _levels(frame.get("asks", ()))
        self.seq = frame.get("seq")
        self.replaces += 1
        return True

    # -- reading ------------------------------------------------------------

    @property
    def best_bid(self) -> Level | None:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> Level | None:
        return self.asks[0] if self.asks else None

    @property
    def two_sided(self) -> bool:
        return bool(self.bids and self.asks)

    def mid_ticks(self) -> float:
        """`(bestBid + bestAsk) / 2`, in ticks. Half-ticks are real, so: float.

        **This is the only place a mid comes from, and `GET /api/summary`'s
        `last` is never it.** That field *was* the book mid and is now the
        ticker's last *traded* price — a semantic change made inside wire
        version `1` with no version bump, because the JSON shape did not move
        (vendored header, §2 `GET /api/summary`, §3.5). Two consequences bite a
        client that reasoned about it as a mid: `""` now means "has not traded
        yet" rather than "the book is empty", and once set it persists after the
        book empties. Anvil's own document names clients that "reasoned about it
        as a mid" as exactly the ones that break — and the arrival price is the
        one number this milestone reports, so reading it from `last` would be
        silently wrong in precisely the place it would never be noticed.
        """
        if not self.two_sided:
            raise ValueError(
                f"ticker {self.ticker} has a one-sided book "
                f"({len(self.bids)} bids, {len(self.asks)} asks); there is no "
                "mid to take an arrival price from"
            )
        return (self.best_bid.price + self.best_ask.price) / 2.0

    def depth(self, side: str) -> int:
        """Total visible quantity on one side, in shares."""
        return sum(level.qty for level in self._side(side))

    def _side(self, side: str) -> tuple[Level, ...]:
        if side == "B":
            return self.bids
        if side == "S":
            return self.asks
        raise ValueError(f"side is 'B' or 'S', got {side!r}")

    def walk(self, side: str, qty: int, limit: int) -> tuple[tuple, int]:
        """What a marketable order of `qty` at `limit` would take from this book.

        `side` is the side being *consumed*: a sell aggresses the bids (`"B"`),
        a buy aggresses the asks (`"S"`). Returns `(fills, unfilled)` where
        `fills` is `((price_ticks, qty), ...)` best-first — the resting (maker)
        price is the trade price, per the settled matching semantics (§3.3).

        This is the prediction and the venue's own arithmetic in one function, on
        purpose. The M6 brief's acceptance criterion is *realised matches
        predicted*, and a prediction computed by different code from the one the
        client reasons with would be checking two implementations of a guess
        rather than checking the client.
        """
        if qty <= 0:
            raise ValueError(f"quantity must be positive, got {qty}")
        crosses = (lambda price: price >= limit) if side == "B" else (
            lambda price: price <= limit
        )
        fills: list[tuple[int, int]] = []
        remaining = int(qty)
        for level in self._side(side):
            if remaining == 0 or not crosses(level.price):
                break
            taken = min(remaining, level.qty)
            fills.append((level.price, taken))
            remaining -= taken
        return tuple(fills), remaining

    def sweep_price(self, side: str, qty: int) -> int | None:
        """The limit that just crosses enough depth to fill `qty`, or `None`.

        The price of the last level the order would need to reach. `None` when
        the side does not hold `qty` at any price — which is a venue fact worth
        reporting rather than a number worth inventing.
        """
        remaining = int(qty)
        for level in self._side(side):
            remaining -= min(remaining, level.qty)
            if remaining == 0:
                return level.price
        return None

    def worst_price(self, side: str) -> int | None:
        """The furthest level published on one side — the full-sweep limit."""
        levels = self._side(side)
        return levels[-1].price if levels else None


def _levels(rows) -> tuple[Level, ...]:
    return tuple(
        Level(
            price=parse_price(str(row["price"])),
            qty=int(row["qty"]),
            orders=int(row.get("orders", 0)),
        )
        for row in rows
    )


@dataclass
class TradeTape:
    """The client's own fills, matched by `takerId` — no inference, no bracketing.

    Vendored §3.3: a `trade` frame carries `takerId` and `makerId`, and
    `POST /api/order` returns the server-minted id. So attribution is an equality
    test against the ids this client was given, and nothing about it is
    statistical.

    Dedupe is by `seq`, which is legitimate for exactly this: `seq` values are
    globally *unique* even though they do not arrive in order, so membership is a
    set test rather than a comparison (§4). It is the only use of `seq` in this
    package other than the reconnect watermark.
    """

    own_ids: set[str] = field(default_factory=set)
    fills: list[dict] = field(default_factory=list)
    seen: set[int] = field(default_factory=set)
    #: Fills on this ticker in which this client was neither side. In a measured
    #: run there must be none, so they are counted rather than discarded.
    third_party: int = 0
    #: Fills in which somebody else *took* against this client's resting ladder.
    #: A distinct category from the above and not a lesser one: it means the
    #: ladder the prediction was computed from is no longer the book that was
    #: traded, which voids a measured run just as surely.
    against_us: int = 0

    def own(self, order_id: str) -> None:
        if order_id:
            self.own_ids.add(order_id)

    def apply(self, frame: dict) -> dict | None:
        """Record a `trade` frame if this client was the taker; else classify it.

        Attribution is by `takerId` and nothing else. A fill where the client is
        the *maker* is not part of the parent order — it is somebody trading
        against the ladder — and counting it would inflate the executed quantity
        with shares the policy never asked for.
        """
        if frame.get("type") != "trade":
            return None
        stamp = frame.get("seq")
        if stamp in self.seen:
            return None
        self.seen.add(stamp)
        taker = str(frame.get("takerId", ""))
        maker = str(frame.get("makerId", ""))
        if taker not in self.own_ids:
            if maker in self.own_ids:
                self.against_us += 1
            else:
                self.third_party += 1
            return None
        fill = {
            "seq": stamp,
            "price": parse_price(str(frame["price"])),
            "qty": int(frame["qty"]),
            "aggr": str(frame.get("aggr", "")),
            "takerId": taker,
            "makerId": maker,
            "ts": frame.get("ts"),
        }
        self.fills.append(fill)
        return fill

    def attributed(self, order_id: str) -> tuple[dict, ...]:
        return tuple(fill for fill in self.fills if fill["takerId"] == order_id)

    @property
    def total_qty(self) -> int:
        return sum(fill["qty"] for fill in self.fills)


def notional(fills) -> int:
    """Price-times-quantity in tick-shares — exact integer arithmetic."""
    return sum(price * qty for price, qty in fills)


def vwap_ticks(fills) -> float:
    """The volume-weighted average price of `((price, qty), ...)`, in ticks."""
    quantity = sum(qty for _, qty in fills)
    if quantity == 0:
        raise ValueError("no fills to average")
    return notional(fills) / quantity


def slippage_bps(arrival_mid: float, vwap: float, side: str) -> float:
    """Arrival slippage in basis points, positive = worse than arrival.

    For a sell, executing below the arrival mid is a cost, so the sign is
    `(mid - vwap) / mid`. The mirror for a buy. Both in ticks, which cancel.
    """
    if arrival_mid <= 0:
        raise ValueError(f"arrival mid must be positive, got {arrival_mid}")
    signed = (arrival_mid - vwap) if side == "S" else (vwap - arrival_mid)
    return 1e4 * signed / arrival_mid
