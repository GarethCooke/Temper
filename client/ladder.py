"""The counterparty ladder the measured run builds, and tops up before each bin.

Anvil publishes books only for roster tickers, and the feeder drives **exactly**
that roster with `cross_frac = 0.10` — it aggresses, sweeping two levels through
the touch. So "a dedicated ticker with a quiet feeder" is not a configuration
that exists: a ticker with a published book is a ticker the feeder is trading.
The measured run therefore turns the feeder off (`ANVIL_FEEDER=0`) and posts its
own two-sided book.

That is circular unless it buys something, and what it buys is a *prediction*.
A committed ladder plus a deterministic policy plus deterministic matching makes
the whole run computable in closed form before it runs: every resting level is
known, so the fill prices and quantities of every bin are known. The number the
run reports then says something about client correctness — which is the one
thing it can certify — instead of about execution quality, which it never could.
The demo trades against its own liquidity, and says so.

The ladder is **two-sided** and the ask side is never touched. It is not
decoration: the arrival price is the book-snapshot mid, and a one-sided book has
no mid. Sizing the two sides symmetrically about a round centre makes the arrival
mid exactly that centre, which is one fewer thing for a reader to have to trust.

Shapes are committed in `configs/m6_anvil.yaml`. Varying the *ladder* rather than
the policy is deliberate (M6 brief, task 4): the prediction is computed from the
ladder, so a second shape tests the pricing and attribution logic directly, while
a second policy would only re-run the same arithmetic on different numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

from client.book import Book, Level


@dataclass(frozen=True)
class Ladder:
    """A symmetric two-sided book: prices in ticks, quantities in shares.

    `quantities` runs touch-first, so `quantities[0]` sits at
    `centre ± half_spread` and each subsequent level is `spacing` further out.
    Shaped like the feeder's own book — larger at the touch, thinning with depth
    — so a run against it is not obviously unlike a run against Anvil's normal
    flow.
    """

    name: str
    centre: int
    half_spread: int
    spacing: int
    quantities: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.half_spread <= 0 or self.spacing <= 0:
            raise ValueError("half spread and spacing are positive tick counts")
        if not self.quantities or any(qty <= 0 for qty in self.quantities):
            raise ValueError("every ladder level carries a positive quantity")
        deepest = self.centre - self.half_spread - self.spacing * (len(self.quantities) - 1)
        if deepest <= 0:
            raise ValueError(
                f"ladder {self.name!r} reaches {deepest} ticks on the bid side; "
                "Anvil rejects a price at or below zero"
            )

    # -- geometry -----------------------------------------------------------

    def prices(self, side: str) -> tuple[int, ...]:
        """Level prices, best-first, for the side that would be *resting*."""
        sign = -1 if side == "B" else 1
        if side not in ("B", "S"):
            raise ValueError(f"side is 'B' or 'S', got {side!r}")
        return tuple(
            self.centre + sign * (self.half_spread + self.spacing * index)
            for index in range(len(self.quantities))
        )

    def targets(self, side: str) -> tuple[tuple[int, int], ...]:
        """`((price, target_qty), ...)` best-first — the shape to hold."""
        return tuple(zip(self.prices(side), self.quantities))

    @property
    def mid(self) -> float:
        """The arrival mid this ladder presents: exactly `centre`, by symmetry."""
        return float(self.centre)

    def depth(self, side: str = "B") -> int:
        return sum(self.quantities)

    def as_book(self, ticker: int) -> Book:
        """The ladder as a fully-replenished :class:`~client.book.Book`.

        What the prediction walks. Orders-per-level is one, because that is what
        posting one order per level produces — the count is not used by the walk
        and is carried only so the predicted book and an observed book compare
        field for field.
        """
        return Book(
            ticker=ticker,
            bids=tuple(Level(price, qty, 1) for price, qty in self.targets("B")),
            asks=tuple(Level(price, qty, 1) for price, qty in self.targets("S")),
        )

    # -- maintenance --------------------------------------------------------

    def shortfall(self, book: Book, side: str) -> tuple[tuple[int, int], ...]:
        """`((price, qty_to_post), ...)` to restore `book`'s `side` to target.

        Compares against the *observed* book rather than against what the client
        believes it posted. In the measured run those agree and the difference is
        theoretical; in the demonstration run they do not, because the feeder has
        been trading, and topping up from a belief rather than from an
        observation is how a client ends up with a book it cannot predict.

        Levels the observed book carries *beyond* the target are left alone. A
        ladder run's book is entirely the client's own, so there is nothing to
        prune; a feeder run's is not the client's to tidy.
        """
        resting = {level.price: level.qty for level in book._side(side)}
        return tuple(
            (price, target - resting.get(price, 0))
            for price, target in self.targets(side)
            if target - resting.get(price, 0) > 0
        )

    def full_sweep_limit(self, side: str) -> int:
        """The limit price that crosses every level of `side`.

        The client prices each bin here: it crosses the whole published ladder,
        so a bin fills to the extent the book can fill it and never rests because
        it was priced too timidly. **Accepted is not filled** (vendored §2) —
        Anvil has no market orders, a sell limit above the best bid simply rests,
        and the REST response still says `accepted: true`. Pricing through the
        far side of the observed book is how this client refuses that failure,
        and the fills are verified afterwards rather than inferred from the
        verdict.
        """
        return self.prices(side)[-1]


def ladder_from_mapping(name: str, document: dict) -> Ladder:
    """One committed shape, from the config's own bytes."""
    return Ladder(
        name=name,
        centre=int(document["centre_ticks"]),
        half_spread=int(document["half_spread_ticks"]),
        spacing=int(document["spacing_ticks"]),
        quantities=tuple(int(qty) for qty in document["quantities"]),
    )
