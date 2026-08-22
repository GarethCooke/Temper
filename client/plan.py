"""What the client decides each bin, and what a committed ladder says will happen.

Two things live here and they are deliberately the same code twice over.

**The decision.** Ask the policy for a fraction of remaining inventory, turn it
into whole shares, and price it to cross. This is used by the prediction *and* by
the live run, because it is the client's decision either way — a prediction made
by a second implementation of the decision would be checking two guesses against
each other.

**The prediction.** Walk that decision against the ladder's own levels. This is
`client.book.Book.walk`, a dozen lines of Python; what it is checked against is
Anvil's C++ matching engine over a wire. Those are genuinely independent, which
is what makes *realised matches predicted* the acceptance criterion the M6 brief
states rather than a tautology. It is M1's differential-oracle pattern applied to
the wire leg: a closed form on one side, the thing itself on the other.

The closed loop is the part the simulator never exercised. `ExecutionEnv.step`
does `shares = min(max(shares, 0), inventory)` — it has *always* filled exactly
what was asked, so a partial fill is a state the trained policy has never seen.
Here bin `k + 1`'s observation carries the inventory bin `k` actually left, which
is the whole reason a replayed schedule could not stand in for a policy on this
venue, and is a generalisation risk to report rather than to discover.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from client.book import notional, vwap_ticks


def requested_quantity(fraction: float, inventory: int, *, final: bool) -> int:
    """Whole shares to work this bin, from a fraction of remaining inventory.

    Anvil's quantities are integers, so the fraction has to land on one. Rounding
    is half-up on a non-negative product, chosen for being *stated* rather than
    for being better than the alternatives: it is applied identically by the
    prediction and by the live client, so any deterministic rule would do and a
    rule nobody wrote down would not.

    `final` forces the whole remainder. That is the env's terminal condition
    (`ARCHITECTURE.md` §4: any remainder at the final step is force-liquidated
    and charged like any other trade), and it is the boundary the trained policy
    was shaped by — its last-bin fraction is essentially 1 already.
    """
    if inventory <= 0:
        return 0
    if final:
        return int(inventory)
    return min(int(inventory), int(fraction * inventory + 0.5))


@dataclass(frozen=True)
class BinPlan:
    """One bin: what was decided, what was priced, and what happened."""

    index: int
    time_remaining: float
    inventory_before: int
    fraction: float
    requested: int
    limit: int | None = None
    fills: tuple[tuple[int, int], ...] = ()
    order_id: str = ""
    #: Set when the bin was priced against an empty or one-sided book.
    skipped: str = ""

    @property
    def filled(self) -> int:
        return sum(qty for _, qty in self.fills)

    @property
    def levels(self) -> tuple[tuple[int, int], ...]:
        """The fills merged by price, in walk order — the *level* granularity.

        The prediction and the venue speak at different granularities and both
        are right. A prediction is computed from the published book, which
        aggregates by price (`snapshot` carries one `qty` and one `orders` count
        per level, §3.1). A `trade` frame is one fill against one *maker order*,
        so a level holding several resting orders produces several frames.

        The client's own replenishment is what makes them differ here: topping
        the ladder back up posts a fresh order at a price that already has one,
        so the touch accumulates orders and a later sweep crosses two of them.
        Realised bin 2 of the first ladder run came back as 78 + 46 at 9.99
        where the prediction said 124.

        Merging is therefore the comparison meeting the prediction where the
        prediction can speak, not a tolerance: *which* resting order supplies a
        share is queue position, which the M6 brief puts explicitly out of scope.
        The unmerged fills stay in the artefact, and `trade_frames` records how
        many there were, so the difference is visible rather than smoothed away.
        """
        merged: list[list[int]] = []
        for price, qty in self.fills:
            if merged and merged[-1][0] == price:
                merged[-1][1] += qty
            else:
                merged.append([price, qty])
        return tuple((price, qty) for price, qty in merged)

    @property
    def unfilled(self) -> int:
        return self.requested - self.filled

    @property
    def inventory_after(self) -> int:
        return self.inventory_before - self.filled

    @property
    def notional(self) -> int:
        return notional(self.fills)

    def as_dict(self) -> dict:
        document = {
            "bin": self.index,
            "time_remaining": self.time_remaining,
            "inventory_before": self.inventory_before,
            "fraction": self.fraction,
            "requested": self.requested,
            "limit_ticks": self.limit,
            "fills": [{"price_ticks": price, "qty": qty} for price, qty in self.fills],
            "levels": [{"price_ticks": price, "qty": qty} for price, qty in self.levels],
            "trade_frames": len(self.fills),
            "filled": self.filled,
            "unfilled": self.unfilled,
            "inventory_after": self.inventory_after,
            "notional_tick_shares": self.notional,
        }
        if self.order_id:
            document["order_id"] = self.order_id
        if self.skipped:
            document["skipped"] = self.skipped
        return document


@dataclass
class Execution:
    """A whole parent order: the bins, and the numbers they add up to."""

    parent: int
    side: str
    arrival_mid: float
    bins: list[BinPlan] = field(default_factory=list)

    @property
    def filled(self) -> int:
        return sum(plan.filled for plan in self.bins)

    @property
    def all_fills(self) -> tuple[tuple[int, int], ...]:
        return tuple(fill for plan in self.bins for fill in plan.fills)

    @property
    def vwap(self) -> float:
        return vwap_ticks(self.all_fills)

    @property
    def complete(self) -> bool:
        return self.filled == self.parent

    def as_dict(self) -> dict:
        return {
            "parent": self.parent,
            "side": self.side,
            "arrival_mid_ticks": self.arrival_mid,
            "filled": self.filled,
            "complete": self.complete,
            "vwap_ticks": self.vwap if self.filled else None,
            "bins": [plan.as_dict() for plan in self.bins],
        }


def predict(policy, ladder, ticker: int, parent: int, side: str = "S") -> Execution:
    """The whole run, in closed form, from the committed ladder and the policy.

    Assumes what the measured run arranges: the ladder is replenished to target
    before every bin, so each bin meets the same book, and nothing else is
    trading. Under those conditions the venue's behaviour is a deterministic
    function of numbers that are all committed, so this is a prediction in the
    strong sense — computed *before* the run, and either matched or not.

    The resting side consumed is the opposite of the side being worked: a sell
    aggresses the bids.
    """
    resting = "B" if side == "S" else "S"
    book = ladder.as_book(ticker)
    limit = ladder.full_sweep_limit(resting)
    n_bins = policy.n_bins
    execution = Execution(parent=parent, side=side, arrival_mid=ladder.mid)

    inventory = int(parent)
    for index in range(n_bins):
        time_remaining = 1.0 - index / n_bins
        fraction = policy.fraction(time_remaining, inventory / parent)
        requested = requested_quantity(
            fraction, inventory, final=index == n_bins - 1
        )
        if requested <= 0:
            execution.bins.append(
                BinPlan(
                    index=index,
                    time_remaining=time_remaining,
                    inventory_before=inventory,
                    fraction=fraction,
                    requested=0,
                    skipped="nothing to work",
                )
            )
            continue
        fills, _ = book.walk(resting, requested, limit)
        plan = BinPlan(
            index=index,
            time_remaining=time_remaining,
            inventory_before=inventory,
            fraction=fraction,
            requested=requested,
            limit=limit,
            fills=fills,
        )
        execution.bins.append(plan)
        inventory = plan.inventory_after
    return execution


def compare(predicted: Execution, realised: Execution) -> dict:
    """Bin by bin, where the two agree and where they do not.

    Returns a document rather than raising: a mismatch is a *result* — it is the
    client, the venue or the prediction being wrong, and which of the three it is
    is decided by looking at the disagreement, not by an exception's message.
    """
    rows = []
    for index, expected in enumerate(predicted.bins):
        actual = realised.bins[index] if index < len(realised.bins) else None
        rows.append(
            {
                "bin": index,
                "requested": [expected.requested, actual.requested if actual else None],
                "filled": [expected.filled, actual.filled if actual else None],
                # Compared at level granularity: see `BinPlan.levels`. The frame
                # counts travel beside it so a reader can see where the venue
                # split a level across maker orders.
                "fills_match": bool(actual) and actual.levels == expected.levels,
                "levels": [
                    [list(fill) for fill in expected.levels],
                    [list(fill) for fill in actual.levels] if actual else None,
                ],
                "trade_frames": [len(expected.fills), len(actual.fills) if actual else None],
                "notional_tick_shares": [
                    expected.notional,
                    actual.notional if actual else None,
                ],
            }
        )
    matched = all(row["fills_match"] for row in rows) and len(
        predicted.bins
    ) == len(realised.bins)
    return {
        "matched": matched,
        "bins": rows,
        "predicted_filled": predicted.filled,
        "realised_filled": realised.filled,
        "predicted_vwap_ticks": predicted.vwap if predicted.filled else None,
        "realised_vwap_ticks": realised.vwap if realised.filled else None,
    }
