"""Work one parent order on a live Anvil book, and write down what happened.

    python -m client.run --config configs/m6_anvil.yaml --dry-run
    python -m client.run --config configs/m6_anvil.yaml

`--dry-run` computes the prediction from the committed ladder and stops. That is
the order the M6 brief requires: the prediction is computed *before* the run, so
acceptance is *realised matches predicted* rather than *a number was produced*.

The shape of the loop is one decision per bin against the book the client has
just observed:

1. top the ladder back up to its committed target (measured run only);
2. drain the stream to the newest full replace, and measure how stale that is
   with a ping — vendored §3's *Keepalive*, which is the only freshness signal
   the wire offers;
3. ask the policy for a fraction of the inventory that is *actually* left;
4. price it to cross the resting side just observed, and send it;
5. collect the `trade` frames it caused, matched by `takerId`;
6. cancel anything that rested, because a marketable order that did not fill has
   become a resting order and would otherwise leak into a later bin.

Then reconcile: the quantity attributed to this client's own order ids must equal
the parent order. A mismatch **voids the measurement** and is reported as void.
There is no reconciliation path and none may be added — the moment one exists,
every number here becomes an estimate.

What this is not: an evaluation. `ARCHITECTURE.md` §7 — the flow is synthetic,
the sample is one order, and there is no baseline it can fairly be compared
against. The number it reports is arrival slippage in bps, labelled a demo.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # `python client/run.py` from a clean clone
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from client.book import Book, TradeTape, slippage_bps, vwap_ticks  # noqa: E402
from client.inference import load_policy  # noqa: E402
from client.ladder import ladder_from_mapping  # noqa: E402
from client.plan import BinPlan, Execution, compare, predict, requested_quantity  # noqa: E402
from client.wire import (  # noqa: E402
    Stream,
    TransportFault,
    Venue,
    cancel_line,
    format_price,
    new_line,
)
from temper.eval.provenance import stamp  # noqa: E402


@dataclass(frozen=True)
class RunConfig:
    """One named run of the committed config. Nothing here defaults to a surprise.

    The config holds four runs because they differ in three fields and share
    everything that decides what the client does; the run's name travels into the
    artefact beside the config digest, so `(config_sha256, run)` identifies a run
    as completely as a filename would and cannot disagree with itself.
    """

    path: Path
    document: dict
    run: str

    @property
    def block(self) -> dict:
        try:
            return self.document["runs"][self.run]
        except KeyError:
            raise SystemExit(
                f"{self.path.name} has no run named {self.run!r}; it has "
                f"{', '.join(sorted(self.document['runs']))}"
            ) from None

    @property
    def venue(self) -> dict:
        return self.document["venue"]

    @property
    def ticker(self) -> int:
        return int(self.venue["ticker"])

    @property
    def side(self) -> str:
        return str(self.document["order"]["side"])

    @property
    def parent(self) -> int:
        return int(self.document["order"]["parent"])

    @property
    def n_bins(self) -> int:
        return int(self.document["grid"]["n_bins"])

    @property
    def bin_seconds(self) -> float:
        return float(self.document["grid"]["bin_seconds"])

    @property
    def settle_seconds(self) -> float:
        return float(self.document["grid"]["settle_seconds"])

    @property
    def checkpoint(self) -> Path:
        return REPO_ROOT / self.document["policy"]["checkpoint"]

    @property
    def builds_ladder(self) -> bool:
        return bool(self.block["build"])

    @property
    def ladder(self):
        name = str(self.block["shape"])
        return ladder_from_mapping(name, self.document["ladder"]["shapes"][name])

    @property
    def results(self) -> Path:
        return REPO_ROOT / self.block["metrics"]

    @property
    def label(self) -> str:
        return str(self.block.get("label", self.run))


def load_config(path: str | Path, run: str) -> RunConfig:
    target = Path(path)
    return RunConfig(
        path=target,
        document=yaml.safe_load(target.read_text(encoding="utf-8")),
        run=run,
    )


# ---------------------------------------------------------------------------
# The live loop
# ---------------------------------------------------------------------------


class Session:
    """One client, one ticker, one parent order."""

    def __init__(self, config: RunConfig, policy) -> None:
        self.config, self.policy = config, policy
        self.venue = Venue(
            host=str(config.venue.get("host", "127.0.0.1")),
            port=int(config.venue.get("port", 18080)),
            scheme=str(config.venue.get("scheme", "http")),
        )
        self.book = Book(ticker=config.ticker)
        self.tape = TradeTape()
        self.stream: Stream | None = None
        self.ladder_ids: list[str] = []
        self.freshness: list[float] = []
        self.events: list[dict] = []

    # -- plumbing -----------------------------------------------------------

    def note(self, **fields) -> None:
        """One line of run narrative, kept in the artefact rather than printed away."""
        self.events.append({"t": round(time.time(), 3), **fields})

    def absorb(self, frames) -> None:
        """Apply frames: books as full replaces, trades by `takerId`."""
        for frame in frames:
            self.book.apply(frame)
            self.tape.apply(frame)

    def refresh(self, seconds: float = 0.0) -> None:
        self.absorb(self.stream.take_pending())
        self.absorb(self.stream.drain(timeout=seconds))

    def post(self, line: str) -> str:
        """Send one order line; return the minted id, or raise on a rejection.

        A rejection here is a client defect rather than a market outcome: every
        line this client sends is built from the book it just read, so
        `accepted: false` means the line was malformed, mispriced beyond the
        engine's bounds, or aimed at an order this session does not own.
        """
        result = self.venue.send(line)
        if not result.accepted:
            raise TransportFault(
                f"the engine rejected {line!r}: {result.reason!r} (id {result.id}). "
                "That is a business verdict, not a transport fault — the client "
                "built a line the engine would not take."
            )
        self.tape.own(result.id)
        return result.id

    # -- the ladder ---------------------------------------------------------

    def build_ladder(self) -> None:
        """Post the committed two-sided ladder. Measured run only.

        Refuses to build on top of anything already resting. The prediction is
        computed from the committed ladder alone, so a leftover order — another
        session's, or this client's from an earlier run — silently makes the
        predicted book and the real one two different books. Task 0 hit exactly
        that: a 300-share bid survived from a previous process, could not be
        cancelled (ownership is the session cookie, and the session had gone),
        and quietly doubled the touch level. Restart the server for a measured
        run; it is a demo book with nothing to preserve.
        """
        if self.book.bids or self.book.asks:
            raise TransportFault(
                f"ticker {self.config.ticker} already has "
                f"{self.book.depth('B')} resting on the bid and "
                f"{self.book.depth('S')} on the ask. A measured run predicts "
                "from the committed ladder and nothing else, so it must build "
                "on an empty book — restart anvil_server with ANVIL_FEEDER=0."
            )
        ladder = self.config.ladder
        for side in ("B", "S"):
            for price, qty in ladder.targets(side):
                self.ladder_ids.append(
                    self.post(new_line(self.config.ticker, side, qty, price))
                )
        self.note(
            event="ladder_posted",
            shape=ladder.name,
            orders=len(self.ladder_ids),
            depth_per_side=ladder.depth(),
        )

    def replenish(self) -> int:
        """Top the ladder back up to target against the *observed* book."""
        ladder, posted = self.config.ladder, 0
        for side in ("B", "S"):
            for price, qty in ladder.shortfall(self.book, side):
                self.ladder_ids.append(
                    self.post(new_line(self.config.ticker, side, qty, price))
                )
                posted += qty
        return posted

    # -- one bin ------------------------------------------------------------

    def work_bin(self, index: int, inventory: int) -> BinPlan:
        config, n_bins = self.config, self.config.n_bins
        resting = "B" if config.side == "S" else "S"
        time_remaining = 1.0 - index / n_bins

        if config.builds_ladder:
            topped = self.replenish()
            if topped:
                self.note(event="replenished", bin=index, shares=topped)
            # The engine applies posted orders asynchronously; wait for the book
            # frame that reflects them rather than pricing against a stale view.
            self.settle(config.settle_seconds)
        self.refresh(0.0)

        fraction = self.policy.fraction(time_remaining, inventory / config.parent)
        requested = requested_quantity(fraction, inventory, final=index == n_bins - 1)
        base = BinPlan(
            index=index,
            time_remaining=time_remaining,
            inventory_before=inventory,
            fraction=fraction,
            requested=requested,
        )
        if requested <= 0:
            return base

        levels = self.book._side(resting)
        if not levels:
            self.note(event="empty_book", bin=index, side=resting)
            return BinPlan(**{**base.__dict__, "skipped": "no resting depth to cross"})

        # Priced through the far side of the book just observed. Anvil has no
        # market orders: a sell limit above the best bid does not error, it
        # *rests*, and the REST response still says accepted (vendored §2). The
        # verdict is therefore never taken as evidence of execution — the fills
        # below are.
        limit = levels[-1].price
        order_id = self.post(
            new_line(config.ticker, config.side, requested, limit)
        )
        before = len(self.tape.fills)
        self.settle(config.settle_seconds)
        fills = self.tape.attributed(order_id)
        self.note(
            event="worked",
            bin=index,
            order=order_id,
            requested=requested,
            limit=format_price(limit),
            fills=len(self.tape.fills) - before,
        )

        filled = sum(fill["qty"] for fill in fills)
        if filled < requested:
            # Whatever did not cross is now a resting order of ours. Cancel it:
            # a remainder left in the book would fill in a later bin against a
            # decision the policy never made, and would be attributed to this
            # bin's id. Ownership is the session cookie (vendored §2).
            self.cancel(order_id, index, requested - filled)
            self.settle(config.settle_seconds)
            fills = self.tape.attributed(order_id)

        return BinPlan(
            index=index,
            time_remaining=time_remaining,
            inventory_before=inventory,
            fraction=fraction,
            requested=requested,
            limit=limit,
            fills=tuple((fill["price"], fill["qty"]) for fill in fills),
            order_id=order_id,
        )

    def cancel(self, order_id: str, index: int, remainder: int) -> None:
        result = self.venue.send(cancel_line(self.config.ticker, order_id))
        self.note(
            event="cancelled_remainder",
            bin=index,
            order=order_id,
            remainder=remainder,
            accepted=result.accepted,
            reason=result.reason,
        )

    def wait_for_two_sided(self, timeout: float) -> float:
        """Block until the book has both sides, or give up. Feeder run only.

        Anvil's feeder is gated on a live WebSocket client — it is wired to the
        registry's 0↔1 hook, so a server with nobody watching publishes an empty
        book. The demonstration run therefore *arrives before its market does*:
        the snapshot on connect is empty, and there is no mid to take an arrival
        price from.

        Waiting is the honest reading of "the book snapshot at t = 0": t = 0 is
        when the parent order starts, and a parent order cannot start on a book
        that does not exist yet. How long it waited goes into the artefact, since
        it is part of what happened.

        A measured run never reaches this — it builds its own book and refuses to
        start on one it did not build.
        """
        started = time.perf_counter()
        deadline = started + timeout
        while time.perf_counter() < deadline:
            self.refresh(0.1)
            if self.book.two_sided:
                return time.perf_counter() - started
        raise TransportFault(
            f"no two-sided book on ticker {self.config.ticker} after {timeout:g}s. "
            "A run that does not build its own book needs Anvil's feeder "
            "running (ANVIL_FEEDER=1); with it off, nothing ever rests."
        )

    def settle(self, seconds: float) -> None:
        """Drain the stream for `seconds`, applying everything that arrives."""
        deadline = time.perf_counter() + seconds
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return
            self.absorb(self.stream.drain(timeout=min(remaining, 0.05)))

    # -- the run ------------------------------------------------------------

    def execute(self) -> Execution:
        config = self.config
        health = self.venue.require_wire_version(int(config.venue["wire_version"]))
        self.note(event="health", **{k: health[k] for k in ("status", "wireVersion")})

        with Stream(self.venue.host, self.venue.port, config.ticker) as stream:
            self.stream = stream
            # On connect: exactly one snapshot, then one summary (vendored §3).
            self.absorb([stream.read(timeout=10.0)])
            if config.builds_ladder:
                self.build_ladder()
                self.settle(max(config.settle_seconds, 1.0))
            else:
                waited = self.wait_for_two_sided(timeout=60.0)
                self.note(event="warmup", seconds=round(waited, 3))
            self.refresh(0.5)

            # The arrival price, and the one place it comes from. NOT
            # `GET /api/summary`'s `last`: that field was the book mid and is now
            # the last *traded* price — changed inside wire version 1 with no
            # bump because the JSON shape did not move (vendored header, §2
            # `GET /api/summary`, §3.5). `""` now means "has not traded yet", and
            # once set it persists after the book empties. Anvil's own document
            # names clients that reasoned about it as a mid as exactly the ones
            # that break, and arrival slippage is the single number this run
            # reports — so reading it from `last` would be silently wrong in the
            # one place nobody would look.
            arrival_mid = self.book.mid_ticks()
            self.note(
                event="arrival",
                mid_ticks=arrival_mid,
                best_bid=self.book.best_bid.price,
                best_ask=self.book.best_ask.price,
                bid_depth=self.book.depth("B"),
                ask_depth=self.book.depth("S"),
            )

            execution = Execution(
                parent=config.parent, side=config.side, arrival_mid=arrival_mid
            )
            inventory = config.parent
            for index in range(config.n_bins):
                started = time.perf_counter()
                try:
                    self.freshness.append(stream.ping(timeout=5.0))
                except TimeoutError:  # pragma: no cover - a stalled stream
                    self.freshness.append(float("nan"))
                plan = self.work_bin(index, inventory)
                execution.bins.append(plan)
                inventory = plan.inventory_after
                if inventory <= 0 and index < config.n_bins - 1:
                    self.note(event="complete_early", bin=index)
                    break
                elapsed = time.perf_counter() - started
                self.settle(max(0.0, config.bin_seconds - elapsed))
            return execution

    def teardown(self) -> None:
        """Cancel the ladder. Best effort: a filled order cannot be cancelled."""
        for order_id in self.ladder_ids:
            try:
                self.venue.send(cancel_line(self.config.ticker, order_id))
            except TransportFault:  # pragma: no cover - the server is going away
                return


# ---------------------------------------------------------------------------
# Reconciliation and the artefact
# ---------------------------------------------------------------------------


def reconcile(execution: Execution, tape: TradeTape, config: RunConfig) -> dict:
    """Does the attributed quantity equal the parent order? Void if not.

    The M6 brief's void condition, implemented as the brief states it and with no
    escape hatch. Two ways to fail: the client's own fills do not add up to the
    parent order, or somebody else traded on this ticker during a measured bin.
    Either is reported as void, with the reason. Reaching for a reconciliation to
    rescue a number would convert a good milestone into a bad one — a client that
    worked the order end to end and voided its measurement is still a successful
    demonstration by `ARCHITECTURE.md` §7's terms, because §7 asks for plumbing
    evidence.
    """
    attributed = sum(
        fill["qty"] for fill in tape.fills if fill["takerId"] in _own_order_ids(execution)
    )
    reasons = []
    if attributed != config.parent:
        reasons.append(
            f"attributed quantity {attributed} != parent order {config.parent}"
        )
    if config.builds_ladder and tape.against_us:
        reasons.append(
            f"{tape.against_us} fill(s) taken against this client's ladder by "
            "another participant during a measured run"
        )
    if config.builds_ladder and tape.third_party:
        reasons.append(
            f"{tape.third_party} third-party fill(s) on this ticker during a "
            "measured run"
        )
    return {
        "attributed": attributed,
        "parent": config.parent,
        "third_party_fills": tape.third_party,
        "fills_against_our_ladder": tape.against_us,
        "void": bool(reasons),
        "reasons": reasons,
    }


def _own_order_ids(execution: Execution) -> set[str]:
    return {plan.order_id for plan in execution.bins if plan.order_id}


def build_document(
    config: RunConfig,
    policy,
    predicted: Execution | None,
    realised: Execution | None,
    tape: TradeTape | None,
    session: Session | None,
) -> dict:
    provenance = stamp(config.path, REPO_ROOT)
    ladder = config.ladder
    document = {
        "milestone": "M6",
        "run": config.run,
        "label": config.label,
        "claim": (
            "A trained execution policy worked one parent order on a live Anvil "
            "book over the versioned wire, as PROTOCOL.md's third independent "
            "client, with zero changes to Anvil. This is plumbing evidence, not "
            "execution-quality evidence: the flow is synthetic, the sample is "
            "one order, and there is no baseline it can fairly be compared "
            "against (ARCHITECTURE.md §7). No epsilon, no capture fraction and "
            "no oracle comparison appears here — none of them is defined on this "
            "venue."
        ),
        "provenance": provenance.as_dict(),
        "policy": {
            "checkpoint": str(
                config.checkpoint.relative_to(REPO_ROOT)
            ).replace("\\", "/"),
            "name": policy.name,
            "n_bins": policy.n_bins,
            "selection": policy.metadata.get("selection"),
            "grade": policy.metadata.get("grade"),
            "source_result": policy.metadata.get("source_result"),
        },
        "venue": dict(config.venue),
        "order": {"side": config.side, "parent": config.parent},
        "ladder": {
            "name": ladder.name,
            "built_by_client": config.builds_ladder,
            "centre_ticks": ladder.centre,
            "half_spread_ticks": ladder.half_spread,
            "spacing_ticks": ladder.spacing,
            "quantities": list(ladder.quantities),
            "depth_per_side": ladder.depth(),
        },
        "grid": {
            "n_bins": config.n_bins,
            "bin_seconds": config.bin_seconds,
            "settle_seconds": config.settle_seconds,
            "note": (
                "13 bins, as the policy was trained on — the bin *count* is part "
                "of the observation's clock and is not a free choice. The "
                "wall-clock length is: the 6.5-hour grid compresses to minutes "
                "because no market parameter enters at inference."
            ),
        },
    }
    if predicted is not None:
        document["predicted"] = predicted.as_dict()
    else:
        document["predicted"] = None
        document["reproducibility"] = (
            "weak. This run does not build its own book: Anvil's feeder does, "
            "and keeps trading it. Book state is wall-clock dependent, so the "
            "run is not byte-regenerable and there is no closed-form prediction "
            "to check it against — there cannot be. It is a demonstration that "
            "the client survives a book that moves between its bins because of "
            "somebody else's flow, which is the one thing the measured runs "
            "cannot show, and it is reported separately for that reason."
        )
    if realised is not None:
        document["realised"] = realised.as_dict()
        document["reconciliation"] = reconcile(realised, tape, config)
        if predicted is not None:
            document["comparison"] = compare(predicted, realised)
        document["measurement"] = _measurement(predicted, realised, document)
    else:
        document["measurement"] = {
            "predicted_slippage_bps": slippage_bps(
                predicted.arrival_mid, predicted.vwap, predicted.side
            )
            if predicted is not None and predicted.filled
            else None,
            "realised_slippage_bps": None,
            "void": None,
            "note": "prediction only; no run has taken place",
        }
    if session is not None:
        document["stream"] = {
            "book_replaces": session.book.replaces,
            "ping_seconds": session.freshness,
            "note": (
                "ping round-trip is the only true freshness signal the wire "
                "offers: the pong is queued behind whatever is already waiting "
                "for the socket, so it cannot overtake stale frames (vendored "
                "§3, Keepalive). seq is used here as a dedupe key and nothing "
                "else — it cannot detect a gap (§1, §4)."
            ),
        }
        document["events"] = session.events
    return document


def _measurement(
    predicted: Execution | None, realised: Execution, document: dict
) -> dict:
    void = document["reconciliation"]["void"]
    realised_bps = (
        slippage_bps(realised.arrival_mid, realised.vwap, realised.side)
        if realised.filled
        else None
    )
    return {
        "arrival_mid_ticks": realised.arrival_mid,
        "realised_vwap_ticks": realised.vwap if realised.filled else None,
        # Void means void. The number is still written down — under a name that
        # cannot be mistaken for the reported one — because hiding it would be a
        # different kind of dishonesty, and because "the run completed and the
        # measurement was void, for this reason" is still a successful M6 by
        # ARCHITECTURE.md §7's terms. What must never happen is a reconciliation
        # that rescues it: the moment one exists, every number here is an
        # estimate.
        "realised_slippage_bps": None if void else realised_bps,
        "unreported_bps": realised_bps if void else None,
        "predicted_slippage_bps": slippage_bps(
            predicted.arrival_mid, predicted.vwap, predicted.side
        )
        if predicted is not None and predicted.filled
        else None,
        "void": void,
        "reasons": document["reconciliation"]["reasons"],
        "label": "demo, not an evaluation (ARCHITECTURE.md §7)",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _report(document: dict) -> None:
    measurement = document["measurement"]
    ladder = document["ladder"]
    where = (
        f"ladder {ladder['name']} ({ladder['depth_per_side']} per side)"
        if ladder["built_by_client"]
        else "the feeder's own book"
    )
    print(
        f"{document['run']} — {document['label']} — parent "
        f"{document['order']['parent']} {document['order']['side']}, "
        f"{document['grid']['n_bins']} bins, {where}"
    )
    if measurement["predicted_slippage_bps"] is not None:
        predicted = document["predicted"]
        print(
            f"  predicted: filled {predicted['filled']}, VWAP "
            f"{predicted['vwap_ticks']:.2f} ticks, slippage "
            f"{measurement['predicted_slippage_bps']:.4f} bps"
        )
    else:
        print("  predicted: none — this run does not build its own book")
    if "realised" in document:
        realised = document["realised"]
        print(
            f"  realised:  filled {realised['filled']} of {realised['parent']}, "
            f"VWAP {realised['vwap_ticks']} ticks, arrival mid "
            f"{realised['arrival_mid_ticks']}"
        )
        comparison = document.get("comparison")
        if comparison is not None:
            print(
                f"  prediction {'MATCHED' if comparison['matched'] else 'MISSED'} "
                f"across {len(comparison['bins'])} bins"
            )
        if measurement["void"]:
            print(f"  VOID — {'; '.join(measurement['reasons'])}")
            print(
                "  the run reached the end and the measurement is void, which is "
                f"a result. Unreported number: {measurement['unreported_bps']} bps"
            )
        else:
            print(
                f"  arrival slippage {measurement['realised_slippage_bps']:.4f} "
                "bps — a demo, not an evaluation"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "m6_anvil.yaml"))
    parser.add_argument(
        "--run",
        default="ladder",
        help="which run of the committed config: ladder, thin, wide or feeder",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute the prediction from the committed ladder and stop",
    )
    parser.add_argument(
        "--no-write", action="store_true", help="report, but write no artefact"
    )
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    config = load_config(args.config, args.run)
    policy = load_policy(config.checkpoint)
    if policy.n_bins != config.n_bins:
        print(
            f"the policy was trained on {policy.n_bins} bins and the config asks "
            f"for {config.n_bins}. The bin count is part of the observation's "
            "clock, so this is not a free choice."
        )
        return 1

    # The prediction is computed *first*, always, and only where it means
    # something: a run that does not build its own book has no committed ladder
    # to predict from, and inventing one would be the opposite of the point.
    predicted = (
        predict(policy, config.ladder, config.ticker, config.parent, config.side)
        if config.builds_ladder
        else None
    )
    if args.dry_run:
        document = build_document(config, policy, predicted, None, None, None)
        _report(document)
        if not args.no_write:
            _write(config, document, suffix="_prediction")
        return 0

    session = Session(config, policy)
    try:
        realised = session.execute()
    finally:
        if session.stream is not None:
            session.teardown()
    document = build_document(config, policy, predicted, realised, session.tape, session)
    _report(document)
    if not args.no_write:
        _write(config, document)
    comparison = document.get("comparison")
    return 0 if comparison is None or comparison["matched"] else 1


def _write(config: RunConfig, document: dict, suffix: str = "") -> None:
    target = config.results
    if suffix:
        target = target.with_name(target.stem + suffix + target.suffix)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {target.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())
