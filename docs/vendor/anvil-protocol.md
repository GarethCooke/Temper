# Vendored artefact — Anvil wire protocol

| | |
| --- | --- |
| **Artefact** | this file: Anvil's `PROTOCOL.md`, snapshotted verbatim below the rule |
| **Source** | Anvil, `PROTOCOL.md` (bindings: `server/protocol.hpp`, `web/src/protocol.ts`) |
| **Source commit** | `da0d255553930154e0d280419b3292d0486ceaf8` (branch `stage3-instrument`, clean working tree) |
| **File last changed at** | `d501652e7b205e36e0c9e647ef3e720559e9f82d`, 2026-07-07 — Stage 1, per-session order ownership |
| **Snapshot sha256** | `e82ecff7252b945cd7e764524c03c15413be300dbb186e5fd4c01da075fb411c` |
| **Wire version** | `1` |
| **Generated** | 2026-08-22 |
| **Consumed by** | `client/` (M6) |

## Why this exists

M6 makes the trained policy Anvil's third independent client, after the browser UI
and DepthCharge. The boundary is a versioned artefact, not shared code, and **zero
upstream changes** (constitution §7) — the same posture Temper takes toward
FrontierView's goldens, and the reason `docs/vendor/frontierview-goldens.md` sits
beside this file.

**This snapshot is the specification `client/` is written against.** Frame schemas
are not restated in the brief or in code comments, because a copy is a thing that
drifts: a comment describing the `trade` frame would be a second, unversioned
contract that nothing checks. Code cites this file by section.

## Regenerating

Read-only with respect to Anvil, and manual on purpose. There is no exporter: the
artefact is one Markdown file, so an exporter would be a script that copies a file
and gets the provenance table wrong less often than a human does — not a trade
worth making at one snapshot per milestone.

```bash
# from the Anvil checkout, on a clean tree
git rev-parse HEAD && shasum -a 256 PROTOCOL.md
```

Update the table above, replace everything below the rule, and re-read §7 of this
repo's `ARCHITECTURE.md` before changing anything in `client/` in response. A wire
version other than the one recorded here is not a snapshot refresh — it is a
breaking change, and `client/` refuses to start against it.

## Two notes on reading the snapshot

- **Its relative links point at Anvil's tree**, not Temper's: `server/protocol.hpp`
  and `web/src/protocol.ts` resolve in the source repo. They are left verbatim
  rather than rewritten, because an edited snapshot is not a snapshot.
- **The header still names branch `rest-interface`.** The snapshot was taken from
  `stage3-instrument`, four merges later. The document's own status line is stale;
  the commit in the table above is what was read.

## Observed upstream quirks (recorded, not corrected)

Read-only observations from taking the snapshot and reading the bindings beside it,
in the shape of the goldens note's own quirks section. **None of these is a defect
Temper may fix** — Anvil's behaviour is the spec, and `client/` is written against
whichever way it actually goes. Each is cited to the source that settles it.

1. **§1 and §4 contradict each other on `seq`, and §1 is the operative one.** §4
   says a `seq` that is not the expected next value means a frame was missed, so
   the client should drop state and resync. §1 says the opposite — one global
   counter stamps every frame across all tickers, a single-ticker socket therefore
   sees a *sparse* subsequence, and "v1 clients apply frames idempotently and do
   not gap-test". `client/` follows §1.

2. **A gap is not evidence of loss, even on a single-ticker roster.** The engine
   stamps a fresh `seq` per roster ticker per publish tick
   (`server/engine_harness.hpp::publish_books`, ~70 ms) and the broadcaster emits
   on its *own* ~70 ms timer, skipping any book whose `seq` was superseded in
   between (`server/broadcaster.hpp::emit_books`). Two engine publishes between two
   broadcaster ticks means one `seq` is never delivered to anybody. Gaps are
   structural.

3. **Delivered `seq` can decrease.** The broadcaster drains the trade ring on every
   loop iteration but publishes books only on its timer, so a trade stamped *after*
   a book is routinely delivered *before* it. `seq` is a stamp, not a delivery
   order, and nothing in §4 says so.

4. **Ring overflow is unsignalled.** `drain_ring` clears the stale flag without
   emitting anything (`server/broadcaster.hpp`), and §3.4 rules out an `error`
   frame. Since a gap cannot be distinguished from the structural gaps above, a
   dropped trade frame is **undetectable from the stream**. M6 defends against it
   by reconciling total attributed quantity against the parent order at end of run,
   not by watching `seq`.

5. **`GET /api/book`'s `depth` default is documented as top-5 and is not.**
   `ANVIL_BOOK_DEPTH` defaults to `0`, which `server/config.hpp` documents as *all*
   resting levels; `server/main.cpp` passes it straight to
   `EngineHarness::Config::top_n`. The `?depth=` query parameter is parsed and then
   ignored — "the published book governs" (`server/app.hpp`), because
   re-aggregating off the engine thread is forbidden. Convenient for M6, which
   needs to price a marketable limit through the whole ladder, but it is a
   configuration default rather than a protocol guarantee.

6. **`POST /api/feeder` is documented as "forward-declared … not in the v1
   bindings" and is implemented.** `server/app.hpp` serves both `GET` and `POST` on
   `/api/feeder`, returning `{enabled, rate, live}`. M6's local runs use it, and the
   deployment's is left alone.

7. **`summary.last` is the mid only when the book is two-sided.** §2's field table
   gives "book mid … `""` if empty". `server/snapshot.hpp::summarise` also returns
   the *lone best* when only one side is populated, which the table does not say.
   M6 takes its arrival price from the book snapshot for this reason among others —
   see the call site in `client/`.

---

*Everything below this line is Anvil's `PROTOCOL.md`, verbatim at the commit named
above. Do not edit it here.*

---

# Anvil Demo — Wire Protocol

**Status:** canonical contract for the live-demo build (branch `rest-interface`).
**Wire version:** `1`. **Bindings (kept in lockstep, this file is the source of truth):**

| Side   | File                                             | Notes                              |
| ------ | ------------------------------------------------ | ---------------------------------- |
| Server | [`server/protocol.hpp`](server/protocol.hpp)     | C++ structs + hand-rolled writers  |
| Client | [`web/src/protocol.ts`](web/src/protocol.ts)     | TypeScript types + parser/guards   |

Any change here must land in both bindings in the same commit. A breaking change
bumps **wire version** (surfaced by `GET /api/health` so a client can detect a
mismatch on connect).

> Scope note: this is the *demo* transport — an unauthenticated, single-shared-book
> "trading floor". The production order-entry gateway (reliable FIX/binary sessions)
> is a named out-of-scope extension point, not this.

---

## 1. Conventions

- **Transport:** REST over HTTP for request/response; a single WebSocket (`GET /ws`)
  for the server→client event stream. The browser never polls the book in steady
  state — it subscribes once and consumes the stream.
- **Encoding:** all bodies and frames are UTF-8 JSON. WebSocket frames are discrete
  text messages, one complete JSON object each (no framing/newlines of our own).
- **Prices are JSON _strings_**, e.g. `"3.2"`, `"7"`, `"6.9"`. The server serialises
  them through the engine's own `append_price`, so a price on the wire is
  byte-identical to the same price in the CLI's trade/dump output (shortest decimal,
  trailing zeros trimmed). Clients render the string verbatim; parse to a number only
  for chart maths, accepting the rounding that implies.
- **Quantities, counts, `seq`, `ts`** are JSON numbers. All stay well under the
  2⁵³ safe-integer ceiling in any realistic demo run (`MAX_QTY` is 10⁹).
- **Sides** are `"B"` / `"S"`, matching the engine's `AggrSide`.
- **Order ids** are the raw id strings (`"A001"`), decoded from the engine's packed
  key — same charset/length the engine validates (`[A-Za-z0-9-]`, ≤10).
- **`seq`** is a per-connection, monotonically increasing sequence number stamped on
  every server→client frame. See [§4 Reconnect](#4-reconnect--idempotency).
- **Ticker scope:** the protocol is ticker-aware (every book/trade frame names its
  `ticker`). A WebSocket subscribes to **one** ticker (`/ws?ticker=`) and receives
  that ticker's `snapshot`/`book`/`trade`; the cross-ticker `summary` frame (§3.5)
  goes to **every** socket regardless. Switching ticker = reconnect with a new
  `?ticker=`. (Phase 8 made this real across feeder + server + UI; the wire shapes
  for `snapshot`/`book`/`trade` were already ticker-scoped and did not change.)
- **`seq` is a single global line.** One engine-thread counter stamps every frame —
  trades, books and the summary across all tickers — so a socket subscribed to one
  ticker sees a *sparse* subsequence of `seq` (the gaps belong to other tickers'
  frames it never receives). v1 clients apply frames idempotently and do not gap-test,
  so this is benign; a per-ticker `seq` line is the change if strict gap detection is
  ever needed.

---

## 2. REST endpoints

### `GET /api/health`

Liveness + contract check. Never requires a body.

```json
{ "status": "ok", "wireVersion": 1, "uptimeMs": 1234567, "clients": 3 }
```

| Field         | Type     | Meaning                                  |
| ------------- | -------- | ---------------------------------------- |
| `status`      | `"ok"`   | constant when serving                    |
| `wireVersion` | number   | server's wire version (compare to yours) |
| `uptimeMs`    | number   | process uptime, milliseconds             |
| `clients`     | number   | connected WebSocket clients              |

### `GET /api/book?ticker=<id>&depth=<n>`

Current top-N book for one ticker, as a **snapshot-shaped** body (the same parser
handles it as a `snapshot` WS frame). `ticker` selects the book (default
`ANVIL_DEFAULT_TICKER`); `depth` is optional (default top-5 levels per side, matching
the CLI dump). An unknown or quiescent ticker returns empty `bids`/`asks` (not a 404)
— idempotent and simpler for the client.

```json
{ "type": "snapshot", "seq": 42, "ticker": 101,
  "bids": [ { "price": "3.1", "qty": 1500, "orders": 2 } ],
  "asks": [ { "price": "3.2", "qty": 1000, "orders": 1 } ] }
```

### `GET /api/summary`

Cross-ticker roster one-shot (Phase 8): the resting-buy / resting-sell totals and a
`last` (book mid) for **every** ticker, for the initial page load and the summary
view's first paint. Live updates ride the `summary` WS frame (§3.5). Empty `tickers`
before the first publish (not a 404).

```json
{ "tickers": [
  { "ticker": 101, "restingBuy": 1820, "restingSell": 1640, "last": "10.0098" },
  { "ticker": 102, "restingBuy": 900,  "restingSell": 1200, "last": "" }
] }
```

| Field         | Type   | Meaning                                              |
| ------------- | ------ | ---------------------------------------------------- |
| `ticker`      | number | product id                                           |
| `restingBuy`  | number | sum of resting qty across **all** bid levels         |
| `restingSell` | number | sum of resting qty across **all** ask levels         |
| `last`        | string | book mid as a wire decimal ("10.0098"); "" if empty  |

> `restingBuy`/`restingSell` walk **all** levels (a true per-side total), not the
> top-N snapshot. `last` is the book **mid** `(bestBid+bestAsk)/2` — computed
> read-side from the book, zero extra engine state (the implementer's-choice
> alternative to tracking last-trade). Wire-formatted through the engine's
> `append_price`, so it is a JSON **string** like every other price.

### `POST /api/order`

Inject one order. **Body is a raw engine CSV line** (`Content-Type: text/plain`),
fed into the engine's existing `parse_line` — the exact validated path the CLI uses.

**The server owns order-id assignment** (Phase 9). The client never mints an id: it
can't see the global id space (the book snapshot is aggregated *quantity*, not
individual ids) and independent clients can't coordinate, so any client-minted scheme
collides across browser restarts and concurrent users. A single server-side monotonic
allocator mints ids for both the feeder and manual orders.

- **New (`N`)** — the body's **id field is empty** (six fields, the third blank — *not*
  five fields): `101,N,,B,500,10.00`. The server mints an id, splices it into field 3
  before `parse_line`, and returns it as `id`.
- **Cancel (`C`) / Amend (`A`)** — the body carries the **server-assigned id** the
  client received on the New; it passes through unchanged and is echoed back as `id`.

| Type         | Request id field   | Response `id`            |
| ------------ | ------------------ | ------------------------ |
| New (`N`)    | empty              | server-assigned (minted) |
| Cancel (`C`) | server-assigned id | echoed                   |
| Amend (`A`)  | server-assigned id | echoed                   |

The splice replaces field 3's content only — it never adds or removes a field — so a
client that drops the field (five fields) still earns a wrong-column rejection rather
than being silently "repaired". A New rejected for any reason (bad qty/price, wrong
column count) still returns its minted id; the value is simply spent and unused (the
counter only advances — rejects are not reclaimed).

```
POST /api/order
Content-Type: text/plain

101,N,,B,500,10.00
```

Response is an `OrderResult`. **Every engine verdict — accept or reject — returns
`200`** with `{accepted, reason, id}`: the POST was well-formed and reached the engine, so
a reject (even of a garbage CSV line) is a *business outcome in the body*, not an
HTTP-layer error. This mirrors how order entry actually works — a FIX session accepts
the message and the rejection comes back as an execution report, not a session-level
error; the client reads `accepted` from the body. Only genuine *non-verdicts* are
non-2xx, so a client can tell "the engine answered" from "it never got there":
**503** `{"reason":"engine busy"}` when the inbound queue is full and **504**
`{"reason":"engine timeout"}` when the engine did not answer in time (Crow itself
still returns **400** for malformed HTTP and **403** for a blocked origin). Any
resulting trades and book changes are observed asynchronously on the WebSocket stream
— they are **not** in this response.

```json
{ "accepted": true, "id": "o1" }
{ "accepted": false, "reason": "out-of-bounds price", "id": "o2" }
```

#### Session & ownership (Stage 1)

`POST /api/order` establishes a session on first contact. If the request carries no
`anvil_session` cookie, the server mints an opaque 128-bit token and returns it:

```
Set-Cookie: anvil_session=<32-hex>; HttpOnly; SameSite=Lax; Path=/
```

(with `; Secure` added under TLS — `ANVIL_SESSION_COOKIE_SECURE=true`). **Possession of
the cookie is the ownership principal — there is no login and no server-side session
store.** A **New** order is recorded to the calling session. A **Cancel** or **Amend**
is accepted only for an order the calling session *owns*; for any other order — another
session's, **or an unknown id** — the response is

```json
{ "accepted": false, "reason": "cannot modify another participant's order", "id": "o5" }
```

at **`200`** (an ownership reject is a business verdict like any other engine reject, not
an HTTP error), and is **rejected before the engine processes it**. The reason is
**uniform** for "not owner" and "unknown id", so it reveals nothing about which ids are
live — an enumeration walk (`C,o1; C,o2; …`) from a session that owns none of them earns
the same reject on every line. Order ids remain server-minted and sequential (§ above);
the **cookie, not the id, is the ownership boundary**, so guessing an id you do not own
buys nothing. The browser sends and receives the `HttpOnly` cookie transparently on
same-origin requests, so a normal client — which only ever cancels its own ids — sees no
behavioural change.

> The `/ws` stream is read-only market data and is **session-agnostic**: ownership is a
> `POST /api/order` concern exclusively. The feeder's own orders are submitted on a
> trusted internal path (a reserved system principal) that bypasses ownership, so a
> client can never cancel a feeder (`f`-prefixed) id.

### `POST /api/feeder` — _forward-declared (Phase 4), not in the v1 bindings_

Viewer control for the server-side dummy-order feeder.

```json
// request
{ "action": "start", "rate": 30 }
// response
{ "running": true, "rate": 30 }
```

> **Rate is clamped server-side.** A requested `rate` is bounded to
> `ANVIL_FEEDER_MAX_RATE` (default `2000/s`) inside `BasicFeeder::set_rate()` before it
> takes effect, so the response `rate` may be lower than requested. The ceiling is a
> safety cap: `ANVIL_FEEDER_MAX_RATE` = 0 or absent keeps the default ceiling — the cap
> cannot be disabled. The feeder and genuine manual orders share the one bounded inbound
> queue; the cap stops synthetic flow from `503`-ing real orders. See
> `docs/ARCHITECTURE.md` and Stage 0.

---

## 3. WebSocket stream — `GET /ws`

### Handshake

Connect with `GET /ws?ticker=<id>&since=<seq>` (HTTP upgrade). `ticker` selects the
subscribed ticker (v1: required, single ticker). `since` is **reserved** for
sequence-based replay; v1 has no replay buffer, so the server always resyncs by
sending a fresh `snapshot` regardless of `since`.

On connect the server sends exactly one **`snapshot`** (establishing the `seq`
baseline and the full visible book), then streams `book` and `trade` frames live.

### Envelope

Every frame is a JSON object with a `type` discriminator and a `seq`:

```ts
{ "type": "snapshot" | "book" | "trade", "seq": <number>, ... }
```

> **The v1 WS stream carries `snapshot` / `book` / `trade` only — no `error` frame.**
> A rejected order's verdict is the `POST /api/order` response, not a broadcast: a
> shared market-data feed shouldn't carry one participant's input errors to every
> watcher. Stream-integrity loss (a dropped frame on overflow) is signalled
> structurally by a gap in `seq` (see [§4](#4-reconnect--idempotency)), not by an
> error frame. The `error` shape below is retained as a **reserved** type — both
> bindings still parse it defensively — for the documented override in which the
> server *deliberately* broadcasts engine rejects (`WsPublishSink::kEmitErrorFrames`).

### 3.1 `snapshot`

Authoritative on-connect / resync baseline for one ticker. Applying it **fully
replaces** the client's view of that ticker's book. `GET /api/book` returns this
same shape.

```json
{"type":"snapshot","seq":1,"ticker":101,"bids":[{"price":"3.1","qty":1500,"orders":2},{"price":"3","qty":800,"orders":1}],"asks":[{"price":"3.2","qty":1000,"orders":1},{"price":"3.3","qty":2200,"orders":3}]}
```

| Field             | Type          | Meaning                                            |
| ----------------- | ------------- | -------------------------------------------------- |
| `seq`             | number        | baseline sequence number for this stream           |
| `ticker`          | number        | the ticker this book is for                        |
| `bids`            | `LevelView[]` | top-N levels, **best-first** (highest price first) |
| `asks`            | `LevelView[]` | top-N levels, **best-first** (lowest price first)  |
| `LevelView.price` | string        | wire decimal at this level                         |
| `LevelView.qty`   | number        | summed resting quantity at this price              |
| `LevelView.orders`| number        | count of resting orders at this price              |

> The aggregate `qty`/`orders` are computed by the read-side helper that walks each
> level's FIFO. The engine's `Level` stores **no** running total — a deliberate
> omission documented in the engine README; the snapshot helper is exactly the kind
> of consumer that would justify adding one later.

### 3.2 `book`

A coalesced top-N **refresh** for one ticker, published on the server's ~10–15 Hz
tick. Identical payload to `snapshot`; it carries the latest full top-N (not a
delta), so it is idempotent — apply it as a full replace of the ticker's book.

```json
{"type":"book","seq":7,"ticker":101,"bids":[{"price":"3.1","qty":1500,"orders":2}],"asks":[{"price":"3.2","qty":1000,"orders":1}]}
```

### 3.3 `trade`

One fill, streamed **individually** (never coalesced) so the trade tape is complete.
`price` is the resting (maker) order's price — the trade price, per the settled
matching semantics.

```json
{"type":"trade","seq":8,"ticker":101,"price":"3.2","qty":400,"aggr":"B","takerId":"A002","makerId":"A001","ts":1718480000000}
```

| Field     | Type   | Meaning                                  |
| --------- | ------ | ---------------------------------------- |
| `ticker`  | number | ticker                                   |
| `price`   | string | resting order's price = trade price      |
| `qty`     | number | fill quantity                            |
| `aggr`    | `"B"`/`"S"` | aggressor side                      |
| `takerId` | string | aggressor (incoming) order id            |
| `makerId` | string | resting order id that was filled         |
| `ts`      | number | server wall-clock, epoch milliseconds    |

### 3.4 `error` — reserved, *not emitted by the v1 server*

The v1 WS stream does **not** broadcast `error` frames (see the note under
[Envelope](#envelope)): a rejected `POST /api/order` is the POST's HTTP response, and
overflow loss surfaces as a `seq` gap, not a frame. The shape is retained here as a
**reserved** type — both bindings still parse it — for the documented override in
which the server deliberately broadcasts engine rejects
(`WsPublishSink::kEmitErrorFrames`). When emitted, `raw` and `ticker` are omitted
when absent.

```json
{"type":"error","seq":9,"code":"rejected","message":"out-of-bounds price","raw":"101,N,A003,B,1,200000","ticker":101}
```

| Field     | Type   | Meaning                                                    |
| --------- | ------ | --------------------------------------------------------- |
| `code`    | string | machine code: `"resync"` \| `"rate_limited"` \| `"rejected"` … |
| `message` | string | human-readable reason (an engine reason string forwards here) |
| `raw`     | string | offending input line, if any (omitted when absent)        |
| `ticker`  | number | ticker scope, if any (omitted when absent)                |

---

## 4. Reconnect & idempotency

- `seq` is per-connection and increments by 1 per frame, starting at the `snapshot`'s
  `seq`. A client tracks the last `seq` it applied.
- **Gap detection:** if a received `seq` is not the expected next value, the client
  has missed a frame → drop local state, reconnect, and take the fresh `snapshot` as
  the new baseline.
- **Idempotent book frames:** `snapshot` and `book` both carry the full top-N, so
  reapplying one is harmless — it is a full replace of the ticker's visible book.
- **Trade tape:** `trade` frames are append-only, keyed by `seq`; on a reconnect
  overlap, dedupe by `seq`.
- v1 has no server-side replay buffer; resync = a fresh `snapshot`. A bounded replay
  window keyed by `seq` is a natural later addition that needs no protocol change.

---

## 5. Phase status

| Area                                   | Status (branch `rest-interface`) |
| -------------------------------------- | -------------------------------- |
| Types + serialisers (both bindings)    | **Phase 0 — done**               |
| `EventSink`/`FillEvent` engine seam    | **Phase 1 — done**               |
| Engine thread, coalesced publication   | **Phase 2 — done**               |
| Crow REST + WS transport               | **Phase 3 — done**               |
| Feeder + `POST /api/feeder`            | Phase 4                          |
| Browser client consuming this contract | Phase 5                          |
