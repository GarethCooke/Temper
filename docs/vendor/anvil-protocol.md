# Vendored artefact — Anvil demo wire protocol

| | |
| --- | --- |
| **Artefact** | this file: a **verbatim** snapshot of Anvil's `PROTOCOL.md`, header prepended |
| **Source** | Anvil, `PROTOCOL.md`, branch **`main`** — *not* `rest-interface`; see the note below |
| **Source commit** | `4801ed8d8b09b62ec4fcee8e68280f16b3c4780c` (clean working tree), committed 2026-08-17 |
| **Source SHA-256** | `350f9c701bb4548edfd0ad2015a1d1bd71e92618c02cccbe338786595f93ef01` — 38,166 bytes, LF endings |
| **Wire version** | `1`; the client refuses to start against any other `GET /api/health` value |
| **Last source change** | `864ee2f` *feat(server): publish a real last-traded price in the summary row* — the `summary.last` semantic change described below and recorded in the snapshot's own header |
| **Vendored** | 2026-08-22, for M6 (`docs/briefs/M6-anvil-live-leg.md`, task 0) |
| **Consumed by** | `client/`, and `tests/test_vendored_protocol.py`, which re-hashes the body |

> **The snapshot's own status line is stale; the commit in the table above is
> what was read.** The vendored body opens with *"canonical contract for the
> live-demo build (branch `rest-interface`)"*, and that branch does not contain
> this snapshot: `git branch -a --contains 4801ed8` returns only `main`, and
> `origin/rest-interface` still points at `6ac288e`, the Phase-0 commit that
> created the file. The same is true of the last source change, `864ee2f`. The
> branch name is a label Anvil has outgrown; the **commit hash and the body
> digest** are what identify this artefact, and they are what the test checks.
> Recorded rather than corrected in the body, because the body is verbatim and
> stays that way — an upstream document is allowed to be stale about itself, and
> a vendoring that quietly fixed it would no longer be a snapshot.

## Why this exists

Anvil is the second contract Temper consumes without changing it, after
FrontierView's goldens (constitution §7: *zero upstream changes*, and the
boundary is a versioned artefact rather than shared code). The two are consumed
the same way and for the same reason — a claim that rests on an upstream
behaviour has to name the revision of that behaviour it was made against.

The difference is what the artefact *is*. FrontierView's boundary is numbers, so
the vendored artefact is a fixture and a differential test compares against it.
Anvil's boundary is prose, so the vendored artefact is the prose, and there is no
mechanical differential to be had: what the snapshot buys is that a client
written today can be read tomorrow against exactly the document it was written
from, rather than against whatever `PROTOCOL.md` says by then.

**The vendored file is the spec.** `client/` does not restate frame schemas in
comments or docstrings: a copy drifts, and a drifted copy is worse than no copy
because it reads as authority. Code comments cite this file by section instead.

## Re-vendoring

Read-only with respect to Anvil, like every other boundary here:

```bash
# from an Anvil checkout, clean:
git -C /path/to/Anvil rev-parse HEAD
sha256sum /path/to/Anvil/PROTOCOL.md
# then replace everything below the BEGIN marker and update the header above
```

`tests/test_vendored_protocol.py` re-hashes the body against the digest recorded
in this header, so an in-place edit of the snapshot goes red. Re-vendoring is
therefore a deliberate act with a fresh digest, never a quiet correction — and a
wire-version bump upstream turns the client's own health check red before any
order is sent, which is the other half of the same property.

## What M6 depends on, and what it must not

Everything below is Anvil's; this section is Temper's reading of it, and exists
so the client can cite a section instead of paraphrasing a schema.

| Depends on | Section |
| --- | --- |
| `GET /api/health` reports `wireVersion`, and a mismatch is detectable before anything is sent | §2 |
| Order entry is a raw engine CSV line, six fields, **id field empty** on a New; the server mints the id and returns it | §2 `POST /api/order` |
| Every engine verdict — accept *or* reject — is `200` with `{accepted, reason, id}`; only `503`/`504`/`400`/`403` are non-verdicts | §2 `POST /api/order` |
| `anvil_session` is the ownership principal, minted on first contact | §2, *Session & ownership* |
| One socket per ticker; `snapshot` and `book` are full replaces, applied idempotently | §3, §3.1, §3.2, §4 |
| `trade` carries `takerId` and `makerId`, so a client's own fills are directly attributable | §3.3 |
| The book mid is read off a `snapshot`/`book` frame | §3.1 |
| **The published book is the whole book** — the client prices each bin at the last level it can see, which is a full sweep only while `ANVIL_BOOK_DEPTH` is `0`. See the note below: this is server *configuration*, not a protocol guarantee | §2 `GET /api/book`, §3, *depth* |

| Must not | Section |
| --- | --- |
| Reason about **ordering** from `seq`, or compute a "next expected `seq`" — it is one global engine-thread stamp, so a per-ticker socket sees a sparse subsequence that can step *backwards* | §1, §4 |
| Detect a dropped frame at all: ring-overflow loss is **not signalled on the wire** — no error frame, no detectable gap | §3.4, §4 |
| Wait for a server ping or a heartbeat frame — the server never initiates either | §3, *Keepalive* |
| Expect an `error` frame — the v1 stream carries `snapshot`/`book`/`trade`/`summary` only | §3, *Envelope*, §3.4 |
| Take the arrival price from `summary.last` — that field **was** the book mid and is now the last *traded* price, changed inside wire version `1` with no bump because the shape did not move. `""` means "has not traded yet", and it persists after the book empties | header, §2 `GET /api/summary`, §3.5 |
| Infer that an order executed from `accepted: true` — Anvil has no market orders, and a mispriced limit **rests** rather than erroring | §2 `POST /api/order` |

The last two are the ones M6's brief calls out as returning plausible wrong
numbers rather than crashing, which is why they are listed as prohibitions
rather than as notes.

### Two things the tables above understate

**A single-ticker socket sees `seq` gaps too, and not because of other tickers.**
§1 explains sparseness as the gaps belonging to other tickers' frames, which
reads as "irrelevant on a one-ticker roster" — and M6 runs exactly one ticker.
The mechanism survives anyway: the engine publishes its coalesced book on a
70 ms deadline (`server/engine_harness.hpp:96`, `coalesce = Millis{70}`) and the
broadcaster samples that slot on its *own* independent 70 ms tick
(`server/broadcaster.hpp:84`). The two are unsynchronised, so a generation the
engine stamped can be overwritten before the broadcaster ever reads it, and that
`seq` is delivered to nobody. The prohibition is unchanged and this is why it is
not merely a multi-ticker artefact: there is no roster small enough to make `seq`
a gap detector.

**Whole-ladder pricing depends on an environment variable.** `ANVIL_BOOK_DEPTH`
(`server/config.hpp:38`) caps the levels per side the server publishes; it
defaults to `0`, meaning every resting level, and the deployed configuration
leaves it there. The client's "price through the far side of the book just
observed" is therefore a genuine full sweep *by operator configuration rather
than by contract*. Set it non-zero and the client would price to the deepest
**published** level while resting depth continued below it — bins would fill
short for a reason invisible in the artefact, since the run's `operator_note`
records `ANVIL_TICKERS`, `ANVIL_DEFAULT_TICKER`, `ANVIL_FEEDER` and `ANVIL_PORT`
and not this. Recorded here rather than defended in code: the measured runs ran
at the default, and a client that tried to detect the truncation could not — a
truncated book is indistinguishable from a shallow one on the wire.

---

<!-- BEGIN VENDORED PROTOCOL.md — verbatim from the commit named above. Do not edit below this line. -->

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

**Semantic change within wire version `1` — no bump, no client change.** The summary
row's `last` was the **book mid** `(bestBid+bestAsk)/2`; it is now the ticker's **last
traded price** ([`GET /api/summary`](#get-apisummary), [§3.5](#35-summary)). The wire
**shape is unchanged** — same `last` key, same JSON *string* type, same `""` sentinel —
so no client parsing changes and the version does **not** bump. Only the *meaning* of
the value moved: `""` now means *this ticker has not traded yet* rather than *the book
is empty*, and once set the price **persists after the book empties**. Clients that
merely render the string need no edit; clients that *reasoned* about it as a mid
(deriving a spread from it, or assuming a price exists whenever the book is two-sided)
do.

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
- **`seq`** is a **single global engine-thread stamp** carried on every server→client
  frame — *not* a per-connection counter, and not an ordering oracle. See the "single
  global line" note below and [§4 Reconnect](#4-reconnect--idempotency).
- **Ticker scope:** the protocol is ticker-aware (every book/trade frame names its
  `ticker`). A WebSocket subscribes to **one** ticker (`/ws?ticker=`) and receives
  that ticker's `snapshot`/`book`/`trade`; the cross-ticker `summary` frame
  ([§3.5](#35-summary)) goes to **every** socket regardless. Switching ticker =
  reconnect with a new `?ticker=`. (Phase 8 made this real across feeder + server +
  UI; the wire shapes for `snapshot`/`book`/`trade` were already ticker-scoped and
  did not change.)
- **`seq` is a single global line.** One engine-thread counter stamps every frame —
  trades, books and the summary across all tickers — so a socket subscribed to one
  ticker sees a *sparse* subsequence of `seq` (the gaps belong to other tickers'
  frames it never receives). It can also **step backwards**. The broadcaster delivers
  from two independent sources and does **not** merge-sort them: the trade ring is
  drained continuously (each fill stamped at *generation*), while the coalesced
  book/summary slots are sampled on the ~14 Hz tick (each stamped at *publish*). A
  frame from either source can therefore be delivered ahead of a lower-`seq` frame
  still queued on the other — a `book` delivered ahead of a lower-`seq` `trade` still
  draining from the ring, or a just-drained `trade` delivered ahead of the lower-`seq`
  `book` that was stamped before it. Both properties make `seq` **unusable for
  per-ticker gap detection**; clients apply frames idempotently and snapshot-heal, so
  this is benign. `seq` values are still globally *unique*, so
  they remain valid as a reconnect watermark and for dedupe (§4) — just never for
  ordering. A delivery-order per-ticker `seq` (a broadcaster merge-sort, or
  per-ticker counters) is the change if strict gap detection is ever wanted — no
  current client needs it.

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

Current book for one ticker, as a **snapshot-shaped** body (the same parser handles it
as a `snapshot` WS frame). `ticker` selects the book (default `ANVIL_DEFAULT_TICKER`).
`depth` is optional and caps the levels per side; **absent or `0` means every level the
server publishes** (`ANVIL_BOOK_DEPTH`, itself `0` = all resting levels by default). An
unknown or quiescent ticker returns empty `bids`/`asks` (not a 404) — idempotent and
simpler for the client.

`depth` only ever **truncates** the published book — see
[§3 `depth`](#depth--per-socket-book-depth) for the full semantics, which are identical
on both surfaces. A value larger than the published depth returns what exists rather
than an error, and a negative or unparseable value is treated as absent.

```json
{ "type": "snapshot", "seq": 42, "ticker": 101,
  "bids": [ { "price": "3.1", "qty": 1500, "orders": 2 } ],
  "asks": [ { "price": "3.2", "qty": 1000, "orders": 1 } ] }
```

### `GET /api/summary`

Cross-ticker roster one-shot (Phase 8): the resting-buy / resting-sell totals and the
**last traded price** for **every** ticker, for the initial page load and the summary
view's first paint. Live updates ride the `summary` WS frame ([§3.5](#35-summary)).
Empty `tickers` before the first publish (not a 404).

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
| `last`        | string | last **traded** price; `""` until first trade        |

> `restingBuy`/`restingSell` walk **all** levels (a true per-side total), not the
> top-N snapshot. `last` is the ticker's **last traded price** — the resting (maker)
> price of the most recent fill, byte-identical to the `price` on the
> [`trade`](#33-trade) frame that carried it. It is recorded by a `LastTradeSink` tee
> child on the engine thread (`server/last_trade.hpp`), **not** derived from the book,
> so the matching core in `src/` stays byte-for-byte unchanged. Wire-formatted through
> the engine's `append_price`, so it is a JSON **string** like every other price.
>
> Two consequences, both deliberate and both different from the book mid this replaced:
>
> - `last` is `""` **until a ticker's first trade** — even when the book already has
>   resting orders on both sides. `""` means *has not traded yet*, **not** *empty book*.
> - Once set it **persists after the book empties**. A trade is a historical fact,
>   independent of current book state; the old mid vanished with the book.

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

Connect with `GET /ws?ticker=<id>&depth=<n>&since=<seq>` (HTTP upgrade). `ticker`
selects the subscribed ticker (v1: required, single ticker). `depth` is optional and
caps this socket's book depth (below). `since` is **reserved** for sequence-based
replay; v1 has no replay buffer, so the server always resyncs by sending a fresh
`snapshot` regardless of `since`.

On connect the server sends exactly one **`snapshot`** (establishing the `seq`
baseline and the full visible book), immediately followed by one **`summary`**
([§3.5](#35-summary)) seeding the cross-ticker roster, then streams `book`, `trade`
and `summary` frames live.

#### `depth` — per-socket book depth

`depth` caps the price levels per side in **this socket's** `snapshot` and `book`
frames. It is negotiated once, on the upgrade request, and cannot be changed without
reconnecting.

| `?depth=` | Levels per side served |
| --------- | ---------------------- |
| absent, or `0` | every level the server publishes — **the default, unchanged** |
| `n` ≤ published depth | **at least** `n`, best-first — `n` rounded up to the next supported tier |
| `n` > published depth | every published level (no error, no padding) |
| negative / unparseable | treated as absent |

**Depth is served in tiers, so you may receive slightly more than you asked for.** The
request is rounded **up** to the next supported depth — `1, 2, 3, 5, 8, 10, 15, 20, 30,
40, 50, 75, 100, 150, 200, 300, 500, 1000`, then unlimited. A client asking for 27 is
served 30. A client needing exactly `n` must truncate locally; you will never be served
*fewer* than you asked for (except where the book simply has fewer levels).

This exists to bound server work: the fan-out formats one frame per *distinct depth in
use*, so free-form depths would let unauthenticated clients dictate how many distinct
frames the server formats per tick. Tiers cap that at a constant no matter how many
sockets connect.

- **It only ever truncates.** The server publishes each ticker at `ANVIL_BOOK_DEPTH`
  (`0` = all resting levels, the deployed default) and a socket slices that published
  view at serialise time. A socket therefore **cannot ask for more depth than the
  server publishes** — deepening would require re-aggregating against live engine
  state, which happens only on the engine thread.
- **It changes the payload, not the stream.** Frame types, cadence, `seq` values and
  the `trade`/`summary` frames are identical on every socket regardless of `depth`; a
  shallow socket receives the same frame *sequence* as a deep one, carrying fewer
  levels. `trade` frames have no depth. The `summary` frame is cross-ticker and
  unaffected.
- **Truncation is a prefix**, so the levels a shallow socket receives are exactly the
  first `n` a deep socket receives — the two views never disagree about a level they
  both carry, and `book` frames stay idempotent full replaces *of the depth requested*
  ([§4](#4-reconnect--idempotency)).
- **Same semantics on `GET /api/book?depth=`**, deliberately: one parameter, one
  meaning, whichever surface you read the book from.

> **Why it exists.** The `book` frame dominates this stream — on the deployed feed it
> is ~98% of all bytes sent, at ~8.4 KB per frame carrying ~200 levels, ~14 times a
> second. A client that renders a shallow ladder (an embedded display, a mobile view, a
> tile) otherwise pays for ~200 levels and discards most of them on arrival. At
> `depth=27` the frame falls to ~2.2 KB — the stream to roughly 28% of its full-depth
> size.
>
> Server-side, the **formatting** work is shared: a ticker's book is serialised once per
> distinct depth in use (in practice one or two — full-depth browsers, one shallow tier
> for boards), not once per socket. Delivery is not shared — each connection still
> receives its own copy of the frame into its own send buffer — so this bounds
> serialisation cost, not bytes on the wire or per-socket memory.

#### Keepalive — client pings are answered, and they measure stream freshness

**The server never initiates a WebSocket ping, and there is no dedicated heartbeat frame
in v1.** A client that waits for either waits forever.

What the server does send unprompted is the **data stream itself, on fixed timers that do
not depend on order flow**: one `book` frame per subscribed ticker on the ~14 Hz publish
tick, and one `summary` frame on the `ANVIL_SUMMARY_HZ` cadence ([§3.5](#35-summary)).
Both continue on a completely idle book — one with nothing resting, nothing arriving and
nothing trading — because the engine thread's publish deadlines are timers, not activity
hooks. A quiet socket is therefore **not** the normal appearance of a quiet market; see
[§3.5](#35-summary) for what may and may not be concluded from silence.

Client-initiated pings are always answered:

- A **ping (opcode `0x9`) is answered with a pong** carrying the same payload,
  unconditionally. No application code is involved, there is no rate limit, and no
  maximum payload size is configured.
- **The pong is queued behind whatever is already waiting for that socket.** It is
  appended to the same per-connection send buffer as `book` and `trade` frames and does
  not jump the queue.

That second property is the useful one, and it makes a client ping the **recommended
way to measure freshness**. Because a pong cannot overtake queued frames, a ping
round-trip measures *true end-to-end stream freshness* rather than mere TCP liveness: a
socket that is 110 seconds behind ([§4](#4-reconnect--idempotency)) gets its pong back
110 seconds late. A transport-level keepalive would answer promptly on a badly
backlogged stream and tell the client nothing — which is exactly the trap the
[slow-consumer note](#slow-consumers--the-server-queues-it-does-not-shed) warns about.

The server does **not** use pongs to detect dead clients, and applies no
application-level idle timeout. The only upper bound on an idle connection is nginx's
`proxy_read_timeout` (3600s in the deployed configuration).

**The ordering is measured, not merely read from the source.** `tests/tools/pong_ordering_probe.py`
opens a raw socket, completes the WebSocket handshake, then deliberately **stops reading**
so the kernel receive buffer fills and frames genuinely back up in the server's
per-connection send queue. It then sends one ping and reports the pong's *position in the
byte stream* — which round-trip timing alone cannot do, since a slow reader inflates the
RTT whether or not the server reordered anything. Captured against the Crow build:

```
not reading for 6.0s to induce backpressure...
drained 425,890 bytes -> 149 frames
pong found at frame index 148
  data frames delivered BEFORE the pong: 148  (425,364 payload bytes)
  frames delivered AFTER the pong:       0
```

The pong arrived **last**, behind every frame queued ahead of it. Re-run it after any
change to the egress path.

> **Implementation note for whoever upgrades Crow.** This behaviour comes from the
> vendored Crow build — its frame handler answers opcode `0x9` directly — not from
> Anvil's own code. The build pins `crow_all.h` by SHA256, so it cannot change silently,
> but a Crow upgrade is an upgrade of a **documented wire promise**: re-run the probe
> above rather than assuming it survived.

### Envelope

Every frame is a JSON object with a `type` discriminator and a `seq`:

```ts
{ "type": "snapshot" | "book" | "trade", "seq": <number>, ... }
```

> **The v1 WS stream carries `snapshot` / `book` / `trade` only — no `error` frame.**
> A rejected order's verdict is the `POST /api/order` response, not a broadcast: a
> shared market-data feed shouldn't carry one participant's input errors to every
> watcher. Stream-integrity loss (a fill dropped on ring overflow) is **not signalled
> on the wire at all** in v1 — not by an error frame, and not by a detectable `seq`
> gap, since a single-ticker socket's `seq` subsequence is already sparse and
> non-monotonic (§1). The book self-heals from the next full-replace `book`/`snapshot`
> and the trade tape is best-effort; see [§4](#4-reconnect--idempotency). The `error`
> shape below is retained as a **reserved** type — both bindings still parse it
> defensively — for the documented override in which the
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
| `seq`             | number        | reconnect-watermark baseline (global stamp — §1)   |
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
>
> **"top-N" is per socket.** N is `min(?depth=, ANVIL_BOOK_DEPTH)`, where `0` on either
> means "no limit from this one" — see [`depth`](#depth--per-socket-book-depth). Two
> sockets on the same ticker can legitimately receive different-length `bids`/`asks`
> for the same publish generation; both are correct and agree level-for-level as far as
> the shallower one goes.

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
overflow loss is not signalled on the wire at all ([§4](#4-reconnect--idempotency)) —
the `"resync"` code below is reserved for it, not emitted. The shape is retained here
as a **reserved** type — both bindings still parse it — for the documented override in
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

### 3.5 `summary`

The cross-ticker roster, delivered to **every** socket regardless of which ticker it
subscribes to ([§1](#1-conventions)): once on connect — right after the `snapshot`, so
a socket can paint its roster without waiting out a cadence tick — and thereafter on
the slow `ANVIL_SUMMARY_HZ` cadence. Coalesced like `book`: each frame carries the
current state of the whole roster, so applying one is an idempotent **full replace** of
the client's roster view — never a delta.
[`GET /api/summary`](#get-apisummary) returns the same rows without the `type`/`seq`
envelope, for the first paint before the stream opens.

**Emission is timer-driven, not activity-driven — this part is a guarantee.** `summary`
is published by the engine thread on a fixed deadline (`EngineHarness::run` →
`publish_summary`), checked every loop iteration alongside the book deadline and reached
by the inbound queue's `wait_until` timeout, so it fires whether or not a message ever
arrives. Order flow determines the *contents*, never the *timing*: an idle book — nothing
resting, nothing arriving, nothing trading — keeps publishing, and each publish stamps a
fresh `seq`, so consecutive frames differ in `seq` even when the roster does not. Measured
with the feeder stopped against a static book: 62 consecutive frames over 31 s of zero
order flow, `tickers` bodies byte-identical throughout; an empty, never-traded book gives
the same with all-zero rows.

**The interval is not part of that guarantee.** It is operator configuration
(`ANVIL_SUMMARY_HZ`), tunable per deployment, and is deliberately not stated here as a
value a client may rely on. What is promised is that frames keep coming when nothing is
happening. **A client that needs a staleness threshold must derive it from the cadence it
observes on the connection it is using**, not from a number read out of this document.

Three further properties bear on any use of this frame as a liveness signal:

- **Coalescing is global, never per socket.** If the broadcaster falls behind the publish
  cadence it emits only the latest roster, so a client sees a *lower rate* of frames
  rather than a gap it could detect. A coalesced summary is still a live signal.
- **A frame is never dropped for one slow socket** — it is queued behind that socket's
  backlog ([§4](#4-reconnect--idempotency)). Arrival therefore proves the server was
  alive when the frame was *generated*, not that it is fresh: on a backlogged socket a
  `summary` can arrive minutes after its contents were true. **Cadence is a liveness
  signal, not a freshness signal** — freshness still requires a client ping timed
  against your own clock ([§4](#slow-consumers--the-server-queues-it-does-not-shed)).
- **There is one unexplained counterexample.** On 2026-08-16 the WS endpoint went silent
  for 2 min 56 s on a live TCP connection and then resumed on that same connection; the
  cause is unresolved, and both leading internal hypotheses (fan-out head-of-line
  blocking, engine-thread publish starvation) would silence `summary` along with
  everything else. A client that treats absence as loss of service must either size its
  threshold above that observation or accept the false positive.

```json
{"type":"summary","seq":12,"tickers":[{"ticker":101,"restingBuy":1820,"restingSell":1640,"last":"10.0098"},{"ticker":102,"restingBuy":900,"restingSell":1200,"last":""}]}
```

| Field                   | Type           | Meaning                                       |
| ----------------------- | -------------- | --------------------------------------------- |
| `seq`                   | number         | global engine-thread stamp, not per-socket    |
| `tickers`               | `SummaryRow[]` | one row per roster ticker                     |
| `tickers[].ticker`      | number         | product id                                    |
| `tickers[].restingBuy`  | number         | sum of resting qty across **all** bid levels  |
| `tickers[].restingSell` | number         | sum of resting qty across **all** ask levels  |
| `tickers[].last`        | string         | last **traded** price; `""` until first trade |

> `last` is the ticker's **last traded price**, identical in meaning and formatting to
> the REST surface — see [`GET /api/summary`](#get-apisummary) for the full semantics.
> In short: it is the resting (maker) price of that ticker's most recent fill, it is
> `""` until the ticker's **first** trade (a two-sided book that has never traded still
> reports nothing), and it **persists after the book empties**. It is *not* a book mid,
> and it is not derived from the book at all — the server records it off the engine's
> fill stream.

---

## 4. Reconnect & idempotency

- `seq` is the **single global engine-thread stamp** of [§1](#1-conventions), not a
  per-connection counter. A single-ticker socket sees a *sparse and non-monotonic*
  subsequence, so a client **cannot compute a "next expected `seq`"**. Track the
  last-applied `seq` only as the reconnect watermark (below), never for ordering.
- **Recovery is transport-driven, not `seq`-driven.** A reconnect is triggered by the
  socket closing, **never** by an unexpected `seq`. On reconnect the fresh `snapshot`
  is the new baseline; discard any buffered frame with `seq ≤` the snapshot's `seq`.
  Because every `snapshot`/`book` is a full replace, the book self-heals and a missed
  frame needs no client-side detection.
- **Idempotent book frames:** `snapshot` and `book` both carry the full top-N, so
  reapplying one is harmless — it is a full replace of the ticker's visible book.
- **Trade tape:** `trade` frames are append-only; on a reconnect overlap, dedupe by
  `seq`. This stays valid because `seq` values are globally *unique* even though they
  do not arrive in order — dedupe is a set membership test, not a comparison.
- **Ring-overflow loss is not signalled on the wire in v1.** When the engine→broadcaster
  ring is full the fill is dropped and a server-side `stale` latch is set; the
  broadcaster clears that latch without emitting anything. There is no `resync` frame,
  and the resulting `seq` gap is **not** client-detectable (per §1 the subsequence is
  already sparse and non-monotonic, so a gap is indistinguishable from another
  ticker's frame). The book is unaffected — the next `book`/`snapshot` is a full
  replace — and the **trade tape is best-effort**: dropped fills are lost silently. The
  reserved `error` frame with code `"resync"` ([§3.4](#34-error--reserved-not-emitted-by-the-v1-server))
  is the wire shape held for making this explicit; the v1 server does not emit it.
- v1 has no server-side replay buffer; resync = a fresh `snapshot`. A bounded replay
  window keyed by `seq` is a natural later addition that needs no protocol change.

### Slow consumers — the server queues, it does not shed

A client that consumes slower than the server publishes gets **every frame, late**. It
does *not* get a thinned, sampled or per-socket-coalesced stream. State this plainly
because the opposite is the intuitive guess and it is wrong:

- **There is no per-socket backpressure policy of any kind.** `Broadcaster::deliver()`
  calls Crow's `conn.send_text()`, which posts the frame onto that connection's
  io_context and appends it to the connection's private `write_buffers_`
  (`std::vector<std::string>`), drained by `do_write()`. On that path there is **no
  cap, no drop rule, no coalescing and no disconnect threshold**. The queue is
  **unbounded in v1** — it grows for as long as the client stays connected and behind.
- **The coalescing that does exist is upstream and global, not per socket.** The
  ~14 Hz book tick ([§3.2](#32-book)) collapses many book changes into one frame *for
  everybody*; that is what bounds egress independent of match rate. A slow socket is
  offered exactly the same frame sequence as a fast one and simply falls behind in it.
- **Therefore lag is unbounded and grows linearly, with no plateau.** A client draining
  at a fraction *f* of full rate accumulates staleness at `(1 − f)` seconds per second
  for as long as it is connected. Measured externally against a matched wire `seq`: a
  socket drained at ~25% of full rate reached **111 s of lag over 150 s (+0.745 s/s,
  linear, no plateau)** with an implied server-side backlog of ~1,746 messages /
  ~12.4 MB still growing at disconnect. Every frame kind thinned by the *same* fraction
  — a delayed byte stream, not per-kind coalescing.

**Client rule: measure freshness against your own clock, never infer it from message
rate.** A reduced arrival rate with roughly-preserved inter-frame cadence is exactly
what a *delayed* stream looks like, so rate and gap statistics cannot distinguish a
shed stream from a queued one — only a comparison against local time can. A client that
cares about freshness must timestamp on arrival and act on the delta itself. (An
application-level ping whose reply the client times works too, and measures true
end-to-end stream freshness rather than TCP liveness: a WebSocket pong is appended to
the *same* per-connection buffer as data, so during a backlog it arrives *behind* the
queued frames rather than jumping them.) **Absence of frames is a separate signal from
lateness of frames:** emission does not depend on order flow, so silence is never merely
a quiet market — see [§3.5](#35-summary).

**Why this is currently safe, and what it is coupled to.** The stream is
lossless-but-stale rather than lossy *only* because `snapshot` and `book` are
idempotent full replaces: a client that falls behind and reconnects heals completely
from the next baseline, so lateness never becomes corruption. That property is
load-bearing for two future changes and neither is free:

- **Bounding the queue** (dropping frames or closing the socket past a threshold)
  trades staleness for loss, which is safe *today* — the next full replace heals a
  dropped `book` — but is not safe against a delta stream.
- **A delta / incremental feed** over this same unbounded queue would be **lossy**, not
  merely late: unlike a full replace, a missed delta never heals. A delta mode
  therefore has to arrive with its own gap detection and resync path, and with a slow
  consumer disconnected or snapshot-reset rather than queued indefinitely.

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
