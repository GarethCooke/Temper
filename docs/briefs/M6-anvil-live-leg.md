# M6 — The Anvil live leg *(stretch)*

**Track:** agentic · **Size:** one evening, plus a ~30 min prerequisite training run ·
**Reads first:** `ARCHITECTURE.md` §7's *The live-Anvil leg is a demo, not an evaluation*
and §1's Phase 3, invariants 1, 6 and **8**; then the vendored
`docs/vendor/anvil-protocol.md` once task 0 has placed it — its §1 (`seq` semantics) and
§3.3 (`trade`) are the two sections this milestone is most able to get wrong.

> **Revision note.** This brief was rewritten after a first draft was checked against
> Anvil's actual source. Five of its premises were wrong: it assumed a committed policy
> checkpoint existed, it missed that the simulator has never produced a partial fill, it
> claimed fills were unattributable (they are not), it told the client to handle `seq`
> gaps (which are undetectable), and it assumed a dedicated ticker could have a quiet
> feeder (it cannot). Everything below is the corrected version. The general lesson is
> recorded because it will recur: **a brief written from documentation gets things wrong
> that only reading the source catches**, and the gate in task 0 exists to catch them
> before code does.

## Amendments (invariant 3 — recorded before the work they govern)

**1. The median rule's tie-break, fixed before the export ran (2026-08-22).** The brief
says "M4a's **median** seed" and M4a has ten of them, so the median is not one of them:
its objective falls exactly between ordinals 7 and 9, whose distances from it differ at
the 1e-17 level. "Nearest the median" is therefore decided by float noise rather than by
a rule. The rule is **by rank**: sort the sweep's seeds ascending by graded objective — a
cost, so best-first — and take index `n // 2`, the *upper* of the two central ranks.
At even `n` that is the worse of the pair, which is the point: a tie-break that can only
cost cannot flatter the artefact.

Applied to `results/m4a_power_law.json` it selects **ordinal 9** — env stream base 36864,
J = 2.383440447509 bps, capture 0.993874, sixth of ten from best — against a sweep median
objective of 2.383429750037 bps and a best seed (ordinal 3) at 2.383397422365. The rule
lives in `temper.eval.grading.median_ordinal` and `tests/test_policy_checkpoint.py`
re-applies it to the committed sweep rather than trusting the artefact's own claim, so an
export done by hand from the strongest seed goes red instead of shipping as "the median".

---

## Objective

The trained policy works a parent order on a real Anvil book, over the wire, as
`PROTOCOL.md`'s **third independent client** — after the browser UI and DepthCharge —
with zero changes to Anvil.

Plumbing evidence: the policy speaks a versioned venue protocol end to end, handles a
session, an order-entry verdict channel, a coalesced event stream, and a book that does
not fill it exactly what it asked for. **Not** execution-quality evidence — the flow is
synthetic, the sample is one order, and there is no baseline it can fairly be compared
against. The milestone says so in the same breath as it reports its number.

## Prerequisite — the checkpoint that does not exist yet

**No trained policy weights exist anywhere in Temper.** `*.pt` and `*.ckpt` are
gitignored, there is no `torch.save` or `state_dict()` in the tree, and every network M2,
M3 and M4a produced was discarded after grading. Results carry trajectories, grades and
provenance — never weights.

That is a repo gap rather than an M6 gap. It means no reported result can be
re-examined, no policy inspected, and the backlog's C++/ONNX inference leg is impossible
by construction. So it lands as **its own commit before M6 starts**, not as an M6 subtask:

- checkpoint export in `tools/train.py`, written beside the results JSON;
- run on the **reference box**, exporting **M4a's median seed** — named explicitly, and
  the *median*, not the best. Choosing the strongest seed for the demo would be a quiet
  cherry-pick;
- committed as `.npz` **with provenance** — config hash, git rev, seed address, and the
  graded objective that seed achieved — plus a test that loads it and reproduces that
  grade. A committed binary that nothing verifies is not an artefact this repo keeps.

M6 then depends on a committed artefact, which is how everything else here works. Note
this is a one-time reference-box step, exactly as M0's golden export was; M6's own work
needs no training and no reference box.

**Why M4a's policy and not M2's.** It ties the live leg to the milestone the portal leads
with, and it is the more demanding test of the client: at λ = 10^−3.5 the power-law
optimum front-loads 42.3 % into bin one against the tangent optimum's 32.6 %, so it
sweeps more depth and puts more pressure on the partial-fill path. Do **not** argue the
choice on realism — Anvil's impact is a discrete ladder, neither of the trained worlds,
so no policy has a correctness claim here.

## Context — five facts that fix this milestone's shape

### 1. The policy is portable, and it has never seen a partial fill

`ExecutionEnv`'s observation is `(time remaining fraction, inventory remaining fraction)`
and its action is a fraction of remaining. No price, no ADV, no σ, no η̃ — **no market
parameter enters at inference at all.** That is why a policy trained on a synthetic AC
world can work an order on a venue it has never seen, why the 6.5-hour grid compresses to
a bin length of the demo's choosing, and why the parent size scales freely.

But `ExecutionEnv.step` does `shares = min(max(shares, 0), inventory)` — **the simulator
has always filled exactly what was asked.** Anvil will not: the client crosses a thin
ladder and bins will partially fill. So M6 is the first place the closed loop is
*exercised* rather than merely present, and the policy will be out-of-distribution in
inventory-remaining in a way it never has been. That is simultaneously the reason a
replayed schedule cannot substitute for a policy here, and a generalisation risk to
report rather than discover.

### 2. Anvil has no market orders, and a mispriced limit rests instead of executing

Order entry is a six-field engine CSV line — `<ticker>,<type>,<id>,<side>,<qty>,<price>`
— with the id field **empty** on a New (six fields, third blank; five fields is a
column-count rejection). Price is required and is a limit. §7's "market-order execution
at the impacted price" is therefore a **marketable limit**, priced through the resting
side so it crosses.

The failure is silent. A sell limit above the best bid does not error — it *rests*, the
REST response still says `accepted: true`, and the schedule quietly under-executes.
**Accepted is not filled.** Price every bin to cross against the book just observed, and
verify execution rather than inferring it from the verdict.

### 3. Fills are directly attributable; what you cannot detect is a dropped frame

The `trade` frame carries `takerId` and `makerId` (§3.3), and `POST /api/order` returns
the server-minted `id`. So the client matches its own fills exactly — no bracketing, no
inference, no construction needed.

The real hazard is elsewhere and it is worse than it looks. **`seq` cannot detect gaps at
all** (§1): it is a single global engine-thread stamp, so a per-ticker socket sees a
sparse subsequence; it can *step backwards*, because the trade ring drains continuously
while book/summary are sampled on the ~14 Hz tick and the broadcaster does not merge-sort
them. And ring overflow is **not signalled on the wire at any point** — no error frame,
no detectable gap (§3.4). `seq` is valid only as a reconnect watermark and for dedupe,
never for ordering.

So a lost trade frame is silent, and the defence is not `seq`:

- apply `snapshot`/`book` **idempotently as full replaces** and let the book self-heal;
- keep the trade ring quiet (task 5's ladder run has no other flow at all), so overflow
  is implausible rather than merely unlikely;
- **reconcile at the end**: total attributed quantity must equal the parent order. A
  mismatch **voids the measurement**, reported as void. There is no reconciliation path
  and none may be added later — the moment one exists, every number here becomes an
  estimate.

### 4. The oracle's numbers do not transfer, and must not be quoted as if they do

Anvil's tickers are synthetic integers, prices sit around 10, quantities in the hundreds,
and there is **no ADV** — `v_hourly`, `sigma_bin` and `η̃` are undefined here. Nothing in
`temper/oracle/` applies.

M6 therefore has no ε, no capture fraction, no red-flag test and no comparison to a
certified optimum. Any sentence pairing an Anvil number with an oracle number is wrong.

### 5. Roster membership is the only control, so the measured run builds its own book

Anvil publishes books only for roster tickers (`ANVIL_TICKERS`), and the feeder drives
**exactly that roster** with `cross_frac = 0.10` — it *aggresses*, sweeping two levels
through the touch. "Dedicated ticker plus quiet feeder" is not available: a ticker with a
published book is a ticker the feeder is trading.

So there are two runs, answering different questions, never merged:

**The measured run — client-built ladder, feeder off.** `ANVIL_FEEDER=0`, single-ticker
roster; the client posts a committed counterparty ladder and replenishes it to target
depth before each bin. Zero third-party flow, fully regenerable from config plus git rev.

The reason this is worth doing rather than circular: **a committed ladder plus a
deterministic policy plus deterministic matching makes the whole run predictable in
closed form.** Every resting level is known, so expected fill prices and quantities are
computable *before* the client runs. That turns M6 from "a demo that produces a number"
into "a demo whose number is checked" — M1's differential applied to the wire leg. The
slippage figure was never going to say anything about execution quality; this makes it
say something about client correctness, which is the one thing it can certify. Yes, the
demo trades against its own liquidity, and it says so plainly.

**The demonstration run — feeder on.** Single-ticker roster, `ANVIL_FEEDER_SEED` pinned,
the feeder aggressing as it normally does. This is the only condition that shows the
client surviving a book that moves between its bins because of someone else's flow, and
the only one that exercises snapshot-healing under real churn. It is **not
byte-regenerable** — book state is wall-clock dependent — and that is acceptable for a
demonstration, which is all §7 asks for. Report it separately, with the non-regenerability
stated.

## Tasks

### 0. Vendor the contract, and size the ladder — *gate*

`docs/vendor/anvil-protocol.md`: `PROTOCOL.md` snapshotted at a named Anvil commit with
generation date and source revision, as `docs/vendor/frontierview-goldens.md` does for the
goldens. The vendored file is the spec; do not restate frame schemas in code comments,
because a copy drifts.

Then, against a local build:

- `GET /api/health` reports the vendored **wire version**; the client refuses to start on
  any other value rather than parsing hopefully.
- Fix the **ladder shape and the parent order** together, so the largest bin is a
  plausible fraction of posted depth rather than a sweep of it. Being unable to fill is a
  venue fact, not an agent result — and with M4a's 42.3 % first bin, this is the number
  that decides whether the run is interesting or degenerate.

**Gate:** both recorded here before a client exists.

#### Gate record — 2026-08-22, against a local build

Anvil at `4801ed8` (clean tree), server built from `864ee2f` — the last commit to
touch `server/`; the three since are documentation. MSVC 19.44, Crow 1.2.1, Boost
1.86, `NDEBUG`. Started as `ANVIL_TICKERS=101 ANVIL_DEFAULT_TICKER=101
ANVIL_FEEDER=0 ANVIL_PORT=18080`.

**1. Wire version.** `GET /api/health` → `{"status":"ok","wireVersion":1,...}`,
matching `docs/vendor/anvil-protocol.md`'s header. The client refuses any other
value before it opens a socket or sends an order.

**2. Ladder and parent order, fixed together.**

| | |
| --- | --- |
| Parent order | **1,000 shares, sell**, ticker 101, 13 bins |
| Reference ladder | centre 100,000 ticks ($10.0000), half-spread 100 ticks, spacing 100 ticks, quantities `[300, 260, 220, 180, 150, 120, 100, 80]` per side |
| Posted depth | 1,410 a side; best bid 9.99, worst bid 9.92; arrival mid exactly $10.0000 by symmetry |
| Largest bin | 421 shares — **42.1 % of the parent, 29.9 % of posted depth**, crossing two levels |
| Also committed | `thin` (375 a side: bin one cannot fill, exercising the partial-fill and cancel path) and `wide` (three times the spread and spacing) — machinery checks, never the reported number |

Everything above lives in `configs/m6_anvil.yaml`, which holds all four runs
because they differ in three fields and share everything that decides what the
client does. Each artefact records the run name beside the config digest, so
`(config_sha256, run)` identifies a run as completely as a filename would.

**3. What the probe confirmed, and one thing the brief did not anticipate.**
Every venue fact the brief predicted from the source held: the six-field New with
a blank id is accepted and the server mints the id (five fields earns *wrong
column count*); `POST /api/order` mints `anvil_session` on first contact; a
cancel from another session and a cancel of an unknown id are refused with the
*identical* reason; a marketable sell larger than the book fills what it can and
**rests the remainder** while still returning `accepted: true`; `summary.last`
was `""` under a two-sided resting book and became `9.99` only after the first
trade. `Book.walk` and Anvil's matching engine agreed level for level on an
eight-level sweep of 1,289 shares.

Two things the brief did not say, both found by running it:

* **The book publishes on the ~14 Hz tick**, so `GET /api/book` and the `book`
  frame lag a `POST /api/order` verdict by up to ~70 ms. A client that priced
  immediately after topping up its ladder would cross a book that predates its
  own orders. Hence `grid.settle_seconds`.
* **A measured run must start on an empty book.** A 300-share bid survived from
  an earlier probe process, could not be cancelled — ownership is the session
  cookie, and that session had gone — and silently doubled the touch level. The
  client now refuses to build a ladder on a non-empty book, and the Makefile says
  to restart the server between measured runs.

*(The first draft's third gate question — does the feeder aggress — is answered: it does,
`cross_frac = 0.10` through two levels, on exactly the roster tickers. That answer is what
produced context §5.)*

### 1. Inference without the training stack

`client/` needs a 64×64 MLP forward pass. **Plain numpy reading the committed `.npz`** —
not torch. This keeps `client/` free of the training stack, which §3's seam already
implies, and it is the first step of the backlog's C++/ONNX inference leg rather than a
detour from it.

The decision stands on those merits. It was briefly forced by an environment constraint
that no longer applies; a decision made for a vanished constraint has to be re-justified
or dropped, and this one re-justifies.

Checkable the way everything here is: the numpy path must reproduce the training-time
policy's action **to float tolerance on the committed evaluation trajectory**, in a test
needing no network. A second implementation of a forward pass is only worth having if
something proves the two agree.

### 2. `client/` — the participant

The only networked code in the repo (invariant 8). `temper/` is untouched and imports
nothing new.

- **Session.** First `POST /api/order` mints `anvil_session`; store and replay it — it is
  the ownership principal for any later cancel.
- **Order entry.** Six-field CSV, id field empty; read `accepted`, `reason`, `id` from the
  `200` body. `200 {accepted: false}` is a **business verdict**; only `503` (engine busy),
  `504` (engine timeout), `400` and `403` are transport faults. A client that collapses
  those cannot tell "the engine said no" from "the engine never heard me".
- **Stream.** One socket per ticker (`/ws?ticker=`). `snapshot` and `book` are **full
  replaces — apply them idempotently**. Never reason about ordering from `seq`; use it
  only as the reconnect watermark and for dedupe. There is no error frame.

Extend `tests/test_repo_invariants.py`: it currently rejects networking imports under
`temper/`; it should also assert `client/` is the **only** place they appear, so the seam
is stated from both sides.

### 3. Attribution and reconciliation

Match fills by `takerId` against the ids returned from `POST /api/order`. Reconcile total
attributed quantity against the parent order at end of run; a mismatch voids the
measurement with the reason recorded. No reconciliation path, now or later.

### 4. The measurement, its prediction, and the trap inside it

Arrival slippage in bps against the **arrival mid**, taken from the book snapshot at
t = 0 as `(bestBid + bestAsk) / 2`.

**Not from `GET /api/summary`'s `last`.** That field was the book mid and is now the last
*traded* price — a semantic change made inside the wire version with no bump, because the
shape did not move. `""` now means "has not traded yet", and once set it persists after
the book empties. `PROTOCOL.md` names clients that "reasoned about it as a mid" as exactly
the ones that break. A client taking arrival price from `last` would be silently wrong in
the one number this milestone reports. Read the mid off the snapshot, and put this
paragraph's reason in a comment at the call site.

**Compute the prediction first.** From the committed ladder and the policy, derive
expected fill prices and quantities before running. Acceptance is *realised matches
predicted*, not merely *a number was produced*. If you want the prediction machinery
tested harder, vary the **ladder** across several committed shapes — thin, deep,
wide-spread — rather than varying the policy: the ladder is what the prediction is
computed from, so varying it tests the pricing and attribution logic directly.

Report arrival mid, per-bin fill prices and quantities, predicted vs realised, realised
slippage in bps, the schedule the policy actually produced, and the wall-clock bin length.
Committed with config hash and git rev — the demo is not exempt from invariant 1.

### 5. The runs

**Ladder run** (feeder off) — yields the measurement and the prediction check.

**Feeder run** (`ANVIL_FEEDER_SEED` pinned) — the demonstration. Report separately, state
that it is reproducible only in the weak sense.

**Deployment run**, optionally, once — same client against anvil.garethcooke.com. The demo
book is a shared, unauthenticated trading floor: ownership stops a stranger cancelling
your orders, not trading against them. A third-party fill makes it a successful
demonstration and a void measurement, and both halves get reported.

### 6. The write-up, and what it hands MP Stage 2

`README.md`'s Phase-3 rung stops being a promise: what ran, against what, and what it does
and does not establish. MP Stage 2 unblocks here — leave a clean handoff: the results
artefact, the schedule figure, and one paragraph a portfolio page can quote without
softening.

## Pre-stated (invariant 3 — loosen only by amending this brief before work starts)

| Item | Value |
| --- | --- |
| Prerequisite | checkpoint export committed first: M4a's **median** seed by amendment 1's rank rule — ordinal 9 — as `.npz` + provenance + a test reproducing its grade |
| Policy on the wire | M4a's power-law policy |
| Inference | plain numpy; must match the training-time action to float tolerance on the committed eval trajectory, no network |
| Venue | Anvil at the vendored wire version; client refuses any other `GET /api/health` value |
| Ladder + parent order | fixed in task 0 together; largest bin a plausible fraction of posted depth |
| Grid | 13 bins; bin length is a client config, minutes not half-hours |
| Aggression | marketable limit priced to cross the observed resting side. **Accepted ≠ filled** |
| Attribution | `takerId` against the ids from `POST /api/order` |
| `seq` | reconnect watermark and dedupe **only** — never ordering, never gap detection |
| Void condition | attributed quantity ≠ parent order, or a third-party fill in a measured bin. Reported, never reconciled |
| Arrival price | book-snapshot mid at t = 0. **Never** `summary.last` |
| Acceptance | realised fills match the closed-form prediction from the committed ladder |
| Reported number | realised arrival slippage, bps, labelled a demo |
| Not reported | any ε, capture fraction, red-flag test, or oracle comparison — undefined on this venue |
| Runs | ladder (measured, regenerable) and feeder (demonstration, weak reproducibility). Never merged |
| Network seam | `client/` only; asserted from both sides |
| Upstream changes | **zero** |
| Artefact | config hash + git rev, `git_dirty: false` |
| Suite impact | `make test` ≤ 3 min; anything needing a live server behind a marker |

## Definition of done

- ☐ Checkpoint export committed as its own change; `.npz` + provenance + grade-reproducing test.
- ☐ `docs/vendor/anvil-protocol.md` placed with provenance; ladder and parent order recorded.
- ☐ numpy inference pinned against the training-time policy, no network needed.
- ☐ `client/` speaks session, order entry and the stream; `temper/` untouched.
- ☐ Repo-invariant test asserts the network seam from both sides.
- ☐ Attribution by `takerId`; end-of-run reconciliation; an induced mismatch shown to void a run.
- ☐ Arrival mid from the snapshot, with the `summary.last` reason at the call site.
- ☐ Prediction computed before the run; realised matches predicted.
- ☐ Ladder run complete, artefact committed and regenerable.
- ☐ Feeder run complete, reported separately with its seed and its caveat.
- ☐ README Phase-3 rung rewritten from promise to result.
- ☐ `ROADMAP.md` M6 row flipped; MP Stage 2 noted unblocked; anything structural → §9.

## Out of scope (resist)

Any change to Anvil, including the one that would make this easier. Any use of
`temper/oracle/` against Anvil prices. Limit-order placement strategy or queue-position
logic — v2. Multiple parent orders or tickers, or any attempt to turn one demo into a
sample. A second policy on the wire: vary the ladder instead. Retraining or touching
`temper/agents/`. A latency claim — this is a Python client over HTTP against a demo
transport, and quoting microseconds beside Anvil's own numbers would be embarrassing.

## Session notes

- **The likely failure is a confident wrong number, not a crash.** Three routes to one: a
  limit that rests instead of crossing, a silently dropped trade frame, and an arrival
  price read from `last`. All three return plausible bps. Each has a specific defence
  above, and each defence is an assertion rather than a convention.
- **Void is a result.** "The client worked the order end to end and the measurement was
  void, for this reason" is still a successful M6 by §7's terms, because §7 asks for
  plumbing evidence. Reaching for a reconciliation to rescue a number converts a good
  milestone into a bad one.
- **Anvil is the second contract this project consumes without changing it**, after
  FrontierView's goldens. Two independent upstreams, both consumed through versioned
  artefacts, neither modified — that is the sentence for the write-up.
- Venue facts here came from Anvil's source and docs, not from a running server. Task 0
  confirms them. Treat them as predictions, like every other brief in this directory.
