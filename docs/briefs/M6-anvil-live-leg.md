# M6 — The Anvil live leg _(stretch)_

**Track:** agentic · **Size:** one laptop evening, plus a one-time twenty-minute run
on the reference box that is not M6's (task 1) · **Reads first:** `ARCHITECTURE.md`
§7's _The live-Anvil leg is a demo, not an evaluation_ and §1's Phase 3, invariants
1, 6 and **8** (which is the one this milestone is most able to break); then the
vendored `docs/vendor/anvil-protocol.md`, which task 0 has placed.

> ### Amended 2026-08-22, before any work started (invariant 3)
>
> The first draft of this brief was written against Anvil's documentation. Reading
> Anvil's *source* found four venue facts wrong and one Temper fact assumed. Because
> invariant 3 says pre-stated items loosen only by amending the brief before work
> starts, they are amended here rather than footnoted later. Two of them make the
> milestone stronger than it was written.
>
> | Was | Is | Where it lands |
> | --- | --- | --- |
> | `trade` frames carry no order id, so attribution needs a `seq` bracket | frames carry **`takerId` and `makerId`** (PROTOCOL.md §3.3, `server/protocol.hpp`) | context §3 rewritten; `seq` bracketing **withdrawn entirely** |
> | a `seq` gap means dropped frames and voids a bin | `seq` is a global stamp, sparse per ticker, **can arrive out of order**, and overflow is unsignalled | context §7 is new; the gap void condition is **withdrawn** |
> | `summary.last` silently became last-traded inside wire v1 | `last` **is** the mid — the claim was never true of this tree | task 5's reasoning replaced with reasons that are checkable |
> | a dedicated ticker plus a quiet feeder gives attribution | books publish **only for roster tickers**, and the feeder drives exactly the roster, aggressing 10 % of the time | context §6 rewritten; the two controls collapse into one |
> | task 1 reads "the committed checkpoint" | **no policy weights exist anywhere in this repo** | task 1 rewritten as a prerequisite that lands first, on its own |
>
> Also added: context §4, that the policy has never seen a partial fill; and the
> predicted-versus-realised acceptance in task 4, which the first draft had no way
> to ask for because it believed fills were unattributable.

## Objective

The trained policy works a parent order on a real Anvil book, over the wire, as
`PROTOCOL.md`'s **third independent client** — after the browser UI and DepthCharge —
with zero changes to Anvil.

This is plumbing evidence: the claim is that the policy speaks a versioned venue
protocol end to end, handles a session, an order-entry verdict channel, a coalesced
event stream that gaps and reorders by design, and comes back with an
arrival-slippage number that was actually measured rather than modelled. It is **not**
execution-quality evidence and the milestone must say so in the same breath it reports
the number: the flow it trades against is synthetic and non-adversarial, the sample is
one parent order, and there is no baseline it can be fairly compared against.

What the amendment adds is a second, sharper claim that the first draft could not
make. Because the book is committed, the policy is deterministic and Anvil's
quantities and prices are integers, the whole run has a **closed form computable
before it starts**. The acceptance is therefore not "a number came back" but
"the realised fills equal the predicted fills, exactly" — M1's differential, applied
to a wire.

## Context — seven facts that fix this milestone's shape

### 1. The policy is portable, because its observation is dimensionless

`ExecutionEnv`'s observation is `(time remaining fraction, inventory remaining
fraction)` and the action is a fraction of remaining inventory. No price, no ADV, no
σ, no η̃ — **no market parameter enters at inference time at all.** That is why this
milestone is possible: the same trained network can work any parent size on any venue
over any grid, because everything it consumes is a ratio.

Two consequences worth taking deliberately rather than discovering. The 6.5-hour,
13-bin grid compresses to a bin length of the demo's choosing and the policy cannot
tell — bin length is a client config, not a model input. And the parent size can be
scaled to whatever Anvil's book can absorb without retraining anything.

### 2. Anvil has no market orders, and a mispriced limit rests instead of executing

Order entry is a six-field engine CSV line — `<ticker>,<type>,<id>,<side>,<qty>,<price>`
— with the id field **empty** on a New (six fields, third blank; five fields is a
column-count rejection). Price is required and is a limit: a positive decimal of at
most four places (`src/parser.cpp`), so the tick is $0.0001. Quantity is a positive
integer. §7's "market-order execution at the impacted price" therefore has to be
implemented as a **marketable limit**, priced through the resting side so it crosses.

This is the milestone's sharpest failure mode and it is silent. A sell limit priced
above the best bid does not fail — it _rests_, the bin appears to have been submitted
successfully, the REST response says `accepted: true`, and the schedule quietly
under-executes while the slippage number goes to nonsense. Accepted is not filled.
Every bin must therefore be priced to cross against the book the client has just
observed, and the client must verify execution rather than infer it from the verdict.

Price it at **exactly** the worst level it intends to consume, not lower. Pricing
deeper is the tempting defensive move and it is the wrong one: on a book that is not
what the client read, a deeper limit sweeps further than predicted and *hides* the
discrepancy inside a plausible fill, while a limit at the intended worst level leaves
a residual and makes the surprise loud. Fills happen at the resting order's price
either way, so the choice costs nothing and buys the assertion.

### 3. Fills are attributable by order id, and the client must still prove it

A `trade` frame carries `ticker`, `price`, `qty`, `aggr`, **`takerId`** and
**`makerId`** (PROTOCOL.md §3.3). The server mints the id on a New and returns it in
the `POST /api/order` body, so the client knows its own ids exactly. Attribution is
therefore direct: a fill is the client's when `takerId` or `makerId` is one of them.
Both, not just the taker — a residual that rests and is later hit is still the
client's fill.

This retires the first draft's `seq`-bracketing construction outright, and with it the
requirement that no third party trade inside a bin. A stranger's trade is now a market
condition, not an attribution failure.

What it does **not** retire is the proof. Attribution is only as good as the stream is
complete, and §7 says the stream cannot tell you when it is not. So the identity is
asserted rather than trusted, at two scales: per bin, realised fills must equal the
prediction of task 4; and at end of run, total attributed quantity must equal the
parent order. A mismatch marks the episode **void and reports it as void**. It is never
reconciled, adjusted or best-guessed — an unattributable fill means the one number this
milestone produces was not measured, and reporting it anyway would be the exact failure
the rest of the repo is built to prevent.

### 4. The policy has never seen a partial fill

`ExecutionEnv.step` executes exactly what it is asked for and force-liquidates the
remainder in the final bin. Inventory evolves from the action and nothing else. Across
M2, M3 and M4a the agent has therefore never once been shown an inventory it did not
choose.

Anvil will show it one. This is both the reason a replayed schedule cannot substitute
for the network — an open-loop trajectory has no response to an under-fill, and the
whole point of running a policy is that it has one — and a generalisation risk to
**report** rather than discover. The observation is in-distribution (an inventory
fraction is an inventory fraction) but the *state the policy is asked about* is one no
training episode reached, and M6 has one sample and no way to say whether the response
is good. Say that; do not measure it and do not defend it.

### 5. The oracle's numbers do not transfer to Anvil, and must not be quoted as if they do

Anvil's tickers are synthetic positive integers, prices sit around 10, quantities in
the hundreds, and there is **no ADV** — so `v_hourly`, `sigma_bin` and `η̃` are
undefined on this venue. Nothing in `temper/oracle/` applies here.

That means M6 has no ε, no capture fraction, no red-flag test and no comparison to a
certified optimum. It reports exactly one measured quantity — realised arrival
slippage in bps — and it reports it as a demo. Any sentence pairing an Anvil number
with an oracle number is wrong, and the out-of-scope section says so.

Note what this costs the policy choice, and take it deliberately: **no trained policy
has a correctness claim on Anvil.** A discrete price ladder is neither the linear
tangent nor the 0.6-power law. The policy running here is not the right policy for
this venue; it is a policy, run on this venue, to show that it can be.

### 6. The book you can see is the roster's, and the roster is what the feeder drives

Two facts from Anvil's source that the first draft treated as independent knobs:

- **Books publish only for roster tickers.** `MarketData` is built once from
  `ANVIL_TICKERS` and never modified; `GET /api/book` and the `snapshot`/`book` frames
  serve only those slots. An order on an off-roster ticker reaches the engine and
  matches — the engine creates a book on demand — and `trade` frames for it are even
  fanned out. But the client would be **blind**: no snapshot, so no arrival mid and
  nothing to price a marketable limit against.
- **The feeder drives exactly the roster, and it aggresses.** `FeederConfig` carries
  `cross_frac = 0.10`, and `make_crosser` prices two levels through the opposite touch
  specifically so trades appear on the tape.

So "a dedicated ticker" and "a quiet feeder" are not two arrangements, they are one:
roster membership. The demo ticker must be in `ANVIL_TICKERS` to be visible, which
means the feeder is on it unless the feeder is off. Locally that is a decision —
`ANVIL_FEEDER=0`, or `POST /api/feeder {"enabled": false}` at runtime. On the
deployment it is not: the roster is twelve tickers at ~200 messages/second aggregate,
so third-party flow inside every bin is a certainty rather than a risk.

### 7. `seq` is a stamp, not a sequence, and cannot detect loss

`PROTOCOL.md` says two incompatible things about `seq`. §4 says a value other than the
expected next one means a frame was missed, so drop state and resync. §1 says one
global counter stamps every frame across every ticker, a single-ticker socket sees a
*sparse* subsequence, and "v1 clients apply frames idempotently and do not gap-test".
**§1 is the operative one**, and the source is more emphatic than either:

- the engine stamps a `seq` per roster ticker per publish tick and the broadcaster
  emits on its own independent timer, skipping any book superseded in between — so
  gaps occur even on a **single-ticker** roster;
- the broadcaster drains the trade ring every iteration but publishes books only on
  that timer, so a trade stamped *after* a book is routinely delivered *before* it —
  **`seq` can go backwards**;
- and ring overflow clears its own flag and emits nothing, with §3.4 ruling out an
  error frame — so a genuinely dropped frame looks exactly like the structural gaps
  above.

The client therefore records `seq` and never reasons about ordering from it. Frames are
applied idempotently, the book heals from any snapshot, and the reconnect rule is "take
the snapshot as the new baseline", not "a gap means resync". This is recorded in
`docs/vendor/anvil-protocol.md`'s quirks section as an observation about upstream, not
as something to fix: **zero upstream changes**, and a documentation contradiction is
upstream's to resolve if it ever wants strict gap detection.

The residual risk is real and is named rather than closed: a silently dropped `trade`
frame under-counts a fill. It is defended by keeping the ring quiet (the feeder off, or
slow) and by the end-of-run reconciliation in §3 — not by watching `seq`, which cannot
see it.

## Tasks

### 0. Amend this brief, vendor the contract — _gate_

Amend this brief first, from the source rather than the docs, because everything below
was specified against facts that were wrong. Then place
`docs/vendor/anvil-protocol.md`: `PROTOCOL.md` snapshotted at a named commit with the
generation date, source revision and file digest, in the shape of
`docs/vendor/frontierview-goldens.md`, and with an **Observed upstream quirks
(recorded, not corrected)** section for the divergences above. The vendored file is the
spec the client is written against; do not restate frame schemas in this brief or in
code comments, because a copy is a thing that drifts.

Then confirm against a local Anvil build — these are now facts to *verify*, not
discover, which is the amendment doing its job:

- `GET /api/health` reports **wire version 1**; the client refuses to start on any
  other value rather than parsing hopefully.
- The feeder aggresses, and `ANVIL_FEEDER=0` silences it. Confirm both, because the
  measured run depends on the second.
- The resting depth a committed ladder actually produces, which fixes the parent order
  size in the pre-stated table.

**Gate:** brief amended, contract vendored with its digest checked against the source,
health check green.

### 1. The checkpoint that does not exist — _prerequisite, its own commit_

The first draft said "reading the committed checkpoint". There is no committed
checkpoint. There is no checkpoint at all: `*.pt` is gitignored, nothing in the repo
calls `torch.save`, and **every network M2, M3 and M4a trained was graded, summarised
and dropped on the floor.** What survives is a number and a trajectory; the object that
produced them is gone.

That is a gap in the repo, not a problem for this milestone to work around, and it is
fixed the way M0's goldens exporter was — as its own capability, in its own commit,
before the milestone that needs it:

- `temper/agents/checkpoint.py` defines the format: the actor mean's affine layers as
  named float32 arrays plus a JSON metadata blob. Not a `state_dict`, for two reasons
  that are the point rather than taste — a `.pt` needs torch to read, and the whole
  purpose is that something *without* the training stack can evaluate the forward pass;
  and `torch.load` executes a pickle, so a committed `.pt` is a committed executable.
- `tools/train.py` gains `--checkpoints DIR` (write each seed's policy beside the
  metrics) and `--only-seed N` (re-derive one seed's policy without the sweep).
  `temper.eval.sweep.train_seed` is already the per-seed entry point and
  `tests/test_m4a_phase1_regression.py` is already the standing evidence that one seed
  retrains bitwise in isolation, so no restructuring is needed.
- **On the reference box**, one run: seed 9 of `configs/m4a_power_law.yaml`, ~20
  minutes. Its acceptance is not that it produced a file but that its grade reproduces
  `results/m4a_power_law.json`'s seed 9 **bitwise** — objective, capture fraction and
  all fourteen trajectory points. That is what makes the committed weights *that* run's
  policy rather than a similar one.
- Commit `results/m6_policy.npz` and `results/m6_policy.json` (provenance, seed
  address, the grade it achieved), and a test that loads the weights, rolls them out
  open-loop and reproduces that trajectory. **No unverifiable blobs.**

Everything after this task is laptop work.

**Which policy, and which seed.** M4a's power-law agent, seed ordinal **9**. Not M2's:
M4a is the milestone the portal already leads with, and its first bin is 42.05 % of the
parent order against M2's 32.6 %, so it sweeps more depth and puts more pressure on the
partial-fill path — a more demanding test of the client, which is the only thing being
tested. Per context §5 the choice is *not* argued on realism, because neither trained
world is Anvil's.

Seed 9 is the **lower median** of the ten by capture fraction (0.993874). Ten seeds
have no median seed, so the rule takes the worse of the two straddling it; stating the
rule in advance is what stops "the median seed" quietly becoming the best one. The
achieved grade travels in the artefact's provenance so a reader can see which seed they
are looking at.

### 2. Inference without the training stack

A plain-numpy forward pass reading the committed `.npz`. Not torch in `client/`: it
keeps the client light, makes the network's arithmetic visible, and is the first step
of the backlog's C++/ONNX inference leg rather than a detour from it.

On an x86-64 Mac it is also the only option, which settles a question the first draft
left open. `torch==2.9.1` ships **arm64-only** macOS wheels — 2.2.2 was the last
release with `macosx_*_x86_64` — so an Intel development host cannot install the pinned
training stack at all. Consequences, both to be recorded rather than worked around:
`requirements.txt` should separate the training stack from the runtime one, and on such
a host the full `make test` becomes a reference-box gate while the oracle, the
differential, the certificate, the client and the ladder prediction all still run
locally.

The path is checkable the way everything here is checkable, in two tests that need no
network:

- the numpy rollout reproduces the **committed evaluation trajectory** — seed 9's
  fourteen points in `results/m4a_power_law.json` — to float tolerance;
- and, wherever torch is installed, it agrees with `PPOPolicy` elementwise over a grid
  of observations.

A second implementation of a forward pass is only worth having if something proves the
two agree, and the first of those tests is the one that runs everywhere.

### 3. `client/` — the participant

The only networked code in the repo (invariant 8). `temper/` is not touched, except for
one figure function in `temper/eval/figures.py` — the single module already allowed
matplotlib — which adds no dependency and no network.

- **Session.** First `POST /api/order` mints `anvil_session`; the client stores and
  replays it, because it is the ownership principal for any later cancel.
- **Order entry.** Build the six-field CSV with the id field empty; read `accepted`,
  `reason` and `id` from the `200` body. Treat **`200 {accepted: false}` as a
  business verdict**, and only `503` (engine busy), `504` (engine timeout), `400` and
  `403` as transport faults — the distinction is the protocol's, and a client that
  collapses them is one that cannot tell "the engine said no" from "the engine never
  heard me".
- **Stream.** One WebSocket, `snapshot` / `book` / `trade` / `summary`. Book frames are
  full replaces and are applied idempotently; trades are accumulated by id. Per context
  §7, `seq` is **recorded and never reasoned about** — out-of-order and gapped values
  are both expected, and a client that resyncs on either would thrash on a healthy
  server.
- **No error frames exist.** Verdicts are REST-only. A client with an error-frame
  handler has misread the contract.

Extend `tests/test_repo_invariants.py`: it currently rejects networking imports under
`temper/`. It should now also assert that `client/` is the **only** place they appear —
across `temper/`, `tools/` and `tests/` alike — so the seam is stated from both sides
rather than one.

### 4. The prediction, and the differential it makes possible

The client posts its own two-sided counterparty ladder from a committed spec, then
works the parent order into it with the feeder off. Because the ladder is committed,
the policy is deterministic, and Anvil's quantities are integers at integer ticks, the
entire execution is computable **before the run starts**: per bin the requested
quantity, the levels consumed, each fill's quantity and maker price, the resulting
inventory, and the total slippage.

**Acceptance is `realised == predicted`, exactly.** Integer and tick equality, per bin
and in total — not `allclose`, because both sides are exact arithmetic and anything
less would pass on the discrepancy worth catching.

This is what makes trading against your own liquidity worth doing rather than circular.
A run against a book nobody can predict produces a number and no way to know whether
the client computed it correctly; this produces the same number *and* a differential
that fails loudly if the pricing, the attribution, the rounding or the terminal
condition is wrong. It is M1's pattern — an independently-derived expectation, checked
against a simulator's realisation — with Anvil's engine in the simulator's place.

Void conditions, replacing the first draft's:

1. realised per-bin fills ≠ predicted (ladder run);
2. total attributed quantity ≠ the parent order at end of run (every run);
3. a trade on the ticker attributable to neither of the client's ids (ladder run — a
   third party, or a frame the ring dropped).

**Not** a void condition: a `seq` gap, an out-of-order `seq`, or third-party flow in the
feeder run. Void is reported, never reconciled, and no reconciliation path exists — the
moment one does, every number this milestone produces becomes an estimate. A
deliberately induced mismatch, both offline and against a ladder posted thinner than
the predictor believes, is shown to void an episode.

### 5. The measurement, and the trap inside it

Arrival slippage in bps against the **arrival mid**, taken from the book snapshot at
t = 0 as `(bestBid + bestAsk) / 2`, parsed from the wire decimals.

**Not from `GET /api/summary`'s `last`.** The first draft's reason for this was wrong —
it claimed the field had silently changed from the mid to the last traded price, and it
never did; `summary.last` is the mid in this tree and the documentation says so. The
conclusion survives on reasons that are actually checkable, and those are what go in the
call-site comment:

- it is computed as `(bid + ask) / 2` in **integer ticks**, so it truncates;
- it degenerates to the **lone best** on a one-sided book, which the field table does
  not mention;
- it rides a 2 Hz cross-ticker publish, a different cadence and a different snapshot
  from the book the order will actually cross.

Any of those makes it the wrong number by a little, silently, in the one quantity the
milestone exists to report. Read the mid off the snapshot. And write *these* reasons at
the call site, not the first draft's — shipping a comment whose provenance claim is
invented would be its own version of the failure this milestone is about.

Report: arrival mid, per-bin fill prices and quantities, predicted beside realised,
realised slippage in bps, the schedule the policy actually produced, and the wall-clock
bin length used. All of it committed as a results artefact with the config hash and git
rev, per invariant 1 — the demo is not exempt from being regenerable.

### 6. The runs

**The ladder, locally, across several committed shapes.** Feeder off, single-ticker
roster. This is the run that yields the number, and the one the prediction differential
applies to. Several shapes rather than several policies: varying the ladder stresses the
pricing, the level walk and the attribution directly, where a second policy would only
re-test them at another point.

**The feeder, locally, once.** Same client, same policy, feeder on with
`ANVIL_FEEDER_SEED` pinned and recorded. A demonstration that the client works an order
against moving synthetic flow, with fills still exactly attributed. It is reproducible
only in the weak sense and the artefact must say so in those words: wall-clock ordering
between the feeder thread and the client makes exact replay impossible, so this is
**not** an invariant-1 artefact and its number is never merged with the ladder run's.

**The deployment, once**, as the wire demonstration: same client, same policy, against
anvil.garethcooke.com. Report it separately, state that the book was shared and
unauthenticated for its duration and that third-party flow was certain rather than
possible, and do not merge its slippage with the local figures.

### 7. The write-up, and what it hands to MP Stage 2

`README.md`'s Phase-3 rung stops being a promise. State what ran, against what, and
what it does and does not establish — the wire is real, the venue is real, the flow is
synthetic and non-adversarial, the sample is one order, and the policy was working a
book of its own making in the run that produced the number.

MP Stage 2 (the live-leg writeup on the portal) unblocks here, so leave it a clean
handoff: the results artefact, the schedule figure, and one paragraph that a portfolio
page can quote without softening.

## Pre-stated (invariant 3 — loosen only by amending this brief before work starts)

| Item                    | Value                                                                                                                                                     |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Venue                   | Anvil, wire version **1**, contract vendored at a named commit with its digest                                                                            |
| Client refuses to start | any `GET /api/health` wire version other than the vendored one                                                                                            |
| Policy                  | M4a power-law agent, seed ordinal **9** — the lower median of ten by capture fraction, rule stated before the export                                      |
| Parent order            | fixed in task 0 from the committed ladder's depth; the whole order a stated fraction of posted depth, not just the largest bin                            |
| Grid                    | 13 bins, matching the trained policy's horizon; bin length is a client config, seconds not half-hours                                                     |
| Quantities              | positive integers; nearest-integer rounding per bin, the final bin submitting all remaining inventory (the env's terminal constraint, mirrored)           |
| Aggression              | marketable limit priced at **exactly** the worst level intended to be consumed. **Accepted ≠ filled**                                                     |
| Attribution             | `takerId` or `makerId` matching a client-minted id. No `seq` bracketing                                                                                   |
| Ladder acceptance       | realised per-bin `(quantity, price)` vectors **equal** the pre-run prediction, exactly                                                                    |
| Void condition          | prediction mismatch, total attributed quantity ≠ parent order, or a trade attributable to neither client id. Void is reported, never reconciled           |
| Not a void condition    | a `seq` gap, an out-of-order `seq`, or third-party flow in the feeder and deployment runs                                                                 |
| Arrival price           | book-snapshot mid at t = 0. **Never** `summary.last`                                                                                                      |
| Reported number         | realised arrival slippage, bps, labelled a demo                                                                                                           |
| Not reported            | any ε, capture fraction, red-flag test, or comparison to `temper/oracle/` — none of them are defined on this venue                                        |
| Inference               | numpy forward pass; reproduces the committed eval trajectory to float tolerance in a test needing no network, and `PPOPolicy` elementwise wherever torch installs |
| Network seam            | `client/` is the only networked code; `tests/test_repo_invariants.py` asserts it from both sides                                                          |
| Upstream changes        | **zero**. Anvil is consumed through its existing public contract; its documentation divergences are recorded, not fixed                                   |
| Artefact                | committed with config hash + git rev, `git_dirty: false` — for the ladder runs. The feeder run is explicitly **not** an invariant-1 artefact              |
| Suite impact            | `make test` ≤ 3 min; anything needing a live server behind a marker, like `deep` and `training`                                                           |

## Definition of done

- ☐ This brief amended from source before work started; `docs/vendor/anvil-protocol.md` placed with provenance and its digest verified; task 0's three facts confirmed.
- ☐ Checkpoint export landed as its own commit; `results/m6_policy.npz` committed with provenance and a test that reproduces its committed grade — no unverifiable blob.
- ☐ Numpy inference path landed and pinned against the committed trajectory, no network needed.
- ☐ `client/` speaks session, order entry, and the stream; `seq` recorded, never reasoned about.
- ☐ Repo-invariant test asserts the network seam from both sides.
- ☐ Ladder prediction landed; realised equals predicted exactly on every committed shape; a deliberately induced mismatch shown to void an episode.
- ☐ Arrival mid taken from the snapshot, with the checkable `summary.last` reasons recorded at the call site.
- ☐ Ladder runs complete; slippage reported with its caveats; artefacts committed and regenerable.
- ☐ Feeder run complete, reported separately, labelled weakly reproducible with its seed recorded.
- ☐ Deployment run complete and reported separately, with the shared-book caveat.
- ☐ README Phase-3 rung rewritten from promise to result.
- ☐ `ROADMAP.md` M6 row flipped; MP Stage 2 noted as unblocked; anything structural → §9.

## Out of scope (resist)

Any change to Anvil, including the one that would make this milestone easier, and
including "fixing" the `seq` documentation contradiction — that is upstream's call and
this repo records it. Any use of `temper/oracle/` against Anvil prices. Limit-order
placement strategy or queue-position logic — the action space is unchanged and that is
v2. Multiple parent orders, multiple tickers, or any attempt to make one demo into a
sample. A second policy on the wire: vary the ladder instead. Retraining, retuning, or
touching `temper/agents/` beyond the checkpoint format. A latency claim: this is a
Python client over HTTP and WebSocket against a demo transport, and quoting microseconds
off it would be embarrassing next to Anvil's own numbers.

## Session notes

- **The failure this milestone is most likely to produce is a confident wrong number**,
  not a crash. Three independent routes to one: a limit that rests instead of crossing,
  a fill the stream under-counted, and an arrival price read from `last`. All three
  return plausible bps. Each has a specific defence above and each defence is an
  assertion rather than a convention — and the prediction differential catches all
  three at once, which is the strongest argument for building it.
- **The first draft was confidently wrong about four venue facts**, all of them read
  from Anvil's documentation rather than its source, and two of them in the direction
  of making the milestone harder than it is. That is worth more than the corrections:
  a vendored contract is evidence about the *contract*, not about the server, and the
  quirks section of `docs/vendor/anvil-protocol.md` exists because the two had already
  diverged in six places before this milestone read them.
- **Void is a result.** If attribution fails, the honest output is "the client worked
  the order end to end and the measurement was void, for this reason". That is still a
  successful M6 by §7's terms, because §7 asks for plumbing evidence. Reaching for a
  reconciliation to rescue a number would convert a good milestone into a bad one.
- **Anvil is the second contract this project consumes without changing it**, after
  FrontierView's goldens. That is the sentence worth landing in the write-up: two
  independent upstreams, both consumed through versioned artefacts, neither modified.
- The static ladder depletes as the order works into it, so slippage grows across bins
  and a front-loaded schedule fares worse than a flat one would. That is mechanical,
  not a result. M6 reports no comparison, and the write-up should say why one would be
  meaningless rather than leave the reader to infer it.
