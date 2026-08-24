# House notes

Practices that outgrew the milestone that found them. Each was measured here and applies to
every project in the portfolio that reports a trained or sampled number — Anvil's latency
distributions, Crucible's benchmark captures, Temper's RL results. `ARCHITECTURE.md` §9 is
where a *structural* decision about Temper is recorded; this file is for the smaller,
portable rules, stated once so a brief can cite them by title rather than restate them.

Cite entries by title. A title that changes is a title every citation of it has to change with — *No code path may be reachable only at the end of a long run* was *The artefact writer is tested on fabricated data, not on the run* until M4b showed that naming the writer had protected the writer and nothing else.

---

## Thread count is a reproducibility axis

**Rule.** Any result produced by a multithreaded numerical library — torch, BLAS, OpenMP —
records the thread count it ran at as a committed input, not a property of the host, and
pins the surrounding pools (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`) *before* the library is
imported. A result that does not state its thread count is not regenerable, and any RL
result in the portfolio that does not pin it is suspect by this argument.

**Measured (M2, 2026-08-06).** The same seed address — same shock streams, same network
initialisation, same minibatch order — scored 0.165 of the TWAP gap on four torch threads
and 0.066 on eight. Nothing the config addressed had changed; what changed was the order in
which torch's CPU reductions summed, which follows the thread count. PPO compounds that
float-level difference over ~1 800 updates, and the objective is flat enough near its
minimum (a 28.8 %-of-X ball costs 0.066 bps) for the compounded difference to move the
trajectory visibly. On one host at one thread count, training is *bitwise* reproducible;
across thread counts it is not.

**How it is applied here.** `ppo.torch_threads` is a committed hyperparameter, set inside
`train` before the first tensor is allocated; `tools/train.py` sets the OpenMP/MKL pools
from it before torch is imported (the runtime half: unpinned, the pools oversubscribe a
16-logical-core box and every seed runs ~25 % slower — which once pushed a seed one second
past a budget and voided a sweep); every results file records the count. `ARCHITECTURE.md`
§9, *`git_dirty` asks whether the source is uncommitted…*, records the structural half.

**Why it generalises.** The failure is invisible in any single run: two runs at different
thread counts each look like a perfectly good experiment. It is only visible when a
committed number is regenerated on a different core count and does not match — at which
point invariant 1 looks broken while nothing was. Pinning costs one config line.

---

## Below n ≈ 10, draw every trace

**Rule.** A figure summarising fewer than about ten samples with a band — an
inter-quartile range, a confidence interval, ± one standard deviation — draws every
individual sample as well, and does so regardless of *n* when the extremes are the story.
Below ten, a band alone is structurally misleading rather than merely incomplete.

**Measured (M2/M1a, 2026-08-06).** M2's first committed trajectory figure drew the seed
median with an IQR band, and the band looked tight. Underneath it one seed of five had
failed outright — a trajectory deviation of 108 000 shares, larger than the parent order.
At *n* = 5 the IQR is computed per time point from seeds 2–4; at *n* = 10 it spans seeds
3–8. Either way the extremes are outside the band *by construction*, so a band cannot show
the thing a small-sample figure most needs to show, and a reader who trusts it is being
told the spread is smaller than it is.

**How it is applied here.** `temper/eval/figures.py` draws every seed behind the band in
the trajectory overlay and every seed at every λ — traced across the grid — in the frontier
figure. The M3 brief makes it a rule for the milestone: "individual traces at any *n* below
~10, and here regardless".

**Why it generalises.** The same arithmetic applies to a benchmark's tail latencies, a
five-run timing comparison, or a bootstrap band over a handful of captures: the summary
statistic is a property of the middle of the sample, and at small *n* the middle is most
of what there is. Drawing the samples costs one line and makes the spread the reader's to
judge rather than the statistic's to hide.


---

## No code path may be reachable only at the end of a long run

**Rule.** Every path that runs *after* an expensive producer — the JSON assembly, the
verdict, the caption, the figure, the line that prints a grade, the line that says where a
file was written — must be exercisable on **fabricated data, without running the producer**,
in a test that takes milliseconds. The rule is about the property, not about any one
function: name the function and the next defect appears in the function beside it.

**Why this inverts an instinct, and why the inversion is the point.** The natural ranking
puts model and training code first and reporting code last: the training loop is where the
hard thinking is, the reporter is "just printing". The ranking is backwards, and the reason
is *when each one fails*. A defect in training code fails in minutes — the first update, the
first batch, the first assertion — and costs a re-launch. A defect in reporting code fails
**after the run**, when every number has been computed and nothing has been written, and it
costs the run. Test-criticality is not proportional to how clever a line is; it is
proportional to how much work is already sunk when the line first executes. By that measure
the last line of a driver is the most test-critical line in the repo, and the loss function
is among the least.

**Measured (Temper M4a, 2026-08-19).** The verdict block was edited to read its pass/fail
bar off a world-dependent field. The edit dropped one line — the one computing `red_flags`
— and the suite stayed green, because every test that reached `build_document` got there
by *training first* and so none of them was ever run. Ten seeds trained for two hours, were
graded correctly, and the driver then raised `NameError: red_flags` while assembling the
file. Nothing was written. Ten correct answers discarded by a missing assignment, and the
only reason it cost two hours rather than two evenings is that the pipeline is deterministic
and the re-run reproduced every seed exactly.

**Measured again (Temper M4b, 2026-08-23), four times, which is why the rule is now stated
as a property.** M4b's brief cited this note and obeyed it *exactly where it was named*:
`build_document`'s new artefact keys were covered on fabricated data before the training run,
and that coverage held. The same defect class then arrived four more times in reporting code
the note did not name.

| # | where | when it fired | what it cost |
| - | --- | --- | --- |
| 1 | `_on_seed` read `Grade` fields off a `LiquidityGrade` | after seed 0 trained | 20 min, caught by watching the launch |
| 2 | `--dry-run` printed the *deterministic* world's advantage as the bar | before the run | nothing — but it understated a pre-stated bar by 1.7x in the flattering direction |
| 3 | `tools/…_adaptivity.py`'s `main` died reporting where it wrote the figure | after rendering | the figure existed; the process did not survive saying so |
| 4 | the closing summary read a key the new world's summary lacks | after **all ten seeds** were graded | nothing, and only because `write_outputs` runs before the printing |

Number 4 is the one to remember. It is the M4a defect exactly, one milestone later, in the
same driver, forty metres down the same function — and it survived a session that had
explicitly set out to obey this note. Naming `build_document` had made `build_document` safe
and had done nothing for the twenty lines under it.

**How it is applied here.** `tests/test_sweep_document.py` grades a nudged copy of each
world's own optimum through the real grader, pairs it with a real `TrainResult` — whose
`as_dict` never touches the network, so one can be constructed without training — and runs
the real `build_document` over the pair in every cost encoding. It then does the same for
every *reporting* path: `print_verdict` and the per-seed line in both grade shapes, the
figure tool's `main` end to end, the caption's width against the canvas, and the two
failure modes that must stay visible (a red flag, and a missing input that should skip the
figure rather than half-draw it). Twenty-nine tests, seconds, no training anywhere. Each of
the four defects above is now one of them.

**A test that would have caught all five.** Take the artefact your producer writes, hand it
to every function that runs after the producer, and require them to complete. If a function
cannot be called that way, that is the finding — extract it until it can.

**Why it generalises.** Any project with an expensive producer and a cheap reporter has this
shape: a benchmark harness writing a results file, a capture tool rendering a report, a
deploy script emitting a summary, a migration printing what it changed. The reporter is
usually the part edited most often and tested least, because reaching it honestly means
paying for the producer. Fabricating its input is not a compromise — the reporter's contract
is over *data*, not over how the data was obtained — and it converts "found after the run"
into "found before it". Two useful smells: if a function's only test is marked slow, ask
whether the function is actually slow or merely *downstream* of something that is; and if a
milestone cites this note, check what it did for the paths the citation did **not** name.
---

## A clock that cannot see the interval reports zero, not an error

**Rule.** Any measurement of a short interval uses `time.perf_counter()` (or the platform's
high-resolution counter), never `time.monotonic()`. Both are monotonic and only one is
*fine*, and the difference is silent: a clock too coarse to see the interval does not fail,
it returns `0.0`. Before reporting a timing, look at the distribution — a column of exact
zeros, or of values that are all multiples of one number, is a measurement of the clock
rather than of the thing.

**Measured (M6, 2026-08-22).** The Anvil client pings its WebSocket once a bin and records
the round-trip, because a pong is queued behind whatever is already waiting for the socket
and is therefore the only true end-to-end freshness signal the wire offers. Thirteen
consecutive bins reported **exactly `0.0` seconds**. Nothing was wrong with the ping: on
Windows CPython implements `time.monotonic()` as `GetTickCount64`, whose resolution is
~15.6 ms, and a loopback round-trip is ~0.23 ms. Switching to `perf_counter` produced
0.20–0.50 ms with one 15.6 ms outlier — which is, of course, the old clock's tick showing
through the scheduler.

**How it is applied here.** `client/wire.py` and `client/run.py` time everything with
`perf_counter`, and `temper/eval/sweep.py` already did. The M6 artefacts carry the per-bin
ping series, and `tests/test_m6_runs.py` asserts a bound on it — a run that priced against
a stale book would still have produced a number, so the freshness series is evidence rather
than decoration.

**Why it generalises.** It is the same shape as a benchmark that reports 0 ns because the
compiler removed the loop: the instrument returns a plausible value for "too small to see",
and a plausible value is exactly what nobody investigates. The tell is *implausible
regularity* — repeated exact zeros, or a quantised column — and it costs nothing to look
for. Anvil's latency work and Crucible's benchmark captures both live or die on this, and
both already use high-resolution counters; the note exists so the next Python tool in the
portfolio does not have to rediscover which of the two obvious clocks is the right one.
