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

Two clauses the rule needed and did not have until M5 paid for them.

**The fabricated data must vary along every axis the path computes over.** Ten identical
grades exercise `worst` ten times and cannot catch a direction error, because the best and
the worst of ten identical numbers are the same number. Constant data proves the imports
resolved and the shapes line up; it says nothing about what the path *computes*. Vary the
input along every dimension the output claims to summarise, and prefer a fabricated case
whose right answer is known independently — one seed dominated on every axis, say, so
"which is worst" needs no agreement with the code to establish.

**The entry point is a path too.** A rule written over *functions* stops at the last
function, and `main` is below it: argument handling, the exit code, the `--expect` check,
the lines that print where things were written. Those are not reachable by calling a
function, so they have to be run — the driver invoked end to end against a fabricated or
committed artefact, in-process, asserting on its exit. `tests/test_m6_runs.py` already does
this for the M6 figure tool. The pattern existed and M5 did not apply it, which is the whole
of why two of its three defects reached a finished run.

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


**Measured again (Temper M5, 2026-08-25), three times, and this time the pre-run pass
existed, ran, and reported green.** M5's brief required a fabricated-data pass over the
*whole* reporting path and named its stages: the three-number computation, the shuffled
control's re-grade, the red-flag evaluation, the figure and its caption, and every line
reporting where a file was written. The pass was written, it called every one of those
functions before the first seed trained, and it caught nothing.

| # | where | when it fired | what it cost |
| - | --- | --- | --- |
| 1 | `main`'s `--expect` check read a bare `verdict`, unbound since the block that set it became `print_verdict` | after the artefact was written and the verdict printed | nothing, purely by ordering — but exit 1 on a sweep that passed |
| 2 | the closing line read `sweep.baselines`, which is `{}` in the alpha world; the four graded baselines live in `sweep.alpha_baselines` | after all ten seeds | the baselines were absent from the report and present in the file, so the report was quietly less than the run |
| 3 | `summarise` calls `worst` the maximum, right for a cost and backwards for a capture fraction | never — it was written to the artefact and stayed there | a wrong number in a shipped file: `alpha_capture.worst = 1.109916`, the sweep's *best* seed, and more alpha than the optimum has to give |

Numbers 1 and 2 are in `main`. The pass is a function that calls functions, and `main` is
not one of them, so no amount of coverage below it could have reached either. Number 3 the
pass *did* call — ten times, on ten identical fabricated grades, where `max` and `min`
return the same number and a direction error is invisible by construction. Three defects,
one pass, zero catches, and each one for a structural reason rather than by bad luck.

Number 3 is the one to remember, for a reason the other two do not have: it is the only
defect in this note that never fired at all. Numbers 1, 2, M4a's and all four of M4b's
announced themselves by crashing. A wrong number does not crash. It is quoted.

**And the first attempt to test it was worse than no test.** The check written for number 3
read the module's own direction table and asked whether the document agreed with it.
Reintroducing the defect — flip one field's declared direction — and the test **passed**,
because the table and the document were now agreeing with each other about a wrong answer.
*The oracle for a test must not be the thing under test.* The version that works fabricates
ten grades in which one seed is worse on every axis at once and requires every reported
`worst` to be that seed's number; "which seed is worst" is then established by construction
rather than by the code under test. Flip the direction now and it fails, naming the field.

**A test that would have caught all five, and the two clauses it needed to catch all
eight.** Take the artefact your producer writes, hand it to every function that runs after
the producer, and require them to complete. If a function cannot be called that way, that is
the finding — extract it until it can. That much catches M4a's and M4b's five. It caught
none of M5's three, so: hand the artefact to the **entry point** as well, running it the way
a user does and asserting on its exit code; and fabricate the artefact with **variation along
every axis the reporting summarises**, ideally with one case whose right answer is known
without asking the code. "Completes" is the weakest possible assertion, and three of these
eight defects were in code that completed.

**Why it generalises.** Any project with an expensive producer and a cheap reporter has this
shape: a benchmark harness writing a results file, a capture tool rendering a report, a
deploy script emitting a summary, a migration printing what it changed. The reporter is
usually the part edited most often and tested least, because reaching it honestly means
paying for the producer. Fabricating its input is not a compromise — the reporter's contract
is over *data*, not over how the data was obtained — and it converts "found after the run"
into "found before it". Three useful smells: if a function's only test is marked slow, ask
whether the function is actually slow or merely *downstream* of something that is; if a
milestone cites this note, check what it did for the paths the citation did **not** name;
and if a fabricated-data test constructs its input by repeating one record, ask which of the
computations under test can tell the copies apart.

**Not the same as the note below it.** *A guard that takes its context as an argument is
only as strong as its call sites* is the neighbour, and the two failures are opposites at
the mechanism even where the symptom rhymes. This note is about paths that only run
**late**, so the remedy is to reach them early, on fabricated data, and coverage is real
evidence. That one is about paths that run **every time**, correctly, on the wrong
context: nothing is unreached, every line executes on every invocation, and running them
earlier shows nothing — because the argument a path is handed in a test is the argument
the test's author chose.

---

## A guard that takes its context as an argument is only as strong as its call sites

**Rule.** A function that refuses one thing and permits another *on the strength of an
argument it is handed* is not a guard on its own. It is half of one; the other half is
every call site, and that is where the defect lives. So test the **callers**, not the
guard. A test that constructs a good argument and checks the refusal exercises the guard
and nothing else — it passes on the day a caller starts handing it the wrong context, and
it passes on the day a caller stops calling it at all.

**Measured (Temper M5, 2026-08-26).** `Experiment.denominator_bps` refuses to hand back
M4a's tangent advantage for an alpha-aware config, with M5's own reasoning in a comment
beside the raise: in that world the tolerance is a fraction of the net signal advantage
`J_M4a − J_DP`, and the deterministic row would return a number 2.2× smaller. The refusal
is right, and it fires — when the function is called with no row at all.

Both of the driver's banner paths got the wrong number anyway, and they got it by the two
different routes a call site has available. `--dry-run` **called the guard and handed it a
row**, so the `reference is None` condition was false, the refusal never evaluated, and the
deterministic `available_advantage` came straight back. `_header` — the banner the
4.5-hour acceptance run printed — **never called the guard**: it read `available_advantage`
off `experiment.reference()` directly, which is the same field the guard exists to
withhold. One caller went round the check; the other went round the function.

The cost is measurable rather than hypothetical. The banner printed **ε = 0.00367 bps**
where the milestone's bar is **0.00808**. The run's own median excess over `J_DP` was
**0.00532** — which *fails* the printed bar and *passes* the real one. For the length of
the run the banner and the artefact disagreed about the verdict, on the milestone's
headline gate. The artefact was right throughout: every graded number was computed from
the alpha row, and `results/m5_alpha.json` records `epsilon_met: true` for the correct
reason. Only the thing a human read was wrong.

And the second half is how it got there. `tests/test_sweep_document.py` has held a test
of this exact guard since M4b — the liquidity world's branch of it, refusing rather than
answering, asserted on the call that reaches the refusal. M5's branch was added to `denominator_bps` **by
analogy** with it, in the same commit, with a comment transposing M4b's argument into M5's
numbers. The analogy carried the implementation and not the test. Nothing asserted the new
branch, so nothing was in a position to notice that both callers were routing around it.

**How it is applied here.** Both banners now branch on the seam and print what they can
check cheaply while refusing the number they cannot: `tools/train.py`'s `_signal_header` is
`_liquidity_header` one seam along. Five tests in `tests/test_m5_sweep_document.py` read
the banners the way a reader gets them — what `--dry-run` says, what it *refuses to print*,
that the run header says it too, and that a config no sweep will start is refused rather
than reported OK. The M4b sibling finally has its M5 twin beside it.

The load-bearing assertion is the negative one: `available_advantage` and the string
`0.00367` are required to be **absent** from the output. A caller test that only checked
the good line would have passed against the old code, because the old code printed a
perfectly well-formed line containing a number about a different milestone.

**Why it generalises.** Any function whose refusal depends on context has this shape — a
permission check taking a role, a validator taking a schema, a formatter taking a locale, a
sanitiser taking an output encoding. Such a guard is cheap to test and its test proves
almost nothing about the system: it establishes that the *right* argument produces the
right answer, and every failure mode is a wrong argument arriving from somewhere else. Two
smells worth the ten seconds. If a guard's context parameter has a default meaning "work it
out yourself", then every call site passing something explicit has opted out of the guard,
and those call sites are the ones to read. And when a case is added to a guarded function
**by analogy** with an existing case, the analogy must carry the test, not only the
implementation — the existing case's test is evidence that the pattern is testable, not
evidence that the new case is tested.

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
