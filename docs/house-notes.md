# House notes

Practices that outgrew the milestone that found them. Each was measured here and applies to
every project in the portfolio that reports a trained or sampled number — Anvil's latency
distributions, Crucible's benchmark captures, Temper's RL results. `ARCHITECTURE.md` §9 is
where a *structural* decision about Temper is recorded; this file is for the smaller,
portable rules, stated once so a brief can cite them by title rather than restate them.

Cite entries by title.

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
