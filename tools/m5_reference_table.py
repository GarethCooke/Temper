"""M5 tasks 0 and 1 — the alpha reference table, its four gates, and its word.

Oracle only. **No training code for this milestone is written, imported or run
until every gate below is recorded green in the repo.** No agent is constructed
here, no env, no optimiser, no training loop; the one import that reaches
``temper.agents`` is ``load_experiment``'s ``PPOConfig``, which is the committed
contract's parser and is the same import M4a's and M4b's task-0 tools have always
made.

    python tools/m5_reference_table.py --config configs/m5_alpha.yaml

**Every number in the brief is a prediction made on unpinned numpy in a cloud
container.** None of it is a committed artefact. This regenerates all of it on the
reference box, and a material disagreement is not a tolerance to loosen — it means
the brief is wrong before the code is. The predicted value is printed beside each
measured one for exactly that reason.

Two assertions before the gates, because both would be silent if wrong
----------------------------------------------------------------------
* **Lambda's static reading is bit-identical to M4a's.** M4b needed a third
  *reading* of the selection rule and a recorded decision between two candidates
  that disagreed, because ``E[L^-beta] > 1`` moved every fixed schedule's
  objective. A zero-mean signal moves nothing, so M5 is the first Phase-2
  milestone with no lambda decision to record — and that is worth *proving* rather
  than stating. Every field of every schedule of all seventeen rows is required to
  compare ``==`` against the power-law table.
* **The decomposition's identity closes.** ``J = impact + risk + alpha +
  invariant`` is asserted at every node of every stage, not once at the root: the
  four quantities are accumulated through four separate interpolations and only
  arithmetic keeps them together.

Four gates, and all four must be green or the milestone re-shapes:

1. **``rho -> 0`` returns M4a's certified value**, to within 1e-4 of the
   advantage. The single most valuable check in the milestone and the direct
   successor to M4b's ``sigma_L -> 0``: it ties the value iteration, the
   quadrature, the stage solve, the inventory grid, the three companion value
   functions and the schedule-invariant constant to a number that *was* certified.
2. **The net advantage is >= 1 % of the objective.** Below that the training point
   is not worth an evening.
3. **The execution premium is between 25 % and 75 % of the gross alpha.** The gate
   that matters most, and the one the brief says everything else follows from.
   Below 25 % the decomposition is decorative and one headline would do; above
   75 % the advantage is a small difference of large numbers and the milestone is
   a different milestone — re-shaped *here*, before training, rather than caveated
   afterwards.
4. **The inherited red flag is measured, retired with its evidence, and replaced.**
   Price clairvoyance is computed and its looseness recorded as a multiple of the
   advantage; the convexity bound ``E[impact + risk] >= J_M4a_varying`` is asserted
   in its place and required to hold for the DP by a margin large enough to grade
   against. Retiring an inherited test needs evidence, not a sentence.

Exit status is 0 only if all four are green, both assertions hold, and the
config's committed lambda is the one the rule selects.

Task 1 — the reference, and what word it earns
-----------------------------------------------
One artefact, because there is one reference and it has two halves of different
kinds. Four things, and the last two are checks M4b did not need:

* **Convergence.** Grid and quadrature sweeps with a Richardson residual, and the
  execution premium reported at each grid — 45.5 % is only a finding if it is not
  a discretisation artefact, and that is cheap to establish now and expensive to
  withdraw later.
* **Sufficiency.** The same solve on a state carrying ``s_{k-1}``. The bar is float
  noise rather than a tolerance: a leak shows up as a systematic improvement, not
  as scatter, so a loose bar would hide exactly the failure the check exists for.
* **Timing.** Point the signal at a shock that has **already landed** and require
  the advantage to collapse. The milestone turns entirely on ``s_k`` being about
  ``xi_{k+1}`` and not ``xi_k``; an off-by-one there is the easiest defect to write
  and, if it lands in the helpful direction, it is invisible in every number the
  milestone reports — every gate would simply be greener. This is M4a's
  antithetic-mirror catch in its M5 form, and it runs before anything trains.
* **Two kinds of confidence, carried rather than narrated.** M4a's certified
  execution floor and the converged dynamic program travel with their own kind,
  role and evidence fields (:class:`~temper.eval.reference.ReferenceKind`). A
  reader who takes the floor for the optimum gets the milestone backwards.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# `python tools/x.py` from a clean clone has no `temper` on its path.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from temper.eval.experiment import (  # noqa: E402
    LAMBDA_GRIDS,
    SIGNAL_READING,
    load_experiment,
)
from temper.eval.provenance import stamp  # noqa: E402
from temper.eval.reference import (  # noqa: E402
    SIGNAL_BOUND_PATHS,
    alpha_reference_row,
    reference_table,
    select_lambda,
    signal_static_table,
)
from temper.oracle import (  # noqa: E402
    CLAIRVOYANT_GRID_POINTS,
    CLAIRVOYANT_PATHS,
    DEFAULT_SIGNAL_GRID_POINTS,
    DEFAULT_SIGNAL_QUADRATURE_NODES,
    ENCODINGS,
    POWER_LAW_ENCODING,
    NoSignal,
    OneStepSignal,
    alpha_coefficient,
    alpha_optimum,
    augmented_alpha_optimum,
    clairvoyant_price_values,
    cost_moments,
    execution_floor_bps,
    power_law_optimum,
    richardson_residual,
    schedule_invariant_bps,
    signal_path_objective_bps,
    trades,
)
from temper.seeding import M5_REFERENCE_POOL, pool_rng  # noqa: E402

#: Task 0's bars, pre-stated in `docs/briefs/M5-alpha-aware-execution.md`.
MAX_RHO_ZERO_FRACTION = 1.0e-4       # of the advantage
MIN_ADVANTAGE_FRACTION = 0.01        # of the objective
MIN_PREMIUM_FRACTION = 0.25          # of the gross alpha
MAX_PREMIUM_FRACTION = 0.75

#: The bar on ``|J - (impact + risk + alpha + invariant)|``. Not a brief number:
#: the brief says "asserted rather than assumed" and this is what asserting it
#: costs. Four interpolations that sum to a fifth agree to ~1e-15 bps, and a
#: mis-attributed *term* would be six orders larger, so anything this side of a
#: nanobasis point is float noise and anything past it is a defect.
IDENTITY_TOLERANCE_BPS = 1.0e-9

#: Task 1's sufficiency bar, and it is **float noise rather than a tolerance**.
#: Carrying ``s_{k-1}`` cannot improve the value, because that signal predicted a
#: shock that has already landed on inventory the previous decision fixed — its
#: information is spent, not merely stale. A leak would show as a *systematic*
#: improvement, so a bar set where a tolerance would sit (M4b used 1e-9 on the
#: spread) would hide exactly the failure being looked for. Measured: the value
#: agrees to 0.0e+00 and the column spread is 4.4e-16 bps at 401 points and
#: 6.7e-16 at 1601 — the same order M4b measured for liquidity.
SUFFICIENCY_TOLERANCE_BPS = 1.0e-14

#: The grid task 1's cross-lambda premium stability is re-read at. Four times
#: coarser than the reference's, which is the point: if 45.5 % were a
#: discretisation artefact it would move here. It does, at three lambdas, and that
#: is the finding — see :func:`render_premium_stability`.
STABILITY_GRID_POINTS = 401

#: Grids the *worst* lambda's premium is escalated through, as the witness for
#: why the cross-lambda claim is stated over a region rather than over the grid.
ESCALATION_GRID_POINTS = (401, 801, DEFAULT_SIGNAL_GRID_POINTS, 3201)

#: A lambda counts as **resolved** when the coarse and fine grids agree on its
#: premium fraction to within this, in percentage points of the gross alpha. Half
#: a point is two orders inside the 45 % it is qualifying, and an order inside the
#: 2-point span the claim is about — tight enough that a lambda which passes is
#: genuinely settled, loose enough not to reject one for float noise.
PREMIUM_RESOLVED_SHIFT = 0.005

#: The span the resolved region's premium fractions must fit inside for the
#: cross-lambda claim to stand at all. Pre-stated here rather than read off the
#: measurement: without it "the premium is roughly constant" is whatever the
#: numbers turn out to be.
PREMIUM_RESOLVED_SPAN = 0.02

#: The invented parameter, at the six values the value-of-signal curve reports.
#: The middle one is trained; the outer ones exist because a single invented
#: parameter with a single number beside it reads as calibration, and it is not —
#: and because the top two are where the milestone stops being about execution.
RHO_CURVE = (0.0025, 0.005, 0.01, 0.02, 0.05, 0.2)

#: The brief's predictions, so a disagreement is visible rather than discoverable.
#: Made on unpinned numpy in a cloud container; **none of it is an artefact**.
PREDICTED = {
    "alpha_coefficient": 42.9893,
    "volatility_multiple": 18.0,
    "j_m4a": 2.383215,
    "j_dp": 2.302456,
    "advantage": 0.080759,
    "advantage_fraction": 0.0339,
    "execution_floor": 1.819586,
    "impact_rho_zero": 0.907718,
    "risk_rho_zero": 0.911870,
    "impact": 0.939129,
    "risk": 0.947764,
    "alpha_available": 0.148067,
    "execution_premium": 0.067305,
    "premium_fraction": 0.455,
    "clairvoyant_bps": -84.39,
    "clairvoyant_half_width": 8.77,
    "clairvoyant_multiple": 1075.0,
    "curve": {
        "0.0025": 0.005698,
        "0.005": 0.021998,
        "0.01": 0.080753,
        "0.02": 0.278472,
        "0.05": 1.267187,
        "0.2": 8.890347,
    },
}


def _delta(measured: float, predicted: float) -> str:
    """How far the box's number is from the brief's, in the brief's own units."""
    if predicted == 0.0:
        return f"{measured:+.6f}"
    return f"{measured - predicted:+.6f} ({(measured - predicted) / predicted:+.2%})"


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


#: Paths for the *refinement* of the feasible upper bound. At the pre-stated
#: M = 20 000 the bound's own half-width is ~1 % of the advantage while the gap it
#: is being asked to resolve is a tenth of that, so the sign of that gap at 20 000
#: is a coin toss and reporting it as if it meant something would be reading
#: noise. Ten times the paths costs seconds and settles it — M4b's mechanism, kept
#: because M5 has the same question about the same kind of bound.
REFINEMENT_PATHS = 200_000


def refine_feasible_bound(experiment, signal, paths: int) -> dict:
    """The feasible upper bound again at ten times the paths.

    The DP's greedy policy is a *real* policy, so its mean conditional cost is an
    unbiased estimate of an attainable value and therefore cannot be below the
    optimum it came from by anything but sampling error and grid discretisation.
    A negative gap at the reported M is the first thing a careful reader stops on,
    and the honest answer is to measure it at a path count where the sign means
    something rather than to argue that it is inside the interval.
    """
    market = experiment.case.market
    order_size = experiment.case.order_size
    lam = experiment.lambda_risk
    optimum = alpha_optimum(market, order_size, lam, signal)
    reference = power_law_optimum(market, order_size, lam)
    level = cost_moments(reference, market).objective(lam)
    weights = trades(reference, market) / order_size

    rng = pool_rng(experiment.seeds.root_seed, M5_REFERENCE_POOL, 901)
    signals = signal.draw(rng, (paths, market.n_bins))
    difference = signal_path_objective_bps(
        optimum.greedy_weights(signals), signals, market, order_size, lam, signal
    ) - signal_path_objective_bps(
        weights, signals, market, order_size, lam, signal
    )
    value = level + float(difference.mean())
    half = float(1.96 * difference.std(ddof=1) / math.sqrt(paths))
    return {
        "paths": paths,
        "value_bps": value,
        "half_width_bps": half,
        "gap_bps": value - optimum.objective_bps,
    }


def render_table(table, rule) -> str:
    """The alpha analogue of M4a's and M4b's task-0 tables, over the whole grid."""
    lines = [
        "| λ | J_twap | J_tangent | J_M4a | J_DP | net advantage | adv / J_M4a "
        "| gross alpha | premium | premium / alpha | TWAP gap | max bin, M4a "
        "| max bin, DP mean | rule |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
        "| ---: | ---: | ---: | :---: |",
    ]
    for row in table:
        admits = rule.admits(row)
        why = "✓" if admits else ("i" if row.twap_gap < rule.min_twap_gap else "ii")
        lines.append(
            f"| {row.lambda_risk:.3e} | {row.twap.objective:.4f} | "
            f"{row.tangent.objective:.4f} | {row.optimal.objective:.4f} | "
            f"{row.adaptive_bps:.4f} | {row.signal_advantage:.5f} | "
            f"{row.advantage_fraction:.2%} | {row.alpha_available:.5f} | "
            f"{row.execution_premium:.5f} | {row.premium_fraction:.1%} | "
            f"{row.twap_gap:.2%} | {row.optimal.max_bin_fraction:.1%} | "
            f"{row.mean_schedule_max_bin:.1%} | {why} |"
        )
    return "\n".join(lines)


def render_lambda_identity(experiment, static_table, rule) -> tuple[bool, dict]:
    """Assert — do not assume — that the static reading is M4a's, to the bit.

    The brief asks for this as a *result*: "a zero-mean signal does not move a
    deterministic schedule's objective by a single float, so the liquidity world's
    third-reading problem does not recur". M4b's session had to choose between two
    readings that disagreed by 0.011 percentage points and record the choice; this
    one has nothing to choose, and the way to say so honestly is to compare every
    float rather than to explain why they must match.

    Every schedule of every row, on ``expected``, ``variance``, ``risk``,
    ``objective`` and the trajectory itself, compared with ``==``.
    """
    market, order_size = experiment.case.market, experiment.case.order_size
    grid = LAMBDA_GRIDS[experiment.lambda_grid]
    m4a_table = reference_table(
        market, order_size, grid, encoding=POWER_LAW_ENCODING
    )

    mismatches: list[str] = []
    compared = 0
    for signal_row, m4a_row in zip(static_table, m4a_table, strict=True):
        assert signal_row.lambda_risk == m4a_row.lambda_risk
        for name in sorted(m4a_row.schedules):
            mine, theirs = signal_row.schedules[name], m4a_row.schedules[name]
            for field in ("expected", "variance", "risk", "excess_risk", "objective"):
                compared += 1
                if getattr(mine, field) != getattr(theirs, field):
                    mismatches.append(
                        f"lambda {m4a_row.lambda_risk:.3e}, {name}.{field}: "
                        f"{getattr(mine, field)!r} != {getattr(theirs, field)!r}"
                    )
            compared += 1
            if not (mine.trajectory == theirs.trajectory).all():
                mismatches.append(
                    f"lambda {m4a_row.lambda_risk:.3e}, {name}.trajectory"
                )

    identical = not mismatches
    signal_choice = select_lambda(static_table, rule).lambda_risk
    readings = {
        encoding: experiment.rule_selected(encoding).lambda_risk
        for encoding in ENCODINGS
    }
    readings[SIGNAL_READING] = signal_choice
    agree = len(set(readings.values())) == 1
    matches_config = experiment.lambda_risk == signal_choice

    print("## Lambda needs no new reading, and that is the result\n")
    for name, lam in sorted(readings.items()):
        print(f"- `{name}` selects **{lam:.6e}** (10^{math.log10(lam):.1f}).")
    print(
        f"- {'**They agree.**' if agree else '**THEY DISAGREE.**'} The config "
        f"commits {experiment.lambda_risk:.6e} — "
        f"{'agree' if matches_config else '**DISAGREE**'}."
    )
    print(
        f"- The signal reading is **bit-identical** to the power-law one, not "
        f"merely equal to three digits: **{compared:,} floats and trajectories "
        f"compared with `==` across {len(static_table)} lambdas, "
        f"{len(mismatches)} mismatches**."
    )
    print(
        "- Why it has to be: a fixed schedule's holdings do not depend on `s`, so "
        "its expected alpha is `-A rho sum_k h_k E[s]` and `E[s]` is a float zero "
        "(`temper.oracle.alpha.expected_alpha_bps`). Adding a float zero to an "
        "objective is the identity operation. **M4b had to decide something here "
        "and record it** — liquidity inflated every fixed schedule's charge by "
        "`E[L^-0.6] = 1.1275`, the static and adaptive readings disagreed, and the "
        "rejected one is recorded with its margin. M5 has nothing to decide, and "
        "this is the arithmetic that says so rather than the argument.\n"
    )
    for line in mismatches[:10]:
        print(f"  - **MISMATCH** {line}")
    if mismatches:
        print()

    return identical and agree and matches_config, {
        "readings": readings,
        "agree": agree,
        "committed_matches": matches_config,
        "recorded_reading": SIGNAL_READING,
        "reading_rule_applied_to": "static",
        "bit_identical_to_m4a": identical,
        "fields_compared": compared,
        "mismatches": mismatches,
    }


def render_convergence(experiment, signal, floor: float) -> dict:
    """The DP's own numbers: grid, quadrature, and the ``rho -> 0`` differential.

    The last of the three is gate 1 and the most valuable check in the milestone,
    because it ties every piece of new machinery to a number that *was* certified.
    The first two are what make its tolerance readable: a residual of 2e-6 bps
    means nothing until the grid it was measured on is shown to be converged.

    Task 1 adds the **execution premium at each grid**. 45.5 % of the gross alpha
    is on its way to being a §9 entry, and an entry that turns out to be a
    discretisation artefact has to be withdrawn — which costs far more than
    reading the number off four solves that were being done anyway.
    """
    market = experiment.case.market
    order_size = experiment.case.order_size
    lam = experiment.lambda_risk

    print("## Task 1 — the reference, and what word it earns\n")
    print("### Grid and quadrature\n")
    grid = {}
    premium = {}
    for points in (201, 401, 801, DEFAULT_SIGNAL_GRID_POINTS):
        started = time.perf_counter()
        optimum = alpha_optimum(market, order_size, lam, signal, points=points)
        grid[points] = optimum.objective_bps
        premium[points] = (optimum.execution_bps - floor) / -optimum.alpha_bps
        print(
            f"- {points:5d} inventory points: **{grid[points]:.6f} bps**, "
            f"execution premium **{premium[points]:.3%}** of the gross alpha "
            f"({time.perf_counter() - started:.1f} s)"
        )
    extrapolant, residual = richardson_residual(
        grid[801], grid[DEFAULT_SIGNAL_GRID_POINTS]
    )
    print(
        f"- Second order in the spacing, so Richardson-extrapolating the last two "
        f"gives **{extrapolant:.6f} bps** and a residual of **{residual:.2e} bps** "
        f"— the numerical uncertainty in a reference this milestone is explicit "
        f"about not having certified.\n"
    )

    quadrature = {}
    for nodes in (3, 5, 7, DEFAULT_SIGNAL_QUADRATURE_NODES, 21):
        quadrature[nodes] = alpha_optimum(
            market, order_size, lam, signal, nodes=nodes
        ).objective_bps
    moments = {}
    for nodes in (5, DEFAULT_SIGNAL_QUADRATURE_NODES, 21):
        values, weights = signal.quadrature(nodes)
        moments[nodes] = (float(weights @ values), float(weights @ values**2))
    print(
        "- Quadrature: "
        + ", ".join(
            f"{nodes} nodes → {value:.6f}" for nodes, value in quadrature.items()
        )
        + f". Pinned at {DEFAULT_SIGNAL_QUADRATURE_NODES}: the quadrature is the "
        "cheap axis (the inventory grid dominates the cost by two orders), so "
        "sitting past the knee removes the node count from the list of things a "
        "later session has to re-argue.\n"
    )
    print(
        "- The quadrature's own moments, because the signal's *mean* is what the "
        "lambda claim rests on: "
        + ", ".join(
            f"{nodes} nodes → E[s] = {first:+.2e}, E[s²] = {second:.15f}"
            for nodes, (first, second) in moments.items()
        )
        + ". The first moment is float noise rather than an exact zero, which is "
        "worth writing down: it means the *quadrature's* mean is 1e-17 and not 0, "
        "and it is why the bit-identity claim above is made about `signal.mean()` "
        "— an exact zero the static route uses — and not about this. At `rho = "
        "0.01` this noise is worth `A rho E[s] h ~ 1e-17` bps to the DP, which is "
        "two orders below the identity residual and eleven below the grid.\n"
    )
    span = max(premium.values()) - min(premium.values())
    print(
        f"- **The execution premium moves {span:.3%} across a factor of eight in "
        f"the grid** — {premium[201]:.3%} at 201 points against "
        f"{premium[DEFAULT_SIGNAL_GRID_POINTS]:.3%} at "
        f"{DEFAULT_SIGNAL_GRID_POINTS}, monotone and settling. At *this* lambda it "
        f"is a property of the problem rather than of the discretisation, which is "
        f"the first half of what a 45 %-of-gross-alpha finding has to establish "
        f"before it is promoted anywhere. The second half — whether the same is "
        f"true across the grid — is below, and it is where the answer stops being "
        f"a formality.\n"
    )
    return {
        "grid": {str(k): v for k, v in grid.items()},
        "premium_fraction_by_grid": {str(k): v for k, v in premium.items()},
        "premium_fraction_grid_span": span,
        "richardson_extrapolant_bps": extrapolant,
        "richardson_residual_bps": residual,
        "quadrature": {str(k): v for k, v in quadrature.items()},
        "quadrature_moments": {
            str(k): {"first": f, "second": s} for k, (f, s) in moments.items()
        },
    }


def render_rho_zero_gate(experiment, advantage: float) -> tuple[bool, dict]:
    """Gate 1 — the differential against a *certified* number, by two routes."""
    market = experiment.case.market
    order_size = experiment.case.order_size
    lam = experiment.lambda_risk

    certified = cost_moments(power_law_optimum(market, order_size, lam), market).objective(
        lam
    )
    varying_route = execution_floor_bps(market, order_size, lam) + schedule_invariant_bps(
        market, order_size
    )
    full = alpha_optimum(market, order_size, lam, OneStepSignal(0.0))
    collapsed = alpha_optimum(market, order_size, lam, NoSignal())
    bar = MAX_RHO_ZERO_FRACTION * advantage
    green = (
        abs(full.objective_bps - certified) <= bar
        and abs(collapsed.objective_bps - certified) <= bar
    )

    print("## Gate 1 — `rho -> 0` returns M4a's certified value\n")
    print(
        f"- M4a's certified `power_law_optimum` value is **{certified:.9f} bps**. "
        f"The two routes to it — `cost_moments` and `varying_objective_bps + "
        f"schedule_invariant_bps` — agree to **{abs(certified - varying_route):.1e} "
        f"bps**, which matters because the DP is built the second way and graded "
        f"against the first."
    )
    print(
        f"- The DP at `rho = 0` with the **full {full.quadrature_nodes}-node "
        f"quadrature** returns **{full.objective_bps:.9f} bps** — a difference of "
        f"**{full.objective_bps - certified:+.3e} bps**. Its alpha term is "
        f"**{full.alpha_bps:+.3e} bps**: the expectation is actually taken and it "
        f"cancels, rather than being skipped by a one-node shortcut."
    )
    print(
        f"- With the signal seam *absent* (`NoSignal`, one node, M4b's analogue): "
        f"**{collapsed.objective_bps:.9f} bps**, "
        f"**{collapsed.objective_bps - certified:+.3e} bps**."
    )
    print(
        f"- Bar: **{MAX_RHO_ZERO_FRACTION:g} of the advantage** = "
        f"**{bar:.3e} bps**. {'**GREEN**' if green else '**RED**'} — the worse of "
        f"the two routes uses "
        f"{max(abs(full.objective_bps - certified), abs(collapsed.objective_bps - certified)) / bar:.1%} "
        f"of it."
    )
    print(
        "- That difference is grid discretisation, not a disagreement, and it "
        "calibrates what \"converged\" is worth here. **This is the check that ties "
        "the value iteration, the quadrature, the stage solve, the three companion "
        "value functions and the constant back to a number with a Cholesky "
        "factorisation and a 1.2e-15 KKT residual behind it** — the direct "
        "successor to M4b's `sigma_L -> 0`, and the only place in this milestone "
        "where new machinery is measured by something that is not also new.\n"
    )
    return green, {
        "green": green,
        "bar_fraction": MAX_RHO_ZERO_FRACTION,
        "bar_bps": bar,
        "certified_bps": certified,
        "certified_varying_route_bps": varying_route,
        "full_quadrature_bps": full.objective_bps,
        "full_quadrature_difference_bps": full.objective_bps - certified,
        "full_quadrature_alpha_bps": full.alpha_bps,
        "collapsed_bps": collapsed.objective_bps,
        "collapsed_difference_bps": collapsed.objective_bps - certified,
    }


def render_clairvoyant_gate(
    experiment, row, floor: float, advantage: float, paths: int
) -> tuple[bool, dict]:
    """Gate 4 — the inherited red flag, measured, retired, and replaced."""
    market = experiment.case.market
    order_size = experiment.case.order_size
    lam = experiment.lambda_risk
    optimum = row.optimum

    rng = pool_rng(experiment.seeds.root_seed, M5_REFERENCE_POOL, 900)
    shocks = rng.standard_normal((paths, market.n_bins))
    measured = {}
    for points in (CLAIRVOYANT_GRID_POINTS, 2 * CLAIRVOYANT_GRID_POINTS - 1):
        started = time.perf_counter()
        values = clairvoyant_price_values(
            market, order_size, lam, shocks, points=points
        )
        measured[points] = {
            "value_bps": float(values.mean()),
            "half_width_bps": float(1.96 * values.std(ddof=1) / math.sqrt(paths)),
            "seconds": time.perf_counter() - started,
        }
    coarse = measured[CLAIRVOYANT_GRID_POINTS]
    fine = measured[2 * CLAIRVOYANT_GRID_POINTS - 1]
    looseness = (row.optimal.objective - coarse["value_bps"]) / advantage

    execution = optimum.execution_bps
    margin = execution - floor
    convex_holds = execution > floor
    green = convex_holds and looseness > 1.0

    print("## Gate 4 — the inherited red flag retires, with its evidence\n")
    print(
        f"- Price clairvoyance over **{paths} shock paths** is worth "
        f"**{coarse['value_bps']:.2f} ± {coarse['half_width_bps']:.2f} bps** on a "
        f"{CLAIRVOYANT_GRID_POINTS}-point inventory grid "
        f"({coarse['seconds']:.0f} s). Brief predicted "
        f"{PREDICTED['clairvoyant_bps']:.2f} ± "
        f"{PREDICTED['clairvoyant_half_width']:.2f}; the two draws differ by "
        f"{coarse['value_bps'] - PREDICTED['clairvoyant_bps']:+.2f} bps against "
        f"half-widths of ~{coarse['half_width_bps']:.1f}."
    )
    print(
        f"- The advantage it would license is "
        f"**{row.optimal.objective - coarse['value_bps']:.2f} bps** = "
        f"**{looseness:,.0f}x** the signal's {advantage:.5f} (brief predicted "
        f"{PREDICTED['clairvoyant_multiple']:,.0f}x). **A bound that loose can "
        f"never fire.** Per-bin volatility is "
        f"{alpha_coefficient(market) / row.optimal.objective:.1f}x the objective, "
        f"so a clairvoyant trader is not executing an order — it is trading one."
    )
    print(
        f"- **The number is grid-converged; what is left is sampling error.** "
        f"Doubling the inventory grid to {2 * CLAIRVOYANT_GRID_POINTS - 1} points "
        f"**on the same paths** moves it by "
        f"{abs(fine['value_bps'] - coarse['value_bps']):.1e} bps — nothing, because "
        f"unlike the reference DP's this value function is dominated by a term "
        f"*linear* in inventory and linear interpolation is exact on one. (What "
        f"error it does carry runs the conservative way: a linear interpolant of a "
        f"convex function lies above it, so this over-estimates the clairvoyant "
        f"value and under-states its looseness.) The ± is Monte Carlo, and at "
        f"{coarse['half_width_bps'] / abs(coarse['value_bps']):.0%} of a number "
        f"three orders from the thing it would have to resolve, a fourth digit "
        f"would not change a word."
    )
    print("\n### The replacement, which is rigorous and *certified*\n")
    print(
        f"- Impact and risk are convex in the trade weights and contain **no "
        f"signal term at all**, so by Jensen `E[impact + risk] >= min over "
        f"deterministic schedules` for **any** policy, adapted or not — with "
        f"equality only at M4a's optimum."
    )
    print(
        f"- That minimum is `J_M4a_varying` = **{floor:.6f} bps** (brief predicted "
        f"{PREDICTED['execution_floor']:.6f}; measured "
        f"{_delta(floor, PREDICTED['execution_floor'])}), and it is **certified** — "
        f"Cholesky-PD Hessian, relative KKT residual 1.2e-15, an independent "
        f"bisection solver agreeing to 3.1e-15 of X."
    )
    print(
        f"- The DP sits at `E[impact + risk]` = **{execution:.6f} bps**, "
        f"**{margin:+.6f} bps** above the floor — "
        f"{'**GREEN**' if convex_holds else '**RED**'}, and a margin of "
        f"{margin / advantage:.0%} of the whole advantage, which is what "
        f"\"large enough to grade against\" means: an agent has to be that far "
        f"wrong before the flag is ambiguous."
    )
    print(
        "- **M4a earned its red flag with an algebraic certificate; M4b's rested "
        "on a relaxation; M5's rests on a certificate again, but only over the "
        "half of the objective where one exists.** The other half is graded "
        "against a converged number and says so.\n"
    )
    return green, {
        "green": green,
        "clairvoyant": {str(k): v for k, v in measured.items()},
        "clairvoyant_paths": paths,
        "clairvoyant_looseness_multiple": looseness,
        "retired": True,
        "replacement": "convexity floor at M4a's certified optimum",
        "execution_floor_bps": floor,
        "execution_bps": execution,
        "margin_bps": margin,
        "margin_as_fraction_of_advantage": margin / advantage,
        "convexity_holds": convex_holds,
    }


def render_sufficiency(experiment, signal) -> tuple[bool, dict]:
    """Task 1 — ``(k, x_k, s_k)`` is a sufficient statistic, measured.

    The reference is the optimum over **all** adapted policies only if carrying
    more of the past cannot help. Under an i.i.d. one-step-ahead signal it cannot,
    and the reason is sharper than M4b's: ``s_{k-1}`` predicted ``xi_k``, that
    shock has already landed, and the inventory it was charged on was fixed by the
    previous decision. The information is **spent**, not merely stale.

    The bar is float noise. A leak — a signal with memory, a seam that lets a past
    draw reach a future one — produces a *systematic* improvement rather than
    scatter, so a bar set where a tolerance would sit would hide it.
    """
    market = experiment.case.market
    order_size = experiment.case.order_size
    lam = experiment.lambda_risk

    print("### Sufficiency — carrying the spent signal buys nothing\n")
    rows = {}
    green = True
    for points in (STABILITY_GRID_POINTS, DEFAULT_SIGNAL_GRID_POINTS):
        started = time.perf_counter()
        plain = alpha_optimum(
            market, order_size, lam, signal, points=points
        ).objective_bps
        augmented = augmented_alpha_optimum(
            market, order_size, lam, signal, points=points
        )
        difference = augmented.objective_bps - plain
        ok = (
            abs(difference) <= SUFFICIENCY_TOLERANCE_BPS
            and augmented.column_spread <= SUFFICIENCY_TOLERANCE_BPS
        )
        green = green and ok
        rows[points] = augmented.as_dict() | {
            "plain_bps": plain,
            "difference_bps": difference,
            "green": ok,
        }
        print(
            f"- {points:5d} points: plain **{plain:.9f}**, augmented "
            f"**{augmented.objective_bps:.9f}**, difference "
            f"**{difference:+.1e} bps**, column spread "
            f"**{augmented.column_spread:.1e} bps** — "
            f"{'**GREEN**' if ok else '**RED**'} "
            f"({time.perf_counter() - started:.1f} s)"
        )
    print(
        f"- Bar **{SUFFICIENCY_TOLERANCE_BPS:g} bps**, and it is float noise rather "
        f"than a tolerance: an augmented state that *improved* the value would mean "
        f"the seam leaks, and a leak is systematic. M4b measured the same order "
        f"(4.4e-16) for liquidity."
    )
    print(
        "- The column spread is the sharper half. It says the continuation is the "
        "same array for **every** previous signal, not merely that two scalars "
        "agreed at one inventory — and it has content only because "
        "`transition_quadrature` is a genuine `(nodes, nodes)` object that *could* "
        "have carried a dependence.\n"
    )
    return green, rows


def render_timing(experiment, signal, floor: float, advantage: float) -> tuple[bool, dict]:
    """Task 1 — point the signal at a shock that has already landed.

    The check M4b did not need and this milestone cannot do without. Everything
    M5 claims rests on ``s_k`` being about ``xi_{k+1}`` and not ``xi_k``. An
    off-by-one in the seam's timing is the easiest defect to write here, and it is
    invisible in every number the milestone reports if it lands in the helpful
    direction: the advantage would simply be bigger and every gate would still be
    green.

    So the milestone's own machinery is pointed one bin the wrong way and required
    to find **nothing**. It must, and for a reason that is arithmetic rather than
    empirical: at lag 0 the alpha term prices the inventory carried *into* the bin,
    which the previous decision already fixed, so it is constant in the action and
    mean zero over the signal. If it comes back non-zero, the term is reaching a
    decision it should not be able to reach.
    """
    market = experiment.case.market
    order_size = experiment.case.order_size
    lam = experiment.lambda_risk
    reference = cost_moments(
        power_law_optimum(market, order_size, lam), market
    ).objective(lam)

    landed = OneStepSignal(signal.correlation(), bins_ahead=0)
    uninformative = OneStepSignal(0.0)
    already = alpha_optimum(market, order_size, lam, landed)
    blind = alpha_optimum(market, order_size, lam, uninformative)
    collapsed = reference - already.objective_bps
    # The collapse has to be to the *rho = 0* value, not merely to something
    # small: at rho = 0 the DP still carries the grid's own discretisation, and
    # calling that residual "an advantage of 1.8e-06 bps" would be reading noise.
    # Float noise, not a tolerance — the same bar the sufficiency check uses, and
    # for the same reason. At 1601 points the two solves land bit-identical; at
    # coarser grids they differ by an ulp, because the lag-0 path threads a state
    # term through additions the lag-1 path does not. An ulp is the answer "worth
    # nothing"; anything a reader could see would not be.
    green = (
        abs(already.objective_bps - blind.objective_bps) <= SUFFICIENCY_TOLERANCE_BPS
        and abs(already.alpha_bps) < 1.0e-12
        and abs(collapsed) < 1.0e-4 * advantage + 2.0e-06
    )

    print("### Timing — a signal about a shock that has already landed is worth zero\n")
    print(
        f"- Same rho ({signal.correlation()}), same machinery, one bin the wrong "
        f"way: `s_k` made to predict `xi_k` instead of `xi_{{k+1}}`. The dynamic "
        f"program returns **{already.objective_bps:.9f} bps** against the "
        f"uninformative `rho = 0` solve's **{blind.objective_bps:.9f}** — "
        f"{'**bit-identical**' if already.objective_bps == blind.objective_bps else f'**{already.objective_bps - blind.objective_bps:+.1e} bps** apart, one ulp'}."
    )
    print(
        f"- Its alpha term is **{already.alpha_bps:+.2e} bps**, and the advantage "
        f"it would license is **{collapsed:+.2e} bps** — against the model's "
        f"{advantage:.6f}. That is the grid's own discretisation with the sign it "
        f"has at `rho = 0`, not a small advantage: **{abs(collapsed) / advantage:.5%}** "
        f"of the real one."
    )
    print(
        f"- {'**GREEN**' if green else '**RED**'}. The whole milestone is one index "
        f"apart from measuring nothing, and this is the arithmetic that says which "
        f"side of it the seam is on — run before anything trains, which is the only "
        f"time it is cheap.\n"
    )
    return green, {
        "green": green,
        "rho": signal.correlation(),
        "bins_ahead": 0,
        "objective_bps": already.objective_bps,
        "uninformative_objective_bps": blind.objective_bps,
        "bit_identical_to_uninformative": (
            already.objective_bps == blind.objective_bps
        ),
        "alpha_bps": already.alpha_bps,
        "collapsed_advantage_bps": collapsed,
        "collapsed_as_fraction_of_advantage": collapsed / advantage,
    }


def render_reference_kinds(row) -> dict:
    """Task 1 — two references, one artefact, and each carries its own word."""
    kinds = row.reference_kinds
    print("### Two references, and which word each earned\n")
    print("| reference | value, bps | kind | certified | what it is |")
    print("| --- | ---: | :---: | :---: | --- |")
    for name, kind in kinds.items():
        role = kind.role.split(".")[0] + "."
        print(
            f"| `{name}` | {kind.value_bps:.6f} | **{kind.kind}** | "
            f"{'yes' if kind.certified else 'no'} | {role} |"
        )
    print(
        f"\n- `execution_floor` — {kinds['execution_floor'].evidence}"
    )
    print(f"- `adaptive_optimum` — {kinds['adaptive_optimum'].evidence}")
    print(
        "- **They are not interchangeable and the artefact carries the difference "
        "rather than narrating it.** A reader who takes the floor for the optimum "
        "makes the agent's job look seven times larger than it is; one who takes "
        "the optimum for a certified object claims a Cholesky factorisation for a "
        "number that has a Richardson residual. `ReferenceKind` admits exactly two "
        "words and refuses a `certified` flag that contradicts the one it was "
        "given.\n"
    )
    return {name: kind.as_dict() for name, kind in kinds.items()}


def render_premium_stability(experiment, signal, table, lambdas) -> dict:
    """Task 1 — the cross-lambda premium range, re-read at a coarser grid.

    The brief's session notes flag 44.9-49.9 % across seventeen lambdas as a
    candidate §9 entry: it is what turns decomposed grading from an accommodation
    for one operating point into a structural claim. An entry that turns out to be
    a discretisation artefact has to be withdrawn, so the range is measured again
    on a grid four times coarser before anything is promoted.

    **And the check fires.** Fourteen of the seventeen lambdas agree between the
    grids to four decimal places; the top three do not, by up to 24 percentage
    points. Escalating the worst one shows why and shows which reading was wrong:
    the coarse grid is not adding noise to a converged number, the *fine* grid had
    not converged either. At ``lambda = 10^-1`` the optimum liquidates 99 % of the
    order in the first bin, the gross alpha falls to 0.001 bps, and the premium
    becomes a difference of quantities the inventory grid cannot resolve.

    So the claim is stated over the region where it is measured rather than over
    the region where it was convenient, and the top of the earlier range goes with
    it: **the 49.9 % was the grid, not the problem**, and the true span over the
    resolved region is *narrower* than the number the session note recorded. That
    is the entry getting stronger for being checked, which is the outcome this
    check was cheap insurance against not getting.
    """
    market = experiment.case.market
    order_size = experiment.case.order_size
    ordered = sorted(lambdas)
    fine = [row.premium_fraction for row in table]

    started = time.perf_counter()
    coarse = []
    for lam in ordered:
        optimum = alpha_optimum(
            market, order_size, lam, signal, points=STABILITY_GRID_POINTS
        )
        floor = execution_floor_bps(market, order_size, lam)
        coarse.append((optimum.execution_bps - floor) / -optimum.alpha_bps)

    shifts = [abs(c - f) for c, f in zip(coarse, fine, strict=True)]
    resolved = [s <= PREMIUM_RESOLVED_SHIFT for s in shifts]
    settled = [f for f, ok in zip(fine, resolved, strict=True) if ok]
    span = max(settled) - min(settled)
    green = span <= PREMIUM_RESOLVED_SPAN

    print("### The cross-lambda premium range, and where it is resolvable\n")
    print(
        f"| λ | premium @ {DEFAULT_SIGNAL_GRID_POINTS} | premium @ "
        f"{STABILITY_GRID_POINTS} | shift | max bin, M4a | resolved |"
    )
    print("| --- | ---: | ---: | ---: | ---: | :---: |")
    for lam, row, f, c, shift, ok in zip(
        ordered, table, fine, coarse, shifts, resolved, strict=True
    ):
        print(
            f"| {lam:.3e} | {f:.3%} | {c:.3%} | {shift:+.3%} | "
            f"{row.optimal.max_bin_fraction:.1%} | {'✓' if ok else '**✗**'} |"
        )

    print(
        f"\n- **{sum(resolved)} of {len(fine)} lambdas are resolved** — the two "
        f"grids agree to better than {PREMIUM_RESOLVED_SHIFT:.1%} of the gross "
        f"alpha — and across those the premium spans **{min(settled):.1%} to "
        f"{max(settled):.1%}**, a range of {span:.2%} against a pre-stated "
        f"{PREMIUM_RESOLVED_SPAN:.0%}. {'**GREEN**' if green else '**RED**'}."
    )

    worst_index = max(range(len(shifts)), key=lambda i: shifts[i])
    worst_lambda = ordered[worst_index]
    escalation = {}
    for points in ESCALATION_GRID_POINTS:
        optimum = alpha_optimum(
            market, order_size, worst_lambda, signal, points=points
        )
        floor = execution_floor_bps(market, order_size, worst_lambda)
        escalation[points] = (optimum.execution_bps - floor) / -optimum.alpha_bps
    print(
        f"- **The three that are not resolved are not noisy — they are "
        f"unconverged, at both grids.** Escalating the worst, λ = "
        f"{worst_lambda:.3e}: "
        + ", ".join(f"{points} → {value:.1%}" for points, value in escalation.items())
        + f". The {DEFAULT_SIGNAL_GRID_POINTS}-point reading of "
        f"{fine[worst_index]:.1%} is itself a way-point, not an answer, so **the "
        f"top of the range recorded in the brief's session notes was the grid "
        f"rather than the problem** and this is where it is corrected."
    )
    print(
        f"- Why there and nowhere else: at λ = {worst_lambda:.3e} the optimum puts "
        f"**{table[worst_index].optimal.max_bin_fraction:.1%}** of the order in the "
        f"first bin, so the whole reaction to the signal lives in the last percent "
        f"of inventory, and the gross alpha it is a fraction of has fallen to "
        f"{table[worst_index].alpha_available:.5f} bps. That is a difference of "
        f"small numbers on a grid whose spacing is "
        f"{order_size / (DEFAULT_SIGNAL_GRID_POINTS - 1):.0f} shares."
    )
    print(
        f"- **Those lambdas are outside the region the milestone operates in.** "
        f"M2's selection rule rejects every one of them on condition (ii) — the "
        f"optimum's largest bin is "
        f"{min(table[i].optimal.max_bin_fraction for i, ok in enumerate(resolved) if not ok):.0%}"
        f"–{max(table[i].optimal.max_bin_fraction for i, ok in enumerate(resolved) if not ok):.0%} "
        f"of the order against a 50 % ceiling — because there the control problem "
        f"is a single trade and there is nothing to schedule. The claim is "
        f"therefore stated over the resolved region, where it is **tighter** than "
        f"the reading it replaces: the decomposition is needed at every lambda the "
        f"testbed is discriminative at, which is what makes it structural rather "
        f"than an accommodation for the one point the rule selected "
        f"({time.perf_counter() - started:.0f} s).\n"
    )
    return {
        "green": green,
        "fine_grid_points": DEFAULT_SIGNAL_GRID_POINTS,
        "coarse_grid_points": STABILITY_GRID_POINTS,
        "resolved_shift_bar": PREMIUM_RESOLVED_SHIFT,
        "resolved_span_bar": PREMIUM_RESOLVED_SPAN,
        "resolved_count": sum(resolved),
        "resolved_range": [min(settled), max(settled)],
        "resolved_span": span,
        "unresolved_lambdas": [
            lam for lam, ok in zip(ordered, resolved, strict=True) if not ok
        ],
        "worst_lambda": worst_lambda,
        "worst_lambda_escalation": {
            str(k): v for k, v in escalation.items()
        },
        "by_lambda": [
            {
                "lambda": lam,
                "fine": f,
                "coarse": c,
                "shift": shift,
                "resolved": ok,
                "max_bin_fraction": row.optimal.max_bin_fraction,
            }
            for lam, row, f, c, shift, ok in zip(
                ordered, table, fine, coarse, shifts, resolved, strict=True
            )
        ],
    }


def render_value_of_signal(experiment, floor: float) -> list[dict]:
    """The oracle's value of the signal at six rho — a curve, because rho is invented."""
    market = experiment.case.market
    order_size = experiment.case.order_size
    lam = experiment.lambda_risk
    j_m4a = cost_moments(power_law_optimum(market, order_size, lam), market).objective(lam)

    print("## The value of the signal against the invented parameter\n")
    print(
        "| rho | R² | J_DP | net advantage | as a multiple of M4b's | % of J_M4a "
        "| gross alpha | premium / alpha | brief predicted |"
    )
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    rows = []
    trained = experiment.signal.correlation()
    for rho in RHO_CURVE:
        signal = OneStepSignal(rho)
        optimum = alpha_optimum(market, order_size, lam, signal)
        advantage = j_m4a - optimum.objective_bps
        alpha = -optimum.alpha_bps
        premium = optimum.execution_bps - floor
        emphasis = "**" if rho == trained else ""
        predicted = PREDICTED["curve"][f"{rho:g}"]
        print(
            f"| {emphasis}{rho:.4g}{emphasis} | {rho**2:.1e} | "
            f"{optimum.objective_bps:.6f} | {emphasis}{advantage:.6f}{emphasis} | "
            f"{advantage / 0.062124:.2f}x | {advantage / j_m4a:.2%} | "
            f"{alpha:.6f} | {premium / alpha:.1%} | {predicted:.6f} |"
        )
        rows.append(
            {
                "rho": rho,
                "explained_variance_fraction": rho**2,
                "objective_bps": optimum.objective_bps,
                "advantage_bps": advantage,
                "advantage_fraction": advantage / j_m4a,
                "alpha_available_bps": alpha,
                "execution_premium_bps": premium,
                "premium_fraction": premium / alpha,
                "predicted_advantage_bps": predicted,
                "decomposition": optimum.as_dict(),
            }
        )
    print(
        f"\n**rho is Temper's own invention** — FrontierView has no alpha model — "
        f"so the result is this curve and not any one row of it. It is also why "
        f"the trained value is a scoping decision rather than a calibration: at "
        f"rho = 0.05 the advantage is more than half the objective and at 0.2 the "
        f"\"cost\" is negative, so the agent would no longer be executing an order. "
        f"A signal explaining one part in ten thousand of next-bin variance is "
        f"already worth more than everything M4b measured.\n"
    )
    return rows


def main() -> int:
    # The table is markdown destined for a UTF-8 document and is full of Greek and
    # arrows. Windows consoles default to cp1252, where printing it raises rather
    # than mangling, so the stream is reconfigured instead of the table being
    # spelled in ASCII.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "configs" / "m5_alpha.yaml")
    )
    parser.add_argument("--paths", type=int, default=SIGNAL_BOUND_PATHS)
    parser.add_argument("--clairvoyant-paths", type=int, default=CLAIRVOYANT_PATHS)
    # Exposed so every path in this file can be exercised in seconds before the
    # real run spends six minutes reaching them. `docs/house-notes.md`, *No code
    # path may be reachable only at the end of a long run* — which this file
    # learned the hard way: the refinement below was added after the smoke run and
    # its first execution, four minutes into a clean-tree run, was a NameError.
    parser.add_argument("--refinement-paths", type=int, default=REFINEMENT_PATHS)
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "results" / "m5_reference.json")
    )
    args = parser.parse_args()

    started = time.perf_counter()
    experiment = load_experiment(args.config)
    market = experiment.case.market
    order_size = experiment.case.order_size
    signal = experiment.signal
    rule = experiment.rule
    grid = LAMBDA_GRIDS[experiment.lambda_grid]
    floor = execution_floor_bps(market, order_size, experiment.lambda_risk)

    print(
        f"# M5 task 0 — {experiment.case.symbol}, X = {order_size:,.0f}, "
        f"T = {market.horizon_hours} h, N = {market.n_bins}; the "
        f"{experiment.lambda_grid} λ grid, {len(grid)} points\n"
    )
    print(
        f"**The alpha signal is invented.** At the decision point for bin `k` the "
        f"observation carries `s_k ~ N(0, 1)` with `E[xi_{{k+1}} | s_k] = rho s_k`, "
        f"`rho = {signal.correlation()}` — Temper's own parameter, not "
        f"FrontierView's, so constitution §7's \"vendored, not invented\" cover "
        f"does not reach any number below. The signal explains "
        f"`rho² = {signal.explained_variance_fraction:.1e}` of next-bin return "
        f"variance: one part in ten thousand.\n"
    )
    print(
        f"**The world is M4a's, not M4b's** — power law, *deterministic* "
        f"liquidity, plus the signal (`{experiment.liquidity.as_dict()['model']}`). "
        f"Bundled with stochastic liquidity a red result could not be attributed, "
        f"and the two adaptivities would compete for the same schedule shape.\n"
    )
    print(
        f"Per-bin volatility is `A = sigma_bin × BPS = "
        f"**{alpha_coefficient(market):.6f} bps**` — "
        f"**{alpha_coefficient(market) / 2.383215:.1f}x the whole objective**, "
        f"which is the single fact that fixes this milestone's shape (brief "
        f"predicted {PREDICTED['alpha_coefficient']:.4f}).\n"
    )
    print(
        f"Monte-Carlo bounds use **M = {args.paths:,}** signal paths from the "
        f"`{M5_REFERENCE_POOL}` pool, common across every policy on a row and "
        f"paired against M4a's certified optimum's closed-form level.\n"
    )

    # --- the table -----------------------------------------------------------
    print("## The alpha table\n")
    table_started = time.perf_counter()
    table = [
        alpha_reference_row(
            market,
            order_size,
            lam,
            signal,
            root_seed=experiment.seeds.root_seed,
            stream_index=index,
            paths=args.paths,
        )
        for index, lam in enumerate(sorted(grid))
    ]
    print(render_table(table, rule))
    print(
        f"\n`J_tangent` is the Almgren–Chriss sinh, derived at the tangent to this "
        f"world's impact function and therefore not its optimum. `J_M4a` is M4a's "
        f"**certified** power-law optimum — the best schedule that cannot see the "
        f"signal, and the denominator's top rung. `J_DP` is the **converged** "
        f"optimum over all policies that can. Unlike M4b there is no fifth rung "
        f"and no level shift: a zero-mean signal gives a fixed schedule nothing to "
        f"re-solve for, so the whole gap is information. "
        f"({time.perf_counter() - table_started:.0f} s)\n"
    )

    static_table = signal_static_table(market, order_size, signal, grid)
    lambda_green, lambda_record = render_lambda_identity(experiment, static_table, rule)
    selected = next(row for row in table if row.lambda_risk == experiment.lambda_risk)
    optimum = selected.optimum
    advantage = selected.signal_advantage

    # --- the decomposition ---------------------------------------------------
    at_zero = alpha_optimum(market, order_size, experiment.lambda_risk, OneStepSignal(0.0))
    identity_green = (
        optimum.identity_residual_bps <= IDENTITY_TOLERANCE_BPS
        and optimum.node_identity_residual_bps <= IDENTITY_TOLERANCE_BPS
    )
    print("## The decomposition at the selected λ, identity asserted\n")
    print("| term | rho = 0 (certified reference) | rho = 0.01 DP | difference | brief |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for label, zero, now, prediction in (
        ("temporary impact", at_zero.impact_bps, optimum.impact_bps, PREDICTED["impact"]),
        ("inventory risk", at_zero.risk_bps, optimum.risk_bps, PREDICTED["risk"]),
        ("alpha", at_zero.alpha_bps, optimum.alpha_bps, -PREDICTED["alpha_available"]),
    ):
        print(
            f"| {label} | {zero:.6f} | **{now:.6f}** | {now - zero:+.6f} | "
            f"{prediction:.6f} |"
        )
    print(
        f"| schedule-invariant | {at_zero.invariant_bps:.6f} | "
        f"{optimum.invariant_bps:.6f} | {0.0:+.6f} | — |"
    )
    print(
        f"| **objective** | **{at_zero.objective_bps:.6f}** | "
        f"**{optimum.objective_bps:.6f}** | "
        f"**{optimum.objective_bps - at_zero.objective_bps:+.6f}** | "
        f"{PREDICTED['j_dp']:.6f} |"
    )
    print(
        f"\n- The identity `J = impact + risk + alpha + invariant` closes at the "
        f"root to **{optimum.identity_residual_bps:.2e} bps** and at **every node "
        f"of every stage** to **{optimum.node_identity_residual_bps:.2e} bps**, "
        f"against a bar of {IDENTITY_TOLERANCE_BPS:g} — "
        f"{'**GREEN**' if identity_green else '**RED**'}. Asserted, not assumed: "
        f"the four quantities ride four separate interpolations through the same "
        f"backward pass and only arithmetic keeps them together."
    )
    print(
        f"- The feasible upper bound — the DP's own greedy policy, a **real** "
        f"policy, simulated on {selected.feasible.paths:,} held-out signal paths — "
        f"is **{selected.feasible.value_bps:.6f} ± "
        f"{selected.feasible.half_width_bps:.6f} bps**, "
        f"{(selected.feasible.value_bps - selected.adaptive_bps) / advantage:+.2%} "
        f"of the advantage from `J_DP`. Pairing against M4a's certified level cuts "
        f"the per-path SD from {selected.feasible.unpaired_sd_bps:.4f} to "
        f"{selected.feasible.paired_sd_bps:.4f} bps — a variance reduction of "
        f"{(selected.feasible.unpaired_sd_bps / selected.feasible.paired_sd_bps) ** 2:.0f}×."
    )
    refinement = refine_feasible_bound(experiment, signal, args.refinement_paths)
    inside = abs(refinement["gap_bps"]) < refinement["half_width_bps"]
    print(
        f"- **That bound sits {(selected.feasible.value_bps - selected.adaptive_bps) / advantage:+.2%} "
        f"of the advantage from `J_DP`, and the sign of that is noise rather than "
        f"a finding.** The greedy policy is a *real* policy, so it cannot beat the "
        f"optimum it came from and a gap of either sign at this M is sampling "
        f"error: the half-width is "
        f"{selected.feasible.half_width_bps / advantage:.2%} of the advantage, "
        f"larger than the gap it is being asked to resolve. Measured rather than "
        f"argued — at **M = {refinement['paths']:,}** the bound is "
        f"{refinement['value_bps']:.6f} ± {refinement['half_width_bps']:.6f} bps, "
        f"a gap of {refinement['gap_bps'] / advantage:+.3%} of the advantage "
        f"against a half-width of "
        f"{refinement['half_width_bps'] / advantage:.3%}, "
        f"{'**inside it**' if inside else '**outside it**'}. A gap that survived "
        f"the path count would mean a bad action map — solving the stage problem "
        f"by snapping to a grid node instead of searching the interpolant costs an "
        f"order of magnitude, and that is the defect this measurement exists to "
        f"catch. The DP's own grid residual, 1.5e-06 bps, is three orders below "
        f"either number."
    )
    print(
        f"- Brief predicted impact {PREDICTED['impact']:.6f} / risk "
        f"{PREDICTED['risk']:.6f} / alpha {-PREDICTED['alpha_available']:.6f} / "
        f"objective {PREDICTED['j_dp']:.6f}; measured "
        f"{_delta(optimum.impact_bps, PREDICTED['impact'])}, "
        f"{_delta(optimum.risk_bps, PREDICTED['risk'])}, "
        f"{_delta(-optimum.alpha_bps, PREDICTED['alpha_available'])}, "
        f"{_delta(optimum.objective_bps, PREDICTED['j_dp'])}.\n"
    )

    # --- gate 1 --------------------------------------------------------------
    rho_zero_green, rho_zero_record = render_rho_zero_gate(experiment, advantage)

    # --- gate 2 --------------------------------------------------------------
    fraction = selected.advantage_fraction
    worth_it = fraction >= MIN_ADVANTAGE_FRACTION
    print("## Gate 2 — the net advantage is ≥ 1 % of the objective\n")
    print(
        f"- `J_M4a − J_DP` = **{advantage:.6f} bps** = **{fraction:.3%}** of "
        f"`J_M4a` = {selected.optimal.objective:.6f} bps "
        f"({advantage / selected.adaptive_bps:.3%} of `J_DP`)."
    )
    print(
        f"- {'**GREEN**' if worth_it else '**RED**'} against the "
        f"{MIN_ADVANTAGE_FRACTION:.0%} bar."
    )
    epsilon = experiment.tolerances.epsilon_fraction
    per_seed = experiment.tolerances.per_seed_fraction
    print(
        f"- Median bar: {epsilon:.0%} of it = **{epsilon * advantage:.6f} bps** ⇒ "
        f"net capture c ≥ {1 - epsilon:.2f}. Per seed: {per_seed:.0%} = "
        f"{per_seed * advantage:.6f} bps ⇒ c ≥ {1 - per_seed:.2f}. M4b's bars, "
        f"transferred because the advantages are on one scale "
        f"({advantage / 0.062124:.2f}x M4b's, "
        f"{advantage / 0.036740:.2f}x M4a's)."
    )
    print(
        f"- Brief predicted advantage {PREDICTED['advantage']:.6f} bps; measured "
        f"{_delta(advantage, PREDICTED['advantage'])}.\n"
    )

    # --- gate 3 --------------------------------------------------------------
    alpha_available = selected.alpha_available
    premium = selected.execution_premium
    premium_fraction = selected.premium_fraction
    premium_range = [row.premium_fraction for row in table]
    premium_green = MIN_PREMIUM_FRACTION <= premium_fraction <= MAX_PREMIUM_FRACTION
    print("## Gate 3 — the execution premium is 25–75 % of the gross alpha\n")
    print(
        f"- The optimum monetises **{alpha_available:.6f} bps** of signal and pays "
        f"**{premium:.6f} bps** of it back in worse impact "
        f"(+{optimum.impact_bps - at_zero.impact_bps:.6f}) and worse risk "
        f"(+{optimum.risk_bps - at_zero.risk_bps:.6f}). That is "
        f"**{premium_fraction:.1%}** of the gross effect, given back."
    )
    print(
        f"- {'**GREEN**' if premium_green else '**RED**'} against the "
        f"{MIN_PREMIUM_FRACTION:.0%}–{MAX_PREMIUM_FRACTION:.0%} band (brief "
        f"predicted {PREDICTED['premium_fraction']:.1%})."
    )
    print(
        f"- **This is the gate the brief says everything else follows from.** "
        f"Against M4b's level shift of 3.8 % of its advantage, {premium_fraction:.0%} "
        f"is a different regime: a single capture fraction against "
        f"{advantage:.6f} bps scores a policy that captures 0.15 of alpha and pays "
        f"0.07 identically to one that captures 0.25 and pays 0.17, at a headline "
        f"that is supposed to be about execution quality and reports neither. **The "
        f"milestone grades alpha capture and execution premium separately or it "
        f"grades nothing** — M4a's finding was that the tolerance's denominator was "
        f"wrong, and M5's is that one denominator is not enough."
    )
    print(
        f"- Had this come back at 5 % or 90 % the milestone would be a different "
        f"milestone and the brief wrong before the code; it came back at "
        f"{premium_fraction:.1%}, {(premium_fraction - MIN_PREMIUM_FRACTION) / (MAX_PREMIUM_FRACTION - MIN_PREMIUM_FRACTION):.0%} "
        f"of the way across the band."
    )
    print(
        f"- **And it is not a property of the selected λ.** Across the whole "
        f"{len(grid)}-point grid — nine decades — the premium fraction stays "
        f"between **{min(premium_range):.1%}** and **{max(premium_range):.1%}**, "
        f"while the advantage it is a fraction of moves from "
        f"{max(row.advantage_fraction for row in table):.1%} of the objective to "
        f"{min(row.advantage_fraction for row in table):.4%}. The gate would have "
        f"read the same at any λ the rule could have chosen, which is what stops "
        f"it being an artefact of one point.\n"
    )

    # --- gate 4 --------------------------------------------------------------
    clairvoyant_green, clairvoyant_record = render_clairvoyant_gate(
        experiment, selected, floor, advantage, args.clairvoyant_paths
    )

    # --- task 1: the reference, and what word it earns -----------------------
    convergence = render_convergence(experiment, signal, floor)
    sufficiency_green, sufficiency = render_sufficiency(experiment, signal)
    timing_green, timing = render_timing(experiment, signal, floor, advantage)
    kinds = render_reference_kinds(selected)
    stability = render_premium_stability(experiment, signal, table, grid)
    stability_green = stability["green"]
    curve = render_value_of_signal(experiment, floor)

    gates_green = (
        rho_zero_green
        and worth_it
        and premium_green
        and clairvoyant_green
        and lambda_green
        and identity_green
    )
    green = (
        gates_green and sufficiency_green and timing_green and stability_green
    )
    print(
        f"## Verdict — {'tasks 0 and 1 GREEN' if green else 'NOT everything is green'}\n"
    )
    print(
        f"- Gate 1 (`rho -> 0` vs certified): "
        f"{'GREEN' if rho_zero_green else 'RED'}\n"
        f"- Gate 2 (net advantage ≥ 1 %): {'GREEN' if worth_it else 'RED'}\n"
        f"- Gate 3 (premium 25–75 % of alpha): "
        f"{'GREEN' if premium_green else 'RED'}\n"
        f"- Gate 4 (clairvoyance retired, convexity asserted): "
        f"{'GREEN' if clairvoyant_green else 'RED'}\n"
        f"- Assertion: lambda's static reading bit-identical to M4a's: "
        f"{'HELD' if lambda_green else 'FAILED'}\n"
        f"- Assertion: the decomposition's identity closes: "
        f"{'HELD' if identity_green else 'FAILED'}\n"
        f"- Task 1: `(k, x_k, s_k)` sufficient: "
        f"{'GREEN' if sufficiency_green else 'RED'}\n"
        f"- Task 1: an already-landed signal is worth zero: "
        f"{'GREEN' if timing_green else 'RED'}\n"
        f"- Task 1: the premium is flat across every resolvable lambda: "
        f"{'GREEN' if stability_green else 'RED'}\n"
    )
    if not green:
        print(
            "The brief's instruction when a gate is red is to report it and "
            "re-shape the milestone here, with the reason — not to relax the bar, "
            "not to raise rho, and not to adjust the brief to fit what was "
            "measured. A red task-1 check is different in kind: sufficiency and "
            "timing are properties of the seam, not thresholds, and a red one "
            "means the reference is wrong rather than the milestone."
        )

    document = {
        "milestone": "M5",
        "task": "0+1",
        "config": experiment.as_dict(),
        "provenance": stamp(Path(args.config), REPO_ROOT).as_dict(),
        "signal": signal.as_dict(),
        "alpha_coefficient_bps": alpha_coefficient(market),
        "paths": args.paths,
        "gates": {
            "rho_zero": rho_zero_record,
            "advantage": {
                "green": worth_it,
                "bar": MIN_ADVANTAGE_FRACTION,
                "advantage_bps": advantage,
                "advantage_fraction": fraction,
                "median_bar_bps": epsilon * advantage,
                "per_seed_bar_bps": per_seed * advantage,
            },
            "execution_premium": {
                "green": premium_green,
                "band": [MIN_PREMIUM_FRACTION, MAX_PREMIUM_FRACTION],
                "alpha_available_bps": alpha_available,
                "execution_premium_bps": premium,
                "premium_fraction": premium_fraction,
                "impact_increase_bps": optimum.impact_bps - at_zero.impact_bps,
                "risk_increase_bps": optimum.risk_bps - at_zero.risk_bps,
                "premium_fraction_across_grid": [
                    min(premium_range),
                    max(premium_range),
                ],
            },
            "clairvoyant_retired": clairvoyant_record,
        },
        "assertions": {
            "lambda_bit_identical": {"green": lambda_green} | lambda_record,
            "decomposition_identity": {
                "green": identity_green,
                "bar_bps": IDENTITY_TOLERANCE_BPS,
                "root_residual_bps": optimum.identity_residual_bps,
                "node_residual_bps": optimum.node_identity_residual_bps,
            },
        },
        "task_1": {
            "green": sufficiency_green and timing_green,
            "sufficiency": {
                "green": sufficiency_green,
                "bar_bps": SUFFICIENCY_TOLERANCE_BPS,
                "by_grid": {str(k): v for k, v in sufficiency.items()},
            },
            "timing": timing,
            "reference_kinds": kinds,
            "premium_stability": stability,
        },
        "all_green": green,
        "gates_green": gates_green,
        "convergence": convergence,
        "feasible_refinement": refinement
        | {"gap_fraction": refinement["gap_bps"] / advantage},
        "decomposition_at_rho_zero": at_zero.as_dict(),
        "predicted_by_brief": PREDICTED,
        "selected": selected.as_dict(),
        "table": [row.as_dict() for row in table],
        "static_table": [row.as_dict() for row in static_table],
        "value_of_signal": curve,
        "elapsed_seconds": time.perf_counter() - started,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    try:
        shown = out.resolve().relative_to(REPO_ROOT)
    except ValueError:  # --out pointed somewhere outside the repo, e.g. a smoke run
        shown = out
    print(f"\nWritten to `{shown}` ({time.perf_counter() - started:.0f} s total).")
    return 0 if green else 1


if __name__ == "__main__":
    raise SystemExit(main())
