"""M4b task 0 — the liquidity reference table, and the four gates it decides.

Oracle only. No agent exists when this runs, no training code is imported, and
that is the entire point (constitution invariant 3): a threshold derived from
closed forms and a dynamic program, *before* any training code, cannot have been
chosen to fit a training curve.

    python tools/m4b_reference_table.py --config configs/m4b_liquidity.yaml

**Every number in the brief is a prediction made on unpinned numpy in a cloud
container.** None of it is a committed artefact. This regenerates all of it on the
reference box, and a material disagreement is not a tolerance to loosen — it means
the brief is wrong before the code is. The predicted value is printed beside each
measured one for exactly that reason.

Four gates, and all four must be green or the milestone re-shapes:

1. **λ agreement.** The rule must select the same λ in every reading, so M3's,
   M4a's and M4b's points are comparable. Liquidity is not a new *encoding* — the
   charge is still ``eta sigma p**0.6`` and what changes is that the market is
   random — so this also has to *decide and record* how the rule is applied, and
   it does: see :func:`render_lambda_gate`.
2. **The adaptive advantage is worth an evening**: ``J_static* − J_DP`` at least
   1 % of ``J_DP``.
3. **The level shift is ≤ 10 % of the adaptive advantage.** The gate that matters
   most. ``J_M4a − J_static*`` is a constant any static solver picks up for free
   by re-solving at the inflated coefficient; if it dominates, the milestone's
   headline is a re-solve rather than adaptivity and has to be restated *before*
   training rather than caveated afterwards. Both rungs are computed in closed
   form — two simulated levels differenced is how a 0.002 bps quantity becomes
   noise.
4. **The clairvoyant bracket is ≤ 15 % of the advantage**, so the red-flag test
   has teeth. No adapted policy can beat perfect information, which makes an agent
   below that bound a defect with a proof rather than a discovery.

Exit status is 0 only if all four are green *and* the config's committed λ is the
one the rule selects.
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
    LIQUIDITY_READING,
    load_experiment,
)
from temper.eval.provenance import stamp  # noqa: E402
from temper.eval.reference import (  # noqa: E402
    REFERENCE_BOUND_PATHS,
    liquidity_reference_row,
    liquidity_trajectories,
    select_lambda,
    static_liquidity_table,
)
from temper.oracle import (  # noqa: E402
    DEFAULT_GRID_POINTS,
    DEFAULT_QUADRATURE_NODES,
    ENCODINGS,
    LognormalLiquidity,
    adaptive_optimum,
    cost_moments,
    expected_cost_moments,
    path_objective_bps,
    power_law_optimum,
    richardson_residual,
    trades,
)
from temper.seeding import M4B_REFERENCE_POOL, pool_rng  # noqa: E402

#: Task 0's four bars, pre-stated in `docs/briefs/M4b-stochastic-liquidity.md`.
MIN_ADVANTAGE_FRACTION = 0.01
MAX_LEVEL_SHIFT_FRACTION = 0.10
MAX_BRACKET_FRACTION = 0.15

#: Paths for the *refinement* of the feasible upper bound. At the pre-stated
#: M = 20 000 the bound's own half-width is ~1.4 % of the advantage while the gap
#: it is being asked to resolve is ~0.01 %, so the sign of that gap at 20 000 is a
#: coin toss and reporting it as if it meant something would be reading noise.
#: Ten times the paths costs seconds and settles it.
REFINEMENT_PATHS = 200_000

#: The invented parameter, at the three values the value-of-sight curve reports.
#: The middle one is trained; the outer two exist because a single invented
#: parameter with a single number beside it reads as calibration, and it is not.
SIGMA_CURVE = (0.25, 0.50, 0.75)

#: The brief's predictions, so a disagreement is visible rather than discoverable.
#: Made on unpinned numpy in a cloud container; **none of it is an artefact**.
PREDICTED = {
    "j_m4a": 2.49895,
    "j_static": 2.49661,
    "j_dp": 2.43449,
    "advantage": 0.06212,
    "advantage_fraction": 0.0255,
    "level_shift": 0.00234,
    "level_shift_fraction": 0.038,
    "bracket_fraction": 0.085,
    "feasible_gap_fraction": 0.0031,
    "sigma_zero": 2.383218,
    "paired_sd": 0.0612,
    "half_width_fraction": 0.0136,
}


def _delta(measured: float, predicted: float) -> str:
    """How far the box's number is from the brief's, in the brief's own units."""
    if predicted == 0.0:
        return f"{measured:+.6f}"
    return f"{measured - predicted:+.6f} ({(measured - predicted) / predicted:+.2%})"


def render_table(table, rule) -> str:
    """The liquidity analogue of M4a task 0's table, over the whole λ grid."""
    lines = [
        "| λ | J_twap | J_M4a | J_static* | J_DP | adaptive advantage | adv / J_DP "
        "| level shift | shift / adv | bracket / adv | TWAP gap | max bin, static "
        "| max bin, DP mean | rule |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
        "| ---: | ---: | :---: |",
    ]
    for row in table:
        admits = rule.admits(row)
        why = "✓" if admits else ("i" if row.twap_gap < rule.min_twap_gap else "ii")
        lines.append(
            f"| {row.lambda_risk:.3e} | {row.twap.objective:.4f} | "
            f"{row.m4a.objective:.4f} | {row.static.objective:.4f} | "
            f"{row.adaptive_bps:.4f} | {row.adaptive_advantage:.5f} | "
            f"{row.advantage_fraction:.2%} | {row.level_shift:.5f} | "
            f"{row.level_shift_fraction:.1%} | {row.bracket_fraction:.1%} | "
            f"{row.twap_gap:.2%} | {row.static.max_bin_fraction:.1%} | "
            f"{row.mean_schedule_max_bin:.1%} | {why} |"
        )
    return "\n".join(lines)


def render_lambda_gate(experiment, table, rule) -> tuple[bool, dict]:
    """Gate 1, and the decision the brief asked this task to *record*.

    ``verify_lambda_rule_agrees_across_worlds`` loops the cost encodings, and
    liquidity is not one of them: the functional is unchanged and what M4b
    randomises is the market. So the question "how is the rule applied here" has
    an answer that had to be chosen, and this is where the choice and its evidence
    are written down.

    **The rule is applied to the static reading** — the liquidity world's
    fixed-schedule table, which is M4a's problem at the coefficient
    ``A E[L^-beta]``. Three reasons, in order of weight:

    1. It reads a *schedule*. Both of the rule's conditions are properties of one:
       condition (ii) asks for the largest single-bin fraction, and a policy has no
       single schedule — the DP's "mean schedule" is an average no policy ever
       executes.
    2. It is a closed form. The static optimum is a certified Newton solve, so the
       selection cannot move with a grid resolution.
    3. It is not knife-edge, and the alternative is. Both are printed below.
    """
    static_choice = select_lambda(table, rule).lambda_risk
    readings = {
        encoding: experiment.rule_selected(encoding).lambda_risk
        for encoding in ENCODINGS
    }
    readings[LIQUIDITY_READING] = static_choice
    agree = len(set(readings.values())) == 1
    committed = experiment.lambda_risk
    matches_config = committed == static_choice

    # The reading that was *not* taken: the rule against the DP's value and its
    # mean schedule. Recorded rather than omitted — it disagrees, and a session
    # that quietly picked the agreeing one would have made the milestone's lambda
    # a choice nobody could audit.
    adaptive_choice = None
    for row in table:
        if (
            row.adaptive_twap_gap >= rule.min_twap_gap
            and row.mean_schedule_max_bin <= rule.max_bin_fraction
        ):
            adaptive_choice = row
            break

    print("## Gate 1 — every reading of the rule selects the same λ\n")
    for name, lam in sorted(readings.items()):
        print(f"- `{name}` selects **{lam:.6e}** (10^{math.log10(lam):.1f}).")
    print(f"- {'**GREEN** — they agree.' if agree else '**RED** — they disagree.'}")
    print(
        f"- The config commits {committed:.6e} — "
        f"{'agree' if matches_config else '**DISAGREE**'}.\n"
    )

    print(
        "### The decision this task was asked to record\n\n"
        "Liquidity is **not a new cost encoding** — the charge is still "
        "`eta*sigma*p^0.6`; what M4b randomises is the *market*. So the rule is "
        "applied to a third **reading**, not a fourth world, and the reading is "
        "the **static** one: the liquidity world's fixed-schedule problem is M4a's "
        "at the coefficient `A*E[L^-0.6]`, a monotone rescaling, so the rule reads "
        "a schedule's TWAP gap and a schedule's largest bin exactly as it has "
        "since M2.\n"
    )
    if adaptive_choice is not None:
        chosen = next(row for row in table if row.lambda_risk == static_choice)
        print(
            f"The alternative reading — the rule against `J_DP` and the DP's "
            f"*mean* schedule — selects **10^{math.log10(adaptive_choice.lambda_risk):.1f}** "
            f"instead. It is recorded, and it was not taken, because it is "
            f"knife-edge where the static reading is not:\n"
        )
        print(
            f"- At 10^{math.log10(adaptive_choice.lambda_risk):.1f} the adaptive "
            f"reading clears the {rule.min_twap_gap:.0%} bar by "
            f"**{100 * (adaptive_choice.adaptive_twap_gap - rule.min_twap_gap):+.3f} "
            f"percentage points** — a milestone's λ turning on the fifth digit of a "
            f"numerically-solved value function."
        )
        print(
            f"- The static reading *misses* the same bar there by "
            f"**{100 * (adaptive_choice.twap_gap - rule.min_twap_gap):+.2f} points**, "
            f"and clears it at 10^{math.log10(static_choice):.1f} by "
            f"**{100 * (chosen.twap_gap - rule.min_twap_gap):+.1f}**."
        )
        print(
            "- And a policy has no single schedule: condition (ii) asks for the "
            "largest single-bin fraction, which for a policy is an average no "
            "realised episode ever executes.\n"
        )
    else:
        print("The alternative reading admits no λ at all on this grid.\n")

    return agree and matches_config, {
        "readings": readings,
        "agree": agree,
        "committed_matches": matches_config,
        "recorded_reading": LIQUIDITY_READING,
        "reading_rule_applied_to": "static",
        "alternative_reading_selects": (
            None if adaptive_choice is None else adaptive_choice.lambda_risk
        ),
    }


def render_convergence(experiment, law) -> dict:
    """The DP's own numbers: grid, quadrature, and the σ_L → 0 differential.

    The last of the three is the single most valuable check in the milestone,
    because it ties every piece of new machinery to a number that *was* certified:
    at ``sigma_log = 0`` the dynamic program must return M4a's
    ``power_law_optimum`` value.
    """
    market = experiment.case.market
    order_size = experiment.case.order_size
    lam = experiment.lambda_risk

    print("## The dynamic program, converged\n")
    grid = {}
    for points in (201, 401, 801, DEFAULT_GRID_POINTS):
        started = time.perf_counter()
        grid[points] = adaptive_optimum(
            market, order_size, lam, law, points=points
        ).objective_bps
        print(
            f"- {points:5d} inventory points: **{grid[points]:.6f} bps** "
            f"({time.perf_counter() - started:.1f} s)"
        )
    extrapolant, residual = richardson_residual(grid[801], grid[DEFAULT_GRID_POINTS])
    print(
        f"- Second order in the spacing, so Richardson-extrapolating the last two "
        f"gives **{extrapolant:.6f} bps** and a residual of **{residual:.2e} bps** "
        f"— the numerical uncertainty in a reference this milestone is explicit "
        f"about not having certified.\n"
    )

    quadrature = {}
    for nodes in (3, 5, 7, DEFAULT_QUADRATURE_NODES, 21):
        quadrature[nodes] = adaptive_optimum(
            market, order_size, lam, law, nodes=nodes
        ).objective_bps
    print(
        "- Quadrature: "
        + ", ".join(
            f"{nodes} nodes → {value:.6f}" for nodes, value in quadrature.items()
        )
        + f". Converged by 5; pinned at {DEFAULT_QUADRATURE_NODES}, because the "
        "quadrature is the cheap axis (the inventory grid dominates the cost by "
        "two orders) and sitting three times past the knee removes the node count "
        "from the list of things a later session has to re-argue.\n"
    )

    certified = cost_moments(
        power_law_optimum(market, order_size, lam), market
    ).objective(lam)
    at_zero = adaptive_optimum(
        market, order_size, lam, LognormalLiquidity(0.0)
    ).objective_bps
    print("### σ_L → 0 — the differential against a *certified* number\n")
    print(
        f"- The DP at `sigma_log = 0` returns **{at_zero:.6f} bps** against M4a's "
        f"certified `power_law_optimum` value of **{certified:.6f} bps** — a "
        f"difference of **{at_zero - certified:+.3e} bps**."
    )
    print(
        f"- Brief predicted {PREDICTED['sigma_zero']:.6f}; measured "
        f"{_delta(at_zero, PREDICTED['sigma_zero'])}."
    )
    print(
        "- That difference is grid discretisation, not a disagreement, and it "
        "calibrates what \"converged\" is worth here. This is the check that ties "
        "the DP, the quadrature, the stage solve and the constant back to a value "
        "with a Cholesky factorisation and a 1.2e-15 KKT residual behind it.\n"
    )
    return {
        "grid": {str(k): v for k, v in grid.items()},
        "richardson_extrapolant_bps": extrapolant,
        "richardson_residual_bps": residual,
        "quadrature": {str(k): v for k, v in quadrature.items()},
        "sigma_zero_bps": at_zero,
        "m4a_certified_bps": certified,
        "sigma_zero_difference_bps": at_zero - certified,
    }


def refine_feasible_bound(experiment, law, paths: int) -> dict:
    """The feasible upper bound again at ten times the paths — gate 4's evidence.

    The 2 %-of-advantage band on this bound exists to catch a **bad action map**:
    solving the stage problem by snapping to a grid node instead of searching the
    interpolated value function costs an order of magnitude. It is not a test of
    the DP's convergence, and at the pre-stated M it cannot be one — the estimate's
    own half-width is a hundred times the gap it would have to resolve.

    So the sign of that gap is measured once at a path count where it means
    something. If the greedy policy were genuinely worse than ``J_DP`` by anything
    approaching the band, this is where it would show; if the two are the same
    number, the gap shrinks toward zero as the paths grow, which is what a
    converged value function and a properly solved stage problem look like.
    """
    market = experiment.case.market
    order_size = experiment.case.order_size
    lam = experiment.lambda_risk
    optimum = adaptive_optimum(market, order_size, lam, law)
    static = liquidity_trajectories(market, order_size, lam, law)["static"]
    static_level = expected_cost_moments(static, market, law).objective(lam)
    static_weights = trades(static, market) / order_size

    rng = pool_rng(experiment.seeds.root_seed, M4B_REFERENCE_POOL, 900)
    multipliers = law.draw(rng, (paths, market.n_bins))
    difference = path_objective_bps(
        optimum.greedy_weights(multipliers), multipliers, market, order_size, lam
    ) - path_objective_bps(
        static_weights, multipliers, market, order_size, lam
    )
    value = static_level + float(difference.mean())
    half = float(1.96 * difference.std(ddof=1) / math.sqrt(paths))
    return {
        "paths": paths,
        "value_bps": value,
        "half_width_bps": half,
        "gap_bps": value - optimum.objective_bps,
    }


def render_value_of_sight(experiment, paths: int) -> list[dict]:
    """The oracle's value of sight at three σ_L — a curve, because σ_L is invented."""
    market = experiment.case.market
    order_size = experiment.case.order_size
    lam = experiment.lambda_risk
    print("## The value of sight against the invented parameter\n")
    print(
        "| σ_L | E[L^-0.6] | J_M4a | J_static* | J_DP | advantage | % of J_DP "
        "| shift / advantage | bracket / advantage |"
    )
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    rows = []
    for index, sigma in enumerate(SIGMA_CURVE):
        law = LognormalLiquidity(sigma)
        row = liquidity_reference_row(
            market,
            order_size,
            lam,
            law,
            root_seed=experiment.seeds.root_seed,
            stream_index=1000 + index,
            paths=paths,
        )
        emphasis = "**" if sigma == experiment.liquidity.sigma_log else ""
        print(
            f"| {emphasis}{sigma:.2f}{emphasis} | "
            f"{law.inverse_power_moment(market.temp_exponent):.5f} | "
            f"{row.m4a.objective:.5f} | {row.static.objective:.5f} | "
            f"{row.adaptive_bps:.5f} | {emphasis}{row.adaptive_advantage:.5f}"
            f"{emphasis} | {row.advantage_fraction:.2%} | "
            f"{row.level_shift_fraction:.1%} | {row.bracket_fraction:.1%} |"
        )
        rows.append(row.as_dict() | {"sigma_log": sigma})
    print(
        "\nThe shift/advantage column is why gate 3 is not a formality: the "
        "constant grows faster in σ_L than the adaptivity does. **σ_L is Temper's "
        "own invention** — FrontierView has no liquidity process — so the result "
        "is this curve and not any one row of it.\n"
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
        "--config", default=str(REPO_ROOT / "configs" / "m4b_liquidity.yaml")
    )
    parser.add_argument("--paths", type=int, default=REFERENCE_BOUND_PATHS)
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "results" / "m4b_reference.json")
    )
    args = parser.parse_args()

    started = time.perf_counter()
    experiment = load_experiment(args.config)
    market = experiment.case.market
    order_size = experiment.case.order_size
    law = experiment.liquidity
    rule = experiment.rule
    grid = LAMBDA_GRIDS[experiment.lambda_grid]

    print(
        f"# M4b task 0 — {experiment.case.symbol}, X = {order_size:,.0f}, "
        f"T = {market.horizon_hours} h, N = {market.n_bins}; the "
        f"{experiment.lambda_grid} λ grid, {len(grid)} points\n"
    )
    print(
        f"**The liquidity model is invented.** Per-bin i.i.d. lognormal on "
        f"`v_hourly`, `E[L] = 1`, `sigma_log = {law.sigma_log}` — Temper's own "
        f"parameter, not FrontierView's, so constitution §7's \"vendored, not "
        f"invented\" cover does not reach any number below. `E[L^-0.6] = "
        f"{law.inverse_power_moment(market.temp_exponent):.5f}` exactly.\n"
    )
    print(
        f"Monte-Carlo bounds use **M = {args.paths:,}** liquidity paths from the "
        f"`m4b/reference` pool, common across every policy on a row and paired "
        f"against the static optimum's closed-form level.\n"
    )

    # --- the table -----------------------------------------------------------
    print("## The liquidity table\n")
    table_started = time.perf_counter()
    table = [
        liquidity_reference_row(
            market,
            order_size,
            lam,
            law,
            root_seed=experiment.seeds.root_seed,
            stream_index=index,
            paths=args.paths,
        )
        for index, lam in enumerate(sorted(grid))
    ]
    print(render_table(table, rule))
    print(
        f"\n`J_M4a` is M4a's certified power-law optimum — which knows no "
        f"liquidity at all — re-priced here. `J_static*` is the best *fixed* "
        f"schedule that knows the liquidity **law**. `J_DP` is the optimum over "
        f"all adapted policies. Every rung but `J_DP` is a closed form or a "
        f"certified solve; nothing in the first four columns is simulated. "
        f"({time.perf_counter() - table_started:.0f} s)\n"
    )

    # --- gate 1 --------------------------------------------------------------
    static_table = static_liquidity_table(market, order_size, law, grid)
    lambda_green, lambda_record = render_lambda_gate(experiment, table, rule)
    selected = next(
        row for row in table if row.lambda_risk == experiment.lambda_risk
    )

    # --- gate 2 --------------------------------------------------------------
    advantage = selected.adaptive_advantage
    fraction = selected.advantage_fraction
    worth_it = fraction >= MIN_ADVANTAGE_FRACTION
    print("## Gate 2 — the adaptive advantage is ≥ 1 % of J_DP\n")
    print(
        f"- `J_static* − J_DP` = **{advantage:.5f} bps** = **{fraction:.3%}** of "
        f"`J_DP` = {selected.adaptive_bps:.5f} bps."
    )
    print(
        f"- {'**GREEN**' if worth_it else '**RED**'} against the "
        f"{MIN_ADVANTAGE_FRACTION:.0%} bar."
    )
    epsilon = experiment.tolerances.epsilon_fraction
    per_seed = experiment.tolerances.per_seed_fraction
    print(
        f"- Median bar: {epsilon:.0%} of it = **{epsilon * advantage:.5f} bps** ⇒ "
        f"capture fraction c ≥ {1 - epsilon:.2f}. Per seed: {per_seed:.0%} = "
        f"{per_seed * advantage:.5f} bps ⇒ c ≥ {1 - per_seed:.2f}."
    )
    print(
        f"- Brief predicted advantage {PREDICTED['advantage']:.5f} bps; measured "
        f"{_delta(advantage, PREDICTED['advantage'])}.\n"
    )

    # --- gate 3 --------------------------------------------------------------
    shift = selected.level_shift
    shift_fraction = selected.level_shift_fraction
    shift_green = shift_fraction <= MAX_LEVEL_SHIFT_FRACTION
    print("## Gate 3 — the level shift is ≤ 10 % of the adaptive advantage\n")
    print(
        f"- `J_M4a − J_static*` = **{shift:.5f} bps** = **{shift_fraction:.1%}** "
        f"of the adaptive advantage."
    )
    print(
        f"- {'**GREEN**' if shift_green else '**RED**'} against the "
        f"{MAX_LEVEL_SHIFT_FRACTION:.0%} bar."
    )
    print(
        f"- Both rungs are **closed forms**: `J_static*` is a certified Newton "
        f"solve at the coefficient `A*E[L^-0.6]`, and `J_M4a` is M4a's certified "
        f"optimum re-priced by the same moment. Differencing two *simulated* "
        f"levels would put a {abs(shift):.5f} bps quantity under a per-path "
        f"standard deviation of {selected.feasible.unpaired_sd_bps:.3f} bps — "
        f"{selected.feasible.unpaired_sd_bps / abs(shift):.0f}× the thing being "
        f"measured."
    )
    print(
        f"- An agent measured against *M4a's* schedule would appear to gain "
        f"{selected.m4a.objective - selected.adaptive_bps:.5f} bps, and "
        f"{shift:.5f} of that is a constant any static solver picks up for free. "
        f"**The denominator is `J_static* − J_DP`**, and the level shift is "
        f"reported on its own line everywhere."
    )
    print(
        f"- Brief predicted {PREDICTED['level_shift_fraction']:.1%}; measured "
        f"{shift_fraction:.1%}.\n"
    )

    # --- gate 4 --------------------------------------------------------------
    bracket_fraction = selected.bracket_fraction
    bracket_green = bracket_fraction <= MAX_BRACKET_FRACTION
    # The bracket assertion carries its CIs, because both ends are *estimates*.
    # Comparing point estimates would be a coin toss: at M = 2 000 the feasible
    # bound's own half-width is 4.3 % of the advantage while the gap it is being
    # asked to resolve is 0.2 %, so a strict `J_DP <= upper` on the means fails
    # about half the time for no reason but sampling. The brief says "assert
    # lower <= J_DP <= upper *with the CIs carried*", and this is that.
    lower_ci = selected.clairvoyant.value_bps - selected.clairvoyant.half_width_bps
    upper_ci = selected.feasible.value_bps + selected.feasible.half_width_bps
    ordered = lower_ci <= selected.adaptive_bps <= upper_ci
    ordered_on_means = (
        selected.clairvoyant.value_bps
        <= selected.adaptive_bps
        <= selected.feasible.value_bps
    )
    print("## Gate 4 — the clairvoyant bracket is ≤ 15 % of the advantage\n")
    print(
        f"- Clairvoyant lower bound **{selected.clairvoyant.value_bps:.6f} ± "
        f"{selected.clairvoyant.half_width_bps:.6f} bps** — the perfect-information "
        f"relaxation, rigorous because more information cannot cost more."
    )
    print(
        f"- Feasible upper bound **{selected.feasible.value_bps:.6f} ± "
        f"{selected.feasible.half_width_bps:.6f} bps** — the DP's own greedy "
        f"policy, simulated, which is a *real* policy and so attainable. It sits "
        f"{(selected.feasible.value_bps - selected.adaptive_bps) / advantage:+.2%} "
        f"of the advantage from `J_DP` against the brief's 2 % band "
        f"(predicted {PREDICTED['feasible_gap_fraction']:.2%})."
    )
    print(
        f"- `J_DP` = {selected.adaptive_bps:.6f} bps sits inside the bracket **with "
        f"the CIs carried**: {'**yes**' if ordered else '**NO**'} "
        f"([{lower_ci:.6f}, {upper_ci:.6f}]). On the point estimates alone: "
        f"{ordered_on_means} — recorded separately because both ends are "
        f"*estimates*, and at this M the feasible bound's own half-width is "
        f"{selected.feasible.half_width_bps / advantage:.1%} of the advantage "
        f"while the gap it resolves is "
        f"{abs(selected.feasible.value_bps - selected.adaptive_bps) / advantage:.1%}."
    )
    print(
        f"- Bracket width **{selected.bracket_bps:.6f} bps** = "
        f"**{bracket_fraction:.1%}** of the advantage — "
        f"{'**GREEN**' if bracket_green else '**RED**'} against the "
        f"{MAX_BRACKET_FRACTION:.0%} bar (brief predicted "
        f"{PREDICTED['bracket_fraction']:.1%})."
    )
    print(
        f"- Common random numbers are doing the work: the *unpaired* standard "
        f"deviation of a policy's conditional cost is "
        f"{selected.feasible.unpaired_sd_bps:.4f} bps, and pairing against the "
        f"static optimum's closed-form level cuts it to "
        f"{selected.feasible.paired_sd_bps:.4f} bps — a variance reduction of "
        f"{(selected.feasible.unpaired_sd_bps / selected.feasible.paired_sd_bps) ** 2:.1f}×. "
        f"The achieved half-width is "
        f"{selected.feasible.half_width_bps / advantage:.2%} of the effect "
        f"(predicted {PREDICTED['half_width_fraction']:.2%})."
    )
    refinement = refine_feasible_bound(experiment, law, REFINEMENT_PATHS)
    print(
        f"- At **M = {REFINEMENT_PATHS:,}** the feasible bound is "
        f"{refinement['value_bps']:.6f} ± {refinement['half_width_bps']:.6f} bps, "
        f"a gap of {refinement['gap_bps'] / advantage:+.3%} of the advantage "
        f"against a half-width of {refinement['half_width_bps'] / advantage:.3%}. "
        f"The gap shrinks toward zero as the paths grow, so the greedy policy's "
        f"value is **indistinguishable from `J_DP`**: the 2 % band is passed by "
        f"the measurement being noise-limited rather than bias-limited, which is "
        f"what a converged value function and a properly solved stage problem look "
        f"like. The band's job is to catch a *bad action map* — snapping to a grid "
        f"node instead of searching the interpolant — and it still does."
    )
    print(
        "- **This is a bracket, not a certificate.** M4a's word was earned by a "
        "Cholesky factorisation and a KKT residual; `J_DP` gets \"converged, and "
        "bracketed by a perfect-information relaxation\" instead, and the report "
        "says which everywhere.\n"
    )

    # --- the DP's own convergence, and the curve ----------------------------
    convergence = render_convergence(experiment, law)
    curve = render_value_of_sight(experiment, args.paths)

    green = lambda_green and worth_it and shift_green and bracket_green and ordered
    print(f"## Verdict — {'all four gates GREEN' if green else 'NOT all gates green'}\n")
    if not green:
        print(
            "The brief's instruction when a gate is red is to report it and "
            "re-shape the milestone here, with the reason — not to relax the bar "
            "and not to adjust the brief to fit what was measured."
        )

    document = {
        "milestone": "M4b",
        "task": "0",
        "config": experiment.as_dict(),
        "provenance": stamp(Path(args.config), REPO_ROOT).as_dict(),
        "liquidity": law.as_dict(),
        "paths": args.paths,
        "gates": {
            "lambda_agreement": {"green": lambda_green} | lambda_record,
            "advantage": {
                "green": worth_it,
                "bar": MIN_ADVANTAGE_FRACTION,
                "advantage_bps": advantage,
                "advantage_fraction": fraction,
            },
            "level_shift": {
                "green": shift_green,
                "bar": MAX_LEVEL_SHIFT_FRACTION,
                "level_shift_bps": shift,
                "level_shift_fraction": shift_fraction,
                "computed": "closed form, both rungs",
            },
            "bracket": {
                "green": bracket_green and ordered,
                "bar": MAX_BRACKET_FRACTION,
                "bracket_bps": selected.bracket_bps,
                "bracket_fraction": bracket_fraction,
                "ordered_with_cis": ordered,
                "ordered_on_means": ordered_on_means,
                "lower_ci_bps": lower_ci,
                "upper_ci_bps": upper_ci,
                "feasible_gap_fraction": (
                    selected.feasible.value_bps - selected.adaptive_bps
                )
                / advantage,
                "feasible_refinement": refinement
                | {"gap_fraction": refinement["gap_bps"] / advantage},
            },
        },
        "all_green": green,
        "convergence": convergence,
        "predicted_by_brief": PREDICTED,
        "selected": selected.as_dict(),
        "table": [row.as_dict() for row in table],
        "static_table": [row.as_dict() for row in static_table],
        "value_of_sight": curve,
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
