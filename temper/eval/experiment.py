"""A committed experiment config, resolved into the objects a run needs.

One loader, read by the training driver *and* by the tests, so "the number in
``results/`` came from the committed config" is arithmetic rather than a claim
about somebody's care. Constitution invariant 1 wants every reported number
regenerable from a config and a seed; invariant 3 wants the thresholds fixed
before the work. Both fail quietly if the driver and the suite each parse the
YAML their own way.

The λ this returns is checked, not trusted. :meth:`Experiment.verify_lambda_rule`
re-derives it from the oracle by task 0's stated rule and refuses to run if the
committed value is not what the rule selects — so the config cannot drift away
from the reasoning that produced it, and a session that "just nudged lambda" gets
a red test rather than a plausible figure.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from temper.agents.ppo import PPOConfig
from temper.eval.provenance import Provenance, stamp
from temper.eval.reference import (
    LambdaRule,
    LiquidityReferenceRow,
    ReferenceRow,
    TrajectoryBand,
    liquidity_reference_row,
    reference_row,
    reference_table,
    select_lambda,
    static_liquidity_table,
    trajectory_band,
)
from temper.oracle import (
    ENCODINGS,
    LINEAR_ENCODING,
    VENDOR_LAMBDA_GRID,
    DeterministicLiquidity,
    LiquidityLaw,
    Market,
    SymbolParams,
    liquidity_for,
)

#: Lambda grids a config may name. ``vendor`` is M0's 17-point log-half-decade
#: sweep, which is the grid task 0's rule is stated over. A config naming
#: anything else would be selecting lambda from a set chosen after the fact.
LAMBDA_GRIDS: dict[str, tuple[float, ...]] = {"vendor": VENDOR_LAMBDA_GRID}

#: Frontier sub-grids a sweep config may sit on. Each is a subset of the vendor
#: grid — the same float values, taken by index, so a sweep point's lambda is
#: bit-identical to the reference table's — and each must contain the lambda
#: M2's rule selects, so every sweep carries a point directly comparable to a
#: committed result. ``m3`` is the nine half-decade points from 1e-5 to 1e-1:
#: below 1e-5 the three schedules coincide to the fourth digit and the frontier
#: is degenerate; above 1e-1 the optimum is a single bin. Sized in M3 task 3
#: from task 1's measured cost (docs/briefs/M3-antithetic-and-frontier.md).
FRONTIER_GRIDS: dict[str, tuple[float, ...]] = {
    "m3": tuple(VENDOR_LAMBDA_GRID[8:17]),
}


@dataclass(frozen=True)
class ExecutionCase:
    """The parent order and the market it is worked on."""

    case_id: str
    params_from: str
    symbol: str
    market: Market
    order_size: float

    def as_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "params_from": self.params_from,
            "symbol": self.symbol,
            "order_size": self.order_size,
            "horizon_hours": self.market.horizon_hours,
            "n_bins": self.market.n_bins,
            "params": {
                "adv": self.market.params.adv,
                "sigma": self.market.params.sigma,
                "half_spread": self.market.params.half_spread,
                "eta": self.market.params.eta,
                "gamma": self.market.params.gamma,
            },
        }


#: The key the liquidity world's reading of the lambda rule is reported under.
#: Deliberately *not* a member of :data:`~temper.oracle.model.ENCODINGS`:
#: liquidity randomises the market, not the cost functional, so it is not a new
#: encoding and §9's *A metric grades the world that charges it* is untouched.
#: What it *is* is a third table the same rule is applied to, and the milestone's
#: comparability claim is that all three select the same point.
LIQUIDITY_READING = "power_law+liquidity"

#: The two denominators a tolerance may be a fraction of, and the ``Grade``
#: attribute each reads. ``twap_gap`` is M2's and M3's: the excess over the
#: optimum as a fraction of the excess TWAP already carries, portable across
#: lambda by construction. ``available_advantage`` is M4a's: the excess as a
#: fraction of what the tangent-derived closed form leaves on the table in a
#: world it was not derived for.
TWAP_GAP = "twap_gap"
AVAILABLE_ADVANTAGE = "available_advantage"
DENOMINATORS: dict[str, str] = {
    TWAP_GAP: "gap_fraction",
    AVAILABLE_ADVANTAGE: "advantage_fraction",
}


@dataclass(frozen=True)
class Tolerances:
    """The pre-stated bar, in the units the brief states it in.

    A tolerance is a *fraction*, and `denominator` says of what. Never of the
    objective itself: "within 0.03 bps" would be trivially met at low lambda and
    unmeetable at high.

    ``twap_gap`` — M2's and M3's — is "within 5 % of the distance TWAP already
    covers", which means the same thing at every lambda. It is also degenerate
    wherever TWAP and the optimum have nearly converged, which is the §9 entry
    *The per-lambda tolerance is meaningful only where the testbed is
    discriminative*.

    ``available_advantage`` — M4a's — is "within 5 % of what the closed form left
    on the table". The switch is not a preference. At M4a's lambda the whole
    available advantage is 0.0367 bps while 5 % of that lambda's TWAP gap is
    0.0743 bps, so an agent held to the older denominator could capture *none* of
    the mis-specification the milestone exists to measure and still pass by a
    factor of two. The denominator is the finding; the number 0.05 is not.
    """

    epsilon_fraction: float         # median across seeds
    per_seed_fraction: float        # no individual seed worse than this
    red_flag_rtol: float            # slack on J_agent >= J_optimal
    denominator: str = TWAP_GAP

    def __post_init__(self) -> None:
        if self.denominator not in DENOMINATORS:
            raise ValueError(
                f"unknown tolerance denominator {self.denominator!r}; expected "
                f"one of {', '.join(sorted(DENOMINATORS))}"
            )
        if self.epsilon_fraction > self.per_seed_fraction:
            raise ValueError(
                "the per-seed floor must be at least as loose as the median "
                f"tolerance, got {self.per_seed_fraction} < "
                f"{self.epsilon_fraction}"
            )

    @property
    def graded_attribute(self) -> str:
        """The :class:`~temper.eval.grading.Grade` field the bars are read on."""
        return DENOMINATORS[self.denominator]

    def as_dict(self) -> dict:
        return {
            "denominator": self.denominator,
            "graded_attribute": self.graded_attribute,
            "epsilon_fraction": self.epsilon_fraction,
            "per_seed_fraction": self.per_seed_fraction,
            "red_flag_rtol": self.red_flag_rtol,
        }


@dataclass(frozen=True)
class SeedPlan:
    """Which streams of which pools this experiment is allowed to spend.

    Training draws from ``train``, evaluation from ``eval``, and the two are
    disjoint by construction (:mod:`temper.seeding`). Streams are *addressed*,
    not drawn: seed ordinal ``i`` gets envs on streams
    ``i * env_stream_stride + 0 .. num_envs - 1``, so adding a sixth seed cannot
    move the streams the first five were reported on.
    """

    root_seed: int
    train_pool: str
    eval_pool: str
    n_seeds: int
    env_stream_stride: int
    eval_streams: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.n_seeds < 5:
            raise ValueError(
                f"constitution invariant 4 requires >= 5 training seeds, got "
                f"{self.n_seeds}"
            )
        if len(self.eval_streams) < 2:
            raise ValueError(
                "the eval-determinism assertion needs at least two shock "
                f"streams, got {list(self.eval_streams)}"
            )

    def env_streams(self, seed_ordinal: int, num_envs: int) -> tuple[int, ...]:
        """The stream indices seed `seed_ordinal`'s parallel envs occupy."""
        if not 0 <= seed_ordinal < self.n_seeds:
            raise ValueError(
                f"seed ordinal {seed_ordinal} is outside 0..{self.n_seeds - 1}"
            )
        if num_envs > self.env_stream_stride:
            raise ValueError(
                f"{num_envs} envs per seed does not fit the committed stride of "
                f"{self.env_stream_stride}; seeds would share shock streams"
            )
        base = seed_ordinal * self.env_stream_stride
        return tuple(base + index for index in range(num_envs))

    def as_dict(self) -> dict:
        return {
            "root_seed": self.root_seed,
            "train_pool": self.train_pool,
            "eval_pool": self.eval_pool,
            "n_seeds": self.n_seeds,
            "env_stream_stride": self.env_stream_stride,
            "eval_streams": list(self.eval_streams),
        }


#: The reward regimes an experiment may train on. ``sampled`` is the realised
#: reward, M2's default; ``control_variate`` subtracts M1a's analytic noise
#: identity (M2's amendment 1); ``antithetic`` averages each episode with its
#: negated-shock mirror (M3). The claim a result may make depends on which.
SAMPLED = "sampled"
CONTROL_VARIATE = "control_variate"
ANTITHETIC = "antithetic"
REGIMES: tuple[str, ...] = (SAMPLED, CONTROL_VARIATE, ANTITHETIC)


@dataclass(frozen=True)
class Estimator:
    """Which reward the agent is trained on, and what that lets the result claim.

    ``sampled`` is M2's default and its headline: vanilla PPO on the realised
    reward. The other two are variance-reduction regimes and each is a *weaker*
    claim, not a better run — the control variate because it subtracts the
    analytic noise form, antithetic pairing because it trains on the average of
    a mirrored pair. The brief requires the amendment to be recorded before the
    run rather than after, so the claim string lives here, beside the switch,
    and is copied verbatim into ``results/`` and the figure caption.

    ``control_variate`` survives as a boolean view for the M2 configs, which
    spell the regime that way; a config may state either form and, if it states
    both, they must agree.
    """

    regime: str
    claim: str

    def __post_init__(self) -> None:
        if self.regime not in REGIMES:
            raise ValueError(
                f"unknown estimator regime {self.regime!r}; expected one of "
                f"{', '.join(REGIMES)}"
            )

    @property
    def control_variate(self) -> bool:
        return self.regime == CONTROL_VARIATE

    @property
    def antithetic(self) -> bool:
        return self.regime == ANTITHETIC

    @property
    def sampled(self) -> bool:
        return self.regime == SAMPLED

    def as_dict(self) -> dict:
        return {
            "regime": self.regime,
            "control_variate": self.control_variate,
            "claim": self.claim,
        }


def _estimator(block: dict) -> Estimator:
    """Resolve the config's `estimator` block, in either spelling."""
    regime = block.get("regime")
    switch = block.get("control_variate")
    if regime is None and switch is None:
        raise ValueError("the estimator block must state `regime` or `control_variate`")
    if regime is None:
        regime = CONTROL_VARIATE if bool(switch) else SAMPLED
    elif switch is not None and bool(switch) != (regime == CONTROL_VARIATE):
        raise ValueError(
            f"estimator regime {regime!r} contradicts control_variate: {switch!r}"
        )
    return Estimator(regime=str(regime), claim=str(block["claim"]).strip())


@dataclass(frozen=True)
class Gate:
    """A validation bar stated against a committed result rather than against ε.

    M3 task 1 is a gate: antithetic pairing must reproduce, to within an order of
    magnitude, the median gap fraction M2's control variate already established
    at the same lambda. That is a different question from "did the agent meet
    the milestone tolerance", so it is a different block, and the verdict
    reports it separately.
    """

    median_gap_fraction: float
    reference: Path

    def as_dict(self) -> dict:
        return {
            "median_gap_fraction": self.median_gap_fraction,
            "reference": self.reference.name,
        }


@dataclass(frozen=True)
class Runtime:
    """Wall-clock budgets from the brief. Exceeding one is reported, not hidden."""

    seconds_per_seed: float
    sweep_seconds: float

    def as_dict(self) -> dict:
        return {
            "seconds_per_seed": self.seconds_per_seed,
            "sweep_seconds": self.sweep_seconds,
        }


@dataclass(frozen=True)
class Experiment:
    """Everything a run needs, and nothing it may decide for itself."""

    path: Path
    document: dict
    milestone: str
    case: ExecutionCase
    lambda_risk: float
    lambda_grid: str
    #: The world this experiment trains and grades in. Defaulted to Phase 1 at
    #: the loader, never inherited from anywhere else: constitution §4 says a
    #: Phase-2 model is an additive alternative and never a silent modification,
    #: so a config that wants one has to name it and
    #: ``tests/test_repo_invariants.py`` checks that it did.
    cost_encoding: str
    #: The liquidity law this experiment's market runs under. Defaulted to
    #: deterministic at the loader and never inherited from anywhere else — the
    #: same rule as `cost_encoding`, for the same reason and enforced by the same
    #: ``tests/test_repo_invariants.py`` check: a config written for M2, M3 or M4a
    #: and re-run after M4b landed must not silently acquire a second noise source.
    liquidity: LiquidityLaw
    rule: LambdaRule
    #: The frontier sub-grid this experiment is one point of, or ``None`` when
    #: its lambda is the rule-selected one (M2, and M3's validation run).
    frontier_grid: str | None
    tolerances: Tolerances
    seeds: SeedPlan
    reward_scale: float
    estimator: Estimator
    ppo: PPOConfig
    runtime: Runtime
    results_metrics: Path
    results_figure: Path
    figure_formats: tuple[str, ...]
    #: Cap on per-update trace points kept per seed in the results file; ``None``
    #: keeps them whole. See :func:`~temper.eval.sweep.thin` for why this is a
    #: committed field rather than a default.
    trace_points: int | None
    #: The validation gate, if this experiment is one (M3 task 1); else ``None``.
    gate: Gate | None = None

    # -- oracle surface, derived on demand ---------------------------------

    def table(self, encoding: str | None = None) -> list[ReferenceRow]:
        """Task 0's reference table over the committed grid, in a world.

        `encoding` defaults to the experiment's own. Passing the other one is how
        :meth:`verify_lambda_rule` checks that both worlds' tables select the
        same lambda — which is M4a's task-0 gate, and the condition under which
        this milestone's result is comparable to M2's and M3's at all.
        """
        return reference_table(
            self.case.market,
            self.case.order_size,
            LAMBDA_GRIDS[self.lambda_grid],
            encoding=self.cost_encoding if encoding is None else encoding,
        )

    def reference(self, encoding: str | None = None) -> ReferenceRow:
        """The world's reference schedules at the committed lambda."""
        return reference_row(
            self.case.market,
            self.case.order_size,
            self.lambda_risk,
            encoding=self.cost_encoding if encoding is None else encoding,
        )

    def rule_selected(self, encoding: str | None = None) -> ReferenceRow:
        """The row M2 task 0's rule selects on the committed grid, in a world."""
        return select_lambda(self.table(encoding), self.rule)

    def liquidity_table(self) -> list[LiquidityReferenceRow]:
        """The liquidity world's *static* table over the committed grid.

        Static, because that is the recorded reading of the rule (task 0 gate 1):
        the liquidity world's fixed-schedule problem is M4a's at the coefficient
        ``A E[L^-beta]``, so the rule reads a schedule as it always has. It is
        also five certified solves per grid point rather than a dynamic program,
        which is what keeps :meth:`verify_lambda_rule` affordable at the top of a
        run.
        """
        return static_liquidity_table(
            self.case.market,
            self.case.order_size,
            self.liquidity,
            LAMBDA_GRIDS[self.lambda_grid],
        )

    def liquidity_reference(self, **kwargs) -> LiquidityReferenceRow:
        """The full liquidity row at the committed lambda — DP and both bounds.

        Minutes, not milliseconds: the caller is a reference-table driver or a
        grading path that needs the adaptive optimum, never a config check.
        """
        kwargs.setdefault("root_seed", self.seeds.root_seed)
        return liquidity_reference_row(
            self.case.market,
            self.case.order_size,
            self.lambda_risk,
            self.liquidity,
            **kwargs,
        )

    def verify_lambda_rule(self) -> ReferenceRow:
        """Re-derive lambda from the rule; raise if the config disagrees.

        Called at the top of every training run as well as from the suite. The
        cost is seventeen closed-form evaluations; the thing it buys is that no
        reported M2 number can have been produced at a lambda somebody chose
        after seeing a curve (invariant 3).

        A frontier sweep point is the one legitimate exception, and it is
        checked rather than exempted: its lambda must be a member of the named
        frontier grid, that grid must be a subset of the committed grid, and it
        must contain the rule-selected lambda — so the sweep is a pre-stated set
        of points around a committed one, not a set chosen after the fact.

        **Both worlds must agree.** From M4a the rule is applied to the table of
        *every* encoding and the selections must match. A Phase-2 milestone whose
        lambda was chosen by a different rule than the Phase-1 results it is
        compared against is not comparable to them, and the resolution is a
        decision rather than a rescue — so this raises instead of preferring one.
        The rule itself is unchanged: smallest lambda whose TWAP gap clears
        `min_twap_gap` and whose optimum's largest bin is inside
        `max_bin_fraction`, read off each world's own optimum.
        """
        self.verify_lambda_rule_agrees_across_worlds()
        selected = self.rule_selected()
        if self.frontier_grid is not None:
            try:
                frontier = FRONTIER_GRIDS[self.frontier_grid]
            except KeyError:
                raise ValueError(
                    f"{self.path.name} names frontier grid {self.frontier_grid!r}; "
                    f"known grids are {', '.join(sorted(FRONTIER_GRIDS))}"
                ) from None
            grid = LAMBDA_GRIDS[self.lambda_grid]
            if not set(frontier) <= set(grid):
                raise ValueError(
                    f"frontier grid {self.frontier_grid!r} is not a subset of the "
                    f"{self.lambda_grid} grid"
                )
            if selected.lambda_risk not in frontier:
                raise ValueError(
                    f"frontier grid {self.frontier_grid!r} does not contain the "
                    f"rule-selected lambda {selected.lambda_risk:.6e}; the sweep "
                    "must include a point comparable to a committed result"
                )
            if self.lambda_risk not in frontier:
                raise ValueError(
                    f"{self.path.name} fixes lambda = {self.lambda_risk:.6e}, which "
                    f"is not a point of frontier grid {self.frontier_grid!r}"
                )
            return self.reference()
        if selected.lambda_risk != self.lambda_risk:
            raise ValueError(
                f"{self.path.name} fixes lambda = {self.lambda_risk:.6e}, but task "
                f"0's rule (twap gap >= {self.rule.min_twap_gap:g}, max bin "
                f"fraction <= {self.rule.max_bin_fraction:g}) selects "
                f"{selected.lambda_risk:.6e} on the {self.lambda_grid} grid. "
                "Change the rule in the brief first, or change the case — not the "
                "committed lambda."
            )
        return selected

    def verify_lambda_rule_agrees_across_worlds(self) -> dict[str, float]:
        """Apply the rule to every world's table; raise unless they agree.

        Cheap — a few dozen closed forms and one Newton solve per grid point —
        and the thing it buys is that M4a's training point sits at the lambda two
        committed milestones already answered, so its capture fraction can be put
        beside their gap fractions without an argument about the x axis.

        **M4b adds a third reading, not a third encoding.** Liquidity randomises
        the market rather than the cost functional, so it is not an entry in
        ``ENCODINGS``; what it is is one more table the same rule is applied to,
        and it is included here whenever the experiment's law is stochastic. The
        reading is the *static* one — the liquidity world at ``A E[L^-beta]`` — for
        the reason recorded in ``tools/m4b_reference_table.py``: it is a closed
        form, and the alternative (reading the rule off the dynamic program's
        value and its mean schedule) clears the 20 % bar at 10^-4 by 0.011
        percentage points, which would make a milestone's lambda turn on the fifth
        digit of a numerically-solved value function.
        """
        selections = {
            encoding: self.rule_selected(encoding).lambda_risk
            for encoding in ENCODINGS
        }
        if self.liquidity.stochastic:
            selections[LIQUIDITY_READING] = select_lambda(
                self.liquidity_table(), self.rule
            ).lambda_risk
        if len(set(selections.values())) > 1:
            described = ", ".join(
                f"{encoding} selects {lam:.6e}"
                for encoding, lam in sorted(selections.items())
            )
            raise ValueError(
                f"task 0's rule selects a different lambda in each world "
                f"({described}). A Phase-2 milestone whose lambda was chosen by a "
                "different rule than the Phase-1 results it is compared against is "
                "not comparable to them; decide which case to run before running "
                "one."
            )
        return selections

    def verify_gate_reference(self) -> float | None:
        """The gate's reference result exists and states a median; else raise.

        Called at the top of a run and by ``--dry-run``, because a gate that
        discovers its reference is missing *after* the night's training has
        finished would discard the run with nothing written. Returns the
        reference's median gap fraction, or ``None`` when there is no gate.
        """
        if self.gate is None:
            return None
        if not self.gate.reference.exists():
            raise FileNotFoundError(
                f"{self.path.name} gates against {self.gate.reference}, which "
                "does not exist; the committed result it validates against must "
                "be present before the run"
            )
        document = json.loads(self.gate.reference.read_text(encoding="utf-8"))
        return float(document["summary"]["gap_fraction"]["median"])

    # -- the derived trajectory band ---------------------------------------

    def denominator_bps(self, reference: ReferenceRow | None = None) -> float:
        """The bps quantity the tolerance is a fraction *of*, in this world.

        The TWAP gap for M2 and M3; the available advantage for M4a. One
        function, so the bar, the band and the reported excess cannot be computed
        against three different denominators by three call sites.
        """
        row = self.reference() if reference is None else reference
        if self.tolerances.denominator == AVAILABLE_ADVANTAGE:
            advantage = row.available_advantage
            if advantage is None:
                raise ValueError(
                    f"{self.path.name} states its tolerance against the available "
                    f"advantage, but the {row.encoding!r} world has no "
                    "tangent-derived schedule to beat — there the closed form is "
                    "the optimum, and an agent that beat it would be a red flag "
                    "rather than a result"
                )
            return advantage
        return row.twap.objective - row.optimal.objective

    def band(self, fraction: float | None = None) -> TrajectoryBand:
        """The trajectory band the tolerance implies, via this world's Hessian.

        `fraction` defaults to the median tolerance, so ``experiment.band()`` is
        "how far from the optimum may a schedule that meets epsilon sit?" — the
        number the figure draws and the brief reports beside the observed
        deviation. In the power-law world the returned band is marked
        :attr:`~temper.eval.reference.TrajectoryBand.local`, because there the
        Hessian moves with the schedule.
        """
        chosen = (
            self.tolerances.epsilon_fraction if fraction is None else fraction
        )
        excess = chosen * self.denominator_bps()
        return trajectory_band(
            self.case.market,
            self.case.order_size,
            self.lambda_risk,
            excess,
            encoding=self.cost_encoding,
        )

    def provenance(self, repo_root: Path | None = None) -> Provenance:
        return stamp(self.path, repo_root)

    def as_dict(self) -> dict:
        """The committed contract, for the results file's `config` block."""
        return {
            "path": self.path.name,
            "milestone": self.milestone,
            "case": self.case.as_dict(),
            "lambda_risk": self.lambda_risk,
            "lambda_grid": self.lambda_grid,
            "cost_encoding": self.cost_encoding,
            "liquidity": self.liquidity.as_dict(),
            "frontier_grid": self.frontier_grid,
            "lambda_rule": {
                "min_twap_gap": self.rule.min_twap_gap,
                "max_bin_fraction": self.rule.max_bin_fraction,
            },
            "tolerances": self.tolerances.as_dict(),
            "seeding": self.seeds.as_dict(),
            "reward_scale": self.reward_scale,
            "estimator": self.estimator.as_dict(),
            "ppo": self.ppo.as_dict(),
            "runtime": self.runtime.as_dict(),
            "gate": None if self.gate is None else self.gate.as_dict(),
        }


def repo_root_of(config_path: str | Path) -> Path:
    """The repository root a config's relative paths are resolved against.

    The nearest ancestor holding ``pyproject.toml`` — so a config one directory
    deeper (``configs/m3_frontier/<point>.yaml``) resolves ``results/...`` to the
    same place ``configs/<experiment>.yaml`` does. Falls back to two levels up,
    the layout every committed config has always had.
    """
    path = Path(config_path).resolve()
    for ancestor in path.parents:
        if (ancestor / "pyproject.toml").exists():
            return ancestor
    return path.parent.parent


def _liquidity(document: dict) -> LiquidityLaw:
    """The liquidity law a config names, defaulting to deterministic.

    The second seam, under the first seam's rule. Constitution §4's "additive
    alternatives behind the same interface, never silent modifications of Phase 1"
    is a sentence about defaults, and M4b hands out *two* models — so a config
    with no ``world.liquidity`` block gets ``L = 1``, exactly the market M0
    through M4a ran in, and one that wants a second noise source has to write it
    down where a reader and a digest can both see it.
    """
    block = (document.get("world") or {}).get("liquidity")
    if block is None:
        return DeterministicLiquidity()
    parameters = {key: value for key, value in block.items() if key != "model"}
    return liquidity_for(str(block["model"]), **parameters)


def _cost_encoding(document: dict) -> str:
    """The world a config names, defaulting to Phase 1.

    Constitution §4: Phase-2 models are *additive alternatives behind the same
    interface, never silent modifications of Phase 1*. The default here is what
    makes that mechanical — a config with no ``world`` block gets the linearised
    world, so no config can inherit a Phase-2 world by omission, and one that
    wants the power law has to write it down where a reader and a digest can both
    see it. ``tests/test_repo_invariants.py`` pins the default and pins that
    every committed config in a Phase-2 world names it.
    """
    block = document.get("world")
    if block is None:
        return LINEAR_ENCODING
    encoding = str(block["cost_encoding"])
    if encoding not in ENCODINGS:
        raise ValueError(
            f"unknown cost encoding {encoding!r}; known worlds are "
            f"{', '.join(sorted(ENCODINGS))}"
        )
    return encoding


def _tolerances(block: dict) -> Tolerances:
    """Resolve the ``tolerances`` block, in either spelling.

    M2 and M3 spell the bars ``epsilon_gap_fraction`` / ``per_seed_gap_fraction``
    because for them the denominator *was* the TWAP gap and there was only one.
    M4a's denominator is the available advantage, and a config that stated
    ``epsilon_gap_fraction: 0.05`` while meaning "5 % of the available advantage"
    would be lying in the one place this milestone cannot afford one — so the
    neutral spelling ``epsilon_fraction`` exists and the ``_gap_`` spelling is
    accepted only alongside the ``twap_gap`` denominator. The committed M2/M3
    configs are byte-frozen by their result digests and keep the name they were
    stamped with.
    """
    denominator = str(block.get("denominator", TWAP_GAP))
    legacy = "epsilon_gap_fraction" in block or "per_seed_gap_fraction" in block
    neutral = "epsilon_fraction" in block or "per_seed_fraction" in block
    if legacy and neutral:
        raise ValueError(
            "the tolerances block states both the `_gap_fraction` and the "
            "`_fraction` spelling; use one"
        )
    if legacy and denominator != TWAP_GAP:
        raise ValueError(
            f"the tolerances block spells its bars `epsilon_gap_fraction` but "
            f"names the {denominator!r} denominator; the `_gap_` spelling means "
            "the TWAP gap and nothing else"
        )
    suffix = "_gap_fraction" if legacy else "_fraction"
    return Tolerances(
        epsilon_fraction=float(block["epsilon" + suffix]),
        per_seed_fraction=float(block["per_seed" + suffix]),
        red_flag_rtol=float(block["red_flag_rtol"]),
        denominator=denominator,
    )


def _market(block: dict) -> Market:
    return Market(
        params=SymbolParams(**block["params"]),
        horizon_hours=float(block["horizon_hours"]),
        n_bins=int(block["n_bins"]),
    )


def load_experiment(path: str | Path) -> Experiment:
    """Parse a committed experiment config. Unknown PPO keys are an error."""
    config_path = Path(path)
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    case_block = document["case"]
    selection = document["lambda_selection"]
    grid = str(selection.get("grid", "vendor"))
    if grid not in LAMBDA_GRIDS:
        raise ValueError(
            f"unknown lambda grid {grid!r}; known grids are "
            f"{', '.join(sorted(LAMBDA_GRIDS))}"
        )

    results = document["results"]
    root = repo_root_of(config_path)
    gate_block = document.get("gate")
    return Experiment(
        path=config_path,
        document=document,
        # M2's configs predate the key and are frozen by their committed digests.
        milestone=str(document.get("milestone", "M2")),
        case=ExecutionCase(
            case_id=case_block["id"],
            params_from=case_block["params_from"],
            symbol=case_block["symbol"],
            market=_market(case_block),
            order_size=float(case_block["order_size"]),
        ),
        lambda_risk=float(selection["lambda_risk"]),
        lambda_grid=grid,
        cost_encoding=_cost_encoding(document),
        liquidity=_liquidity(document),
        frontier_grid=(
            None
            if selection.get("frontier_grid") is None
            else str(selection["frontier_grid"])
        ),
        rule=LambdaRule(
            min_twap_gap=float(selection["rule"]["min_twap_gap"]),
            max_bin_fraction=float(selection["rule"]["max_bin_fraction"]),
        ),
        tolerances=_tolerances(document["tolerances"]),
        seeds=SeedPlan(
            root_seed=int(document["seeding"]["root_seed"]),
            train_pool=str(document["seeding"]["train_pool"]),
            eval_pool=str(document["seeding"]["eval_pool"]),
            n_seeds=int(document["seeding"]["n_seeds"]),
            env_stream_stride=int(document["seeding"]["env_stream_stride"]),
            eval_streams=tuple(int(s) for s in document["seeding"]["eval_streams"]),
        ),
        reward_scale=float(document["reward"]["scale"]),
        estimator=_estimator(document["estimator"]),
        ppo=PPOConfig.from_mapping(document["ppo"]),
        runtime=Runtime(
            seconds_per_seed=float(document["runtime"]["seconds_per_seed"]),
            sweep_seconds=float(document["runtime"]["sweep_seconds"]),
        ),
        results_metrics=root / results["metrics"],
        results_figure=root / results["figure"],
        figure_formats=tuple(str(f) for f in results.get("formats", ("png",))),
        trace_points=(
            None
            if results.get("trace_points") is None
            else int(results["trace_points"])
        ),
        gate=(
            None
            if gate_block is None
            else Gate(
                median_gap_fraction=float(gate_block["median_gap_fraction"]),
                reference=root / gate_block["reference"],
            )
        ),
    )


def golden_parameters_match(case: ExecutionCase, golden: dict) -> Sequence[str]:
    """Names of any case field that disagrees with the vendored golden case.

    The config repeats the symbol parameters so that ``temper/`` and ``tools/``
    never have to reach into ``tests/golden/``, which is a test fixture and not
    package data. That repetition is the thing that could drift, so the suite
    closes it: ``tests/test_m2_reference.py`` calls this against the case the
    config's ``params_from`` names, and a single changed digit is red.
    """
    market, params = case.market, case.market.params
    expected = {
        "symbol": (case.symbol, golden["symbol"]),
        "horizon_hours": (market.horizon_hours, golden["horizon_hours"]),
        "n_bins": (market.n_bins, golden["n_bins"]),
        "adv": (params.adv, golden["params"]["adv"]),
        "sigma": (params.sigma, golden["params"]["sigma"]),
        "half_spread": (params.half_spread, golden["params"]["half_spread"]),
        "eta": (params.eta, golden["params"]["eta"]),
        "gamma": (params.gamma, golden["params"]["gamma"]),
    }
    return [name for name, (mine, theirs) in expected.items() if mine != theirs]
