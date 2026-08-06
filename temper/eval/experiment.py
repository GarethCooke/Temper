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

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from temper.agents.ppo import PPOConfig
from temper.eval.provenance import Provenance, stamp
from temper.eval.reference import (
    LambdaRule,
    ReferenceRow,
    TrajectoryBand,
    reference_row,
    reference_table,
    select_lambda,
    trajectory_band,
)
from temper.oracle import VENDOR_LAMBDA_GRID, Market, SymbolParams

#: Lambda grids a config may name. ``vendor`` is M0's 17-point log-half-decade
#: sweep, which is the grid task 0's rule is stated over. A config naming
#: anything else would be selecting lambda from a set chosen after the fact.
LAMBDA_GRIDS: dict[str, tuple[float, ...]] = {"vendor": VENDOR_LAMBDA_GRID}


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


@dataclass(frozen=True)
class Tolerances:
    """The pre-stated bar, in the units the brief states it in.

    Both tolerances are fractions of the *TWAP gap* rather than of the objective
    itself. That is deliberate and it is what makes them portable: "within 5 % of
    the distance TWAP already covers" means the same thing at every lambda, while
    "within 0.03 bps" would be trivially met at low lambda and unmeetable at
    high.
    """

    epsilon_gap_fraction: float     # median across seeds
    per_seed_gap_fraction: float    # no individual seed worse than this
    red_flag_rtol: float            # slack on J_agent >= J_optimal

    def __post_init__(self) -> None:
        if self.epsilon_gap_fraction > self.per_seed_gap_fraction:
            raise ValueError(
                "the per-seed floor must be at least as loose as the median "
                f"tolerance, got {self.per_seed_gap_fraction} < "
                f"{self.epsilon_gap_fraction}"
            )

    def as_dict(self) -> dict:
        return {
            "epsilon_gap_fraction": self.epsilon_gap_fraction,
            "per_seed_gap_fraction": self.per_seed_gap_fraction,
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


@dataclass(frozen=True)
class Estimator:
    """Which reward the agent is trained on, and what that lets the result claim.

    ``control_variate: false`` is M2's default and its headline: vanilla PPO on
    sampled rewards. Turning it on is a *weaker* claim, not a better run, and the
    brief requires the amendment to be recorded before the run rather than after
    — so the claim string lives here, beside the switch, and is copied verbatim
    into ``results/`` and the figure caption.
    """

    control_variate: bool
    claim: str

    def as_dict(self) -> dict:
        return {"control_variate": self.control_variate, "claim": self.claim}


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
    case: ExecutionCase
    lambda_risk: float
    lambda_grid: str
    rule: LambdaRule
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

    # -- oracle surface, derived on demand ---------------------------------

    def table(self) -> list[ReferenceRow]:
        """Task 0's reference table over the committed grid."""
        return reference_table(
            self.case.market, self.case.order_size, LAMBDA_GRIDS[self.lambda_grid]
        )

    def reference(self) -> ReferenceRow:
        """The three schedules at the committed lambda."""
        return reference_row(self.case.market, self.case.order_size, self.lambda_risk)

    def verify_lambda_rule(self) -> ReferenceRow:
        """Re-derive lambda from the rule; raise if the config disagrees.

        Called at the top of every training run as well as from the suite. The
        cost is seventeen closed-form evaluations; the thing it buys is that no
        reported M2 number can have been produced at a lambda somebody chose
        after seeing a curve (invariant 3).
        """
        selected = select_lambda(self.table(), self.rule)
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

    # -- the derived trajectory band ---------------------------------------

    def band(self, gap_fraction: float | None = None) -> TrajectoryBand:
        """The trajectory band the tolerance implies, via task 0's Hessian.

        `gap_fraction` defaults to the median tolerance, so ``experiment.band()``
        is "how far from the sinh may a schedule that meets epsilon sit?" — the
        number the figure draws and the brief reports beside the observed
        deviation.
        """
        fraction = (
            self.tolerances.epsilon_gap_fraction if gap_fraction is None
            else gap_fraction
        )
        reference = self.reference()
        excess = fraction * (
            reference.twap.objective - reference.optimal.objective
        )
        return trajectory_band(
            self.case.market, self.case.order_size, self.lambda_risk, excess
        )

    def provenance(self, repo_root: Path | None = None) -> Provenance:
        return stamp(self.path, repo_root)

    def as_dict(self) -> dict:
        """The committed contract, for the results file's `config` block."""
        return {
            "path": self.path.name,
            "case": self.case.as_dict(),
            "lambda_risk": self.lambda_risk,
            "lambda_grid": self.lambda_grid,
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
        }


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
    root = config_path.resolve().parent.parent
    return Experiment(
        path=config_path,
        document=document,
        case=ExecutionCase(
            case_id=case_block["id"],
            params_from=case_block["params_from"],
            symbol=case_block["symbol"],
            market=_market(case_block),
            order_size=float(case_block["order_size"]),
        ),
        lambda_risk=float(selection["lambda_risk"]),
        lambda_grid=grid,
        rule=LambdaRule(
            min_twap_gap=float(selection["rule"]["min_twap_gap"]),
            max_bin_fraction=float(selection["rule"]["max_bin_fraction"]),
        ),
        tolerances=Tolerances(
            epsilon_gap_fraction=float(
                document["tolerances"]["epsilon_gap_fraction"]
            ),
            per_seed_gap_fraction=float(
                document["tolerances"]["per_seed_gap_fraction"]
            ),
            red_flag_rtol=float(document["tolerances"]["red_flag_rtol"]),
        ),
        seeds=SeedPlan(
            root_seed=int(document["seeding"]["root_seed"]),
            train_pool=str(document["seeding"]["train_pool"]),
            eval_pool=str(document["seeding"]["eval_pool"]),
            n_seeds=int(document["seeding"]["n_seeds"]),
            env_stream_stride=int(document["seeding"]["env_stream_stride"]),
            eval_streams=tuple(int(s) for s in document["seeding"]["eval_streams"]),
        ),
        reward_scale=float(document["reward"]["scale"]),
        estimator=Estimator(
            control_variate=bool(document["estimator"]["control_variate"]),
            claim=str(document["estimator"]["claim"]).strip(),
        ),
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
