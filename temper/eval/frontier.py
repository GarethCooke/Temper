"""The M3 frontier sweep: one committed config per lambda, one aggregate at the end.

A sweep is not one experiment, it is nine — each lambda point is a full
:class:`~temper.eval.experiment.Experiment` with its own config, its own
results file, its own tolerances (per-lambda: 5 % of *that* lambda's TWAP gap),
its own derived trajectory band and its own three baselines through the
identical grader. Keeping the points as ordinary experiments means every
per-point check the suite already makes for M2 and for the validation run
(provenance digest, red flags, monotone schedules, objective consistency)
applies unchanged at every lambda, and ``tools/train.py`` runs a point the same
way it runs anything else.

What this module adds is the layer above the points:

* the **manifest** (``configs/m3_frontier.yaml``) — which frontier grid, which
  template config the points are stamped from, and exactly which fields a point
  may change (lambda, the update budget task 2 amended, the runtime bounds, the
  trace budget, the results paths). Point configs are *generated* from it and
  the suite asserts the committed files are byte-identical to what the generator
  produces, so a sweep point cannot quietly drift from the template;
* the **aggregate** — every point's results file read back into one document
  with, per lambda, the agent's ``(E, V - floor)`` per seed, the median and IQR,
  the three baselines, the per-lambda tolerance verdict, the band, the runtime,
  and the reward-variance evidence — plus the dense oracle curves the figure
  draws the frontier with. The frontier figure is a view of that document and of
  nothing else, so it redraws byte-identically without a single training step.

Nothing here imports matplotlib; :func:`temper.eval.figures.frontier_figure`
does the drawing.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from temper.eval.experiment import FRONTIER_GRIDS, Experiment, load_experiment, repo_root_of
from temper.eval.grading import summarise
from temper.eval.provenance import Provenance, config_digest, stamp
from temper.eval.reference import reference_row, variance_floor_bps2
from temper.oracle import Market

#: Top-level keys of a point config a generated point may differ from its
#: template in. Everything else is the template's, byte for byte in intent and
#: field for field in the loaded document; ``tests/test_m3_frontier.py`` asserts
#: it against every committed point.
POINT_MAY_CHANGE: frozenset[str] = frozenset(
    {"lambda_selection", "ppo", "runtime", "results", "gate", "estimator"}
)

#: Within ``estimator``, a point may restate the *validation* sentence of the
#: claim (the template's says "validated here", which is true of the validation
#: run and not of a sweep point) — never the regime, and never the mechanism
#: paragraph, which ``tests/test_m3_frontier.py`` requires to be the template's
#: word for word up to this marker.
CLAIM_MECHANISM_END = "This is NOT a claim about learning under noise"

#: Header stamped onto every generated point config. Part of the bytes the
#: config digest covers, so it is fixed text with a fixed vocabulary.
POINT_HEADER = """\
# GENERATED — do not edit. One point of the M3 frontier sweep, stamped from the
# template named in configs/m3_frontier.yaml by `tools/m3_frontier.py configs`;
# tests/test_m3_frontier.py asserts this file is byte-identical to what the
# generator produces, so a sweep point cannot drift from the template.
#
#   lambda        {lambda_repr}   (10^{log10:+.1f}, point {index} of {count} on the {grid} grid)
#   template      {template}
#   differs in    lambda_selection (frontier_grid, lambda_risk), ppo.total_timesteps
#                 (the update budget task 2 amended), runtime (per-point bounds),
#                 results (per-point paths, trace_points 128), no gate, and the
#                 claim's closing sentence (which names the validation run rather
#                 than calling itself one)
#
# Everything else — the case, the tolerances (5 % / 10 % of THIS lambda's TWAP
# gap), the seed addressing, the reward scale, the estimator and its claim, the
# PPO hyperparameters — is the template's, which is M2's control-variate config
# with ten seeds and the antithetic regime.
"""


def point_name(lambda_risk: float) -> str:
    """``lambda_1e-4.5`` for ``10^-4.5`` — the file stem of a point."""
    exponent = round(2.0 * math.log10(lambda_risk)) / 2.0
    return f"lambda_1e{exponent:+.1f}".replace("+", "")


@dataclass(frozen=True)
class FrontierManifest:
    """``configs/m3_frontier.yaml``, resolved."""

    path: Path
    document: dict
    milestone: str
    grid: str
    template: Path
    points_dir: Path
    overrides: dict
    results_metrics: Path
    results_figure: Path
    figure_formats: tuple[str, ...]
    results_points_dir: Path
    full_budget_point: Path | None

    @property
    def lambdas(self) -> tuple[float, ...]:
        return FRONTIER_GRIDS[self.grid]

    def point_config(self, lambda_risk: float) -> Path:
        return self.points_dir / f"{point_name(lambda_risk)}.yaml"

    def point_metrics(self, lambda_risk: float) -> Path:
        return self.results_points_dir / f"{point_name(lambda_risk)}.json"

    def point_figure(self, lambda_risk: float) -> Path:
        return self.results_points_dir / point_name(lambda_risk)

    def provenance(self, repo_root: Path | None = None) -> Provenance:
        return stamp(self.path, repo_root)


def load_manifest(path: str | Path) -> FrontierManifest:
    manifest_path = Path(path)
    document = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    root = repo_root_of(manifest_path)
    block = document["frontier"]
    grid = str(block["grid"])
    if grid not in FRONTIER_GRIDS:
        raise ValueError(
            f"unknown frontier grid {grid!r}; known grids are "
            f"{', '.join(sorted(FRONTIER_GRIDS))}"
        )
    results = block["results"]
    full = block.get("full_budget_point")
    return FrontierManifest(
        path=manifest_path,
        document=document,
        milestone=str(document.get("milestone", "M3")),
        grid=grid,
        template=root / block["template"],
        points_dir=root / block["points_dir"],
        overrides=dict(block.get("overrides", {})),
        results_metrics=root / results["metrics"],
        results_figure=root / results["figure"],
        figure_formats=tuple(str(f) for f in results.get("formats", ("png",))),
        results_points_dir=root / results["points_dir"],
        full_budget_point=None if full is None else root / full,
    )


# ---------------------------------------------------------------------------
# Point configs, generated from the template
# ---------------------------------------------------------------------------


def _deep_update(target: dict, changes: dict) -> None:
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def point_document(manifest: FrontierManifest, lambda_risk: float, repo_root: Path) -> dict:
    """The point's config as a document: the template, changed where allowed."""
    if lambda_risk not in manifest.lambdas:
        raise ValueError(f"{lambda_risk!r} is not on frontier grid {manifest.grid!r}")
    document = copy.deepcopy(
        yaml.safe_load(manifest.template.read_text(encoding="utf-8"))
    )
    document.pop("gate", None)
    document["lambda_selection"]["frontier_grid"] = manifest.grid
    document["lambda_selection"]["lambda_risk"] = float(lambda_risk)
    _deep_update(document, copy.deepcopy(manifest.overrides))
    document["results"] = {
        **document.get("results", {}),
        "metrics": manifest.point_metrics(lambda_risk).relative_to(repo_root).as_posix(),
        "figure": manifest.point_figure(lambda_risk).relative_to(repo_root).as_posix(),
        "formats": list(manifest.figure_formats),
        "trace_points": document.get("results", {}).get("trace_points"),
    }
    return document


def point_text(manifest: FrontierManifest, lambda_risk: float, repo_root: Path) -> str:
    """The point config's exact bytes (CRLF, like every committed config)."""
    lambdas = manifest.lambdas
    header = POINT_HEADER.format(
        lambda_repr=repr(float(lambda_risk)),
        log10=round(2.0 * math.log10(lambda_risk)) / 2.0,
        index=lambdas.index(lambda_risk) + 1,
        count=len(lambdas),
        grid=manifest.grid,
        template=manifest.template.relative_to(repo_root).as_posix(),
    )
    body = yaml.safe_dump(
        point_document(manifest, lambda_risk, repo_root),
        sort_keys=False,
        allow_unicode=True,
        width=88,
        default_flow_style=False,
    )
    return (header + "\n" + body).replace("\r\n", "\n").replace("\n", "\r\n")


def write_point_configs(manifest: FrontierManifest, repo_root: Path) -> list[Path]:
    manifest.points_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for lambda_risk in manifest.lambdas:
        path = manifest.point_config(lambda_risk)
        path.write_bytes(point_text(manifest, lambda_risk, repo_root).encode("utf-8"))
        written.append(path)
    return written


def stale_point_configs(manifest: FrontierManifest, repo_root: Path) -> list[Path]:
    """Committed point configs that are not what the generator would write."""
    stale = []
    for lambda_risk in manifest.lambdas:
        path = manifest.point_config(lambda_risk)
        expected = point_text(manifest, lambda_risk, repo_root).encode("utf-8")
        if not path.exists() or path.read_bytes() != expected:
            stale.append(path)
    return stale


def point_experiments(manifest: FrontierManifest) -> list[Experiment]:
    """Every point, in grid order (ascending lambda) — the reporting order."""
    return [load_experiment(manifest.point_config(lam)) for lam in manifest.lambdas]


def run_order(manifest: FrontierManifest) -> list[Experiment]:
    """Every point in the order a sweep should *run* them: the check point first.

    The rule-selected lambda goes first because it is the one point comparable to
    two committed results — M2's control-variate sweep and M3's full-budget
    validation run — so it is where the amended update budget is checked against
    a known answer. A budget that is wrong is then wrong after one point rather
    than after nine. Everything else follows in grid order; reporting order is
    unchanged (:func:`point_experiments`).
    """
    points = point_experiments(manifest)
    selected = points[0].rule_selected().lambda_risk
    first = [p for p in points if p.lambda_risk == selected]
    return first + [p for p in points if p.lambda_risk != selected]


# ---------------------------------------------------------------------------
# The aggregate
# ---------------------------------------------------------------------------


def _seed_rows(document: dict, floor: float) -> list[dict]:
    rows = []
    for record in document["seeds"]:
        grade, training = record["grade"], record["training"]
        rows.append(
            {
                "ordinal": record["ordinal"],
                "expected_bps": grade["expected_bps"],
                "variance_bps2": grade["variance_bps2"],
                "excess_variance_bps2": grade["variance_bps2"] - floor,
                "objective_bps": grade["objective_bps"],
                "relative_excess": grade["relative_excess"],
                "gap_fraction": grade["gap_fraction"],
                "deviation_shares": grade["deviation_shares"],
                "red_flag": grade["red_flag"],
                "seconds": training["seconds"],
                "updates": training["updates"],
                "timed_out": training["timed_out"],
            }
        )
    return rows


def _point_entry(experiment: Experiment, document: dict, *, role: str) -> dict:
    """One lambda's row of the aggregate, from its experiment and results file."""
    reference = document["reference"]
    floor = reference["variance_floor_bps2"]
    seeds = _seed_rows(document, floor)
    summary = {
        name: summarise(name, [row[name] for row in seeds]).as_dict()
        for name in (
            "gap_fraction",
            "relative_excess",
            "expected_bps",
            "excess_variance_bps2",
            "objective_bps",
            "deviation_shares",
        )
    }
    verdict = document["verdict"]
    return {
        "role": role,
        "lambda": experiment.lambda_risk,
        "log10_lambda": math.log10(experiment.lambda_risk),
        "name": point_name(experiment.lambda_risk),
        "config": experiment.path.name,
        "metrics": experiment.results_metrics.name,
        "provenance": document["provenance"],
        "updates": experiment.ppo.num_updates,
        "n_seeds": len(seeds),
        "twap_gap": reference["twap_gap"],
        "kappa_horizon": reference["kappa_horizon"],
        "variance_floor_bps2": floor,
        "baselines": {
            name: {
                "expected_bps": schedule["expected_bps"],
                "variance_bps2": schedule["variance_bps2"],
                "excess_variance_bps2": schedule["variance_bps2"] - floor,
                "objective_bps": schedule["objective_bps"],
                "gap_fraction": document["baselines"][name]["gap_fraction"],
            }
            for name, schedule in reference["schedules"].items()
        },
        "tolerances": document["config"]["tolerances"],
        "band": document["bands"]["epsilon"],
        "seeds": seeds,
        "summary": summary,
        "epsilon_met": verdict["epsilon_met"],
        "per_seed_met": verdict["per_seed_met"],
        "red_flags": verdict["red_flags"],
        "timed_out": verdict["timed_out"],
        "sweep_seconds": verdict["sweep_seconds"],
        "reward_variance": document["summary"].get("reward_variance"),
    }


def _dense_curves(market: Market, order_size: float, lambdas: Sequence[float], per_decade: int = 16) -> dict:
    """The oracle's frontier as smooth curves between the first and last grid points.

    Denser than the grid so the figure draws a curve rather than a polygon;
    the values at the grid points themselves are the reference rows' exactly.
    """
    lo, hi = math.log10(min(lambdas)), math.log10(max(lambdas))
    count = int(round((hi - lo) * per_decade)) + 1
    dense = [10.0 ** (lo + (hi - lo) * i / (count - 1)) for i in range(count)]
    dense = sorted(set(dense) | set(lambdas))
    floor = variance_floor_bps2(market)
    curves = {"lambda": dense, "twap": [], "ac": [], "optimal": []}
    for lam in dense:
        row = reference_row(market, order_size, lam)
        for name in ("twap", "ac", "optimal"):
            schedule = row.schedules[name]
            curves[name].append(
                {
                    "expected_bps": schedule.expected,
                    "excess_variance_bps2": schedule.variance - floor,
                    "objective_bps": schedule.objective,
                }
            )
    curves["variance_floor_bps2"] = floor
    return curves


def aggregate(
    manifest: FrontierManifest,
    repo_root: Path,
    *,
    require_complete: bool = True,
) -> dict:
    """Every point's results file, read back into one document."""
    experiments = point_experiments(manifest)
    points = []
    missing = []
    for experiment in experiments:
        # The manifest's results directory, not the point config's own path:
        # the two agree for every generated point (the suite asserts it), and
        # reading through the manifest is what lets a test aggregate fabricated
        # points from a scratch directory without touching results/.
        path = manifest.point_metrics(experiment.lambda_risk)
        if not path.exists():
            missing.append(path)
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        if document["provenance"]["config_sha256"] != config_digest(experiment.path):
            raise ValueError(
                f"{path.name} was produced by a different version of "
                f"{experiment.path.name}; re-run the point or restore the config"
            )
        points.append(_point_entry(experiment, document, role="sweep"))
    if missing and require_complete:
        raise FileNotFoundError(
            "frontier points without a results file: "
            + ", ".join(p.name for p in missing)
        )

    full_budget = None
    if manifest.full_budget_point is not None and manifest.full_budget_point.exists():
        template = load_experiment(manifest.template)
        document = json.loads(manifest.full_budget_point.read_text(encoding="utf-8"))
        full_budget = _point_entry(template, document, role="full_budget")

    template = experiments[0]
    market, order_size = template.case.market, template.case.order_size
    lambdas = list(manifest.lambdas)
    met = [p["lambda"] for p in points if p["epsilon_met"]]
    missed = [p["lambda"] for p in points if not p["epsilon_met"]]
    red = [(p["lambda"], p["red_flags"]) for p in points if p["red_flags"]]
    return {
        "milestone": manifest.milestone,
        "claim": template.estimator.claim,
        "provenance": manifest.provenance(repo_root).as_dict(),
        "grid": {"name": manifest.grid, "lambdas": lambdas},
        "template": manifest.template.name,
        "overrides": manifest.overrides,
        "case": template.case.as_dict(),
        "estimator": template.estimator.as_dict(),
        "variance_floor_bps2": variance_floor_bps2(market),
        "curves": _dense_curves(market, order_size, lambdas),
        "points": points,
        "missing_points": [p.name for p in missing],
        "full_budget_point": full_budget,
        "verdict": {
            "points": len(points),
            "complete": not missing,
            "epsilon_met_at": met,
            "epsilon_missed_at": missed,
            "per_seed_missed_at": [p["lambda"] for p in points if not p["per_seed_met"]],
            "red_flags": red,
            "red_flag_free": not red,
            "timed_out_at": [p["lambda"] for p in points if p["timed_out"]],
            "total_sweep_seconds": float(sum(p["sweep_seconds"] for p in points)),
            "per_seed_seconds": {
                "median": float(np.median([s["seconds"] for p in points for s in p["seeds"]]))
                if points else float("nan"),
                "max": float(max((s["seconds"] for p in points for s in p["seeds"]), default=float("nan"))),
            },
        },
    }
