"""The only module under ``temper/`` that may import matplotlib.

The core stays headless and import-light: the oracle is pure numpy, the env is
pure numpy, and the agent adds torch and nothing else. A plotting stack that
leaked into any of them would put a rendering backend on the import path of
every test, every training run and eventually the M6 Anvil client.
``tests/test_repo_invariants.py`` enforces the confinement statically — this file
is the allow-list, and it is one entry long — and the ``Agg`` backend is selected
here, before ``pyplot`` is imported, so nothing ever tries to open a window.

Every figure carries its provenance in the footer (invariant 1): the config
digest and the git revision that produced it. A chart without that line is not a
result, it is a picture.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # before pyplot: headless, deterministic, no display needed

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402

from temper.eval.provenance import Provenance  # noqa: E402
from temper.eval.reference import ReferenceRow, TrajectoryBand  # noqa: E402

#: Colours for the three references and the agent. Chosen for a grey-scale
#: print as much as for a screen: the agent is the only filled band, and the
#: three references differ in dash pattern as well as in hue.
STYLE: dict[str, dict] = {
    "twap": {"color": "#8c8c8c", "linestyle": (0, (6, 3)), "linewidth": 1.6},
    "ac": {"color": "#c1663c", "linestyle": (0, (2, 2)), "linewidth": 1.6},
    "optimal": {"color": "#1f4e79", "linestyle": "-", "linewidth": 2.0},
    "agent": {"color": "#227a4b", "linestyle": "-", "linewidth": 2.0},
}

LABELS = {
    "twap": "TWAP",
    "ac": "AC (vendored $\\kappa$)",
    "optimal": "optimal (exact discrete)",
    "agent": "PPO agent, median",
}


def trajectory_overlay(
    path: str | Path,
    *,
    hours: Sequence[float],
    agent_trajectories: Sequence[Sequence[float]],
    reference: ReferenceRow,
    order_size: float,
    band: TrajectoryBand,
    provenance: Provenance,
    caption: str,
    formats: Sequence[str] = ("png",),
) -> list[Path]:
    """The M2 hero figure: the agent's schedule against the three references.

    Two panels, because the interesting number is invisible in one. The top
    panel is inventory over the session, where a converged agent sits *on* the
    optimal curve and the reader learns only that it is not TWAP. The bottom
    panel is the residual against the optimum in shares, with the band that task
    0's Hessian implies drawn across it — which is where "within epsilon" becomes
    something the eye can check, and where the flatness of the objective near its
    minimum becomes visible rather than asserted.

    `agent_trajectories` is one row per training seed; the line is their median
    and the shaded band their inter-quartile range (constitution invariant 4 — no
    single-run numbers anywhere).
    """
    schedules = np.asarray(agent_trajectories, dtype=float)
    if schedules.ndim != 2 or schedules.shape[0] < 1:
        raise ValueError(
            f"agent_trajectories must be (seeds, n_bins + 1), got {schedules.shape}"
        )
    times = np.asarray(hours, dtype=float)
    optimum = np.asarray(reference.optimal.trajectory, dtype=float)

    median = np.median(schedules, axis=0)
    q1 = np.percentile(schedules, 25.0, axis=0)
    q3 = np.percentile(schedules, 75.0, axis=0)

    figure, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(8.0, 7.4),
        sharex=True,
        gridspec_kw={"height_ratios": (2.2, 1.0), "hspace": 0.12},
    )

    for name in ("twap", "ac", "optimal"):
        top.plot(
            times,
            reference.schedules[name].trajectory / order_size,
            label=LABELS[name],
            **STYLE[name],
        )
    top.fill_between(
        times,
        q1 / order_size,
        q3 / order_size,
        color=STYLE["agent"]["color"],
        alpha=0.22,
        linewidth=0.0,
        label=f"PPO agent, IQR ({schedules.shape[0]} seeds)",
    )
    # Every seed, thin, behind the band. The IQR is computed per time point, so a
    # single seed that failed outright sits *outside* the band at every point and
    # leaves it looking narrow — which is exactly the impression a rediscovery
    # figure must not give. Drawing the seeds costs one line and makes the spread
    # the reader's to judge rather than the summary statistic's to hide.
    for index, schedule in enumerate(schedules):
        top.plot(
            times,
            schedule / order_size,
            color=STYLE["agent"]["color"],
            alpha=0.45,
            linewidth=0.7,
            zorder=1.5,
            label="individual seeds" if index == 0 else None,
        )
    top.plot(times, median / order_size, label=LABELS["agent"], **STYLE["agent"])

    top.set_ylabel("inventory remaining / $X$")
    top.set_ylim(-0.02, 1.02)
    top.grid(True, alpha=0.25, linewidth=0.6)
    top.legend(frameon=False, fontsize=9, loc="upper right")
    top.set_title(caption, fontsize=9.5, loc="left", pad=8)

    # -- residual panel -----------------------------------------------------
    bottom.axhline(0.0, **STYLE["optimal"])
    bottom.fill_between(
        times,
        (q1 - optimum),
        (q3 - optimum),
        color=STYLE["agent"]["color"],
        alpha=0.22,
        linewidth=0.0,
    )
    for schedule in schedules:
        bottom.plot(
            times,
            schedule - optimum,
            color=STYLE["agent"]["color"],
            alpha=0.45,
            linewidth=0.7,
            zorder=1.5,
        )
    bottom.plot(times, median - optimum, **STYLE["agent"])
    bottom.plot(
        times,
        reference.twap.trajectory - optimum,
        label="TWAP $-$ optimal",
        **STYLE["twap"],
    )

    # The band is a bound on the L2 norm of the whole interior deviation vector,
    # not on any one bin, so it is drawn as a horizontal reference rather than as
    # an envelope the curve is required to sit inside. Labelled to say so.
    for sign in (+1.0, -1.0):
        bottom.axhline(
            sign * band.bound_shares,
            color="#b03a2e",
            linestyle=(0, (1, 2)),
            linewidth=1.2,
            label=(
                f"$\\|\\delta\\|_2 \\leq$ {band.bound_shares:,.0f} shares "
                f"({band.bound_fraction:.1%} of $X$), derived"
            )
            if sign > 0
            else None,
        )

    bottom.set_xlabel("hours into the session")
    bottom.set_ylabel("shares $-$ optimal")
    bottom.grid(True, alpha=0.25, linewidth=0.6)
    bottom.legend(frameon=False, fontsize=8.5, loc="upper right")

    figure.text(
        0.01,
        0.014,
        provenance.short,
        fontsize=7.5,
        color="#666666",
        family="monospace",
    )
    # Explicit margins rather than `tight_layout`: the two panels share an x-axis
    # through a `gridspec` with a fixed hspace, which tight_layout warns it
    # cannot honour — and the suite turns warnings into errors. Fixed margins are
    # also what keeps a regenerated figure byte-identical.
    figure.subplots_adjust(left=0.105, right=0.98, top=0.885, bottom=0.095)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix in formats:
        out = target.with_suffix(f".{suffix.lstrip('.')}")
        # No timestamp of any kind. matplotlib stamps a `Date` chunk into a PNG
        # by default, which makes every redraw a diff — and a figure that always
        # shows as modified is one nobody checks, so a *real* change to it would
        # go unnoticed. With Date/Software suppressed, redrawing an unchanged
        # result is byte-identical and `git status` stays meaningful.
        figure.savefig(
            out,
            dpi=160,
            metadata={"Software": None, "Creator": None, "Date": None},
        )
        written.append(out)
    plt.close(figure)
    return written


# ---------------------------------------------------------------------------
# M3 — the risk–cost frontier
# ---------------------------------------------------------------------------


def _fmt_lambda(value: float) -> str:
    exponent = np.log10(value)
    half = round(2.0 * exponent) / 2.0
    return f"$10^{{{half:g}}}$"


def frontier_figure(
    path: str | Path,
    *,
    aggregate: dict,
    provenance: Provenance,
    caption: str,
    formats: Sequence[str] = ("png",),
) -> list[Path]:
    """The M3 hero figure: agent, TWAP, AC and the exact optimum on one frontier.

    Two panels. The top is the frontier itself — expected cost against the
    **excess** of variance over the ``sigma_bin^2 X^2`` floor, on a log axis,
    because every schedule pays the floor and plotting total ``V`` compresses
    the separation exactly where the schedules differ most (the high-lambda
    end). The exact optimum and the vendored AC schedule are drawn as dense
    oracle curves; TWAP is one point, since it does not move with lambda; the
    agent is every seed at every lambda (small dots, and a thin trace per seed
    ordinal across the grid), with the per-lambda median joined and IQR bars in
    both coordinates. Grid points are labelled with their lambda.

    The bottom panel is the per-lambda tolerance table drawn: the gap fraction
    of every seed at every lambda on a log axis, the median with its IQR band,
    the pre-stated epsilon and per-seed floor, and — where the aggregate has one
    — the full-budget validation run at M2's lambda as a distinct marker, so the
    effect of the update-budget cut is visible on the same axes.

    `aggregate` is the document :func:`temper.eval.frontier.aggregate` builds;
    nothing is recomputed here, so a redraw is a pure function of the JSON.
    """
    points = sorted(aggregate["points"], key=lambda p: p["lambda"])
    if not points:
        raise ValueError("the aggregate has no frontier points to draw")
    curves = aggregate["curves"]
    full = aggregate.get("full_budget_point")

    figure, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(8.0, 9.2),
        gridspec_kw={"height_ratios": (2.0, 1.15), "hspace": 0.28},
    )

    # -- top: the frontier ---------------------------------------------------
    for name in ("ac", "optimal"):
        top.plot(
            [c["excess_variance_bps2"] for c in curves[name]],
            [c["expected_bps"] for c in curves[name]],
            label=LABELS[name],
            **STYLE[name],
        )
    twap = points[0]["baselines"]["twap"]
    top.plot(
        [twap["excess_variance_bps2"]],
        [twap["expected_bps"]],
        marker="s",
        markersize=7,
        color=STYLE["twap"]["color"],
        linestyle="none",
        label=r"TWAP (one point: it does not move with $\lambda$)",
    )

    seeds_x = np.array([[s["excess_variance_bps2"] for s in p["seeds"]] for p in points])
    seeds_y = np.array([[s["expected_bps"] for s in p["seeds"]] for p in points])
    agent = STYLE["agent"]["color"]
    for ordinal in range(seeds_x.shape[1]):
        top.plot(
            seeds_x[:, ordinal],
            seeds_y[:, ordinal],
            color=agent,
            alpha=0.35,
            linewidth=0.6,
            zorder=2,
            label=r"individual seeds, traced across $\lambda$" if ordinal == 0 else None,
        )
    top.plot(
        seeds_x.reshape(-1),
        seeds_y.reshape(-1),
        marker="o",
        markersize=2.6,
        color=agent,
        alpha=0.55,
        linestyle="none",
        zorder=2.5,
    )
    med_x = np.array([p["summary"]["excess_variance_bps2"]["median"] for p in points])
    med_y = np.array([p["summary"]["expected_bps"]["median"] for p in points])
    top.errorbar(
        med_x,
        med_y,
        xerr=[
            med_x - np.array([p["summary"]["excess_variance_bps2"]["q1"] for p in points]),
            np.array([p["summary"]["excess_variance_bps2"]["q3"] for p in points]) - med_x,
        ],
        yerr=[
            med_y - np.array([p["summary"]["expected_bps"]["q1"] for p in points]),
            np.array([p["summary"]["expected_bps"]["q3"] for p in points]) - med_y,
        ],
        color=agent,
        linewidth=STYLE["agent"]["linewidth"],
        marker="o",
        markersize=4.5,
        capsize=2.5,
        elinewidth=1.0,
        zorder=3,
        label=rf"PPO agent, median with IQR ({seeds_x.shape[1]} seeds per $\lambda$)",
    )
    for index, p in enumerate(points):
        opt = p["baselines"]["optimal"]
        top.annotate(
            _fmt_lambda(p["lambda"]),
            (opt["excess_variance_bps2"], opt["expected_bps"]),
            textcoords="offset points",
            # Alternate above and below the curve: at the low-lambda end the
            # optimum sits within a few percent of TWAP's variance and the
            # labels would otherwise pile onto one another.
            xytext=(7, -12) if index % 2 == 0 else (7, 5),
            fontsize=7.5,
            color="#444444",
        )
    for name in ("ac", "optimal"):
        top.plot(
            [p["baselines"][name]["excess_variance_bps2"] for p in points],
            [p["baselines"][name]["expected_bps"] for p in points],
            marker="o",
            markersize=3.5,
            color=STYLE[name]["color"],
            linestyle="none",
            zorder=2.8,
        )
    top.set_xscale("log")
    # The x-range is the *optimum's* over the grid, with margin. The vendored AC
    # schedule collapses onto the variance floor at high lambda (excess ~1e-10
    # bps^2 by lambda = 0.1) and would otherwise stretch the log axis by ten
    # decades of nothing; where it leaves the canvas it is sitting on the floor,
    # which the caption says.
    grid_optimal = [p["baselines"]["optimal"]["excess_variance_bps2"] for p in points]
    top.set_xlim(0.25 * min(grid_optimal), 2.5 * twap["excess_variance_bps2"])
    top.set_xlabel(r"$V - \sigma_{bin}^2 X^2$ — variance in excess of the floor (bps$^2$)")
    top.set_ylabel(r"$E$[cost] (bps of notional)")
    top.grid(True, which="both", alpha=0.25, linewidth=0.6)
    top.legend(frameon=False, fontsize=8.5, loc="upper right")
    top.set_title(caption, fontsize=9.0, loc="left", pad=8)

    # -- bottom: the per-lambda tolerance table ------------------------------
    lambdas = np.array([p["lambda"] for p in points])
    gaps = np.array([[s["gap_fraction"] for s in p["seeds"]] for p in points])
    positive = np.clip(gaps, 1e-7, None)
    for ordinal in range(gaps.shape[1]):
        bottom.plot(
            lambdas,
            positive[:, ordinal],
            marker="o",
            markersize=2.6,
            color=agent,
            alpha=0.45,
            linewidth=0.5,
            zorder=2,
        )
    q1 = np.array([p["summary"]["gap_fraction"]["q1"] for p in points])
    q3 = np.array([p["summary"]["gap_fraction"]["q3"] for p in points])
    med = np.array([p["summary"]["gap_fraction"]["median"] for p in points])
    bottom.fill_between(
        lambdas,
        np.clip(q1, 1e-7, None),
        np.clip(q3, 1e-7, None),
        color=agent,
        alpha=0.22,
        linewidth=0.0,
        label="IQR across seeds",
    )
    bottom.plot(
        lambdas,
        np.clip(med, 1e-7, None),
        marker="o",
        markersize=4.5,
        label="median gap fraction",
        **STYLE["agent"],
    )
    tolerances = points[0]["tolerances"]
    bottom.axhline(
        tolerances["epsilon_gap_fraction"],
        color="#b03a2e",
        linestyle=(0, (4, 2)),
        linewidth=1.2,
        label=rf"$\varepsilon$ = {tolerances['epsilon_gap_fraction']:g} of the TWAP gap (median)",
    )
    bottom.axhline(
        tolerances["per_seed_gap_fraction"],
        color="#b03a2e",
        linestyle=(0, (1, 2)),
        linewidth=1.2,
        label=f"per-seed floor {tolerances['per_seed_gap_fraction']:g}",
    )
    if full is not None:
        bottom.plot(
            [full["lambda"]] * len(full["seeds"]),
            np.clip([s["gap_fraction"] for s in full["seeds"]], 1e-7, None),
            marker="D",
            markersize=3.2,
            markerfacecolor="none",
            markeredgecolor=STYLE["optimal"]["color"],
            linestyle="none",
            zorder=3,
            label=rf"validation run at M2's $\lambda$, full budget ({full['updates']} updates)",
        )
    bottom.set_xscale("log")
    bottom.set_yscale("log")
    bottom.set_xlabel(r"$\lambda$ (risk aversion)")
    bottom.set_ylabel(r"$(J - J^*)\,/\,(J_{TWAP} - J^*)$")
    bottom.grid(True, which="both", alpha=0.25, linewidth=0.6)
    bottom.legend(frameon=False, fontsize=8.0, loc="lower left", ncol=1)

    figure.text(
        0.01, 0.012, provenance.short, fontsize=7.5, color="#666666", family="monospace"
    )
    figure.subplots_adjust(left=0.105, right=0.98, top=0.905, bottom=0.075)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix in formats:
        out = target.with_suffix(f".{suffix.lstrip('.')}")
        figure.savefig(
            out, dpi=160, metadata={"Software": None, "Creator": None, "Date": None}
        )
        written.append(out)
    plt.close(figure)
    return written
