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
import matplotlib.ticker as ticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
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
    # M4a. In the power-law world `optimal` is the *certified* optimum of that
    # world and this is the sinh the closed form actually produces — a schedule
    # like any other there, and the one the agent's advantage is measured over.
    # Given its own hue rather than sharing the optimum's, because the whole
    # point of the figure is that they are not the same curve.
    "tangent": {"color": "#7b5ea7", "linestyle": (0, (5, 2, 1, 2)), "linewidth": 1.8},
    "agent": {"color": "#227a4b", "linestyle": "-", "linewidth": 2.0},
}

#: Where the frontier's x axis switches from linear to logarithmic. Below this
#: the axis is linear so that an excess variance of exactly zero — the agent's
#: schedule at high lambda, which liquidates in bin 0 and lands on the floor —
#: has a place on the chart instead of being dropped by a log transform.
SYMLOG_THRESHOLD = 0.5

LABELS = {
    "twap": "TWAP",
    "ac": "AC (vendored $\\kappa$)",
    "optimal": "optimal (exact discrete)",
    "tangent": "AC at the tangent — the closed form's answer",
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

    # `tangent` only exists in a world where the closed form is not the optimum
    # (M4a). Drawn when it is there, because the distance from it to `optimal` is
    # the milestone; absent from the Phase-1 figures, which stay byte-identical.
    drawn = ("twap", "ac", "optimal")
    if "tangent" in reference.schedules:
        drawn = ("twap", "ac", "tangent", "optimal")
    for name in drawn:
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
    if reference.tangent is not None:
        # The residual panel is where M4a's claim is legible: the agent sits on
        # zero and the schedule the vendored library would have run does not.
        bottom.plot(
            times,
            reference.tangent.trajectory - optimum,
            label="AC at the tangent $-$ optimal",
            **STYLE["tangent"],
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
        # Appended, never `with_suffix`: a frontier point is named for its
        # lambda (`lambda_1e-3.5`), and `with_suffix` reads the `.5` as a suffix
        # to replace — which silently collided the half-decade points with the
        # integer-decade ones. Every path M2 committed has no dot in its final
        # component, so the two spellings agree there byte for byte.
        out = target.with_name(f"{target.name}.{suffix.lstrip('.')}")
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
    # symlog, not log, and the reason is a result rather than a preference. At
    # high lambda the agent liquidates the whole order in bin 0, so its schedule
    # sits *exactly* on the variance floor and `V - floor` is 0 — for eight of
    # ten seeds at 10^-1.5 and for all ten at 1e-1. A log axis cannot place zero:
    # those seeds would vanish, and the lines reaching them would run off the
    # canvas edge looking like data. symlog is linear inside `SYMLOG_THRESHOLD`
    # and logarithmic outside it, so "on the floor" is a position on the axis and
    # the decades above it still read as decades. The exact optimum never reaches
    # the floor (it keeps a small tail: 0.68 bps^2 of excess at lambda = 1e-1),
    # so the separation at the top of the frontier is exactly what the reader
    # should be able to see.
    top.set_xscale("symlog", linthresh=SYMLOG_THRESHOLD, linscale=0.6)
    grid_optimal = [p["baselines"]["optimal"]["excess_variance_bps2"] for p in points]
    top.set_xlim(-0.25 * SYMLOG_THRESHOLD, 2.5 * twap["excess_variance_bps2"])
    # Explicit ticks: symlog's default locator emits both a 0 and a decade tick
    # inside the linear region, and at this width their labels overlap into an
    # unreadable smudge. Zero is the tick that has to be legible — it is where
    # the agent's high-lambda schedules sit.
    decades = 1
    while decades <= twap["excess_variance_bps2"]:
        decades *= 10
    top.set_xticks([0.0] + [10.0**k for k in range(0, len(str(int(decades))))])
    top.xaxis.set_minor_locator(ticker.NullLocator())
    del grid_optimal
    top.set_xlabel(
        r"$V - \sigma_{bin}^2 X^2$ — variance in excess of the floor (bps$^2$); "
        rf"linear below {SYMLOG_THRESHOLD:g}, so $0$ is on the axis"
    )
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
    # M3's committed points spell these `epsilon_gap_fraction`; M4a renamed the
    # field because its denominator is no longer the TWAP gap. Read either, so
    # this figure still redraws byte-identically from the artefact it was
    # aggregated from (invariant 1).
    epsilon = tolerances.get("epsilon_fraction", tolerances.get("epsilon_gap_fraction"))
    per_seed = tolerances.get(
        "per_seed_fraction", tolerances.get("per_seed_gap_fraction")
    )
    bottom.axhline(
        epsilon,
        color="#b03a2e",
        linestyle=(0, (4, 2)),
        linewidth=1.2,
        label=rf"$\varepsilon$ = {epsilon:g} of the TWAP gap (median)",
    )
    bottom.axhline(
        per_seed,
        color="#b03a2e",
        linestyle=(0, (1, 2)),
        linewidth=1.2,
        label=f"per-seed floor {per_seed:g}",
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
    figure.subplots_adjust(left=0.105, right=0.98, top=0.875, bottom=0.075)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix in formats:
        # Appended, never `with_suffix`: a frontier point is named for its
        # lambda (`lambda_1e-3.5`), and `with_suffix` reads the `.5` as a suffix
        # to replace — which silently collided the half-decade points with the
        # integer-decade ones. Every path M2 committed has no dot in its final
        # component, so the two spellings agree there byte for byte.
        out = target.with_name(f"{target.name}.{suffix.lstrip('.')}")
        figure.savefig(
            out, dpi=160, metadata={"Software": None, "Creator": None, "Date": None}
        )
        written.append(out)
    plt.close(figure)
    return written


# ---------------------------------------------------------------------------
# M4a — how far the closed form is from the truth, and how much learning recovered
# ---------------------------------------------------------------------------


def degradation_figure(
    path: str | Path,
    *,
    curves: dict,
    provenance: Provenance,
    caption: str,
    formats: Sequence[str] = ("png",),
) -> list[Path]:
    """The M4a hero figure: excess over the certified power-law optimum, against lambda.

    Two things in one picture, which is the whole reason the milestone draws it.
    The curves are oracle-only and therefore free: TWAP, the vendored
    Almgren-Chriss schedule and the tangent-derived sinh, each priced under
    FrontierView's power law and expressed as excess over that world's own
    optimum. They say *how far the closed form is from the truth* across four
    decades of risk aversion. The markers are the agent's ten seeds at the one
    lambda that was trained, drawn individually rather than summarised
    (*below n ~ 10, draw every trace*), and they say *how much of that distance
    learning recovered*.

    The y axis is relative excess, ``(J - J_pow*) / J_pow*``, on a log scale:
    TWAP runs from 1e-6 to nearly 4 across the grid, and a linear axis would show
    one point and a floor. The tangent's curve is the milestone's denominator —
    the available advantage — so the vertical distance from it down to a seed
    marker is the capture, read directly off the chart.

    `curves` is the oracle table plus the seeds, assembled by
    ``tools/m4a_degradation.py``: nothing here computes a cost, so the figure is
    a *view* of a committed result and redraws byte-identically from it.
    """
    lambdas = np.asarray(curves["lambdas"], dtype=float)
    figure, axes = plt.subplots(figsize=(8.6, 5.4))

    order = ("twap", "ac", "tangent")
    labels = {
        "twap": "TWAP",
        "ac": "AC (vendored $\\kappa$)",
        "tangent": "AC at the tangent (exact discrete $\\kappa$)",
    }
    style = {
        "twap": STYLE["twap"],
        "ac": STYLE["ac"],
        # The tangent-derived sinh is what the closed form actually produces and
        # what M4a's advantage is measured against, so it gets the optimum's own
        # colour: in Phase 1 it *was* the optimum, and the point of the figure is
        # how far that is from being true here.
        "tangent": STYLE["optimal"],
    }
    floor = 1e-7
    for name in order:
        excess = np.asarray(curves["excess"][name], dtype=float)
        axes.plot(
            lambdas,
            np.clip(excess, floor, None),
            marker="o",
            markersize=3.6,
            label=labels[name],
            **style[name],
        )

    trained = float(curves["trained_lambda"])
    seeds = np.asarray(curves["seed_excess"], dtype=float)
    axes.plot(
        [trained] * len(seeds),
        np.clip(seeds, floor, None),
        marker="D",
        markersize=5.0,
        markerfacecolor=STYLE["agent"]["color"],
        markeredgecolor="#123f27",
        markeredgewidth=0.6,
        linestyle="none",
        zorder=4,
        label=f"PPO agent, {len(seeds)} seeds (each drawn)",
    )

    # The available advantage at the trained lambda, as a bracket the reader can
    # measure the capture against.
    tangent_at_trained = float(curves["tangent_at_trained"])
    axes.annotate(
        "",
        xy=(trained, max(float(np.median(seeds)), floor)),
        xytext=(trained, tangent_at_trained),
        arrowprops={
            "arrowstyle": "<->",
            "color": "#227a4b",
            "linewidth": 1.1,
            "shrinkA": 2.0,
            "shrinkB": 2.0,
        },
    )
    axes.annotate(
        f"captured {curves['median_capture']:.1%} of\n"
        f"{curves['available_advantage_bps']:.4f} bps available",
        xy=(trained, tangent_at_trained),
        xytext=(1.9 * trained, tangent_at_trained * 2.2),
        fontsize=8.5,
        color="#1c5e3a",
    )

    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel(r"$\lambda$ (risk aversion)")
    axes.set_ylabel(r"excess over the certified power-law optimum, $(J - J^*_{pow})\,/\,J^*_{pow}$")
    axes.grid(True, which="both", alpha=0.25, linewidth=0.6)
    axes.legend(frameon=False, fontsize=8.5, loc="upper left")

    # Headline and caption as one text block rather than a title plus a caption.
    # A three-line wrapped caption and an axes title occupy the same strip of
    # canvas, and matplotlib will happily draw them on top of each other without
    # a word of complaint — which is how a figure ends up illegible in the one
    # place nobody re-reads it, the committed artefact.
    figure.text(
        0.012,
        0.985,
        "M4a — the Almgren-Chriss schedule in the world it was linearised from",
        fontsize=11.5,
        color="#111111",
        va="top",
    )
    figure.text(
        0.012,
        0.948,
        caption,
        fontsize=7.4,
        color="#333333",
        va="top",
        wrap=True,
    )
    figure.text(
        0.01,
        0.014,
        provenance.short,
        fontsize=7.5,
        color="#666666",
        family="monospace",
    )
    figure.subplots_adjust(left=0.105, right=0.985, top=0.800, bottom=0.105)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix in formats:
        out = target.with_name(f"{target.name}.{suffix.lstrip('.')}")
        figure.savefig(
            out,
            dpi=160,
            metadata={"Software": None, "Creator": None, "Date": None},
        )
        written.append(out)
    plt.close(figure)
    return written


def adaptivity_figure(
    path: str | Path,
    *,
    rungs: dict,
    curve: dict,
    provenance: Provenance,
    caption: str,
    formats: Sequence[str] = ("png",),
) -> list[Path]:
    """M4b's hero: what seeing liquidity is worth, and how much of it was captured.

    Two panels, because M4b makes two claims and only one of them is about the
    agent.

    **Left — the ladder, at the trained sigma_L.** Four levels in bps of the
    objective: M4a's schedule, which knows no liquidity at all; ``J_static*``, the
    best *fixed* schedule that knows the liquidity **law**; the ten trained seeds,
    drawn individually (*below n ~ 10, draw every trace*); and ``J_DP``, the
    optimum over adapted policies, with the clairvoyant relaxation under it as a
    floor. The gap the milestone is *about* is `static -> DP`, and it is the only
    one the agent is measured over. The gap `M4a -> static` is drawn as its own
    hatched band and labelled a **level shift**, because it is a constant any
    static solver picks up for free by re-solving at an inflated coefficient, and
    an agent measured against M4a's schedule would appear to gain it.

    Drawing the level shift rather than mentioning it is the point. A chart whose
    top rung was M4a's schedule would make the agent look 3.8 % better than it is,
    which is precisely the direction §9's denominator entry warns about.

    **Right — the value of sight against the invented parameter.** ``sigma_L`` is
    Temper's own; FrontierView has no liquidity process. So the oracle's adaptive
    advantage is drawn as a *curve* across three values with the trained point
    marked, because a single invented parameter with a single number beside it
    reads as calibration and it is not.

    `rungs` and `curve` are read off committed artefacts by
    ``tools/m4b_adaptivity.py``: nothing here computes a cost, so the figure is a
    view of a result and redraws byte-identically from it.
    """
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(11.6, 6.8), gridspec_kw={"width_ratios": (1.35, 1.0)}
    )

    # ---- left panel: the ladder ------------------------------------------
    m4a = float(rungs["m4a"])
    static = float(rungs["static"])
    adaptive = float(rungs["adaptive"])
    clairvoyant = float(rungs["clairvoyant"])
    clairvoyant_half = float(rungs["clairvoyant_half_width"])
    seeds = np.asarray(rungs["seeds"], dtype=float)
    span = (0.0, 1.0)

    left.axhspan(
        static,
        m4a,
        xmin=0.02,
        xmax=0.98,
        facecolor="#c1663c",
        alpha=0.10,
        hatch="///",
        edgecolor="#c1663c",
        linewidth=0.0,
        zorder=0,
    )
    left.axhspan(
        adaptive,
        static,
        xmin=0.02,
        xmax=0.98,
        facecolor="#227a4b",
        alpha=0.09,
        zorder=0,
    )

    for value, key, label in (
        (m4a, "tangent", "M4a's schedule — knows no liquidity"),
        (static, "ac", "$J_{static*}$ — best fixed schedule, knows the law"),
        (adaptive, "optimal", "$J_{DP}$ — adaptive optimum (converged, bracketed)"),
    ):
        left.plot(span, (value, value), label=label, **STYLE[key])

    left.fill_between(
        span,
        clairvoyant - clairvoyant_half,
        clairvoyant + clairvoyant_half,
        color="#1f4e79",
        alpha=0.14,
        linewidth=0.0,
        zorder=0,
    )
    left.plot(
        span,
        (clairvoyant, clairvoyant),
        color="#1f4e79",
        linestyle=(0, (1, 2)),
        linewidth=1.3,
        label="clairvoyant bound — no policy can go below",
    )

    # Every seed, individually. The house note is about exactly this n.
    jitter = np.linspace(0.16, 0.84, seeds.size)
    left.plot(
        jitter,
        seeds,
        marker="o",
        markersize=6.0,
        linestyle="none",
        markerfacecolor=STYLE["agent"]["color"],
        markeredgecolor="white",
        markeredgewidth=0.7,
        label=f"PPO, {seeds.size} seeds (each drawn)",
        zorder=5,
    )
    median = float(np.median(seeds))
    left.plot(
        span,
        (median, median),
        color=STYLE["agent"]["color"],
        linewidth=1.2,
        linestyle=(0, (4, 2)),
        alpha=0.85,
        zorder=4,
    )

    advantage = static - adaptive
    left.annotate(
        "",
        xy=(0.955, adaptive),
        xytext=(0.955, static),
        arrowprops={"arrowstyle": "<->", "color": "#227a4b", "linewidth": 1.2},
    )
    left.text(
        0.945,
        0.5 * (static + adaptive),
        f"adaptive advantage\n{advantage:.5f} bps",
        ha="right",
        va="center",
        fontsize=8.5,
        color="#227a4b",
    )
    # The level shift is ~4 % of the panel's height, so the two rungs it separates
    # are all but on top of each other and there is nowhere *between* them to put
    # a label. A leader line from clear air is the honest way to point at a gap
    # too small to annotate in place — and the gap being too small to see is
    # itself the reading: it is a constant, and a small one.
    left.annotate(
        f"level shift {m4a - static:.5f} bps —\na re-solve, NOT the agent's",
        xy=(0.46, 0.5 * (m4a + static)),
        xytext=(0.40, static - 0.10 * advantage),
        fontsize=8.0,
        color="#a0522d",
        ha="left",
        va="top",
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#c1663c",
            "linewidth": 1.0,
            "shrinkA": 2,
            "shrinkB": 1,
        },
    )

    left.set_xlim(0.0, 1.0)
    left.set_xticks([])
    left.set_ylabel("objective $E + \\lambda V$, bps")
    left.set_title(
        f"The ladder at $\\sigma_L$ = {rungs['sigma_log']:g}", fontsize=11, pad=8
    )
    left.grid(axis="y", color="#e6e6e6", linewidth=0.7)
    left.set_axisbelow(True)
    # In the empty middle of the advantage band, which is the one region of this
    # panel guaranteed to be clear: everything the figure draws is either at the
    # top (the two static rungs) or at the bottom (the seeds, the DP and its
    # bound), and the space between them is the quantity being measured.
    left.legend(
        fontsize=7.8,
        loc="center left",
        bbox_to_anchor=(0.02, 0.47),
        framealpha=0.94,
        borderpad=0.5,
    )

    # ---- right panel: the value of sight ---------------------------------
    sigmas = np.asarray(curve["sigma_log"], dtype=float)
    advantage = np.asarray(curve["advantage_bps"], dtype=float)
    shift = np.asarray(curve["level_shift_bps"], dtype=float)
    trained = float(rungs["sigma_log"])

    right.plot(
        sigmas,
        advantage,
        marker="o",
        color="#227a4b",
        linewidth=2.0,
        markersize=5.5,
        label="adaptive advantage $J_{static*} - J_{DP}$",
    )
    right.plot(
        sigmas,
        shift,
        marker="s",
        color="#c1663c",
        linewidth=1.5,
        linestyle=(0, (5, 2)),
        markersize=4.5,
        label="level shift $J_{M4a} - J_{static*}$",
    )
    right.axvline(trained, color="#8c8c8c", linewidth=1.0, linestyle=(0, (1, 3)))
    index = int(np.argmin(np.abs(sigmas - trained)))
    right.plot(
        [trained],
        [advantage[index]],
        marker="o",
        markersize=10.0,
        markerfacecolor="none",
        markeredgecolor="#227a4b",
        markeredgewidth=1.6,
    )
    right.annotate(
        "trained here",
        xy=(trained, advantage[index]),
        xytext=(8, -18),
        textcoords="offset points",
        fontsize=8.5,
        color="#227a4b",
    )
    right.set_xlabel("$\\sigma_L$ — Temper's own invented parameter")
    right.set_ylabel("bps of the objective")
    right.set_yscale("log")
    right.set_title("What sight is worth, against $\\sigma_L$", fontsize=11, pad=8)
    right.grid(color="#e6e6e6", linewidth=0.7)
    right.set_axisbelow(True)
    right.legend(fontsize=8.0, loc="upper left", framealpha=0.94)

    figure.suptitle(
        "M4b — stochastic liquidity: the advantage no fixed schedule can capture",
        fontsize=12.5,
        y=0.985,
    )
    # The caption is hard-wrapped by the caller and drawn in its own reserved
    # band at the bottom. Both halves of that are the house note's *A figure
    # caption that runs off the canvas* arriving as code: matplotlib will not tell
    # you that text overflowed, so the width is bounded where the string is built
    # and the space is reserved here rather than hoped for.
    figure.text(0.008, 0.010, caption, fontsize=7.6, color="#333333", va="bottom")
    figure.text(
        0.992,
        0.955,
        provenance.short,
        fontsize=7.5,
        color="#666666",
        family="monospace",
        ha="right",
        va="top",
    )
    figure.subplots_adjust(left=0.070, right=0.985, top=0.895, bottom=0.235, wspace=0.20)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix in formats:
        out = target.with_name(f"{target.name}.{suffix.lstrip('.')}")
        figure.savefig(
            out,
            dpi=160,
            metadata={"Software": None, "Creator": None, "Date": None},
        )
        written.append(out)
    plt.close(figure)
    return written


# ---------------------------------------------------------------------------
# M6 — the live leg: a closed form against a venue, in three tiers of claim
# ---------------------------------------------------------------------------

#: One colour per M6 run, taken from the module palette rather than a new one.
#: The three client-built ladders take the three *reference* hues because that is
#: what they are here — schedules whose every fill was computable before the run
#: started. The feeder run takes the agent's green: same policy, same client, a
#: book it did not build. The deployment run takes TWAP's grey, which is the
#: honest colour for the one row on this figure that is not a measurement.
M6_COLOURS = {
    "ladder": STYLE["optimal"]["color"],
    "thin": STYLE["ac"]["color"],
    "wide": STYLE["tangent"]["color"],
    "feeder": STYLE["agent"]["color"],
    "deployment": STYLE["twap"]["color"],
}

#: How a withheld measurement is drawn. Outlined and struck through rather than
#: filled, and never in a run's own colour: the whole point of the third tier is
#: that its number must not read as a data point beside the four that are.
VOID_EDGE = "#6f6f6f"

#: The half-width of the residual strip's y axis when every residual is
#: exactly zero. Float64's last bit on a value near 39 bps is about 9e-15,
#: so this window is roughly eleven of them: wide enough that a single-ulp
#: disagreement would be visible rather than clipped, narrow enough that
#: the number on the axis says what precision is being claimed.
ZERO_RESIDUAL_WINDOW_BPS = 1e-13

#: Vertical gap between two rows of the same tier, and between two tiers. The
#: tiers are the argument, so they are separated by more than the runs inside
#: them — the grouping has to survive being glanced at.
TIER_ROW_STEP = 1.0
TIER_GAP_STEP = 1.75


def prediction_figure(
    path: str | Path,
    *,
    ladders: dict,
    tiers: dict,
    provenance: Provenance,
    caption: str,
    formats: Sequence[str] = ("png",),
) -> list[Path]:
    """M6's figure: what the closed form said, what the venue did, and what that is worth.

    Two panels, because the milestone's claim is not one number but a *ladder of
    claims*, and the strength of each rung is the finding.

    **Left — per bin, for the three runs that can be predicted.** Cumulative
    arrival slippage after each of the 13 bins, predicted as a line and realised
    as an open marker on top of it. A committed ladder plus a deterministic
    policy plus deterministic matching makes every fill computable in closed form
    *before* the run, and the markers sitting on the lines at every bin is what
    that sentence looks like when it is true. Per bin rather than per run on
    purpose: three run-level numbers agreeing could be three coincidences, and
    the run-level number is also the one place a bin that went wrong can hide.

    The thin ladder's first bin is ringed and annotated, because it is the one
    point on the panel where the closed loop is exercised rather than merely
    present — the policy asked for more than the whole book held, was filled
    short, had its remainder cancelled, and carried the shortfall into a state
    ``ExecutionEnv`` has never produced. It is also the panel's largest value; a
    figure that let it pass as an ordinary point would be hiding the most
    interesting thing in the milestone.

    **Under it — the residual strip, which is what turns the claim into
    evidence.** On a 10-to-39 bps axis a marker sitting on a line looks the same
    whether the two agree to the last bit or to one percent: the panel above is
    scaled to the size of the *data*, and the claim is about a quantity three
    orders of magnitude smaller. So the strip differences the very arrays the
    panel plots — not a second route to them, or it would stop being a check on
    the panel — and shows all 39 per-bin comparisons on an axis scaled to the
    precision actually achieved. Where every residual is exactly zero the axis
    has no scale to take from the data, so a fixed narrow window is used and the
    strip says so: *identical to the last bit* is a stronger statement than any
    nonzero bound, and it should be made rather than hidden inside an autoscaled
    empty axis.

    **Right — the three tiers on one bps axis.** Tier 1 predicts and measures.
    Tier 2 measures and cannot predict, and the artefact says why rather than
    apologising for it. Tier 3 *withholds*: 236 third-party fills on a public
    floor make the run a successful demonstration and a void measurement, so its
    number is drawn outlined and struck through, labelled with its void reason,
    and is not a data point beside the others.

    `ladders` and `tiers` are read off the five committed run artefacts by
    ``tools/m6_prediction.py``: nothing here recomputes a fill, so the figure is a
    *view* of results and redraws byte-identically from them.
    """
    figure = plt.figure(figsize=(11.6, 7.6))
    # The left column is two rows sharing an x-axis; the right panel spans both,
    # so the tier layout is untouched by the strip's arrival.
    grid = figure.add_gridspec(
        2, 2, width_ratios=(1.52, 1.0), height_ratios=(5.0, 1.0)
    )
    left = figure.add_subplot(grid[0, 0])
    strip = figure.add_subplot(grid[1, 0], sharex=left)
    right = figure.add_subplot(grid[:, 1])

    # ---- left panel: per bin, predicted against realised -------------------
    bins = np.asarray(ladders["bins"], dtype=float)
    # Kept as they are drawn, so the strip below differences the arrays this
    # panel actually plotted rather than recomputing them from the source.
    drawn_series: list[tuple[str, np.ndarray]] = []
    for row in ladders["runs"]:
        colour = M6_COLOURS[row["run"]]
        predicted = np.asarray(row["predicted_bps"], dtype=float)
        realised = np.asarray(row["realised_bps"], dtype=float)
        left.plot(
            bins,
            predicted,
            color=colour,
            linewidth=1.8,
            zorder=2,
            label=row["label"],
        )
        left.plot(
            bins,
            realised,
            linestyle="none",
            marker="o",
            markersize=7.0,
            markerfacecolor="none",
            markeredgecolor=colour,
            markeredgewidth=1.4,
            zorder=3,
        )
        drawn_series.append((colour, realised - predicted))

    # The bin where the loop closed. Ringed first so the annotation's leader has
    # something to point at that is visibly not one of the ordinary markers.
    highlight = ladders["highlight"]
    marked = next(row for row in ladders["runs"] if row["run"] == highlight["run"])
    x_mark = float(highlight["bin"])
    y_mark = float(marked["realised_bps"][int(highlight["bin"]) - 1])
    left.plot(
        [x_mark],
        [y_mark],
        marker="o",
        markersize=16.0,
        markerfacecolor="none",
        markeredgecolor=M6_COLOURS[highlight["run"]],
        markeredgewidth=1.7,
        zorder=5,
    )
    left.annotate(
        highlight["text"],
        xy=(x_mark, y_mark),
        xytext=(0.022, 0.992),
        textcoords="axes fraction",
        fontsize=8.0,
        color="#8a4a28",
        ha="left",
        va="top",
        arrowprops={
            "arrowstyle": "-|>",
            "color": M6_COLOURS[highlight["run"]],
            "linewidth": 1.1,
            "shrinkA": 2.0,
            "shrinkB": 9.0,
        },
    )

    shapes = left.legend(
        loc="lower right",
        bbox_to_anchor=(0.995, 0.105),
        fontsize=8.5,
        frameon=False,
        title="ladder shape — the client built all three",
        title_fontsize=8.5,
    )
    left.add_artist(shapes)
    # A second legend for what the two *marks* mean, in neutral grey. Folding it
    # into the first would give six entries for three runs and bury the reading
    # the panel exists to support: line and marker coincide everywhere.
    left.legend(
        handles=[
            Line2D(
                [],
                [],
                color="#555555",
                linewidth=1.8,
                label="predicted — closed form, before the run",
            ),
            Line2D(
                [],
                [],
                color="#555555",
                linestyle="none",
                marker="o",
                markersize=7.0,
                markerfacecolor="none",
                markeredgewidth=1.4,
                label="realised — the live venue's fills",
            ),
        ],
        loc="upper right",
        bbox_to_anchor=(0.995, 0.855),
        fontsize=8.5,
        frameon=False,
    )

    left.set_ylabel(
        "cumulative arrival slippage, bps\n(positive = worse than arrival)"
    )
    left.set_title(
        "Per bin: what the closed form said, and what the venue did",
        fontsize=11,
        pad=8,
    )
    left.tick_params(labelbottom=False)
    left.set_xlim(0.4, 13.6)
    # Headroom above the tallest point, reserved rather than left to the
    # autoscaler: the annotation is placed in axes fractions and would
    # otherwise sit on top of the wide ladder's first bin. Trimmed from 0.42
    # when the strip arrived — the top of the panel was the only empty space
    # the strip could come out of without a taller canvas.
    drawn = np.concatenate(
        [np.asarray(row["realised_bps"], dtype=float) for row in ladders["runs"]]
    )
    low, high = float(np.nanmin(drawn)), float(np.nanmax(drawn))
    left.set_ylim(low - 0.08 * (high - low), high + 0.30 * (high - low))
    left.grid(color="#e6e6e6", linewidth=0.7)
    left.set_axisbelow(True)

    # ---- the residual strip ------------------------------------------------
    residuals = np.concatenate([series for _, series in drawn_series])
    comparisons = int(residuals.size)
    worst = float(np.nanmax(np.abs(residuals))) if comparisons else 0.0
    if worst == 0.0:
        # No scale to take from the data. A fixed window, stated on the axis and
        # in the strip, rather than an autoscaled axis that would silently invent
        # one and make bit-identity look like a measurement with error bars.
        window = ZERO_RESIDUAL_WINDOW_BPS
        verdict = (
            f"all {comparisons} comparisons EXACTLY 0.0 — identical to the last bit "
            f"of a float64; window fixed, not fitted"
        )
        verdict_colour = "#1c5e3a"
    else:
        window = 1.6 * worst
        verdict = (
            f"worst |realised − predicted| over {comparisons} comparisons: "
            f"{worst:.2e} bps"
        )
        verdict_colour = "#8a4a28"

    strip.axhline(0.0, color="#9a9a9a", linewidth=1.0, zorder=1)
    for colour, series in drawn_series:
        strip.plot(
            bins,
            series,
            linestyle="none",
            marker="o",
            markersize=5.5,
            markerfacecolor="none",
            markeredgecolor=colour,
            markeredgewidth=1.2,
            zorder=3,
        )
    strip.text(
        0.5,
        0.80,
        verdict,
        transform=strip.transAxes,
        fontsize=7.5,
        color=verdict_colour,
        ha="center",
        va="top",
        zorder=4,
    )

    strip.set_ylim(-window, window)
    strip.set_yticks([-window, 0.0, window])
    # Spelled out rather than left to an offset-notation exponent parked in the
    # corner. The magnitude *is* the point of this strip, so it is on every tick
    # a reader's eye lands on.
    strip.set_yticklabels(
        [f"-{window:.0e}", "0", f"+{window:.0e}"], fontsize=7.5
    )
    strip.set_ylabel("residual, bps", fontsize=8.0)
    strip.set_xlabel("bin (13, the count the policy was trained on)")
    strip.set_xticks(bins)
    strip.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    strip.grid(color="#e6e6e6", linewidth=0.7)
    strip.set_axisbelow(True)

    # ---- right panel: the three tiers on one bps axis ----------------------
    rows = tiers["rows"]
    positions: list[float] = []
    cursor = 0.0
    for index, row in enumerate(rows):
        if index:
            same = row["tier"] == rows[index - 1]["tier"]
            cursor += TIER_ROW_STEP if same else TIER_GAP_STEP
        positions.append(cursor)

    values = [float(row["bps"]) for row in rows]
    span = max(values) - min(values)
    x_low = min(values) - 0.42 * span
    x_high = max(values) + 0.14 * span

    for tier in sorted({row["tier"] for row in rows}):
        members = [y for y, row in zip(positions, rows) if row["tier"] == tier]
        void = all(row["void"] for row in rows if row["tier"] == tier)
        right.axhspan(
            min(members) - 0.5,
            max(members) + 0.5,
            facecolor="#f2f2f2" if not void else "#f7f7f7",
            edgecolor="#d2d2d2" if void else "none",
            hatch="///" if void else None,
            linewidth=0.0,
            alpha=0.75,
            zorder=0,
        )
        right.text(
            x_low + 0.02 * (x_high - x_low),
            min(members) - 0.40,
            tiers["captions"][tier],
            fontsize=8.0,
            color="#444444",
            ha="left",
            va="top",
            zorder=1,
        )

    for y, row in zip(positions, rows):
        colour = M6_COLOURS[row["run"]]
        rightwards = (row["bps"] - x_low) / (x_high - x_low) < 0.5
        offset = 12 if rightwards else -12
        align = "left" if rightwards else "right"
        if row["void"]:
            # Outlined and struck through, in grey. A withheld number drawn like
            # a measured one is the single way this figure could mislead.
            right.plot(
                [row["bps"]],
                [y],
                marker="o",
                markersize=12.0,
                markerfacecolor="none",
                markeredgecolor=VOID_EDGE,
                markeredgewidth=1.7,
                zorder=4,
            )
            right.plot(
                [row["bps"]],
                [y],
                marker="x",
                markersize=8.0,
                color=VOID_EDGE,
                markeredgewidth=1.7,
                zorder=5,
            )
        else:
            right.plot(
                [row["bps"]],
                [y],
                marker="o",
                markersize=9.5,
                markerfacecolor=colour,
                markeredgecolor="white",
                markeredgewidth=0.9,
                zorder=4,
            )
        right.annotate(
            row["value"],
            xy=(row["bps"], y),
            xytext=(offset, 1),
            textcoords="offset points",
            fontsize=8.6,
            color="#444444" if row["void"] else "#222222",
            ha=align,
            va="center",
        )
        right.annotate(
            row["note"],
            xy=(row["bps"], y),
            xytext=(offset, -13),
            textcoords="offset points",
            fontsize=7.6,
            color="#666666",
            ha=align,
            va="center",
        )

    right.set_yticks(positions)
    right.set_yticklabels([row["label"] for row in rows], fontsize=9.0)
    right.set_ylim(max(positions) + 0.85, min(positions) - 0.85)
    right.set_xlim(x_low, x_high)
    right.set_xlabel("arrival slippage, bps")
    right.set_title("Three tiers of claim, one axis", fontsize=11, pad=8)
    right.grid(axis="x", color="#e6e6e6", linewidth=0.7)
    right.set_axisbelow(True)

    figure.suptitle(
        "M6 — a trained policy on a live book: predicted, measured, withheld",
        fontsize=12.5,
        y=0.985,
    )
    # Caption hard-wrapped by the caller and drawn in its own reserved band, for
    # the reason the house note gives: matplotlib will not tell you that text ran
    # off the canvas, so the width is bounded where the string is built and the
    # space is reserved here rather than hoped for.
    figure.text(0.008, 0.010, caption, fontsize=7.6, color="#333333", va="bottom")
    figure.text(
        0.992,
        0.958,
        provenance.short,
        fontsize=7.5,
        color="#666666",
        family="monospace",
        ha="right",
        va="top",
    )
    figure.subplots_adjust(
        left=0.068, right=0.988, top=0.900, bottom=0.355, wspace=0.28, hspace=0.10
    )

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix in formats:
        out = target.with_name(f"{target.name}.{suffix.lstrip('.')}")
        figure.savefig(
            out,
            dpi=160,
            metadata={"Software": None, "Creator": None, "Date": None},
        )
        written.append(out)
    plt.close(figure)
    return written


# ---------------------------------------------------------------------------
# M5 - the alpha signal: what was available, and how it was spent
# ---------------------------------------------------------------------------

#: The three iso-net-capture lines worth drawing as lines, each with the height
#: its label sits at and the words that say what it IS. Three rather than a
#: ladder of them: a contour that nobody can name is a contour nobody reads, and
#: the shading behind them already carries the gradient.
NET_LINES = (
    (0.0, 1.14, "net 0 - no better than M4a's schedule", "#6b6b6b", (0, (5, 3)), 1.1),
    (0.90, 0.62, "net 0.90 - the pre-stated bar", "#1a5c38", "-", 2.0),
    (1.00, 1.56, "net 1.00 - the DP's own line", "#1f4e79", (0, (2, 2)), 1.3),
)


def alpha_figure(
    path,
    *,
    plane: dict,
    curve: dict,
    provenance: Provenance,
    caption: str,
    formats=("png",),
) -> list[Path]:
    """M5's hero: a DECOMPOSITION, because the headline alone cannot be read.

    M5's methodological content is that net capture is a *difference of two
    quantities* — the gross alpha a policy monetised, minus the execution premium
    it paid to monetise it — and that one number cannot tell a policy which traded
    the signal well from one which traded it badly and executed well. A figure
    showing a single capture fraction would repeat that mistake in pictures, so
    neither panel here reports one.

    **Left — the plane the three numbers live in.** Alpha capture along x, premium
    ratio up y, both as multiples of what the converged DP achieves, so the DP is
    the point (1, 1) by construction. Net capture is *linear* on this plane

        net = (A a - P p) / D,     D = A - P

    with A the gross alpha available and P the premium the optimum itself pays, so
    the iso-net-capture lines are straight and their slope, A/P = 2.2, is the
    exchange rate: one part of gross alpha is worth 2.2 parts of premium. The ten
    seeds are drawn individually (*below n ~ 10, draw every trace*), and the ten
    shuffled controls sit directly beneath them — same premium, no alpha — which
    is the milestone's claim in one glance rather than one sentence.

    The DP does not sit on the 1.00 line, and the offset is drawn rather than
    smoothed: the graded M4a schedule, which provably monetises no alpha, reads an
    alpha capture of -0.0035 instead of 0. That is the empirical mean of the
    200,000 shared signal paths, which is not exactly zero at 1/sqrt(M) ~ 2e-3. It
    cancels out of net capture, which is a paired difference against that very
    schedule, and does not cancel out of alpha capture, which is a level.

    **Right — what was available, against the invented parameter.** ``rho`` is
    Temper's own; FrontierView vendored no signal. So the advantage is drawn as a
    curve across the six values the oracle table carries, split into the two halves
    it is a difference of, with the trained point marked. The share the optimum
    gives back falls from 49 % to 20 % as the signal grows, on the thin line
    against the right-hand axis: a bigger signal is not merely worth more, it is
    worth *proportionally more*, and a single point with a single number beside it
    would read as calibration and it is not.

    `plane` and `curve` are read off committed artefacts by
    ``tools/m5_alpha_figure.py``. Nothing here computes a cost.
    """
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(11.9, 6.6), gridspec_kw={"width_ratios": (1.15, 1.0)}
    )

    # ---- left panel: the decomposition plane ------------------------------
    alpha_available = float(plane["alpha_available_bps"])
    premium = float(plane["premium_bps"])
    advantage = float(plane["advantage_bps"])
    intercept = float(plane["net_intercept"])

    def net_of(a, p):
        return intercept + (alpha_available * a - premium * p) / advantage

    x_lo, x_hi = -0.12, 1.28
    y_lo, y_hi = -0.14, 1.90
    grid_a, grid_p = np.meshgrid(
        np.linspace(x_lo, x_hi, 300), np.linspace(y_lo, y_hi, 300)
    )
    net = net_of(grid_a, grid_p)

    left.contourf(grid_a, grid_p, net, levels=24, cmap="BuGn", alpha=0.40, zorder=0)
    left.set_xlim(x_lo, x_hi)
    left.set_ylim(y_lo, y_hi)

    def alpha_on(level, p):
        """Where the iso-net line at `level` crosses premium ratio `p`."""
        return ((level - intercept) * advantage + premium * p) / alpha_available

    for level, label_p, words, colour, dash, width in NET_LINES:
        ends = [(alpha_on(level, y_lo), y_lo), (alpha_on(level, y_hi), y_hi)]
        left.plot(
            [e[0] for e in ends],
            [e[1] for e in ends],
            color=colour,
            linestyle=dash,
            linewidth=width,
            zorder=1,
        )
        # Rotated to lie along its own line, which needs the axes' aspect and so
        # cannot be computed before the limits are set above.
        first = left.transData.transform(ends[0])
        second = left.transData.transform(ends[1])
        angle = np.degrees(np.arctan2(second[1] - first[1], second[0] - first[0]))
        left.text(
            alpha_on(level, label_p),
            label_p,
            words,
            fontsize=7.3,
            color=colour,
            rotation=angle,
            rotation_mode="anchor",
            ha="center",
            va="bottom",
            zorder=3,
        )

    seeds_a = np.asarray(plane["seed_alpha_capture"], dtype=float)
    seeds_p = np.asarray(plane["seed_premium_ratio"], dtype=float)
    shuffled_a = np.asarray(plane["shuffled_alpha_capture"], dtype=float)
    shuffled_p = np.asarray(plane["shuffled_premium_ratio"], dtype=float)

    left.plot(
        [1.0],
        [1.0],
        marker="*",
        markersize=17.0,
        markerfacecolor=STYLE["optimal"]["color"],
        markeredgecolor="white",
        markeredgewidth=0.8,
        linestyle="none",
        label="$J_{DP}$ - the converged optimum, by construction (1, 1)",
        zorder=6,
    )
    left.plot(
        seeds_a,
        seeds_p,
        marker="o",
        markersize=6.4,
        linestyle="none",
        markerfacecolor=STYLE["agent"]["color"],
        markeredgecolor="white",
        markeredgewidth=0.7,
        label=f"PPO, {seeds_a.size} seeds (each drawn)",
        zorder=5,
    )
    left.plot(
        shuffled_a,
        shuffled_p,
        marker="v",
        markersize=5.8,
        linestyle="none",
        markerfacecolor="#c1663c",
        markeredgecolor="white",
        markeredgewidth=0.6,
        label="the same seeds, signal SHUFFLED - the control",
        zorder=5,
    )

    # The two reference schedules that fit on this window. TWAP (22.1x) and AC
    # (1.95x) do not, and the caption says so rather than the panel rescaling
    # around two policies that are not what the milestone is about.
    for key in ("optimal", "tangent"):
        point = plane["baselines"].get(key)
        if point is None:
            continue
        left.plot(
            [point["alpha_capture"]],
            [point["premium_ratio"]],
            marker="s",
            markersize=5.4,
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="#5a5a5a",
            markeredgewidth=1.1,
            zorder=5,
        )
    tangent = plane["baselines"].get("tangent")
    if tangent is not None:
        left.annotate(
            "AC at the tangent, graded",
            xy=(tangent["alpha_capture"], tangent["premium_ratio"]),
            xytext=(9, -3),
            textcoords="offset points",
            fontsize=7.4,
            color="#5a5a5a",
        )

    # One annotation on the M4a marker, not two: what the square IS, and the one
    # number about it that a reader would otherwise take for a defect. A schedule
    # that provably monetises no alpha reads -0.0035, and it is why the DP sits
    # above its own 1.00 line rather than on it.
    offset = float(plane["baselines"]["optimal"]["alpha_capture"])
    # One line with a bbox rather than three across the middle of the panel. The
    # arithmetic behind the offset belongs in the caption; what the picture has to
    # carry is that this square is NOT at zero and that the miss is explained.
    left.annotate(
        f"M4a's certified optimum, graded - monetises no alpha, reads "
        f"{offset:+.4f} (caption)",
        xy=(offset, 0.0),
        xytext=(0.075, 0.035),
        fontsize=7.3,
        color="#8a5a2b",
        ha="left",
        va="center",
        bbox={"facecolor": "white", "alpha": 0.80, "edgecolor": "none",
              "boxstyle": "round,pad=0.22"},
        arrowprops={"arrowstyle": "-|>", "color": "#c1663c", "linewidth": 0.9,
                    "shrinkA": 2, "shrinkB": 4},
    )

    left.axhline(1.0, color="#9a9a9a", linewidth=0.7, linestyle=(0, (1, 3)), zorder=1)
    left.axvline(1.0, color="#9a9a9a", linewidth=0.7, linestyle=(0, (1, 3)), zorder=1)
    left.set_xlabel("alpha capture - gross signal monetised, over the optimum's")
    left.set_ylabel("premium ratio - execution paid for it, over the optimum's")
    left.set_title("The plane the three numbers live in", fontsize=11, pad=8)
    left.grid(color="#ececec", linewidth=0.6)
    left.set_axisbelow(True)
    # Upper left, above the shuffled column's ceiling at 1.34: the y axis runs to
    # 1.90 to open that band rather than to fit any data, which is the cheapest
    # way to keep a legend off a scatter without moving the scatter.
    left.legend(fontsize=7.6, loc="upper left", framealpha=0.93, borderpad=0.5)

    # ---- right panel: what the signal is worth ----------------------------
    rho = np.asarray(curve["rho"], dtype=float)
    available = np.asarray(curve["alpha_available_bps"], dtype=float)
    paid = np.asarray(curve["execution_premium_bps"], dtype=float)
    net_advantage = np.asarray(curve["advantage_bps"], dtype=float)
    trained = float(curve["trained_rho"])

    right.plot(rho, available, marker="o", markersize=5.0, linewidth=1.8,
               color="#1f4e79", label="gross alpha available $A$")
    right.plot(rho, paid, marker="s", markersize=4.4, linewidth=1.5,
               linestyle=(0, (5, 2)), color="#c1663c",
               label="premium the optimum pays $P$")
    right.plot(rho, net_advantage, marker="D", markersize=4.4, linewidth=2.0,
               color="#227a4b", label="net advantage $A - P$ - the denominator")
    right.set_xscale("log")
    right.set_yscale("log")
    right.axvline(trained, color="#8c8c8c", linewidth=1.0, linestyle=(0, (1, 3)))
    index = int(np.argmin(np.abs(rho - trained)))
    right.plot([trained], [net_advantage[index]], marker="D", markersize=11.0,
               markerfacecolor="none", markeredgecolor="#227a4b", markeredgewidth=1.6)
    right.annotate(
        f"trained here\n$\\rho$ = {trained:g}, {curve['trained_explained']:.0e} of\n"
        "next-bin return variance",
        xy=(trained, net_advantage[index]),
        xytext=(9, -30),
        textcoords="offset points",
        fontsize=7.8,
        color="#227a4b",
    )

    share = right.twinx()
    fraction = paid / available
    share.plot(rho, 100.0 * fraction, color="#7b5ea7", linewidth=1.2,
               linestyle=(0, (2, 2)), marker=".", markersize=4.0)
    share.set_ylabel("premium as a share of gross alpha, %", color="#7b5ea7",
                     fontsize=9.0)
    share.tick_params(axis="y", labelcolor="#7b5ea7", labelsize=8.0)
    share.set_ylim(0.0, 60.0)
    share.annotate(
        f"the optimum gives back {100 * fraction[index]:.0f} % here,\n"
        f"and {100 * fraction[-1]:.0f} % at $\\rho$ = {rho[-1]:g}",
        xy=(rho[-1], 100.0 * fraction[-1]),
        xytext=(-4, 44),
        textcoords="offset points",
        fontsize=7.4,
        color="#7b5ea7",
        ha="right",
    )

    right.set_xlabel("$\\rho$ - Temper's own INVENTED signal strength")
    right.set_ylabel("bps of the objective")
    right.set_title("What the signal is worth, and what it costs", fontsize=11, pad=8)
    right.grid(color="#ececec", linewidth=0.6)
    right.set_axisbelow(True)
    right.legend(fontsize=8.0, loc="upper left", framealpha=0.94)

    figure.suptitle(
        "M5 - an invented alpha signal: what was available, and how it was spent",
        fontsize=12.5,
        y=0.985,
    )
    figure.text(0.008, 0.010, caption, fontsize=7.6, color="#333333", va="bottom")
    figure.text(0.992, 0.955, provenance.short, fontsize=7.5, color="#666666",
                family="monospace", ha="right", va="top")
    figure.subplots_adjust(left=0.062, right=0.930, top=0.895, bottom=0.245, wspace=0.30)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix in formats:
        out = target.with_name(f"{target.name}.{suffix.lstrip('.')}")
        figure.savefig(
            out,
            dpi=160,
            metadata={"Software": None, "Creator": None, "Date": None},
        )
        written.append(out)
    plt.close(figure)
    return written
