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
