"""TWAP and the two Almgren–Chriss schedules, wrapped as policies.

Constitution §5: *baselines are policies, not special cases*. Nothing here gets a
private path through the environment or the eval harness — a baseline exposes
``act(observation)`` exactly as a trained network will, steps the same
:class:`~temper.env.ExecutionEnv`, and is scored by the same code. A baseline
that took a shortcut would stop being a baseline: whatever bug the shortcut
skipped would show up as agent skill.

A deterministic schedule is open-loop, so the policy could simply count its own
calls. It reads the bin index out of the observation's time-remaining fraction
instead, which keeps it a genuine function of the observation and means a
mis-ordered rollout loop fails loudly rather than silently replaying bin 0.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np

from temper.oracle import (
    LINEAR_ENCODING,
    POWER_LAW_ENCODING,
    Market,
    ac_trajectory,
    optimal_trajectory,
    power_law_optimum,
    twap_trajectory,
)


class Policy(Protocol):
    """Anything gradable: a name, a per-episode reset, and ``act``."""

    name: str

    def reset(self) -> None:
        """Called once at the start of each episode."""

    def act(self, observation: np.ndarray) -> float:
        """Shares to execute this interval, given ``(time left, inventory left)``."""


class SchedulePolicy:
    """A fixed inventory trajectory replayed as a policy.

    The trajectory is the same representation the oracle uses: ``n_bins + 1``
    inventory levels, so trades are its negated first difference. Schedules whose
    tail does not reach zero (the ``sinh`` overflow asymptote leaves ``X·e^-κT``)
    are replayed as-is and the env's terminal force-liquidation deals with the
    remainder — the policy does not quietly repair the schedule, because whether
    that remainder is charged correctly is exactly what M1 is testing.
    """

    def __init__(self, trajectory, name: str) -> None:
        levels = np.asarray(trajectory, dtype=np.float64)
        if levels.ndim != 1 or levels.size < 2:
            raise ValueError(f"trajectory must be 1-D with >= 2 points, got {levels.shape}")
        self.name = name
        self.trajectory = levels
        self.trades = -np.diff(levels)
        self._n_bins = int(self.trades.size)

    def reset(self) -> None:
        """Nothing to forget: the schedule is a pure function of the bin index."""

    def act(self, observation: np.ndarray) -> float:
        """The bin's planned trade, located by the observation's clock."""
        bin_index = self._n_bins - int(round(float(observation[0]) * self._n_bins))
        if not 0 <= bin_index < self._n_bins:
            raise ValueError(
                f"observation clock {observation[0]!r} is outside the schedule's "
                f"{self._n_bins} bins"
            )
        return float(self.trades[bin_index])

    def __repr__(self) -> str:  # keeps pytest ids and failure messages readable
        return f"SchedulePolicy({self.name!r}, {self._n_bins} bins)"


def twap_policy(market: Market, order_size: float, lambda_risk: float = 0.0):
    """Flat participation. Ignores `lambda_risk`; it takes one for a uniform registry."""
    return SchedulePolicy(twap_trajectory(market, order_size), "twap")


def ac_policy(market: Market, order_size: float, lambda_risk: float):
    """FrontierView's vendored Almgren–Chriss schedule (``ac_kappa``)."""
    return SchedulePolicy(ac_trajectory(market, order_size, lambda_risk), "ac")


def optimal_policy(market: Market, order_size: float, lambda_risk: float):
    """The exact discrete minimiser of the frozen objective (``optimal_kappa``)."""
    return SchedulePolicy(optimal_trajectory(market, order_size, lambda_risk), "optimal")


def tangent_policy(market: Market, order_size: float, lambda_risk: float):
    """The same sinh, under its power-law-world name.

    In the linearised world that schedule *is* the optimum. In M4a's world it is
    the closed form's answer to a problem the closed form was not derived for —
    still exactly optimal for the tangent, and 16 878 shares from the optimum of
    the world it is actually run in. Two names for one trajectory because it
    plays two roles, and conflating them is the whole mistake M4a measures.
    """
    return SchedulePolicy(optimal_trajectory(market, order_size, lambda_risk), "tangent")


def power_law_optimum_policy(market: Market, order_size: float, lambda_risk: float):
    """The certified optimum of the power-law world — what M4a grades against."""
    return SchedulePolicy(power_law_optimum(market, order_size, lambda_risk), "optimal")


#: The named baselines each world carries, by the key its reference row uses.
#: Both kappa conventions are here on purpose: `ac` is what the goldens pin and
#: what every chart carries as the vendored comparison, `optimal` is what the
#: agent is graded against (``ARCHITECTURE.md`` §9, 2026-08-04) — and `optimal`
#: means *this world's* optimum, so in the power-law world it is the certified
#: solve and the sinh appears beside it as `tangent`.
WORLD_BASELINES: dict[str, dict[str, Callable[[Market, float, float], SchedulePolicy]]] = {
    LINEAR_ENCODING: {
        "twap": twap_policy,
        "ac": ac_policy,
        "optimal": optimal_policy,
    },
    POWER_LAW_ENCODING: {
        "twap": twap_policy,
        "ac": ac_policy,
        "tangent": tangent_policy,
        "optimal": power_law_optimum_policy,
    },
}

#: The Phase-1 baselines, under the name every committed config and test uses.
BASELINES: dict[str, Callable[[Market, float, float], SchedulePolicy]] = (
    WORLD_BASELINES[LINEAR_ENCODING]
)


def baseline(
    name: str,
    market: Market,
    order_size: float,
    lambda_risk: float,
    *,
    encoding: str = LINEAR_ENCODING,
):
    """Build a named baseline for a world, or fail with the names that exist."""
    try:
        world = WORLD_BASELINES[encoding]
    except KeyError:
        raise ValueError(
            f"unknown cost encoding {encoding!r}; expected one of "
            f"{', '.join(sorted(WORLD_BASELINES))}"
        ) from None
    try:
        factory = world[name]
    except KeyError:
        raise ValueError(
            f"unknown baseline {name!r} for the {encoding!r} world; expected one "
            f"of {', '.join(sorted(world))}"
        ) from None
    return factory(market, order_size, lambda_risk)
