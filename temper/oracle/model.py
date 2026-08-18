"""Market parameters, the execution grid, and the units contract.

Every quantity in `temper.oracle` obeys the conventions fixed here, which are the
conventions of the vendored FrontierView goldens (constitution §4: *the goldens,
not the prose, are the numeric spec*).

Units
-----
====================  =========================================================
inventory / trades    shares
participation ``p``   shares-per-hour divided by ``v_hourly`` (dimensionless)
time                  trading hours; the day is ``TRADING_HOURS_PER_DAY`` long
``sigma``             *daily* fractional return volatility
``half_spread``       basis points
cost                  basis points of notional
variance              basis points squared (execution shortfall, not P&L)
====================  =========================================================

The grid is FrontierView's: half-hour bins, ``n_bins = max(2, round(2T))``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: Length of a trading day in hours (09:30–16:00 ET).
TRADING_HOURS_PER_DAY = 6.5

#: Exponent of the power-law temporary impact function (Almgren et al. 2005).
TEMP_EXPONENT = 0.6

#: Basis-points scale factor. Impact functions are fractional; costs are bps.
BPS = 1.0e4

#: The frozen Phase-1 objective: linear temporary impact at the tangent eta_tilde.
LINEAR_ENCODING = "linear"

#: FrontierView's 0.6-power temporary charge — reporting context until M4a, and
#: from M4a the world an agent may be trained in and graded against.
POWER_LAW_ENCODING = "power_law"

#: The cost encodings that exist. A *world* (an env's temporary-impact model), a
#: *metric* and an *optimum* each declare one of these, and M4a's rule is that a
#: metric grades the world that charges it — so the name lives here, beside the
#: units, rather than in either of the two packages that have to agree on it.
#: ``temper/env`` and ``temper/eval`` both import it from the oracle; the oracle
#: is normative (invariant 2) and neither of them may depend on the other.
ENCODINGS: tuple[str, ...] = (LINEAR_ENCODING, POWER_LAW_ENCODING)


@dataclass(frozen=True)
class SymbolParams:
    """Observable market parameters for one symbol.

    ``eta`` and ``gamma`` are the Almgren et al. (2005) Table 3 literature
    constants in FrontierView; they are per-symbol here only because the oracle
    must be able to reproduce any case the goldens contain.
    """

    adv: float          # average daily volume, shares/day
    sigma: float        # daily return volatility, fractional
    half_spread: float  # half bid-ask spread, bps
    eta: float          # temporary impact coefficient
    gamma: float        # permanent impact coefficient

    def __post_init__(self) -> None:
        if self.adv <= 0.0:
            raise ValueError(f"adv must be positive, got {self.adv}")
        if self.sigma < 0.0:
            raise ValueError(f"sigma must be non-negative, got {self.sigma}")


def default_n_bins(horizon_hours: float) -> int:
    """Canonical bin count for a horizon: half-hour slots, minimum two.

    Mirrors FrontierView's own rule, including its banker's rounding — a 2.25 h
    horizon gives 4 bins, not 5. The goldens pin this.
    """
    return max(2, round(horizon_hours * 2))


@dataclass(frozen=True)
class Market:
    """A symbol plus the execution grid it is worked on.

    All the derived quantities the closed forms need (`dt`, `v_hourly`,
    `sigma_bin`, `times`) hang off here so that no caller re-derives them and
    drifts.
    """

    params: SymbolParams
    horizon_hours: float
    n_bins: int
    temp_exponent: float = TEMP_EXPONENT

    # Cached because `times` is touched once per trajectory evaluation.
    _times: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.horizon_hours <= 0.0:
            raise ValueError(f"horizon must be positive, got {self.horizon_hours}")
        if self.n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {self.n_bins}")
        times = np.arange(self.n_bins + 1, dtype=float) * self.dt
        object.__setattr__(self, "_times", times)
        times.flags.writeable = False

    @classmethod
    def for_horizon(
        cls,
        params: SymbolParams,
        horizon_hours: float,
        temp_exponent: float = TEMP_EXPONENT,
    ) -> "Market":
        """Build a market on the canonical grid for `horizon_hours`."""
        return cls(
            params=params,
            horizon_hours=horizon_hours,
            n_bins=default_n_bins(horizon_hours),
            temp_exponent=temp_exponent,
        )

    @property
    def dt(self) -> float:
        """Bin length in hours."""
        return self.horizon_hours / self.n_bins

    @property
    def v_hourly(self) -> float:
        """ADV expressed as shares per trading hour."""
        return self.params.adv / TRADING_HOURS_PER_DAY

    @property
    def sigma_bin(self) -> float:
        """Per-bin volatility: daily sigma scaled by sqrt of the day fraction."""
        return self.params.sigma * math.sqrt(self.dt / TRADING_HOURS_PER_DAY)

    @property
    def times(self) -> np.ndarray:
        """Bin boundaries in hours, `n_bins + 1` points from 0 to the horizon."""
        return self._times
