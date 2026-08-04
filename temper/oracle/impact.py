"""Impact functions and the linearisation the AC schedule is derived at.

Two temporary-impact regimes live here and the distinction matters everywhere
downstream:

* **Power law** — ``eta * sigma * p ** 0.6``, what FrontierView charges a
  schedule and therefore what the goldens pin. Convex in participation, so the
  cost functional it induces has no sinh closed form.
* **Linear** — ``eta_tilde * v``, the tangent to that power law at the order's
  own TWAP participation rate. This is the model the Almgren–Chriss closed form
  actually solves, and the one the constitution's §4 dynamics describe.

Permanent impact is linear in participation in both regimes.
"""

from __future__ import annotations

import numpy as np

from .model import BPS, Market

#: Floor on the linearisation participation rate; below it the tangent slope
#: blows up. FrontierView's value, pinned by the `participation-floor` golden.
PARTICIPATION_FLOOR = 1e-4

#: Floor on the linearised slope, guarding a division by zero downstream.
#: Unreachable for any realistic parameter set; present to match the vendor.
ETA_TILDE_FLOOR = 1e-12


def temporary_impact_bps(participation, market: Market):
    """Power-law temporary impact in bps, spread excluded.

    ``h(p) = eta * sigma_daily * |p| ** beta``, in bps. The participation rate
    is dimensionless so no time scaling appears inside the power law; `sigma` is
    the *daily* volatility throughout.
    """
    p = np.abs(np.asarray(participation, dtype=float))
    return market.params.eta * market.params.sigma * p**market.temp_exponent * BPS


def permanent_drift_bps_per_hour(participation, market: Market):
    """Linear permanent impact in bps per hour of trading at that rate.

    ``g(p) = gamma * sigma_daily * |p|``, unsigned: permanent impact hurts the
    seller and the buyer alike.
    """
    p = np.abs(np.asarray(participation, dtype=float))
    return market.params.gamma * market.params.sigma * p * BPS


def linearised_eta(market: Market, order_size: float) -> float:
    """Tangent slope of temporary impact, per unit of shares-per-hour.

    Linearised at the participation rate this specific order would run at under
    TWAP, so the AC schedule is calibrated to the regime it will operate in::

        p0        = (X / T) / v_hourly            (floored at PARTICIPATION_FLOOR)
        eta_tilde = eta * sigma * beta * p0**(beta - 1) / v_hourly

    The result is a slope in *fractional* impact per share-per-hour: multiply by
    :data:`~temper.oracle.model.BPS` to get bps. Note the tangent's intercept is
    dropped: it contributes a constant to any fully-executing schedule's cost
    (the trade weights sum to one) and so cannot move the optimiser.
    """
    if order_size <= 0.0:
        raise ValueError(f"order_size must be positive, got {order_size}")
    beta = market.temp_exponent
    p0 = max((order_size / market.horizon_hours) / market.v_hourly, PARTICIPATION_FLOOR)
    slope = market.params.eta * market.params.sigma * beta * p0 ** (beta - 1.0)
    return max(slope / market.v_hourly, ETA_TILDE_FLOOR)
