"""The liquidity multiplier — Temper's own invented process, owned as invented.

Everything else in :mod:`temper.oracle` reproduces something FrontierView already
computes, and constitution §7's "vendored, not invented" cover is what lets the
project say its market is somebody else's calibration rather than a shape chosen
to make an agent look good. **That cover does not extend to this module.**
FrontierView has no liquidity process, so M4b invents one, and every number
downstream of it carries the invention in the same sentence as the result.

The model is therefore the smallest one that makes the question well-posed: a
per-bin i.i.d. lognormal multiplier ``L_k`` on ``v_hourly`` with ``E[L] = 1``, one
parameter ``sigma_log``. Participation becomes ``p_k = n_k / (dt v_hourly L_k)``
and nothing else in the world changes — the price shock, permanent impact, the
half-spread and the frozen objective are all untouched.

Why i.i.d. rather than something that looks like a real day
-----------------------------------------------------------
Three consequences, all of them load-bearing for what M4b is allowed to claim:

* **All of the measured advantage is adaptivity.** The best *static* schedule
  already absorbs the level shift ``E[L^-beta]``, because a fixed schedule's
  expected cost under i.i.d. liquidity is the deterministic cost at an inflated
  impact coefficient (:meth:`LiquidityLaw.inverse_power_moment`, and
  :func:`~temper.oracle.adaptive.liquidity_charge` is the inflation). Nothing
  about the *shape* of the law is capturable by a schedule, so the numerator and
  the denominator of the milestone's headline are clean. A U-shaped intraday
  profile would not have this property — the profile is deterministic and a
  static schedule eats it — which is why the realistic-looking model is the
  harder one to report honestly, and why it is backlog.
* **``(k, x_k, L_k)`` is a sufficient statistic**, so the dynamic-programming
  optimum over that state is the optimum over *all* adapted policies. Past
  liquidity carries no information about future liquidity. That is what makes
  "the agent could have done better with a richer observation" answerable rather
  than arguable, and ``tests/test_m4b_adaptive_oracle.py`` checks it by re-running
  the DP on an augmented state instead of asserting it.
* **``sigma_log`` is invented, so the result is a curve and not a point.** The
  milestone trains at one value and reports the oracle's value-of-sight at three.

The moments are closed forms
----------------------------
``L = exp(sigma_log Z - sigma_log^2 / 2)`` with ``Z ~ N(0, 1)``, so ``E[L] = 1``
by construction and

.. code::

    E[L^-beta] = exp(sigma_log^2 * beta * (1 + beta) / 2)

exactly. That is not a convenience: the static rung and M4a's schedule re-priced
in this world differ by ~0.002 bps, and differencing two *simulated* levels turns
that into noise — which is the level-shift gate, and the gate is what decides
whether this milestone's headline is adaptivity or a re-solve.

The law is also what the DP quadratures over and what the env draws from, and
those are deliberately different routes to the same distribution: M4b's
differential (invariant 6) measures the env's realised draws against the closed
forms here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

#: The liquidity laws a config may name. Order is not a contract here — unlike
#: :data:`~temper.seeding.POOLS`, nothing is addressed positionally — but the
#: names are, because a committed config and a results file both write them down.
DETERMINISTIC_LIQUIDITY = "deterministic"
LOGNORMAL_LIQUIDITY = "lognormal"
LIQUIDITY_MODELS: tuple[str, ...] = (DETERMINISTIC_LIQUIDITY, LOGNORMAL_LIQUIDITY)


class LiquidityLaw:
    """A per-bin multiplier on ``v_hourly``, and the moments the oracle needs.

    Deliberately a tiny base class rather than a Protocol, for the same reason
    :class:`~temper.env.impact.TemporaryImpact` is one: the env stores the law it
    was handed and republishes it, a config resolves to it, and a results file
    names it — so the type is a thing, not only a shape a call site satisfies.

    Subclasses expose the two moments the analytic reference needs
    (:meth:`mean_multiplier`, :meth:`inverse_power_moment`), the quadrature the
    dynamic program integrates against, and the draw the env samples. A
    distribution moment re-derived at three call sites is the pattern that put a
    derived-quantities object on FrontierView's own backlog; here there is one
    owner and everything else asks it.
    """

    #: The name a config writes down and a results file records.
    name: str

    @property
    def stochastic(self) -> bool:
        """Whether this law introduces a second noise source at all."""
        raise NotImplementedError

    def mean_multiplier(self) -> float:
        """``E[L]``. One for every law here — the multiplier is a *reallocation*
        of liquidity across bins, not a change in how much there is on average."""
        raise NotImplementedError

    def variance(self) -> float:
        """``Var[L]``, for the differential to measure the env's draws against."""
        raise NotImplementedError

    def inverse_power_moment(self, exponent: float) -> float:
        """``E[L^-exponent]`` — the factor a fixed schedule's temporary cost pays.

        Temporary cost is ``A * L^-beta * w^(1+beta)`` per bin (see
        :func:`~temper.oracle.adaptive.liquidity_charge` for the derivation), so a
        schedule chosen without knowing ``L`` pays this multiple of what it would
        pay in the deterministic world. Jensen, and by ``E[L] = 1`` it is always
        ``>= 1``: dispersion in liquidity is a cost even when its mean is not.
        """
        raise NotImplementedError

    def quadrature(self, nodes: int) -> tuple[np.ndarray, np.ndarray]:
        """``(L values, weights)`` integrating ``E[f(L)]`` — the DP's expectation."""
        raise NotImplementedError

    def transition_quadrature(self, nodes: int) -> tuple[np.ndarray, np.ndarray]:
        """``(L values, Q)`` with ``Q[p, j]`` the weight on node ``j`` given node ``p``.

        The *conditional* form of :meth:`quadrature`, and the reason it exists is
        that M4b's sufficiency claim has to be **checkable rather than asserted**.
        ``(k, x_k, L_k)`` is a sufficient statistic only because liquidity is
        i.i.d.; :func:`~temper.oracle.adaptive.augmented_optimum` re-solves the
        dynamic program on a state that also carries ``L_{k-1}`` and requires the
        same value, and that check has content only if the machinery it runs on
        could have represented a dependence.

        For every law in this module the answer is one row repeated — the
        transition ignores the previous multiplier, so ``Q[p, :]`` is
        :meth:`quadrature`'s weight vector for every ``p``. A persistent or AR(1)
        law (backlog) would override this and nothing downstream would change,
        which is the point: the augmented solve is a generalisation, not a
        restatement.
        """
        values, weights = self.quadrature(nodes)
        return values, np.tile(weights, (values.size, 1))

    def draw(self, rng: Generator, size) -> np.ndarray:
        """Sample multipliers. The env's route; the moments above are the oracle's."""
        raise NotImplementedError

    def as_dict(self) -> dict:
        raise NotImplementedError


@dataclass(frozen=True)
class DeterministicLiquidity(LiquidityLaw):
    """``L = 1`` always — Phase 1, M4a, and the default everywhere.

    **Draws no randomness.** Not an optimisation: the liquidity stream is a
    separate seed address (:data:`~temper.seeding.LIQUIDITY_POOL`), so consuming
    from it could not disturb a price path anyway — but a law that never touches a
    generator makes "M4a's committed seeds retrain bitwise through the new seam"
    true for a second, independent reason, and the regression asserts the digits
    rather than the argument.
    """

    name: str = DETERMINISTIC_LIQUIDITY

    @property
    def stochastic(self) -> bool:
        return False

    def mean_multiplier(self) -> float:
        return 1.0

    def variance(self) -> float:
        return 0.0

    def inverse_power_moment(self, exponent: float) -> float:
        return 1.0

    def quadrature(self, nodes: int) -> tuple[np.ndarray, np.ndarray]:
        # One node carries a point mass exactly; `nodes` is accepted and ignored
        # so a caller can sweep the quadrature without branching on the law.
        return np.array([1.0]), np.array([1.0])

    def draw(self, rng: Generator, size) -> np.ndarray:
        return np.ones(size, dtype=float)

    def as_dict(self) -> dict:
        return {"model": self.name}


@dataclass(frozen=True)
class LognormalLiquidity(LiquidityLaw):
    """``L = exp(sigma_log Z - sigma_log^2 / 2)``, i.i.d. per bin, ``E[L] = 1``.

    ``sigma_log`` is **Temper's invented parameter**. It is not calibrated, it is
    not FrontierView's, and no number derived from it may be reported without
    saying so. :meth:`as_dict` puts it in the results file for that reason.

    ``sigma_log = 0`` is admissible and is the degenerate point mass at one: it is
    how the milestone's most valuable differential is stated, because the DP at
    ``sigma_log = 0`` must return M4a's *certified* ``power_law_optimum`` value
    and so ties the whole new machinery to a number that was certified rather than
    merely converged.
    """

    sigma_log: float
    name: str = LOGNORMAL_LIQUIDITY

    def __post_init__(self) -> None:
        if not math.isfinite(self.sigma_log) or self.sigma_log < 0.0:
            raise ValueError(
                f"sigma_log must be finite and non-negative, got {self.sigma_log}"
            )

    @property
    def stochastic(self) -> bool:
        return self.sigma_log > 0.0

    def mean_multiplier(self) -> float:
        return 1.0

    def variance(self) -> float:
        """``exp(sigma_log^2) - 1`` — the lognormal variance at unit mean."""
        return float(math.expm1(self.sigma_log**2))

    def inverse_power_moment(self, exponent: float) -> float:
        return float(math.exp(self.sigma_log**2 * exponent * (1.0 + exponent) / 2.0))

    def quadrature(self, nodes: int) -> tuple[np.ndarray, np.ndarray]:
        """Gauss–Hermite in ``log L``, which is where the density is Gaussian.

        ``hermgauss`` integrates against ``exp(-t^2)``, so the standard normal
        substitution is ``z = sqrt(2) t`` with weights divided by ``sqrt(pi)``.
        The weights sum to one to float precision and integrate a polynomial in
        ``z`` of degree ``2 nodes - 1`` exactly; what is being integrated here is
        a value function in ``L^-beta``, which is smooth but not polynomial, so
        the node count is a *measured* convergence rather than an exact-order
        claim — ``tests/test_m4b_adaptive_oracle.py`` reports the sweep.
        """
        if nodes < 1:
            raise ValueError(f"quadrature needs at least one node, got {nodes}")
        if self.sigma_log == 0.0:
            return np.array([1.0]), np.array([1.0])
        abscissa, weights = np.polynomial.hermite.hermgauss(nodes)
        z = math.sqrt(2.0) * abscissa
        return (
            np.exp(self.sigma_log * z - 0.5 * self.sigma_log**2),
            weights / math.sqrt(math.pi),
        )

    def draw(self, rng: Generator, size) -> np.ndarray:
        return np.exp(
            self.sigma_log * rng.standard_normal(size) - 0.5 * self.sigma_log**2
        )

    def as_dict(self) -> dict:
        return {"model": self.name, "sigma_log": self.sigma_log, "invented": True}


def liquidity_for(model: str, **kwargs) -> LiquidityLaw:
    """The law a config names. One place that maps the names to the classes."""
    if model == DETERMINISTIC_LIQUIDITY:
        if kwargs:
            raise ValueError(
                f"the deterministic liquidity law takes no parameters, got "
                f"{', '.join(sorted(kwargs))}"
            )
        return DeterministicLiquidity()
    if model == LOGNORMAL_LIQUIDITY:
        return LognormalLiquidity(sigma_log=float(kwargs["sigma_log"]))
    raise ValueError(
        f"unknown liquidity model {model!r}; known models are "
        f"{', '.join(LIQUIDITY_MODELS)}"
    )
