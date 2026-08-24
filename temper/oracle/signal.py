"""The alpha signal — Temper's second invented process, owned as invented.

:mod:`~temper.oracle.liquidity` already carries the sentence this module inherits:
constitution §7's "vendored, not invented" cover is what lets the project say its
market is somebody else's calibration rather than a shape chosen to make an agent
look good, and **that cover does not reach here either**. FrontierView has no
alpha model. M5 invents one, and every number downstream of it carries the
invention in the same sentence as the result.

What the signal is
------------------
At the decision point for bin ``k`` the observation carries ``s_k ~ N(0, 1)``,
and ``s_k`` is correlated with the price shock of the bin that has **not yet
happened**:

.. code::

    E[xi_{k+1} | s_k] = rho * s_k,        Corr(s_k, xi_{k+1}) = rho

Every other pair is independent: ``s`` is i.i.d. across bins, each ``s_k``
predicts exactly one shock, and a shock that has landed is predicted by nothing.
One parameter, ``rho``, and :attr:`explained_variance_fraction` is ``rho**2`` —
the fraction of next-bin return variance the signal accounts for.

Why ``rho`` is small, and why "weak" was not a specification
------------------------------------------------------------
Per-bin volatility at the reference case is ``sigma_bin * BPS = 42.99 bps``, which
is **18x the whole objective**. Information about price is therefore worth vastly
more per unit than information about cost, and a signal an equities researcher
would call vanishingly weak still dominates the milestone if it is let: at
``rho = 0.05`` the objective halves, and at ``rho = 0.2`` it goes negative and the
agent is no longer executing an order, it is trading one.

``rho = 0.01`` — one part in ten thousand of next-bin return variance — is chosen
so that M4a, M4b and M5 report advantages on **one scale** (0.037 / 0.062 / 0.081
bps). That is what lets M4b's tolerance machinery transfer instead of being
redesigned, and it is a scoping decision recorded in
``docs/briefs/M5-alpha-aware-execution.md`` rather than a calibration.

Why the mean is exactly zero, and why that is load-bearing
-----------------------------------------------------------
A *deterministic* schedule's inventory path does not depend on ``s``, so the alpha
term of its expected cost is ``-A rho sum_k h_k E[s]`` and ``E[s] = 0`` **exactly**
— not approximately. That is what makes M5 the first Phase-2 milestone with no
lambda decision to record: the liquidity world needed a third *reading* of the
selection rule because ``E[L^-beta] > 1`` moved every fixed schedule's objective,
and here nothing moves at all. :meth:`AlphaSignal.mean` returns a float zero for
that reason, and ``tools/m5_reference_table.py`` asserts the resulting table is
**bit-identical** to M4a's rather than merely agreeing with it.

Why i.i.d. and one step ahead
------------------------------
The same three-part argument :mod:`~temper.oracle.liquidity` makes, and for the
same reasons:

* **All of the measured advantage is information.** A zero-mean i.i.d. signal
  leaves every fixed schedule's objective exactly where M4a left it, so there is
  no level shift for a static solver to pick up for free — M4b needed a whole gate
  to establish that its headline was not a re-solve, and here it is arithmetic.
* **``(k, x_k, s_k)`` is a sufficient statistic**, so the dynamic program over
  that state is the optimum over *all* adapted policies rather than merely over
  the ones with that observation. ``s_{k-1}`` predicted a shock that has already
  landed; its cost is sunk and its information is spent.
* **``rho`` is invented, so the result is a curve and not a point.** The milestone
  trains at one value and the oracle reports the value of the signal at six.

Persistent or AR(1) alpha, and a multi-step-ahead signal, both change the state of
the dynamic program and neither is needed to make the point: they are named in the
brief's *Out of scope* and belong to a later milestone if they belong anywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

#: The signal models a config may name. Like :data:`LIQUIDITY_MODELS` the order is
#: not a contract — nothing is addressed positionally — but the names are, because
#: a committed config and a results file both write them down.
NO_SIGNAL = "none"
ONE_STEP_SIGNAL = "one_step"
SIGNAL_MODELS: tuple[str, ...] = (NO_SIGNAL, ONE_STEP_SIGNAL)


class AlphaSignal:
    """A per-bin observation that is partially predictive of the next shock.

    A tiny base class rather than a Protocol, for the reason
    :class:`~temper.oracle.liquidity.LiquidityLaw` is one: the env will store the
    signal it was handed and republish it, a config resolves to it, and a results
    file names it — so the type is a thing, not only a shape a call site
    satisfies.

    Subclasses expose the moments the analytic reference needs (:meth:`mean`,
    :meth:`variance`, :meth:`correlation`), the quadrature the dynamic program
    integrates against, and the *joint* draw of a signal path with the shock path
    it predicts. One owner for the distribution; everything else asks it.
    """

    #: The name a config writes down and a results file records.
    name: str

    @property
    def informative(self) -> bool:
        """Whether this signal carries any information about any shock at all."""
        raise NotImplementedError

    def mean(self) -> float:
        """``E[s]``. Exactly zero for every signal here, and that is the point.

        Returned as a float rather than assumed at the call sites because the
        whole of "lambda needs no new reading" rests on a *fixed* schedule's
        objective being untouched, and a mean that was merely small would move it
        in the last bits.
        """
        raise NotImplementedError

    def variance(self) -> float:
        """``Var[s]``, for the differential to measure the env's draws against."""
        raise NotImplementedError

    def correlation(self) -> float:
        """``rho = Corr(s_k, xi_{k+1})``. Zero when the signal is uninformative."""
        raise NotImplementedError

    @property
    def explained_variance_fraction(self) -> float:
        """``rho**2`` — the fraction of next-bin return variance the signal explains.

        The unit the brief states the signal's weakness in, because it is the one
        an equities researcher reads: ``rho = 0.01`` is one part in ten thousand.
        """
        return float(self.correlation() ** 2)

    def conditional_shock_mean(self, signals):
        """``E[xi_{k+1} | s_k] = rho s_k`` — the whole model, in one line."""
        return self.correlation() * np.asarray(signals, dtype=float)

    def quadrature(self, nodes: int) -> tuple[np.ndarray, np.ndarray]:
        """``(s values, weights)`` integrating ``E[f(s)]`` — the DP's expectation."""
        raise NotImplementedError

    def draw(self, rng: Generator, size) -> np.ndarray:
        """Sample signal paths alone. The grader's route: cost is closed form given `s`."""
        raise NotImplementedError

    def draw_pair(self, rng: Generator, shape) -> tuple[np.ndarray, np.ndarray]:
        """Sample ``(signals, shocks)`` jointly, obeying the correlation.

        The *definition* of the joint law lives here rather than in the env, for
        the reason :meth:`~temper.oracle.liquidity.LiquidityLaw.draw` does: the
        oracle is normative (invariant 2), and the env's draw will be measured
        against these moments rather than trusted to match them.

        ``shape`` is ``(paths, n_bins)``. ``shocks[:, k]`` is the standardised
        shock that lands *before* bin ``k`` executes — the env's convention
        (``temper/env/execution_env.py``: the shock lands before the bin, M1a's
        §9 entry) — so ``signals[:, k]`` predicts ``shocks[:, k + 1]`` and
        ``shocks[:, 0]`` is predicted by nothing, which is why the first bin's
        shock never appears in a conditional cost: every schedule holds the whole
        order through it.
        """
        raise NotImplementedError

    def as_dict(self) -> dict:
        raise NotImplementedError


@dataclass(frozen=True)
class NoSignal(AlphaSignal):
    """No signal at all — Phase 1, M4a, M4b, and the default everywhere.

    **Draws no randomness.** The same argument :class:`DeterministicLiquidity`
    makes: the signal stream is a separate seed address, so consuming from it
    could not disturb a price path anyway, but a signal that never touches a
    generator makes "M4a's committed seeds reproduce bitwise through the new
    seam" true for a second, independent reason.
    """

    name: str = NO_SIGNAL

    @property
    def informative(self) -> bool:
        return False

    def mean(self) -> float:
        return 0.0

    def variance(self) -> float:
        return 0.0

    def correlation(self) -> float:
        return 0.0

    def quadrature(self, nodes: int) -> tuple[np.ndarray, np.ndarray]:
        # One node carries the point mass at zero exactly; `nodes` is accepted and
        # ignored so a caller can sweep the quadrature without branching.
        return np.array([0.0]), np.array([1.0])

    def draw(self, rng: Generator, size) -> np.ndarray:
        return np.zeros(size, dtype=float)

    def draw_pair(self, rng: Generator, shape) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(shape, dtype=float), rng.standard_normal(shape)

    def as_dict(self) -> dict:
        return {"model": self.name}


@dataclass(frozen=True)
class OneStepSignal(AlphaSignal):
    """``s_k ~ N(0, 1)`` with ``Corr(s_k, xi_{k+1}) = rho``, i.i.d. across bins.

    ``rho`` is **Temper's invented parameter**. It is not calibrated, it is not
    FrontierView's, and no number derived from it may be reported without saying
    so — :meth:`as_dict` puts it in the results file, marked, for that reason.

    ``rho = 0`` is admissible and is the degenerate uninformative case: it is how
    the milestone's most valuable differential is stated, because the dynamic
    program at ``rho = 0`` must return M4a's *certified* ``power_law_optimum``
    value and so ties the whole new machinery to a number that was certified
    rather than merely converged. It is deliberately **not** the same object as
    :class:`NoSignal`: that one is the absence of the seam, this one is the seam
    carrying an uninformative draw, and the differential wants the second.
    """

    rho: float
    name: str = ONE_STEP_SIGNAL

    def __post_init__(self) -> None:
        if not math.isfinite(self.rho) or not -1.0 <= self.rho <= 1.0:
            raise ValueError(f"rho must be finite and in [-1, 1], got {self.rho}")

    @property
    def informative(self) -> bool:
        return self.rho != 0.0

    def mean(self) -> float:
        return 0.0

    def variance(self) -> float:
        return 1.0

    def correlation(self) -> float:
        return float(self.rho)

    def quadrature(self, nodes: int) -> tuple[np.ndarray, np.ndarray]:
        """Gauss–Hermite over the standard normal, which is where ``s`` lives.

        ``hermgauss`` integrates against ``exp(-t^2)``, so the standard normal
        substitution is ``z = sqrt(2) t`` with weights divided by ``sqrt(pi)``,
        exactly as :meth:`~temper.oracle.liquidity.LognormalLiquidity.quadrature`
        does one transformation later. The nodes are symmetric about zero and the
        weights are symmetric with them, so the quadrature's own first moment is
        zero to the last bit — which is the numerical half of the claim that a
        zero-mean signal cannot move a deterministic schedule's objective, and
        ``tests/test_m5_alpha_oracle.py`` measures it rather than assuming it.

        **Deliberately not collapsed at ``rho = 0``.**
        :meth:`~temper.oracle.liquidity.LognormalLiquidity.quadrature` returns a
        point mass at ``sigma_log = 0`` because there the distribution *is* one;
        here the distribution is still ``N(0, 1)`` and only its usefulness is
        gone. Keeping the full node set is what makes the ``rho -> 0``
        differential check the whole machinery rather than a one-node shortcut
        through it: the value has to come back to M4a's certified number with the
        expectation actually taken. :class:`NoSignal` is the collapsed case, and
        the two are reported side by side.
        """
        if nodes < 1:
            raise ValueError(f"quadrature needs at least one node, got {nodes}")
        abscissa, weights = np.polynomial.hermite.hermgauss(nodes)
        return math.sqrt(2.0) * abscissa, weights / math.sqrt(math.pi)

    def draw(self, rng: Generator, size) -> np.ndarray:
        return rng.standard_normal(size)

    def draw_pair(self, rng: Generator, shape) -> tuple[np.ndarray, np.ndarray]:
        """One generator, two streams' worth of standard normals, then the mixture.

        ``xi_{k+1} = rho s_k + sqrt(1 - rho^2) e_{k+1}`` with ``e`` independent of
        ``s`` gives a unit-variance shock correlated ``rho`` with the signal that
        precedes it, and ``xi_0`` is a pure ``e`` because nothing predicts it. The
        shock keeps unit variance whatever ``rho`` is, which is what stops the
        signal quietly changing the market it is a signal about: M1's variance
        identity and the frozen objective (invariant 7) are statements about
        ``sigma_bin``, and a mixture that did not renormalise would move them.
        """
        signals = rng.standard_normal(shape)
        independent = rng.standard_normal(shape)
        shocks = np.empty_like(independent)
        shocks[..., 0] = independent[..., 0]
        shocks[..., 1:] = (
            self.rho * signals[..., :-1]
            + math.sqrt(1.0 - self.rho**2) * independent[..., 1:]
        )
        return signals, shocks

    def as_dict(self) -> dict:
        return {
            "model": self.name,
            "rho": self.rho,
            "explained_variance_fraction": self.explained_variance_fraction,
            "invented": True,
        }


def signal_for(model: str, **kwargs) -> AlphaSignal:
    """The signal a config names. One place that maps the names to the classes."""
    if model == NO_SIGNAL:
        if kwargs:
            raise ValueError(
                f"the {NO_SIGNAL!r} signal takes no parameters, got "
                f"{', '.join(sorted(kwargs))}"
            )
        return NoSignal()
    if model == ONE_STEP_SIGNAL:
        return OneStepSignal(rho=float(kwargs["rho"]))
    raise ValueError(
        f"unknown signal model {model!r}; known models are {', '.join(SIGNAL_MODELS)}"
    )
