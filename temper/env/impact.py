"""Temporary impact as an injected model, so the world is a choice and not a habit.

Until M4a the env's temporary charge was a precomputed per-share constant,
``eta_tilde * BPS / dt``, and it was the only one there had ever been. M4a makes
the *vendored* power law the world, so there are two — and constitution §4 says
Phase-2 models arrive as "additive alternatives behind the same interface, never
silent modifications of Phase 1". This module is that interface.

Each model declares two things: the concession, in bps per share traded, that a
bin of ``n`` shares pays; and the :data:`~temper.oracle.model.ENCODINGS` name of
the cost functional it charges. The second is what makes M4a's registry rule
mechanical — *a metric grades the world that charges it* — because an env can
then be asked which world it is, and refuse a grader that speaks the other one.

Two models, and the second is not new arithmetic
------------------------------------------------
:class:`LinearTemporary` is the tangent ``eta_tilde * v`` the Almgren–Chriss
closed form solves, built exactly as :class:`~temper.env.ExecutionEnv` built it
before this module existed — the Phase-1 regression is *bitwise*, so "exactly" is
checkable rather than intended. :class:`PowerLawTemporary` is FrontierView's
``eta * sigma * p ** 0.6``, which is what the vendored goldens pin and what
:func:`~temper.oracle.cost.cost_moments` charges.

The env still reaches its cost the long way round, bin by bin: these are
per-share price concessions, not schedule cost functionals, and nothing here
imports a trajectory, a moment or a closed form. ``tests/test_repo_invariants.py``
keeps it that way, which is what stops M1's differential quietly becoming a
comparison of the oracle with itself.

Binding
-------
A model carries the parameters of the *impact function*; the execution grid is
the env's. :meth:`TemporaryImpact.bind` folds the two together into a callable
that holds one precomputed float, because ``step`` runs seventy million times in
the deep differential tier and the grid never changes inside an episode.
"""

from __future__ import annotations

from dataclasses import dataclass

from temper.oracle import BPS, LINEAR_ENCODING, POWER_LAW_ENCODING, Market, linearised_eta


class TemporaryImpact:
    """The concession a bin of ``n`` shares pays, in bps per share.

    Subclasses declare :attr:`encoding` and implement :meth:`bind`. Deliberately
    a tiny base class rather than a Protocol: the env stores the model it was
    handed and republishes its encoding, so the type is something a config
    resolves to and a results file names, not only a shape a call site satisfies.
    """

    #: Which of :data:`~temper.oracle.model.ENCODINGS` this model charges.
    encoding: str

    def bind(self, market: Market):
        """A callable ``shares -> bps per share`` on `market`'s grid."""
        raise NotImplementedError


@dataclass(frozen=True)
class _LinearCharge:
    """``eta_tilde * BPS / dt * n`` — one multiply, per step."""

    coefficient: float

    def __call__(self, shares: float) -> float:
        return self.coefficient * shares


@dataclass(frozen=True)
class _PowerLawCharge:
    """``eta * sigma * BPS * (n / (dt * v_hourly)) ** beta`` — the vendored charge."""

    coefficient: float
    rate: float
    exponent: float

    def __call__(self, shares: float) -> float:
        return self.coefficient * (shares * self.rate) ** self.exponent


@dataclass(frozen=True)
class LinearTemporary(TemporaryImpact):
    """Phase 1: linear in the trading rate, at the tangent slope ``eta_tilde``.

    ``eta_tilde`` is a property of the *parent order* — the tangent is taken at
    the participation rate that order would run at under TWAP — so it is passed
    in rather than derived here, and :func:`linear_temporary` is the constructor
    that derives it the one sanctioned way.
    """

    eta_tilde: float
    encoding: str = LINEAR_ENCODING

    def bind(self, market: Market) -> _LinearCharge:
        return _LinearCharge(coefficient=self.eta_tilde * BPS / market.dt)


@dataclass(frozen=True)
class PowerLawTemporary(TemporaryImpact):
    """M4a: FrontierView's ``eta * sigma * |p| ** beta``, the world the goldens pin.

    Concave in the participation rate, so concentrating a trade is cheaper than
    the closed form's tangent believes and the true optimum is *faster* than the
    Almgren–Chriss schedule. That difference is the whole of M4a's earned
    advantage; it is worth 1.54 % of expected cost at the reference case.

    No ``eta_tilde``, and therefore no dependence on the parent order: unlike the
    tangent, the power law is the same function whatever size is worked through
    it. That is the mis-specification, stated as a signature.
    """

    eta: float
    sigma: float
    exponent: float
    encoding: str = POWER_LAW_ENCODING

    def bind(self, market: Market) -> _PowerLawCharge:
        return _PowerLawCharge(
            coefficient=self.eta * self.sigma * BPS,
            rate=1.0 / (market.dt * market.v_hourly),
            exponent=self.exponent,
        )


def linear_temporary(market: Market, order_size: float) -> LinearTemporary:
    """The Phase-1 model for this order — the env's default, built as it always was."""
    return LinearTemporary(eta_tilde=linearised_eta(market, order_size))


def power_law_temporary(market: Market) -> PowerLawTemporary:
    """FrontierView's model for this market, at the vendored exponent."""
    return PowerLawTemporary(
        eta=market.params.eta,
        sigma=market.params.sigma,
        exponent=market.temp_exponent,
    )


def impact_for(encoding: str, market: Market, order_size: float) -> TemporaryImpact:
    """The model an encoding names. One place that maps the two worlds.

    Callers name a *world* — a config's ``world.cost_encoding``, a reference
    table's encoding — and get the model that charges it, so an env and the
    optimum it is graded against cannot be built from two different readings of
    the same string.
    """
    if encoding == LINEAR_ENCODING:
        return linear_temporary(market, order_size)
    if encoding == POWER_LAW_ENCODING:
        return power_law_temporary(market)
    raise ValueError(f"unknown cost encoding {encoding!r}")
