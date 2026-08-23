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
        """A callable ``(shares, liquidity=1.0) -> bps per share`` on `market`'s grid.

        The liquidity argument is M4b's, and it defaults to the deterministic
        world so that a Phase-1 or M4a env reaches bitwise the same charge it
        always did. Liquidity multiplies ``v_hourly``, so it is a property of the
        *market* on the bin rather than of the impact function, which is why it is
        a call argument and not a field.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class _LinearCharge:
    """``eta_tilde * BPS / dt * n`` — one multiply, per step.

    `liquidity` scales ``v_hourly``, so it divides the participation rate and
    therefore the charge. It defaults to ``1.0`` and at that value the arithmetic
    is **bitwise** what it was before the argument existed: ``x / 1.0 == x``
    exactly in IEEE, which is what keeps every Phase-1 number where it was.

    The linear tangent under stochastic liquidity is not a world M4b reports —
    ``eta_tilde`` is taken at a participation rate computed *without* the
    multiplier, so the tangent point itself would move — and the generalisation is
    here for consistency of the interface rather than as a claim about that world.
    """

    coefficient: float

    def __call__(self, shares: float, liquidity: float = 1.0) -> float:
        return self.coefficient * shares / liquidity


@dataclass(frozen=True)
class _PowerLawCharge:
    """``eta * sigma * BPS * (n / (dt * v_hourly * L)) ** beta`` — the vendored charge.

    M4b's world. ``L`` is the per-bin liquidity multiplier and enters exactly
    where the participation rate does and nowhere else: the exponent, the
    coefficient and the shock model are untouched, which is why liquidity does not
    make a new *cost encoding* and §9's *A metric grades the world that charges it*
    needs no amendment. At ``L = 1`` the expression is bitwise the M4a one.
    """

    coefficient: float
    rate: float
    exponent: float

    def __call__(self, shares: float, liquidity: float = 1.0) -> float:
        return self.coefficient * (shares * self.rate / liquidity) ** self.exponent


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
