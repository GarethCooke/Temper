"""M4a task 1 — a certificate that ``power_law_optimum`` is the optimum.

M1 task 0 certified a *formula*: ``optimal_trajectory`` is a sinh, and the check
reassembled the frozen objective as a quadratic and did what one does with a
quadratic. The power law has no formula. It has a solve — and a solved reference
is only normative (invariant 2) if the solve is certified, so this module is M1
task 0's shape adapted to one:

* **(a) convexity** — Cholesky of the Hessian at ``x*`` succeeds, so the
  stationary point is the unique global minimum rather than a saddle. ``λ_min``
  and the condition number are recorded, because the trajectory band is derived
  from the first of them;
* **(b) KKT** — every bin's marginal cost is the same number to 1e-12 relative,
  and no lower bound is active;
* **(c) perturbation** — 200 random feasible interior directions at two scales,
  all uphill;
* **(d) independence** — a second solver that shares no code path with the first
  reproduces ``x*`` to 1e-10 of ``X``;
* **(e) the Phase-1 limit** — the same solver at exponent 1, handed the tangent
  charge, returns ``optimal_trajectory`` to 1e-12 of ``X``.

(e) is the one that makes the rest cheap to believe. The two worlds are the same
problem at different exponents; if the machinery were wrong, it would be wrong in
the world where the right answer is a closed form M0 already pinned against the
vendored goldens.

The band, and why it is stated differently from M2's
----------------------------------------------------
Phase 1's Hessian is constant in ``x``, so
``|d|_2 <= sqrt(2 delta / lambda_min(H))`` is a global bound and M2 asserted the
quadratic inequality directly. Here the curvature of ``w ** 1.6`` is
``w ** -0.4``, so the Hessian moves with the schedule and the same expression is
a statement **at the optimum**. It is validated the honest way — by evaluating
the objective on random directions at the band radius — rather than by asserting
an inequality that no longer holds globally.

Thresholds come from ``configs/m4a_differential.yaml``. No scipy: the optimum is
dense numpy and a scalar bisection.
"""

from __future__ import annotations

import numpy as np
import pytest

from temper.oracle import (
    BPS,
    VENDOR_LAMBDA_GRID,
    cost_moments,
    kkt_residual,
    local_curvature_floor,
    objective_curvature_floor,
    optimal_trajectory,
    optimum_by_shooting,
    optimum_for_charge,
    power_law_charge,
    power_law_optimum,
    power_law_optimum_by_shooting,
    tangent_charge,
    trades,
    twap_trajectory,
    varying_objective_bps,
)
from temper.oracle.powerlaw import charge_hessian, marginal_costs

from .conftest import M4A_CONFIG, case_by_id

CERTIFICATE = M4A_CONFIG["certificate"]
CASES = [case_by_id(case_id) for case_id in CERTIFICATE["cases"]]
GRID_CASE = case_by_id(CERTIFICATE["grid_case"])
KKT_RTOL = float(CERTIFICATE["kkt_rtol"])
DIRECTIONS = int(CERTIFICATE["directions"])
SCALES = [float(scale) for scale in CERTIFICATE["scales"]]
TOLERANCE_REL = float(CERTIFICATE["tolerance_rel"])
INDEPENDENT_RTOL = float(CERTIFICATE["independent_rtol"])
TANGENT_RTOL = float(CERTIFICATE["tangent_rtol"])
BAND_DIRECTIONS = int(CERTIFICATE["band_directions"])
HESSIAN_RTOL = float(CERTIFICATE["hessian_cross_check_rtol"])
SEED = int(CERTIFICATE["seed"])


#: What each part actually observed, so the certificate reports itself rather
#: than only passing — M1 task 0's habit, and the reason its acceptance could say
#: "the generic solve matched to 4e-16 of X on nine cases" instead of "green".
_OBSERVED: dict[str, list[float]] = {
    "(a) Cholesky": [],
    "(b) KKT residual": [],
    "(c) perturbation": [],
    "(d) shooting vs Newton": [],
    "(e) tangent limit": [],
    "band, evaluated": [],
    "band, flattest direction": [],
}
_COUNTS: dict[str, int] = dict.fromkeys(_OBSERVED, 0)

#: Parts where the *largest* number seen is the worst one — distances from the
#: right answer. The rest are margins, where the smallest is worst.
_LARGEST_IS_WORST = {
    "(b) KKT residual",
    "(d) shooting vs Newton",
    "(e) tangent limit",
}


def _record(part: str, value: float, count: int = 1) -> None:
    _OBSERVED[part].append(value)
    _COUNTS[part] += count


@pytest.fixture(scope="module", autouse=True)
def report_certificate(request):
    """Print the five parts individually, with the worst number each saw."""
    yield
    if not any(_OBSERVED.values()):
        return
    writer = request.config.get_terminal_writer()
    writer.line("")
    writer.line(f"power-law certificate, {len(CASES)} cases:")
    units = {
        "(a) Cholesky": "smallest pivot",
        "(b) KKT residual": f"worst relative (band {KKT_RTOL:.0e})",
        "(c) perturbation": f"worst dJ / |J| (floor -{TOLERANCE_REL:.0e})",
        "(d) shooting vs Newton": f"worst, of X (band {INDEPENDENT_RTOL:.0e})",
        "(e) tangent limit": f"worst, of X (band {TANGENT_RTOL:.0e})",
        "band, evaluated": "worst dJ / delta at the band radius",
        "band, flattest direction": "dJ / delta where the bound is attained",
    }
    for part, values in _OBSERVED.items():
        if not values:
            continue
        worst = max(values) if part in _LARGEST_IS_WORST else min(values)
        writer.line(
            f"  {part:24s} green, {_COUNTS[part]:5d} checks   "
            f"{worst:+.3e}  {units[part]}"
        )


@pytest.fixture(params=CASES, ids=str)
def case(request):
    """The 3 x 3 golden grid the config names for the certificate."""
    return request.param


def _optimum(case):
    return power_law_optimum(case.market, case.order_size, case.lambda_risk)


def _charge(case):
    return power_law_charge(case.market, case.order_size)


def _objective(x, case) -> float:
    """``J(x)`` — the schedule-varying part of the power-law objective, in bps."""
    return varying_objective_bps(
        x, case.market, case.order_size, case.lambda_risk, _charge(case)
    )


# ---------------------------------------------------------------------------
# The objective this module minimises really is the oracle's
# ---------------------------------------------------------------------------


def test_the_varying_part_reproduces_the_oracles_power_law_objective(case):
    """``J(x) + constants == cost_moments(x).objective(lambda)``.

    Without this the rest of the module would certify a functional of its own
    invention. The constants are permanent cost and the half-spread, both fixed
    for any monotone schedule that fully liquidates — the same two M1 task 0
    dropped, and dropped for the same reason.
    """
    market, order_size = case.market, case.order_size
    constant = (
        market.params.gamma * market.params.sigma * BPS * order_size
        / (2.0 * market.v_hourly)
        + market.params.half_spread
    )
    schedules = [
        _optimum(case),
        twap_trajectory(market, order_size),
        optimal_trajectory(market, order_size, case.lambda_risk),
        np.asarray(case.ac["trajectory"], dtype=float),
    ]
    for schedule in schedules:
        assert np.all(trades(schedule, market) >= -1e-9 * order_size)
        oracle = cost_moments(schedule, market).objective(case.lambda_risk)
        assert _objective(schedule, case) + constant == pytest.approx(oracle, rel=1e-12)


# ---------------------------------------------------------------------------
# (a) convexity
# ---------------------------------------------------------------------------


def test_the_hessian_at_the_optimum_is_positive_definite(case):
    """Cholesky succeeds: the stationary point is the unique global minimum.

    ``w ** 1.6`` is strictly convex on ``w >= 0`` and the inventory term is a
    convex quadratic, so this is expected — but expected is not measured, and the
    condition number it records is the number the band divides by.
    """
    optimum = _optimum(case)
    hessian = charge_hessian(
        optimum, case.market, case.order_size, case.lambda_risk, _charge(case)
    )
    factor = np.linalg.cholesky(hessian)  # raises LinAlgError if not PD
    assert np.all(np.diag(factor) > 0.0)
    assert np.allclose(factor @ factor.T, hessian, rtol=1e-12, atol=0.0)
    assert np.allclose(hessian, hessian.T, rtol=0.0, atol=0.0), "the Hessian is not symmetric"
    _record("(a) Cholesky", float(np.min(np.diag(factor))))


def test_the_hessian_assembly_reproduces_phase_ones_curvature_floor(case):
    """The cheap cross-check that the new assembly is right.

    Hand the *same* assembly the tangent charge at exponent 1 and it must return
    Phase 1's ``objective_curvature_floor`` — a closed form
    (``4A(1 - cos(pi/N)) + 2B``) that predates this module and was derived a
    completely different way. A transcription error in the tridiagonal, the
    factor of ``(1+beta)beta``, or the ``X^2`` scaling would show up here rather
    than as a quietly wrong band three tasks later.
    """
    market, order_size, lam = case.market, case.order_size, case.lambda_risk
    tangent = optimum_for_charge(market, order_size, lam, tangent_charge(market, order_size))
    mine = local_curvature_floor(
        tangent, market, order_size, lam, tangent_charge(market, order_size)
    )
    theirs = objective_curvature_floor(market, order_size, lam)
    assert mine == pytest.approx(theirs, rel=HESSIAN_RTOL)


def test_the_cross_check_recovers_m3s_committed_band():
    """And the same assembly, at M3's committed lambda, gives M3's committed band.

    28 797 shares is a number in ``results/m3_frontier/``. Reproducing it from the
    M4a Hessian is a stronger statement than matching a closed form: it says the
    new machinery lands on a figure a previous milestone already published.
    """
    from temper.eval.reference import reference_row

    market, order_size = GRID_CASE.market, GRID_CASE.order_size
    lam = 10.0**-3.5
    row = reference_row(market, order_size, lam)
    delta = 0.05 * (row.twap.objective - row.optimal.objective)
    tangent = optimum_for_charge(market, order_size, lam, tangent_charge(market, order_size))
    floor = local_curvature_floor(
        tangent, market, order_size, lam, tangent_charge(market, order_size)
    )
    band = float(np.sqrt(2.0 * delta / floor))
    assert band == pytest.approx(28797.0, abs=1.0), (
        f"the M4a Hessian gives a Phase-1 band of {band:.1f} shares at "
        "lambda = 10^-3.5, not M3's committed 28 797"
    )


# ---------------------------------------------------------------------------
# (b) KKT
# ---------------------------------------------------------------------------


def test_every_bin_pays_the_same_marginal_cost(case):
    """Stationarity on the simplex *is* "the marginal cost is equal everywhere".

    Selling a share now pays the impact of a bigger bin; holding it pays one more
    bin of variance on everything still outstanding. The optimum is where those
    two prices agree in every bin, and the residual is how far from agreeing they
    are, relative to the size of the two terms being differenced.
    """
    optimum = _optimum(case)
    residual = kkt_residual(
        optimum, case.market, case.order_size, case.lambda_risk, _charge(case)
    )
    assert residual <= KKT_RTOL, (
        f"{case.case_id}: relative KKT residual {residual:.3e} exceeds "
        f"{KKT_RTOL:.0e}; the marginal costs do not agree across bins"
    )
    _record("(b) KKT residual", residual, count=case.market.n_bins)


def test_no_lower_bound_is_active_at_the_optimum(case):
    """Every bin trades a strictly positive amount, so the KKT check is complete.

    The stationarity condition above is the whole of optimality *only* where no
    ``w_i >= 0`` bound binds. If one did, its multiplier would enter and equal
    marginals would be the wrong condition — so this is not a nicety, it is what
    makes part (b) a proof rather than half of one.
    """
    optimum = _optimum(case)
    weights = trades(optimum, case.market) / case.order_size
    assert np.all(weights > 0.0), (
        f"{case.case_id}: the optimum has a bin at the lower bound "
        f"(min weight {float(np.min(weights)):.3e}); the equal-marginal condition "
        "is not sufficient there"
    )
    # And the marginals really are one number, not merely close in aggregate.
    marginal = marginal_costs(
        optimum, case.market, case.order_size, case.lambda_risk, _charge(case)
    )
    assert marginal.size == case.market.n_bins


# ---------------------------------------------------------------------------
# (c) perturbation
# ---------------------------------------------------------------------------


def test_no_perturbation_of_the_optimum_lowers_the_objective(case):
    """200 random interior directions at two scales, all uphill.

    The scales are fractions of X rather than of the local inventory, for M1 task
    0's reason: a front-loaded optimum has late bins of a few shares, and a
    perturbation scaled to *those* moves the objective by less than a double can
    represent, which would make this pass on rounding noise instead of on
    convexity. Directions are rejected if they would make a bin trade negative —
    that is outside the reachable set, not a cheaper schedule.
    """
    optimum = _optimum(case)
    best = _objective(optimum, case)
    floor = -TOLERANCE_REL * abs(best)

    rng = np.random.default_rng(SEED)
    interior = case.market.n_bins - 1
    for scale in SCALES:
        directions = rng.normal(size=(DIRECTIONS, interior))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        directions *= scale * case.order_size

        worst, checked = np.inf, 0
        for direction in directions:
            perturbed = optimum.copy()
            perturbed[1:-1] += direction
            if np.any(np.diff(perturbed) > 0.0):
                continue  # buys back: outside the reachable set
            worst = min(worst, _objective(perturbed, case) - best)
            checked += 1
        assert checked > 0, f"every direction at |d| = {scale:g}X was infeasible"
        assert worst >= floor, (
            f"{case.case_id}: a perturbation at |d| = {scale:g}X lowered the "
            f"objective by {-worst:.3e} bps (floor {floor:.3e})"
        )
        _record("(c) perturbation", worst / abs(best), count=checked)


# ---------------------------------------------------------------------------
# (d) independence
# ---------------------------------------------------------------------------


def test_an_independent_solver_reproduces_the_optimum(case):
    """Bisection on the equal-marginal condition, against damped Newton.

    Deliberately unlike each other: the shooting solver has no Hessian, no linear
    solve and no line search — it takes a trial first-bin weight, walks the
    marginal down the schedule bin by bin, and bisects on the inventory left at
    the horizon. Two implementations agreeing to 1e-10 of X is evidence about the
    answer; one implementation agreeing with itself is not.
    """
    newton = _optimum(case)
    shot = power_law_optimum_by_shooting(
        case.market, case.order_size, case.lambda_risk
    )
    worst = float(np.max(np.abs(newton - shot))) / case.order_size
    assert worst <= INDEPENDENT_RTOL, (
        f"{case.case_id}: the two solvers differ by {worst:.3e} of X "
        f"(band {INDEPENDENT_RTOL:.0e})"
    )
    _record("(d) shooting vs Newton", worst, count=case.market.n_bins - 1)


def test_the_independent_solver_agrees_across_the_whole_committed_grid():
    """Every lambda the reference table reads, not only the certificate's nine.

    Task 0's table calls the optimum at all seventeen grid points and applies the
    selection rule to what comes back, so a solve that failed at one of them
    would move a *gate* rather than a footnote.
    """
    market, order_size = GRID_CASE.market, GRID_CASE.order_size
    for lam in VENDOR_LAMBDA_GRID:
        newton = power_law_optimum(market, order_size, lam)
        shot = power_law_optimum_by_shooting(market, order_size, lam)
        worst = float(np.max(np.abs(newton - shot))) / order_size
        assert worst <= INDEPENDENT_RTOL, (
            f"lambda = {lam:.3e}: the two solvers differ by {worst:.3e} of X"
        )
        residual = kkt_residual(
            newton, market, order_size, lam, power_law_charge(market, order_size)
        )
        assert residual <= KKT_RTOL, (
            f"lambda = {lam:.3e}: relative KKT residual {residual:.3e}"
        )


# ---------------------------------------------------------------------------
# (e) the Phase-1 limit
# ---------------------------------------------------------------------------


def test_the_same_solver_at_exponent_one_returns_the_sinh(case):
    """Replace the power law with its tangent and the closed form comes back.

    One test, and the two worlds are demonstrably the same machinery at different
    exponents. It is also the tightest check in this module that the solver is
    right at all, because the answer it must reproduce is a formula M0 pinned
    against the vendored goldens before any of this existed.
    """
    market, order_size, lam = case.market, case.order_size, case.lambda_risk
    solved = optimum_for_charge(market, order_size, lam, tangent_charge(market, order_size))
    closed = optimal_trajectory(market, order_size, lam)
    worst = float(np.max(np.abs(solved - closed))) / order_size
    assert worst <= TANGENT_RTOL, (
        f"{case.case_id}: the tangent solve and the closed form differ by "
        f"{worst:.3e} of X (band {TANGENT_RTOL:.0e})"
    )
    _record("(e) tangent limit", worst, count=market.n_bins - 1)


def test_the_tangent_limit_holds_across_the_whole_committed_grid():
    """At every lambda the frontier grid visits, including its degenerate ends."""
    market, order_size = GRID_CASE.market, GRID_CASE.order_size
    charge = tangent_charge(market, order_size)
    for lam in VENDOR_LAMBDA_GRID:
        solved = optimum_for_charge(market, order_size, lam, charge)
        closed = optimal_trajectory(market, order_size, lam)
        worst = float(np.max(np.abs(solved - closed))) / order_size
        assert worst <= TANGENT_RTOL, (
            f"lambda = {lam:.3e}: tangent solve vs closed form {worst:.3e} of X"
        )


def test_the_two_worlds_disagree_where_they_should():
    """Non-vacuity for (e): the exponent is doing something.

    If the power-law optimum and the tangent's were the same schedule, part (e)
    would pass for the wrong reason and there would be no milestone. At the
    reference lambda they are 16 878 shares apart — 16.9 % of the parent order.
    """
    market, order_size = GRID_CASE.market, GRID_CASE.order_size
    lam = 10.0**-3.5
    separation = float(
        np.linalg.norm(
            power_law_optimum(market, order_size, lam)[1:-1]
            - optimal_trajectory(market, order_size, lam)[1:-1]
        )
    )
    assert separation == pytest.approx(16878.0, abs=2.0), (
        f"the two optima are {separation:.1f} shares apart at lambda = 10^-3.5, "
        "not the 16 878 the brief pre-stated"
    )


# ---------------------------------------------------------------------------
# The band, stated as local and validated by evaluation
# ---------------------------------------------------------------------------


def test_the_derived_band_is_validated_by_direct_evaluation(case):
    """At the band radius the objective really has risen by about ``delta``.

    Phase 1 could assert ``d' H d / 2 >= lambda_min |d|^2 / 2`` and be done: the
    Hessian was constant, so the inequality was exact everywhere. Here it is not,
    so the band is checked the way a local statement has to be — evaluate ``J`` on
    random directions at the radius and require the rise to be at least the
    ``delta`` the band was derived from, up to the quadratic approximation's own
    error at that distance.

    The direction is the point of the test as much as the magnitude. The bound is
    attained along the flattest eigenvector, so a schedule *at* the band radius
    may cost exactly ``delta`` more and no direction may cost less than the
    quadratic predicts by more than the cubic term.

    **Where the reachable set runs out first.** At the top of the lambda grid the
    optimum front-loads hard — 63.8 % of the order in bin 0 at ``lambda = 1e-3``,
    with late bins of a thousandth of it — and *every* random direction at the
    full band radius makes some bin buy back, which
    :class:`~temper.env.ExecutionEnv`'s clip to ``[0, remaining]`` forbids. That
    is not a failure of the band; it is the band being larger than the set the
    agent can move in, which makes it a *conservative* bound rather than a wrong
    one. Each direction is therefore walked in to the largest radius the
    reachable set admits and the quadratic is checked there, against a prediction
    that scales as the square of how far it got.
    """
    market, order_size, lam = case.market, case.order_size, case.lambda_risk
    optimum = _optimum(case)
    best = _objective(optimum, case)
    floor = local_curvature_floor(optimum, market, order_size, lam, _charge(case))

    # A delta small enough that the quadratic still describes the bowl: 5 % of the
    # available advantage at this case, which is M4a's own median bar.
    from temper.eval.reference import reference_row

    row = reference_row(market, order_size, lam, encoding="power_law")
    delta = 0.05 * row.available_advantage
    radius = float(np.sqrt(2.0 * delta / floor))

    rng = np.random.default_rng(SEED + 1)
    interior = market.n_bins - 1
    directions = rng.normal(size=(BAND_DIRECTIONS, interior))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    ratios, reached = [], []
    for direction in directions:
        step = 1.0
        for _ in range(60):
            perturbed = optimum.copy()
            perturbed[1:-1] += step * radius * direction
            if np.all(np.diff(perturbed) <= 0.0):
                break
            step *= 0.5
        else:
            continue
        # The quadratic prediction at `step * radius` is `delta * step**2`, since
        # `delta` was defined at `step = 1` along the flattest direction.
        ratios.append((_objective(perturbed, case) - best) / (delta * step**2))
        reached.append(step)
    assert ratios, f"{case.case_id}: no feasible direction at any radius"

    smallest = min(ratios)
    assert smallest >= 0.99, (
        f"{case.case_id}: a direction at {min(reached):.3g} of the band radius "
        f"raised the objective by only {smallest:.4f} of what the local quadratic "
        "predicts; the bound does not hold there"
    )
    _record("band, evaluated", smallest, count=len(ratios))

    # Random directions establish that the bound *holds*; they cannot establish
    # that it is *tight*, because a random unit vector in twelve dimensions
    # samples the mean eigenvalue rather than the smallest — which is why the
    # worst random ratio above is comfortably above one rather than near it. The
    # bound is attained along the flattest eigenvector, so that direction is
    # walked deliberately: there the rise must be about `delta` and not far more,
    # or the band is loose and every trajectory tolerance derived from it is
    # vacuous.
    #
    # "About" is the local statement, and the measurement is the interesting
    # part. In Phase 1 this ratio would be exactly 1 — the objective *is* the
    # quadratic. Here the worst case on the golden grid is 0.988 at
    # lambda = 1e-3, i.e. the true objective rises 1.2 % *less* than the Hessian
    # at x* predicts, because the third-order term curves away over a radius that
    # is 4.7 % of the parent order. That the deviation is small and downward is
    # exactly what makes the band a usable local bound; a ratio far from 1 in
    # either direction would mean the radius had left the neighbourhood the
    # Hessian describes, and the derived tolerance with it.
    hessian = charge_hessian(optimum, market, order_size, lam, _charge(case))
    flattest = np.linalg.eigh(hessian)[1][:, 0]
    for sign in (1.0, -1.0):
        step = 1.0
        for _ in range(60):
            perturbed = optimum.copy()
            perturbed[1:-1] += sign * step * radius * flattest
            if np.all(np.diff(perturbed) <= 0.0):
                break
            step *= 0.5
        else:
            continue
        attained = (_objective(perturbed, case) - best) / (delta * step**2)
        assert 0.95 <= attained <= 1.05, (
            f"{case.case_id}: along the flattest eigenvector at {step:.3g} of the "
            f"band radius the objective rose by {attained:.4f} of delta; the "
            "quadratic no longer describes the bowl at that radius"
        )
        _record("band, flattest direction", attained)


def test_the_band_is_labelled_local_in_the_power_law_world():
    """A global bound and a local one are different claims; the type says which.

    Reporting a local bound as though the Hessian were constant is the quiet
    version of the mistake this whole milestone is about — using a linearisation
    outside where it holds.
    """
    from temper.eval.reference import trajectory_band

    market, order_size = GRID_CASE.market, GRID_CASE.order_size
    lam = 10.0**-3.5
    power = trajectory_band(market, order_size, lam, 0.0018371, encoding="power_law")
    linear = trajectory_band(market, order_size, lam, 0.0018371)
    assert power.local is True
    assert linear.local is False
    assert power.encoding == "power_law"
    # The power-law bowl is *sharper* at its optimum than Phase 1's is anywhere,
    # so the same objective excess confines the schedule slightly more tightly.
    assert power.curvature_floor > linear.curvature_floor


def test_the_bands_predicted_numbers_are_what_the_oracle_returns():
    """The brief pre-stated these; task 0 regenerates them on the reference box.

    Reproduced here rather than only in the reference table because they are the
    numbers the milestone's third gate is read off: the band a schedule meeting
    the median bar may sit in, against the distance the Almgren–Chriss schedule
    actually sits from the optimum.
    """
    from temper.eval.reference import reference_row, trajectory_band

    market, order_size = GRID_CASE.market, GRID_CASE.order_size
    lam = 10.0**-3.5
    row = reference_row(market, order_size, lam, encoding="power_law")
    optimum = row.optimal.trajectory
    floor = local_curvature_floor(
        optimum, market, order_size, lam, power_law_charge(market, order_size)
    )
    assert floor == pytest.approx(1.636e-10, rel=1e-3)

    hessian = charge_hessian(
        optimum, market, order_size, lam, power_law_charge(market, order_size)
    )
    eigenvalues = np.linalg.eigvalsh(hessian)
    assert eigenvalues[-1] / eigenvalues[0] == pytest.approx(34.6, rel=1e-2)

    band = trajectory_band(
        market, order_size, lam, 0.05 * row.available_advantage, encoding="power_law"
    )
    assert band.bound_shares == pytest.approx(4739.0, abs=2.0)
    separation = float(
        np.linalg.norm(row.tangent.trajectory[1:-1] - optimum[1:-1])
    )
    assert separation / band.bound_shares == pytest.approx(3.56, abs=0.05), (
        "the band and the AC separation are no longer a factor of 3.6 apart; "
        "task 0's third gate was read off that ratio"
    )
