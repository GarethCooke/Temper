"""Market models behind one Gymnasium interface.

Phase 1 (`execution_env`) is arithmetic Brownian motion with linear permanent and
linear temporary impact — the world the Almgren–Chriss closed form solves exactly,
so the oracle can grade it. Phase-2 models (transient impact with decay, stochastic
liquidity, alpha) arrive as *additive alternatives* behind the same interface, never
as silent modifications of this one (constitution §4).

Pure numpy: no torch below this package, and no import of
:func:`~temper.oracle.cost.cost_moments` — Phase 1 is the linearised world
end-to-end (``ARCHITECTURE.md`` §9, 2026-08-04) and both are enforced by
``tests/test_repo_invariants.py``.
"""

from .execution_env import EPISODE_KEY, SHOCK_KEY, ExecutionEnv

__all__ = ["EPISODE_KEY", "SHOCK_KEY", "ExecutionEnv"]
