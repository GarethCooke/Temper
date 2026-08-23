"""Market models behind one Gymnasium interface.

`execution_env` is the one simulator and the one ``step`` loop. What changes
between phases is the *temporary-impact model* it is handed (`impact`), not the
loop: Phase 1 is arithmetic Brownian motion with linear permanent and linear
temporary impact — the world the Almgren–Chriss closed form solves exactly, so
the oracle can grade it — and M4a injects FrontierView's 0.6-power law instead.
Phase-2 models arrive as *additive alternatives behind the same interface*, never
as silent modifications of Phase 1, and never by default: an env or a config has
to name the world it wants (constitution §4).

Pure numpy: no torch below this package, and no import of
:func:`~temper.oracle.cost.cost_moments` — the env reaches its cost the long way
round, bin by bin, or M1's differential is checking the oracle against itself.
Both are enforced by ``tests/test_repo_invariants.py``.
"""

from .execution_env import EPISODE_KEY, LIQUIDITY_KEY, SHOCK_KEY, ExecutionEnv
from .impact import (
    LinearTemporary,
    PowerLawTemporary,
    TemporaryImpact,
    impact_for,
    linear_temporary,
    power_law_temporary,
)
from .liquidity import (
    DETERMINISTIC_LIQUIDITY,
    LiquidityStream,
    liquidity_stream,
)

__all__ = [
    "DETERMINISTIC_LIQUIDITY",
    "EPISODE_KEY",
    "LIQUIDITY_KEY",
    "LiquidityStream",
    "SHOCK_KEY",
    "ExecutionEnv",
    "LinearTemporary",
    "PowerLawTemporary",
    "TemporaryImpact",
    "impact_for",
    "linear_temporary",
    "liquidity_stream",
    "power_law_temporary",
]
