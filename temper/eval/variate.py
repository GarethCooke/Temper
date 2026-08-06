"""The sanctioned control variate: subtract the noise, keep the cost.

**This is M2 task 3's fallback, not its default.** The default is vanilla PPO on
sampled rewards, because that is the honest claim — *reinforcement learning under
realistic execution noise recovers Almgren–Chriss*. What is here weakens that
claim to *reinforcement learning optimises a deterministic function*, and may
only be used with an amendment recorded in ``docs/briefs/M2-ppo-rediscovery.md``
**before** the run it is used for, with the headline restated in ``results/`` and
in the figure caption. Nothing in this module decides that; the experiment
driver does, from the committed config.

What it does
------------
M1a pinned the exact noise identity: with the shock landing before each bin,
one episode's realised cost differs from its expectation by

.. code::

    C - E[cost] = - sum_k (n_k / X) * walk_k

where ``walk_k`` is the cumulative price shock bin ``k`` executed against and
``n_k`` the shares it traded (``ARCHITECTURE.md`` §9, *The shock lands before the
bin executes...*). Both are published per step in the env's ``info``, so the
noise is not estimated — it is *known*, term by term, and subtracting it leaves
the deterministic cost exactly. Reward variance goes to zero rather than merely
down, which is why this is a control variate with a coefficient of one and no
regression: ``tests/test_m2_variate.py`` pins that two unrelated shock streams
produce the same rewards under it to within a couple of ulps, some fifteen orders
of magnitude below the noise removed. Not bitwise, and for a stated reason — the
env forms ``weight * price_bps`` on the summed price, so subtracting
``weight * walk`` afterwards rounds differently from forming
``weight * (price_bps - walk)`` in one operation. Closing that last ulp would
mean editing the env, which M2 may not do.

Under Phase-1 certainty equivalence the optimal policy is unchanged — the agent
is trained on the *expected* reward, whose minimiser is the same deterministic
sinh trajectory — so this does not alter what is being rediscovered. It alters
what the sentence about noise is allowed to say.

Why it lives in ``temper/eval/`` and not next to the training loop
------------------------------------------------------------------
Because computing it means reading the env's shock key, and
``tests/test_repo_invariants.py`` rejects that key — by literal *or* by name —
anywhere under ``temper/agents/``. That guard is M1a's, it closed M2's
observation leak before M2 existed, and it stays green. The variate is an
*estimator*, not a policy: the observation stays two-dimensional, the eval policy
never sees a shock, and the transform is handed to
:func:`~temper.agents.ppo.train` as a parameter by the driver. Putting it here
keeps the seam visible instead of making the guard something to be worked around.
"""

from __future__ import annotations

import numpy as np
from gymnasium import Env, Wrapper

from temper.env import SHOCK_KEY

#: The ``info`` key the env publishes the bin's realised trade under.
SHARES_KEY = "shares"


class DeterministicReward(Wrapper):
    """Replace the reward with its noise-free part, exactly.

    Wrap *inside* any reward scaling: this works in the env's own bps and a
    scale applied first would have to be undone here, which is one more place
    for the two constants to disagree.
    """

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.order_size = float(env.unwrapped.order_size)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        noise = (float(info[SHARES_KEY]) / self.order_size) * float(info[SHOCK_KEY])
        return observation, float(reward) - noise, terminated, truncated, info


def deterministic_reward(env: Env) -> Env:
    """`reward_wrapper` form, for :func:`~temper.agents.execution.execution_env_factory`."""
    return DeterministicReward(env)


def noise_component(shares, walks, order_size: float) -> float:
    """``sum_k (n_k / X) * walk_k`` for a whole episode — the identity, in one place.

    Used by the tests to check the wrapper against the episode-level statement
    rather than against a re-typed copy of its own per-step line.
    """
    n = np.asarray(shares, dtype=float)
    w = np.asarray(walks, dtype=float)
    if n.shape != w.shape:
        raise ValueError(f"shares and walks must align, got {n.shape} and {w.shape}")
    return float(np.sum(n * w) / order_size)
