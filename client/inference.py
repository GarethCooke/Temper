"""The trained policy, in numpy — a second implementation of one forward pass.

`temper/agents/` runs the policy through torch, because that is what trained it.
This runs the same weights through `numpy.tanh` and three matrix multiplies, and
the whole value of having two is that something proves they agree: the committed
checkpoint carries the observations and actions the torch policy produced on
M4a's evaluation episode, and `tests/test_client_inference.py` replays them
through this code and requires a match to float tolerance — with no network, no
env and no torch anywhere in the test.

Why not simply import the training stack. Three reasons, in the order they
matter. `client/` is a *participant*: it should be installable and runnable
without the machinery that produced the policy, which is the seam
`ARCHITECTURE.md` §3 already draws between the two. A checkpoint that only torch
can read is a checkpoint that cannot become the backlog's C++/ONNX inference leg
without being rewritten first, and this is that leg's first step rather than a
detour from it. And a policy that runs on two independent implementations is a
policy whose *file format* has been tested, not merely its weights.

Arithmetic is float32 throughout, matching `temper.agents.ppo.AGENT_DTYPE`.
Promoting to float64 would be more accurate and *less* faithful: the number to
reproduce is the one the trained network emits, not the one a better-conditioned
version of it would.

The network is read from the file rather than assumed. `metadata["network"]`
lists the actor's layers in order, so this loop works for any depth and any
`hidden_sizes` without a constant of its own — the reason the checkpoint stores
its own layer names instead of torch's `nn.Sequential` indices.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: The checkpoint schema this reader understands. Duplicated deliberately from
#: `temper.agents.checkpoint.POLICY_FORMAT` — importing it would put the
#: training stack back on this module's import path, which is the one thing the
#: module exists to avoid. `tests/test_client_inference.py` asserts the two
#: constants agree, so the duplication is checked rather than trusted.
POLICY_FORMAT = "temper-policy-1"

#: The dtype the network was trained and evaluated in.
DTYPE = np.float32


@dataclass(frozen=True)
class Policy:
    """A trained execution policy: observation in, fraction of remaining out.

    Stateless and pure. The observation is `(time remaining fraction, inventory
    remaining fraction)` and carries no price, no volatility and no impact
    parameter — which is exactly why these weights can work an order on a venue
    they have never seen (M6 brief, context §1). Nothing in this class knows what
    a share is worth.
    """

    layers: tuple[tuple[np.ndarray, np.ndarray], ...]
    metadata: dict
    source: Path | None = None

    @property
    def name(self) -> str:
        return str(self.metadata.get("name", "policy"))

    @property
    def n_bins(self) -> int:
        """The grid the policy was trained on — 13 bins, and it is not a free choice.

        The observation's clock is `1 - k/N`, so a policy asked for its action on
        a different number of bins is being asked off-distribution. The wall-clock
        *length* of a bin is free (the M6 brief compresses 6.5 hours to minutes);
        the count is not.
        """
        return int(self.metadata["world"]["n_bins"])

    def fraction(self, time_remaining: float, inventory_remaining: float) -> float:
        """The fraction of *remaining* inventory to work this bin, in ``[0, 1]``.

        Both arguments are fractions in ``[0, 1]``. The squash is a clip, not a
        sigmoid, matching `temper.agents.execution.as_fraction`: the policy's
        density lives on the unsquashed action, and a clip reaches the endpoints
        — the last bin's TWAP fraction is exactly 1, which a sigmoid could only
        approach.
        """
        activation = np.array(
            [time_remaining, inventory_remaining], dtype=DTYPE
        ).reshape(-1)
        for index, (weight, bias) in enumerate(self.layers):
            activation = weight @ activation + bias
            if index < len(self.layers) - 1:
                activation = np.tanh(activation)
        return float(np.clip(activation[0], 0.0, 1.0))

    def schedule(self, n_bins: int | None = None) -> np.ndarray:
        """The open-loop fractions the policy produces if every bin fills exactly.

        The schedule it *would* run against a venue that behaves like the
        simulator — which no venue does. Useful for the pre-run prediction and
        for reporting what the policy asked for beside what it got; the live
        client never uses it, because the point of a closed loop is that bin
        `k + 1`'s observation carries the inventory bin `k` actually left.
        """
        bins = self.n_bins if n_bins is None else int(n_bins)
        remaining = 1.0
        fractions = np.empty(bins, dtype=np.float64)
        for step in range(bins):
            fractions[step] = self.fraction(1.0 - step / bins, remaining)
            remaining -= fractions[step] * remaining
        return fractions


def load_policy(path: str | Path) -> Policy:
    """Read a committed policy `.npz`, refusing any format but this one.

    `allow_pickle` stays at its default `False`: loading a policy must not be
    able to execute one.
    """
    target = Path(path)
    with np.load(target) as handle:
        metadata = json.loads(str(handle["metadata"]))
        if metadata.get("format") != POLICY_FORMAT:
            raise ValueError(
                f"{target} declares format {metadata.get('format')!r}, not "
                f"{POLICY_FORMAT!r}; refusing to interpret its arrays"
            )
        network = metadata["network"]
        if network.get("activation") != "tanh":
            raise ValueError(
                f"{target} was trained with a {network.get('activation')!r} "
                "activation; this reader implements tanh only"
            )
        layers = tuple(
            (
                np.array(handle[weight], dtype=DTYPE),
                np.array(handle[bias], dtype=DTYPE),
            )
            for weight, bias in network["actor_layers"]
        )
    _check_shapes(layers, network, target)
    return Policy(layers=layers, metadata=metadata, source=target)


def _check_shapes(layers, network: dict, source: Path) -> None:
    """The stored layers compose into `obs_dim -> ... -> action_dim`, or refuse.

    Cheap, and it is the difference between a mis-read file failing here and a
    mis-read file producing plausible fractions on a live venue.
    """
    width = int(network["obs_dim"])
    for index, (weight, bias) in enumerate(layers):
        out_features, in_features = weight.shape
        if in_features != width or bias.shape != (out_features,):
            raise ValueError(
                f"{source}: actor layer {index} has shape {weight.shape} with "
                f"bias {bias.shape}, which does not accept {width} inputs"
            )
        width = out_features
    if width != int(network["action_dim"]) or width != 1:
        raise ValueError(
            f"{source}: the actor produces {width} outputs; an execution policy "
            "produces exactly one, the fraction of remaining inventory"
        )


def load_pin(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The checkpoint's evaluation arrays: observations, fractions, trajectory.

    What the *torch* policy did on M4a's evaluation episode, carried inside the
    same archive as the weights so the pin cannot go missing. This is the only
    reason `client/` can assert agreement with an implementation it does not
    import.
    """
    with np.load(Path(path)) as handle:
        return (
            np.array(handle["eval_observations"], dtype=np.float64),
            np.array(handle["eval_fractions"], dtype=np.float64),
            np.array(handle["eval_trajectory"], dtype=np.float64),
        )
