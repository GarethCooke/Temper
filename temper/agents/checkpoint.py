"""A trained policy, written down — the artefact this repo did not keep.

Through M4a every network the project trained was discarded the moment it had
been graded. Results carry trajectories, grades and provenance, which is enough
to *check* a claim and not enough to *re-examine* one: no policy could be
inspected after the fact, no reported schedule re-derived from the thing that
produced it, and the backlog's C++/ONNX inference leg was impossible by
construction. This module is the one-line answer — a committed `.npz` beside the
committed results JSON, carrying the weights, the address they were trained at,
the grade they earned, and enough of the run's provenance to tie the three
together.

Three properties are deliberate.

**It is `.npz`, not `torch.save`.** A pickle is a program, its contents are
opaque to anything but torch, and reading one requires the training stack.
``np.load(path)`` needs numpy and nothing else, which is what lets ``client/``
run this policy on a venue without importing torch (constitution §3's seam, and
M6 task 1). The arrays are stored in the dtype they were trained in — float32,
:data:`~temper.agents.ppo.AGENT_DTYPE` — so a round trip is exact rather than
nearly exact.

**The layer names are the checkpoint's, not torch's.** ``nn.Sequential``'s
``0.weight`` / ``2.weight`` indices encode where the ``Tanh`` modules sit, which
is an implementation detail of how the MLP happens to be assembled. A reader
that hard-coded them would break on a network of a different depth for no reason
that matters. The metadata lists the actor's layers in order, so a forward pass
is a loop over that list and works for any ``hidden_sizes``.

**The metadata carries the grade and the evaluation trajectory.** A committed
binary that nothing verifies is not an artefact this repo keeps: with the
schedule and the graded objective stored beside the weights,
``tests/test_policy_checkpoint.py`` can pin the checkpoint to the committed
sweep without retraining anything, and ``client/`` can pin a second
implementation of the forward pass without a network or a torch import.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from temper.agents.execution import PPOPolicy
from temper.agents.ppo import AGENT_DTYPE, Agent

#: The checkpoint schema's name and version, stored in the metadata and checked
#: on load. A reader that does not recognise it must refuse rather than guess:
#: the arrays are unlabelled floats and a mis-read is a silently wrong policy.
POLICY_FORMAT = "temper-policy-1"

#: Metadata key inside the ``.npz``. A single JSON document rather than a
#: scattering of scalar arrays, so the stamp stays readable with ``np.load`` and
#: ``json.loads`` alone and gains fields without a format bump.
METADATA_KEY = "metadata"


@dataclass(frozen=True)
class PolicyCheckpoint:
    """The contents of one ``.npz``: named arrays plus the metadata document."""

    arrays: dict[str, np.ndarray]
    metadata: dict
    path: Path | None = None

    @property
    def actor_layers(self) -> list[tuple[str, str]]:
        """``[(weight_key, bias_key), ...]`` for the policy mean net, in order."""
        return [tuple(pair) for pair in self.metadata["network"]["actor_layers"]]

    @property
    def grade(self) -> dict:
        """The graded objective this checkpoint's seed earned, as committed."""
        return self.metadata["grade"]

    @property
    def ordinal(self) -> int:
        return int(self.metadata["selection"]["ordinal"])


def _linear_layers(module: nn.Sequential) -> list[nn.Linear]:
    """The ``Linear`` layers of an MLP, in forward order."""
    return [layer for layer in module if isinstance(layer, nn.Linear)]


def _named_arrays(prefix: str, module: nn.Sequential) -> dict[str, np.ndarray]:
    """``{prefix}.w{i}`` / ``{prefix}.b{i}`` for each ``Linear``, as numpy.

    ``weight`` is stored in torch's ``(out, in)`` orientation, unchanged, so a
    reader computes ``W @ x + b`` and the arrays never need transposing on
    either side of the boundary.
    """
    arrays: dict[str, np.ndarray] = {}
    for index, layer in enumerate(_linear_layers(module)):
        arrays[f"{prefix}.w{index}"] = layer.weight.detach().cpu().numpy().copy()
        arrays[f"{prefix}.b{index}"] = layer.bias.detach().cpu().numpy().copy()
    return arrays


def policy_arrays(agent: Agent) -> dict[str, np.ndarray]:
    """Every trained parameter of `agent`, under the checkpoint's own names.

    The actor's mean net is the policy; the critic and ``log_std`` are stored
    beside it because they cost 17 kB and are what makes the difference between
    a saved *policy* and a saved *agent*. Only the actor is needed to act.
    """
    if not getattr(agent, "continuous", False):
        raise TypeError(
            "only the continuous (Gaussian-head) agent is a Temper execution "
            "policy; the categorical head is the CartPole smoke test's"
        )
    arrays = _named_arrays("actor", agent.head.mean)
    arrays |= _named_arrays("critic", agent.critic)
    arrays["log_std"] = agent.head.log_std.detach().cpu().numpy().copy()
    return arrays


def network_description(agent: Agent, hidden_sizes: Sequence[int]) -> dict:
    """How to run the actor, written down beside the weights.

    Everything a forward pass needs that is not a number in the file: the layer
    order, the activation between them, and what the output *means* at the
    environment boundary. The last of those is the one a second implementation
    gets wrong — the action is a fraction of *remaining* inventory and reaches
    the environment through a clip, not a sigmoid
    (:func:`~temper.agents.execution.as_fraction`).
    """
    layers = _linear_layers(agent.head.mean)
    return {
        "obs_dim": int(layers[0].in_features),
        "action_dim": int(layers[-1].out_features),
        "hidden_sizes": [int(width) for width in hidden_sizes],
        "activation": "tanh",
        "actor_layers": [
            [f"actor.w{index}", f"actor.b{index}"] for index in range(len(layers))
        ],
        "output": (
            "the Gaussian head's mean, taken as the deterministic evaluation "
            "action; clipped to [0, 1] and multiplied by remaining inventory "
            "(temper.agents.execution.as_fraction / fraction_to_shares)"
        ),
        "observation": "(time remaining fraction, inventory remaining fraction)",
        "dtype": "float32",
    }


def save_policy(
    path: str | Path,
    agent: Agent,
    metadata: dict,
    extra: dict[str, np.ndarray] | None = None,
) -> Path:
    """Write `agent`'s parameters, `extra` and `metadata` to a ``.npz`` at `path`.

    `extra` is for arrays that are not parameters but belong in the same file —
    the evaluation observations, actions and trajectory this checkpoint is
    pinned against. They travel *inside* the archive rather than beside it,
    because a pin in a second file is a pin that can go missing.

    Uncompressed ``savez``: the file is ~50 kB either way and an uncompressed
    archive is byte-stable across numpy versions, which a committed artefact
    wants and a compressed one does not guarantee.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    document = dict(metadata)
    document["format"] = POLICY_FORMAT
    arrays = policy_arrays(agent)
    for key, value in (extra or {}).items():
        if key in arrays:
            raise ValueError(f"{key!r} collides with a parameter name")
        arrays[key] = np.asarray(value)
    if METADATA_KEY in arrays:  # pragma: no cover - defensive
        raise ValueError(f"{METADATA_KEY!r} collides with a parameter name")
    with target.open("wb") as handle:
        np.savez(handle, **arrays, **{METADATA_KEY: np.array(json.dumps(document))})
    return target


def load_checkpoint(path: str | Path) -> PolicyCheckpoint:
    """Read a ``.npz`` back, refusing any format but this one.

    ``allow_pickle`` stays at its default ``False``. The point of the format is
    that loading it executes nothing.
    """
    target = Path(path)
    with np.load(target) as handle:
        metadata = json.loads(str(handle[METADATA_KEY]))
        arrays = {
            key: np.array(handle[key]) for key in handle.files if key != METADATA_KEY
        }
    if metadata.get("format") != POLICY_FORMAT:
        raise ValueError(
            f"{target} declares format {metadata.get('format')!r}, not "
            f"{POLICY_FORMAT!r}; refusing to interpret its arrays"
        )
    return PolicyCheckpoint(arrays=arrays, metadata=metadata, path=target)


def agent_from_checkpoint(checkpoint: PolicyCheckpoint) -> Agent:
    """Rebuild the torch :class:`~temper.agents.ppo.Agent` a checkpoint holds.

    The shapes come from the arrays themselves rather than from a config, so
    this reads a checkpoint written by an experiment it knows nothing about —
    which is what makes it usable as the *independent* side of the pin in
    ``tests/test_policy_checkpoint.py``.
    """
    from gymnasium import spaces

    from temper.agents.ppo import PPOConfig

    network = checkpoint.metadata["network"]
    observation_space = spaces.Box(
        low=0.0, high=1.0, shape=(int(network["obs_dim"]),), dtype=np.float64
    )
    action_space = spaces.Box(
        low=0.0, high=1.0, shape=(int(network["action_dim"]),), dtype=np.float64
    )
    config = PPOConfig(
        hidden_sizes=tuple(int(width) for width in network["hidden_sizes"])
    )
    agent = Agent(observation_space, action_space, config)

    for prefix, module in (("actor", agent.head.mean), ("critic", agent.critic)):
        for index, layer in enumerate(_linear_layers(module)):
            for kind, parameter in (("w", "weight"), ("b", "bias")):
                key = f"{prefix}.{kind}{index}"
                array = checkpoint.arrays[key]
                stored = tuple(array.shape)
                expected = tuple(getattr(layer, parameter).shape)
                if stored != expected:
                    raise ValueError(
                        f"{key} has shape {stored} but the rebuilt layer wants "
                        f"{expected}; the checkpoint and this code disagree "
                        "about the network"
                    )
                with torch.no_grad():
                    getattr(layer, parameter).copy_(
                        torch.as_tensor(array, dtype=AGENT_DTYPE)
                    )
    with torch.no_grad():
        agent.head.log_std.copy_(
            torch.as_tensor(checkpoint.arrays["log_std"], dtype=AGENT_DTYPE)
        )
    agent.eval()
    return agent


def policy_from_checkpoint(
    checkpoint: PolicyCheckpoint, order_size: float, name: str | None = None
) -> PPOPolicy:
    """A checkpoint as a gradable policy, on the baselines' interface.

    `order_size` is the caller's, not the checkpoint's, and deliberately: the
    observation is scale-free, so the *same* weights work a parent order of any
    size, and a policy that silently insisted on the size it was trained at
    would hide the one property M6 depends on.
    """
    label = name or f"checkpoint:{checkpoint.metadata.get('name', 'policy')}"
    return PPOPolicy(agent_from_checkpoint(checkpoint), order_size, name=label)
