"""Persist a trained policy's forward pass, so a graded result can be re-examined.

Every network M2, M3 and M4a trained was graded, summarised into
``results/*.json`` and then dropped on the floor. What survives is a number and a
trajectory; what does not is the object that produced them. That is a gap in the
repo rather than a milestone's problem: invariant 1 says every reported number
regenerates from a committed config and seed, and it does — at twenty minutes a
seed — but "regenerable" and "inspectable" are not the same property, and nothing
here could answer "what does that agent do at an inventory it never visited?"
without spending the twenty minutes again.

So a sweep may now write, beside its metrics, the *policy* each seed arrived at.

What is written, and what is not. The file holds the actor mean's affine layers —
the whole of what :meth:`~temper.agents.ppo.Agent.deterministic_action` evaluates
— plus the head's ``log_std``, which is not used at evaluation and is recorded
because dropping half a distribution silently is worse than carrying four unused
bytes. The critic is **not** written: it is the training loop's scaffolding, not
the policy, and a file that contained it would invite someone to believe the
value estimates were part of what was graded.

The layout is six named arrays and a JSON blob rather than a ``state_dict``,
which is a deliberate choice with two reasons. A ``.pt`` needs torch to read, and
the point of persisting the forward pass is that something *without* the training
stack can evaluate it — `client/` (M6) and eventually the backlog's C++/ONNX leg.
And ``torch.load`` executes a pickle, so a committed ``.pt`` is a committed
executable; six float arrays are not.

The bytes are self-describing: :data:`CHECKPOINT_FORMAT`, the layer shapes, the
activation and the output squash all travel in the metadata, so a reader that has
never seen this module can still evaluate the network correctly — and
``tests/test_policy_checkpoint.py`` holds a reader to that claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from temper.agents.ppo import Agent

#: Bumped when the on-disk layout changes in a way an existing reader would get
#: wrong. A reader must refuse a format it does not know rather than guess: the
#: failure mode of guessing is a plausible action from a misread weight matrix,
#: which is exactly the class of wrong number that survives review.
CHECKPOINT_FORMAT = 1

#: The array names, in evaluation order. ``weight_k`` is ``(out, in)`` — torch's
#: ``nn.Linear`` layout, kept verbatim so a reader computes ``W @ x + b`` and not
#: its transpose. Getting that backwards is silent whenever the layer is square,
#: which two of these three are.
WEIGHT_KEY = "weight_{index}"
BIAS_KEY = "bias_{index}"

#: The one key that is not an array: a JSON object, stored as a 0-d unicode array
#: because ``np.savez`` has no other way to carry a string.
METADATA_KEY = "metadata"

#: What the recorded arrays mean. Both are properties of the *training* code
#: (:class:`~temper.agents.ppo.GaussianHead`,
#: :func:`~temper.agents.execution.as_fraction`) and are written into the file so
#: an independent reader implements the right ones rather than the usual ones.
ACTIVATION = "tanh"
OUTPUT_SQUASH = "clip[0,1]"


def actor_layers(agent: Agent) -> list[nn.Linear]:
    """The affine layers of the actor mean, in evaluation order.

    Reads the module list rather than the ``state_dict`` keys: the keys are
    positional indices into an ``nn.Sequential`` (``0.weight``, ``2.weight``,
    ``4.weight``), so inserting a layer renumbers them and a reader keyed on the
    old numbers loads the wrong tensor into the right-shaped slot.
    """
    head = agent.head
    mean = getattr(head, "mean", None)
    if mean is None:
        raise TypeError(
            f"{type(head).__name__} has no `mean` module; only the continuous "
            "(Gaussian) head has a forward pass worth persisting — a categorical "
            "head is the CartPole smoke test's, not an execution policy's"
        )
    layers = [module for module in mean.modules() if isinstance(module, nn.Linear)]
    if not layers:
        raise TypeError("the actor mean contains no linear layers")
    return layers


def actor_arrays(agent: Agent) -> dict[str, np.ndarray]:
    """The actor mean's weights and biases as float32 numpy arrays.

    float32 because that is the dtype the network was trained in and evaluated
    at: :data:`~temper.agents.ppo.AGENT_DTYPE` is ``torch.float32`` and
    :meth:`~temper.agents.execution.PPOPolicy.fraction` casts the observation
    through it. Widening to float64 on the way out would make the file disagree
    with the run it came from in the last bits of every action.
    """
    arrays: dict[str, np.ndarray] = {}
    for index, layer in enumerate(actor_layers(agent)):
        arrays[WEIGHT_KEY.format(index=index)] = (
            layer.weight.detach().to(torch.float32).numpy().copy()
        )
        arrays[BIAS_KEY.format(index=index)] = (
            layer.bias.detach().to(torch.float32).numpy().copy()
        )
    log_std = getattr(agent.head, "log_std", None)
    if log_std is not None:
        arrays["log_std"] = log_std.detach().to(torch.float32).numpy().copy()
    return arrays


def describe(agent: Agent) -> dict:
    """The shape half of the metadata: what a reader needs to evaluate the file."""
    layers = actor_layers(agent)
    return {
        "format": CHECKPOINT_FORMAT,
        "n_layers": len(layers),
        "obs_dim": int(layers[0].in_features),
        "action_dim": int(layers[-1].out_features),
        "hidden_sizes": [int(layer.out_features) for layer in layers[:-1]],
        "activation": ACTIVATION,
        "output_squash": OUTPUT_SQUASH,
        "dtype": "float32",
        "weight_layout": "(out, in) — evaluate W @ x + b",
    }


def export_policy(agent: Agent, path: str | Path, *, metadata: dict) -> Path:
    """Write `agent`'s deterministic forward pass to `path` as a ``.npz``.

    `metadata` is the caller's provenance — which config, which revision, which
    seed address, what the run graded. It is merged *under* :func:`describe`'s
    shape fields, which win: a caller cannot accidentally mislabel the layout of
    the arrays sitting beside it in the same file.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = {**metadata, **describe(agent)}
    np.savez(
        destination,
        **actor_arrays(agent),
        **{METADATA_KEY: np.array(json.dumps(document, sort_keys=True))},
    )
    # np.savez appends `.npz` when the name lacks it; return what was written.
    if destination.suffix != ".npz":
        destination = destination.with_suffix(destination.suffix + ".npz")
    return destination


def load_policy(path: str | Path) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    """``([(W, b), ...], metadata)`` — the writer reading its own output.

    Deliberately *not* the only reader: `client/` (M6) parses the same file
    independently, and the two are held to the same actions by a test. A format
    with one reader is a format whose documentation is untested.
    """
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive[METADATA_KEY].item()))
        if metadata.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(
                f"checkpoint format {metadata.get('format')!r} is not "
                f"{CHECKPOINT_FORMAT}; refuse rather than guess at the layout"
            )
        layers = [
            (
                archive[WEIGHT_KEY.format(index=index)],
                archive[BIAS_KEY.format(index=index)],
            )
            for index in range(int(metadata["n_layers"]))
        ]
    return layers, metadata
