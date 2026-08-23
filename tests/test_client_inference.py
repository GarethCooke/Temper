"""M6 task 1 — two implementations of one forward pass, and they agree.

`temper/agents/` runs the trained policy through torch. `client/inference.py`
runs the same weights through numpy. A second implementation is only worth having
if something proves the two agree, and this is that proof: the committed
checkpoint carries the observations and actions the *torch* policy produced on
M4a's evaluation episode, so replaying them through the numpy path is a
comparison against the training-time policy with **no network, no environment
and no torch** anywhere in the test.

The comparison is a float tolerance rather than a bitwise claim, and the
tolerance is not a hedge. Both sides compute in float32
(`temper.agents.ppo.AGENT_DTYPE`), but torch's `addmm` and numpy's `@` reduce a
64-wide dot product in different orders, so the last bits differ. What matters is
whether the difference could move a decision: it is measured at ~3e-8 on the
committed trajectory, against a first-bin fraction of 0.42 on a 1,000-share
parent order — six orders of magnitude below one share.

Two structural claims are asserted here as well, because they are what make the
seam real rather than intended: nothing under `client/` imports torch, and
importing the whole client package does not drag it in transitively.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from client.inference import DTYPE, POLICY_FORMAT, Policy, load_pin, load_policy

from .conftest import REPO_ROOT

CHECKPOINT = REPO_ROOT / "results" / "m4a_power_law_policy.npz"
CLIENT_ROOT = REPO_ROOT / "client"

#: How far the numpy forward pass may sit from the torch one, per action. A
#: fraction of remaining inventory, so the unit is dimensionless; 1e-6 of a
#: fraction is 1e-3 of a share on this milestone's 1,000-share parent order, and
#: the measured difference is ~3e-8.
ACTION_ATOL = 1e-6

if not CHECKPOINT.exists():  # pragma: no cover - the artefact is committed
    pytest.skip(
        f"{CHECKPOINT.relative_to(REPO_ROOT)} has not been exported in this tree",
        allow_module_level=True,
    )

POLICY = load_policy(CHECKPOINT)
OBSERVATIONS, FRACTIONS, TRAJECTORY = load_pin(CHECKPOINT)


# ---------------------------------------------------------------------------
# The pin
# ---------------------------------------------------------------------------


def test_the_numpy_forward_pass_reproduces_the_training_time_actions():
    """The whole point of task 1, in one assertion.

    Replays the checkpoint's committed evaluation observations and requires the
    numpy policy's fraction to match the torch policy's, action for action.
    """
    reproduced = np.array(
        [POLICY.fraction(clock, inventory) for clock, inventory in OBSERVATIONS]
    )
    worst = float(np.max(np.abs(reproduced - FRACTIONS)))
    assert worst <= ACTION_ATOL, (
        f"the numpy forward pass differs from the training-time policy by "
        f"{worst:.3e} on the committed evaluation trajectory, against a "
        f"tolerance of {ACTION_ATOL:g}. That is not a rounding difference."
    )


def test_the_reproduced_actions_rebuild_the_committed_trajectory():
    """Actions agreeing is necessary; the *schedule* agreeing is the claim.

    An action error that stayed inside the tolerance every bin could still
    compound over thirteen of them, because each bin's observation carries the
    inventory the previous one left. So the schedule is rebuilt from the numpy
    fractions and compared against the trajectory the torch policy realised —
    which is the quantity M4a graded, and the one a live client's inventory
    tracks.
    """
    order_size = float(TRAJECTORY[0])
    inventory, rebuilt = order_size, [order_size]
    for index in range(len(OBSERVATIONS)):
        fraction = POLICY.fraction(1.0 - index / len(OBSERVATIONS), inventory / order_size)
        inventory -= fraction * inventory
        rebuilt.append(inventory)
    # The env force-liquidates the final bin (ARCHITECTURE.md §4), so the
    # comparison is over the interior holdings; the terminal zero is the
    # boundary condition rather than a decision.
    np.testing.assert_allclose(
        rebuilt[:-1], TRAJECTORY[:-1], rtol=0.0, atol=ACTION_ATOL * order_size
    )


def test_the_stored_observations_are_the_ones_the_client_would_construct():
    """`(1 - k/N, inventory/X)` — the client builds this, the env built that.

    The live client never sees an `ExecutionEnv`; it constructs the observation
    itself from a clock it keeps and an inventory it reconciles off the wire. So
    the two constructions have to be the same construction, and this is where
    that is checked rather than assumed.
    """
    n_bins = len(OBSERVATIONS)
    order_size = float(TRAJECTORY[0])
    expected = np.column_stack(
        (
            1.0 - np.arange(n_bins, dtype=float) / n_bins,
            TRAJECTORY[:n_bins] / order_size,
        )
    )
    np.testing.assert_allclose(OBSERVATIONS, expected, rtol=0.0, atol=0.0)


def test_the_policy_reads_its_own_shape_out_of_the_file():
    """A self-describing checkpoint, used as one."""
    assert POLICY.n_bins == 13
    assert len(POLICY.layers) == 3
    assert POLICY.layers[0][0].shape == (64, 2)
    assert POLICY.layers[-1][0].shape == (1, 64)
    assert all(weight.dtype == DTYPE for weight, _ in POLICY.layers)
    assert POLICY.metadata["format"] == POLICY_FORMAT


def test_the_two_format_constants_agree():
    """`client/` duplicates the format name; the duplication is checked.

    Importing `temper.agents.checkpoint.POLICY_FORMAT` would put the training
    stack back on the client's import path, which is the one thing the module
    exists to avoid. So the constant is duplicated — and this is what stops the
    two drifting into a silent mis-read.
    """
    from temper.agents.checkpoint import POLICY_FORMAT as TRAINING_SIDE

    assert POLICY_FORMAT == TRAINING_SIDE


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------


def test_an_unknown_format_is_refused(tmp_path: Path):
    forged = tmp_path / "forged.npz"
    with np.load(CHECKPOINT) as handle:
        arrays = {key: np.array(handle[key]) for key in handle.files}
    metadata = json.loads(str(arrays.pop("metadata")))
    metadata["format"] = "something-else"
    np.savez(forged, **arrays, metadata=np.array(json.dumps(metadata)))
    with pytest.raises(ValueError, match="refusing to interpret"):
        load_policy(forged)


def test_a_non_tanh_network_is_refused(tmp_path: Path):
    """The reader implements one activation and says so, rather than guessing."""
    forged = tmp_path / "relu.npz"
    with np.load(CHECKPOINT) as handle:
        arrays = {key: np.array(handle[key]) for key in handle.files}
    metadata = json.loads(str(arrays.pop("metadata")))
    metadata["network"]["activation"] = "relu"
    np.savez(forged, **arrays, metadata=np.array(json.dumps(metadata)))
    with pytest.raises(ValueError, match="implements tanh only"):
        load_policy(forged)


def test_a_different_output_squash_is_refused(tmp_path: Path):
    """The one boundary fact the numeric pin cannot check.

    The committed `eval_fractions` run 0.293-0.421 and never reach 0 or 1, so
    replaying them agrees whether this reader clips to [0, 1], clips to some
    other range, or does not clip at all. Declaring the squash in the metadata
    and refusing an unknown one is what closes that gap — and it is why the
    field was worth adding to `network_description` rather than leaving the fact
    in a docstring on the training side.
    """
    forged = tmp_path / "sigmoid.npz"
    with np.load(CHECKPOINT) as handle:
        arrays = {key: np.array(handle[key]) for key in handle.files}
    metadata = json.loads(str(arrays.pop("metadata")))
    metadata["network"]["output_squash"] = "sigmoid"
    np.savez(forged, **arrays, metadata=np.array(json.dumps(metadata)))
    with pytest.raises(ValueError, match="output squash"):
        load_policy(forged)


def test_a_checkpoint_predating_the_squash_field_still_loads():
    """`.get` with a default, and the committed artefact is why.

    `results/m4a_power_law_policy.npz` was exported before `output_squash`
    existed. A reader that required the key would have refused the one file the
    milestone actually ran on, and buying that strictness costs a retrain.
    """
    assert "output_squash" not in POLICY.metadata["network"]
    assert load_policy(CHECKPOINT).n_bins == 13


def test_layers_that_do_not_compose_are_refused(tmp_path: Path):
    """A mis-read file must fail here, not produce plausible fractions live."""
    forged = tmp_path / "mismatched.npz"
    with np.load(CHECKPOINT) as handle:
        arrays = {key: np.array(handle[key]) for key in handle.files}
    metadata = json.loads(str(arrays.pop("metadata")))
    metadata["network"]["obs_dim"] = 3
    np.savez(forged, **arrays, metadata=np.array(json.dumps(metadata)))
    with pytest.raises(ValueError, match="does not accept 3 inputs"):
        load_policy(forged)


def test_the_action_is_clipped_not_squashed():
    """A clip reaches the endpoints; a sigmoid does not.

    `temper.agents.execution.as_fraction` clips, because the policy's density
    lives on the unsquashed action and the last bin's TWAP fraction is exactly 1.
    The numpy path has to do the same thing or the two disagree at precisely the
    boundary the trained policy was shaped by.
    """
    extreme = Policy(
        layers=(
            (np.array([[0.0, 0.0]], dtype=DTYPE), np.array([50.0], dtype=DTYPE)),
        ),
        metadata={"world": {"n_bins": 13}},
    )
    assert extreme.fraction(1.0, 1.0) == 1.0
    negative = Policy(
        layers=(
            (np.array([[0.0, 0.0]], dtype=DTYPE), np.array([-50.0], dtype=DTYPE)),
        ),
        metadata={"world": {"n_bins": 13}},
    )
    assert negative.fraction(1.0, 1.0) == 0.0


# ---------------------------------------------------------------------------
# The seam, from the client's side
# ---------------------------------------------------------------------------


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_no_module_under_client_imports_torch():
    """The participant runs the policy; it does not carry the thing that trained it."""
    sources = sorted(CLIENT_ROOT.rglob("*.py"))
    assert sources, "no client sources found"
    for source in sources:
        assert "torch" not in _imported_roots(source), (
            f"{source.relative_to(REPO_ROOT)} imports torch; client/ reads the "
            "committed policy with numpy (ARCHITECTURE.md §3's seam, M6 task 1)"
        )


def test_importing_the_client_does_not_drag_torch_in_transitively():
    """A static check is worth little if an indirect import undoes it.

    Run in a subprocess because torch is already loaded in this one — the suite
    trains and grades — so `sys.modules` here cannot answer the question.
    """
    program = (
        "import sys; import client.run; "
        "print('torch' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False", (
        "importing client.run loaded torch through something it depends on; "
        "the client's import path must stay free of the training stack"
    )
