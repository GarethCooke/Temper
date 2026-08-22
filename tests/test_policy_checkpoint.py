"""The committed policy is the seed the rule chose, and it earns the grade it claims.

`results/m4a_power_law_policy.npz` is the first binary this repo has ever
committed, and a committed binary that nothing verifies is not an artefact this
repo keeps. So the file is required to answer three questions on every commit,
none of which needs a training run:

**Is it the right seed?** The median rule is re-derived here from
`results/m4a_power_law.json` and compared with the ordinal the checkpoint says it
holds. That is the anti-cherry-pick: the artefact does not merely *assert* it is
the median seed, the rule is applied to the committed sweep and has to agree.

**Does the schedule it carries earn the grade it carries?** The stored
`eval_trajectory` is re-graded through `temper.eval.grading`, against the same
certified power-law optimum M4a was scored on, and has to reproduce the stored
objective. This half is exact arithmetic on a fixed vector, so it is checked
tightly.

**Do the weights produce that schedule?** The torch network is rebuilt from the
arrays and rolled out through the real `ExecutionEnv`, and has to reproduce both
the stored actions and the stored trajectory. This half runs a forward pass, so
it carries a float tolerance rather than a bitwise claim — see
``test_m2_rediscovery.test_one_seed_retrains_to_the_same_verdict`` for the
measurement that says why digits are a property of the host.

Together those three make the checkpoint's own claim checkable end to end:
*this file holds M4a's median seed and it scores what M4a reported.* The fourth
question — does a second, torch-free implementation of the forward pass agree —
belongs to M6 and is asked in `client/`, against the same three stored arrays.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from temper.agents.checkpoint import (
    POLICY_FORMAT,
    load_checkpoint,
    policy_from_checkpoint,
)
from temper.eval.experiment import load_experiment
from temper.eval.grading import (
    deterministic_schedule,
    grade_trajectory,
    median_ordinal,
)
from temper.eval.provenance import config_digest
from temper.env import impact_for

from .conftest import REPO_ROOT

M4A = load_experiment(REPO_ROOT / "configs" / "m4a_power_law.yaml")
CHECKPOINT_PATH = REPO_ROOT / "results" / "m4a_power_law_policy.npz"

#: How closely the re-graded stored schedule must reproduce the stored objective.
#: The schedule is a fixed vector and grading it is deterministic float
#: arithmetic through the same closed form, so anything above rounding is a
#: disagreement about the *world*, not about the host.
GRADE_RTOL = 1e-12

#: How closely the rebuilt network must reproduce the stored actions. A forward
#: pass in float32 through two 64-wide layers; torch's reduction order is a
#: property of the machine, so this is a float tolerance rather than a bitwise
#: assertion. It is still four orders tighter than anything that could move the
#: graded objective (M4a's whole available advantage is 0.037 bps and the
#: per-seed band is 4,739 shares of a 100,000-share order).
ACTION_ATOL = 1e-9

#: Portability, in one number: the same weights working a parent order 20x
#: smaller must produce the *same fractions*, so the trajectory scales exactly.
#: Not bitwise, because the shares each side is compared in are `X * fraction`
#: for two different `X`.
PORTABILITY_RTOL = 1e-12


def _document() -> dict:
    path = M4A.results_metrics
    if not path.exists():  # pragma: no cover - the sweep is committed
        pytest.skip(
            f"{path.relative_to(REPO_ROOT)} has not been generated in this tree",
            allow_module_level=True,
        )
    return json.loads(path.read_text(encoding="utf-8"))


if not CHECKPOINT_PATH.exists():
    pytest.skip(
        f"{CHECKPOINT_PATH.relative_to(REPO_ROOT)} has not been exported in this "
        "tree. Run it from a committed tree:\n"
        "  python tools/train.py --config configs/m4a_power_law.yaml "
        "--export-checkpoint\n"
        "(one seed, ~15 min on the reference box)",
        allow_module_level=True,
    )

DOCUMENT = _document()
CHECKPOINT = load_checkpoint(CHECKPOINT_PATH)


# ---------------------------------------------------------------------------
# The file says what it is
# ---------------------------------------------------------------------------


def test_the_format_is_declared_and_the_arrays_are_all_present():
    """A reader that cannot name the format must not interpret the floats."""
    assert CHECKPOINT.metadata["format"] == POLICY_FORMAT
    network = CHECKPOINT.metadata["network"]
    assert network["obs_dim"] == 2
    assert network["action_dim"] == 1
    assert network["activation"] == "tanh"
    assert network["hidden_sizes"] == list(M4A.ppo.hidden_sizes)

    for weight, bias in CHECKPOINT.actor_layers:
        assert weight in CHECKPOINT.arrays
        assert bias in CHECKPOINT.arrays
    for name in ("eval_observations", "eval_fractions", "eval_trajectory"):
        assert name in CHECKPOINT.arrays, f"the pin needs {name}"

    n_bins = M4A.case.market.n_bins
    assert CHECKPOINT.arrays["eval_observations"].shape == (n_bins, 2)
    assert CHECKPOINT.arrays["eval_fractions"].shape == (n_bins,)
    assert CHECKPOINT.arrays["eval_trajectory"].shape == (n_bins + 1,)


def test_the_layer_shapes_chain_into_a_single_number():
    """`(2 -> 64 -> 64 -> 1)`, checked by composing the stored shapes.

    Read off the file rather than off the config: the point of a self-describing
    checkpoint is that a reader can validate the network without being told what
    it should be, which is exactly what `client/`'s numpy path will do.
    """
    layers = CHECKPOINT.actor_layers
    width = CHECKPOINT.metadata["network"]["obs_dim"]
    for weight, bias in layers:
        out_features, in_features = CHECKPOINT.arrays[weight].shape
        assert in_features == width, f"{weight} does not accept {width} inputs"
        assert CHECKPOINT.arrays[bias].shape == (out_features,)
        width = out_features
    assert width == CHECKPOINT.metadata["network"]["action_dim"] == 1


def test_the_provenance_is_complete_and_the_tree_was_clean():
    """Invariant 1, on a binary. The stamp is the whole reason it is committable."""
    provenance = CHECKPOINT.metadata["provenance"]
    assert provenance["config"] == "m4a_power_law.yaml"
    assert provenance["config_sha256"] == config_digest(
        REPO_ROOT / "configs" / "m4a_power_law.yaml"
    )
    assert provenance["git_dirty"] is False, (
        "the checkpoint was exported from a dirty tree, so its recorded revision "
        "does not contain the code that produced it"
    )
    assert len(provenance["git_rev"]) == 40

    source = CHECKPOINT.metadata["source_result"]
    assert source["path"] == "results/m4a_power_law.json"
    assert source["sha256"] == config_digest(M4A.results_metrics), (
        "the committed sweep has changed since the checkpoint was exported; the "
        "seed it names may no longer be the seed the rule selects"
    )


# ---------------------------------------------------------------------------
# It is the seed the rule chose
# ---------------------------------------------------------------------------


def test_the_exported_seed_is_the_one_the_median_rule_selects():
    """The anti-cherry-pick, and the only assertion here that needs the sweep.

    The rule is applied to the committed objectives rather than trusted from the
    metadata, so an artefact exported by hand from the best seed goes red here
    instead of shipping as "the median".
    """
    objectives = [record["grade"]["objective_bps"] for record in DOCUMENT["seeds"]]
    selected = median_ordinal(objectives)
    assert CHECKPOINT.ordinal == selected, (
        f"the checkpoint holds seed {CHECKPOINT.ordinal} but the median rule "
        f"selects seed {selected} from the committed sweep"
    )

    ordering = np.argsort(objectives, kind="stable").tolist()
    rank = ordering.index(selected) + 1
    assert CHECKPOINT.metadata["selection"]["rank_from_best"] == rank
    assert rank > 1, "the median rule must never land on the best seed"

    # The upper central rank at even `n`, so the exported policy is *worse* than
    # the sweep's reported median rather than better. Stated as an inequality
    # because that direction is the property, not the specific gap.
    assert objectives[selected] >= DOCUMENT["summary"]["objective"]["median"]


def test_the_seed_address_matches_the_sweep_record():
    """Same ordinal, same shock streams — one reproducible object (invariant 5)."""
    seed = CHECKPOINT.metadata["seed"]
    record = DOCUMENT["seeds"][CHECKPOINT.ordinal]
    assert seed["ordinal"] == record["ordinal"]
    assert seed["env_streams"][0] == record["env_stream_base"]
    assert seed["torch_seed"] == record["training"]["seed"]
    assert seed["root_seed"] == M4A.seeds.root_seed
    assert seed["train_pool"] == M4A.seeds.train_pool
    assert tuple(seed["eval_streams"]) == tuple(M4A.seeds.eval_streams)


# ---------------------------------------------------------------------------
# The schedule it carries earns the grade it carries
# ---------------------------------------------------------------------------


def test_the_stored_schedule_regrades_to_the_stored_objective():
    """Exact half: a fixed trajectory, through the same certified optimum."""
    regrade = grade_trajectory(
        np.asarray(CHECKPOINT.arrays["eval_trajectory"], dtype=float),
        M4A.case.market,
        M4A.case.order_size,
        M4A.reference(),
        name="checkpoint",
    )
    stored = CHECKPOINT.grade
    assert regrade.encoding == stored["encoding"] == M4A.cost_encoding
    assert regrade.objective == pytest.approx(stored["objective_bps"], rel=GRADE_RTOL)
    assert regrade.excess == pytest.approx(stored["excess_bps"], rel=GRADE_RTOL)
    assert regrade.capture_fraction == pytest.approx(
        stored["capture_fraction"], rel=GRADE_RTOL
    )
    assert not regrade.red_flag


def test_the_checkpoint_meets_the_bar_its_sweep_was_reported_under():
    """M4a's per-seed floor, applied to the artefact rather than to the report.

    The exported seed is one of the ten M4a graded, so this is not a new claim —
    it is the guard that stops the file drifting away from the result it stands
    for. Both numbers travel together, as everywhere else in this milestone: a
    capture fraction near 1 on an advantage of 0.037 bps is a small absolute
    claim and reads as one.
    """
    stored = CHECKPOINT.grade
    bar = M4A.tolerances.per_seed_fraction
    assert stored["advantage_fraction"] <= bar
    assert stored["capture_fraction"] >= 1.0 - bar
    assert stored["red_flag"] is False
    assert stored["excess_bps"] > 0.0, (
        "an excess at or below zero over the certified optimum is a defect, "
        "never a result (ARCHITECTURE.md §1.1)"
    )


def test_the_stored_grade_matches_the_seed_the_sweep_committed():
    """The artefact and the report agree about what this seed scored.

    Not bitwise. The checkpoint was produced by *retraining* the named seed, and
    the digits of a trained network depend on the host's thread count — the
    metadata records whether it reproduced bitwise, and this asserts the claim
    that survives a different machine: the same seed, still inside the same
    per-seed floor, still no red flag, and an objective within the band M4a's
    own dispersion already spans.
    """
    stored = CHECKPOINT.grade
    committed = CHECKPOINT.metadata["committed_grade"]
    assert committed == DOCUMENT["seeds"][CHECKPOINT.ordinal]["grade"]

    spread = DOCUMENT["summary"]["objective"]
    band = max(spread["values"]) - min(spread["values"])
    assert abs(stored["objective_bps"] - committed["objective_bps"]) <= band, (
        f"the exported policy scored {stored['objective_bps']:.9f} bps against "
        f"the {committed['objective_bps']:.9f} bps this seed was committed at — "
        f"further apart than the whole 10-seed spread ({band:.2e} bps)"
    )
    assert isinstance(CHECKPOINT.metadata["reproduced_bitwise"], bool)


# ---------------------------------------------------------------------------
# The weights produce that schedule
# ---------------------------------------------------------------------------


def test_the_rebuilt_network_reproduces_the_stored_actions_and_schedule():
    """The weights, through the real env, on the committed eval streams.

    This is the assertion that makes the arrays load-bearing rather than
    decorative: it is the only thing standing between "a file of floats" and
    "the policy that produced M4a's median row".
    """
    policy = policy_from_checkpoint(CHECKPOINT, M4A.case.order_size)
    fractions = [
        policy.fraction(observation)
        for observation in CHECKPOINT.arrays["eval_observations"]
    ]
    np.testing.assert_allclose(
        fractions, CHECKPOINT.arrays["eval_fractions"], rtol=0.0, atol=ACTION_ATOL
    )

    trajectory = deterministic_schedule(
        policy,
        M4A.case.market,
        M4A.case.order_size,
        M4A.lambda_risk,
        root_seed=M4A.seeds.root_seed,
        pool=M4A.seeds.eval_pool,
        streams=M4A.seeds.eval_streams,
        temporary_impact=impact_for(
            M4A.cost_encoding, M4A.case.market, M4A.case.order_size
        ),
        expect_encoding=M4A.cost_encoding,
    )
    np.testing.assert_allclose(
        trajectory,
        CHECKPOINT.arrays["eval_trajectory"],
        rtol=0.0,
        atol=ACTION_ATOL * M4A.case.order_size,
    )


def test_the_stored_observations_are_the_ones_the_schedule_implies():
    """`(1 - k/N, x_k/X)` — the observation is a function of the trajectory.

    A cheap independent derivation of what the recorder wrote down. If the two
    disagreed, the stored actions would be a rollout of something other than the
    stored schedule, and every pin built on them would be pinning the wrong
    thing.
    """
    n_bins = M4A.case.market.n_bins
    trajectory = CHECKPOINT.arrays["eval_trajectory"]
    expected = np.column_stack(
        (
            1.0 - np.arange(n_bins, dtype=float) / n_bins,
            trajectory[:n_bins] / M4A.case.order_size,
        )
    )
    np.testing.assert_allclose(
        CHECKPOINT.arrays["eval_observations"], expected, rtol=0.0, atol=0.0
    )


def test_the_policy_is_portable_to_a_different_parent_order():
    """No market parameter enters inference, so the size is free.

    The property M6 is built on, asserted here where the artefact is rather than
    in the milestone that consumes it: hand the same weights a parent order
    twenty times smaller and the *fractions* are unchanged, so the schedule is
    the same shape scaled by X. That is why a policy trained on a synthetic
    Almgren–Chriss world can work an order on a venue whose prices sit around
    ten and whose quantities are in the hundreds.
    """
    small = M4A.case.order_size / 20.0
    policy = policy_from_checkpoint(CHECKPOINT, small)
    scaled = np.array(
        [
            policy.act(np.array([observation[0], observation[1]]))
            for observation in CHECKPOINT.arrays["eval_observations"]
        ]
    )
    full = CHECKPOINT.arrays["eval_fractions"] * (
        CHECKPOINT.arrays["eval_trajectory"][:-1] / M4A.case.order_size
    )
    np.testing.assert_allclose(scaled / small, full, rtol=PORTABILITY_RTOL, atol=0.0)


def test_loading_a_checkpoint_executes_nothing():
    """`allow_pickle=False`, stated as a test rather than as a habit."""
    with np.load(CHECKPOINT_PATH) as handle:
        assert handle.allow_pickle is False
        assert "metadata" in handle.files


def test_an_unknown_format_is_refused(tmp_path: Path):
    """A reader that does not recognise the schema must not guess."""
    forged = tmp_path / "forged.npz"
    with np.load(CHECKPOINT_PATH) as handle:
        arrays = {key: np.array(handle[key]) for key in handle.files}
    metadata = json.loads(str(arrays.pop("metadata")))
    metadata["format"] = "temper-policy-99"
    np.savez(forged, **arrays, metadata=np.array(json.dumps(metadata)))
    with pytest.raises(ValueError, match="refusing to interpret"):
        load_checkpoint(forged)
