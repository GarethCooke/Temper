"""Structural invariants of the repo that a code review would otherwise have to catch.

Covers the M0 definition-of-done items that are properties of the *repository*
rather than of any one function: no network in the test path, no torch in the
oracle, no GPU dependence, and a golden fixture that is genuinely vendored
rather than generated here.
"""

from __future__ import annotations

import ast
import shutil
import socket
import subprocess
from pathlib import Path

import numpy as np
import pytest

from temper.oracle import Market, SymbolParams, ac_trajectory, cost_moments

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "temper"

#: Constitution invariant 8 — `temper/` performs no network I/O. The Anvil
#: participant lives in `client/` and consumes the package.
NETWORK_MODULES = {
    "aiohttp",
    "asyncio",
    "http",
    "httpx",
    "requests",
    "socket",
    "socketserver",
    "ssl",
    "urllib",
    "urllib3",
    "websockets",
    "xmlrpc",
}


def _package_sources() -> list[Path]:
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_the_package_has_sources_to_check():
    """Guard against the two tests below passing because they found nothing."""
    assert len(_package_sources()) >= 3


def test_package_imports_nothing_that_can_reach_a_network():
    """Invariant 8, checked statically across the whole package."""
    for source in _package_sources():
        offenders = _imported_roots(source) & NETWORK_MODULES
        assert not offenders, (
            f"{source.relative_to(REPO_ROOT)} imports {', '.join(sorted(offenders))}; "
            "constitution invariant 8 keeps network code in client/"
        )


@pytest.mark.parametrize("package", ["oracle", "env"])
def test_the_oracle_and_the_env_do_not_import_torch(package):
    """Pure numpy below the agents.

    The oracle stays light because it is the reference engine. The env stays
    light because M2's agent will cast to torch at *its* boundary and nowhere
    else: an env that returned tensors would make every baseline drag a training
    framework behind it, and would put a dtype conversion between the simulator
    and the closed form it is checked against.
    """
    sources = sorted((PACKAGE_ROOT / package).rglob("*.py"))
    assert sources, f"no sources found under temper/{package}"
    for source in sources:
        assert "torch" not in _imported_roots(source), (
            f"{source.relative_to(REPO_ROOT)} imports torch; temper/{package} is pure numpy"
        )


def test_the_env_does_not_import_the_power_law_moments():
    """Invariant 7's quarantine, at the import level (ARCHITECTURE.md §9).

    Phase 1 is the linearised world end-to-end. `cost_moments` is FrontierView's
    0.6-power charge and is reporting context only — an env that imported it
    would be one edit away from paying out a functional the oracle does not
    minimise, which is the exact failure invariant 7 exists to prevent.
    """
    for source in sorted((PACKAGE_ROOT / "env").rglob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert "cost_moments" not in imported, (
            f"{source.relative_to(REPO_ROOT)} imports cost_moments; Phase 1 charges "
            "the linear tangent (see temper/eval/metrics.py)"
        )


def test_no_config_can_inherit_a_phase_two_world_by_default():
    """Constitution §4, made mechanical: a Phase-2 world has to be *named*.

    "Phase-2 models are additive alternatives behind the same interface, never
    silent modifications of Phase 1" is a sentence about defaults. So the default
    is checked in three places at once — the env's constructor, the experiment
    loader, and the reference table — and every committed config that ends up in
    a non-linear world is required to say so in its own bytes.

    The failure this prevents is specific and would be very quiet: a config
    written for M2 or M3, re-run after M4a landed, silently training in the
    power-law world because some default moved. Every number in ``results/``
    would still regenerate; they would just regenerate from a different world
    than the one they claim.
    """
    import inspect

    import yaml

    from temper.env import ExecutionEnv
    from temper.eval.experiment import load_experiment
    from temper.eval.reference import reference_row
    from temper.oracle import LINEAR_ENCODING, Market, SymbolParams

    market = Market.for_horizon(
        SymbolParams(adv=6e7, sigma=0.0155, half_spread=0.3, eta=0.142, gamma=0.314), 6.5
    )
    env = ExecutionEnv(market, 100_000.0, 1e-4, root_seed=1)
    assert env.cost_encoding == LINEAR_ENCODING, (
        "ExecutionEnv's default world is not Phase 1"
    )
    assert reference_row(market, 100_000.0, 1e-4).encoding == LINEAR_ENCODING
    assert (
        inspect.signature(reference_row).parameters["encoding"].default
        == LINEAR_ENCODING
    )

    configs = sorted((REPO_ROOT / "configs").rglob("*.yaml"))
    assert configs, "no committed configs found"
    for path in configs:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if "lambda_selection" not in document:  # not an experiment config
            named = document.get("world", {}).get("cost_encoding")
            assert named is None or "world" in document
            continue
        experiment = load_experiment(path)
        if experiment.cost_encoding == LINEAR_ENCODING:
            continue
        assert "world" in document, (
            f"{path.relative_to(REPO_ROOT)} resolves to the "
            f"{experiment.cost_encoding!r} world without naming it; a Phase-2 "
            "world is never inherited (constitution §4)"
        )
        assert document["world"]["cost_encoding"] == experiment.cost_encoding


def test_a_config_that_names_an_unknown_world_is_refused():
    """The named world is checked against the oracle's list, not merely read."""
    import tempfile

    import yaml

    from temper.eval.experiment import load_experiment

    source = REPO_ROOT / "configs" / "m3_antithetic_validation.yaml"
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    document["world"] = {"cost_encoding": "vibes"}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "bad.yaml"
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
        with pytest.raises(ValueError, match="unknown cost encoding"):
            load_experiment(path)


#: The packages a policy's code lives in. M2's PPO lands under `agents/`, so
#: `rglob` covers "any future training path" without this list needing an edit.
POLICY_PACKAGES = ("agents",)


def _string_literals(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def _referenced_names(path: Path) -> set[str]:
    """Every bare name, attribute and imported name a module mentions."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add(node.asname or node.name.split(".")[-1])
    return names


def test_the_shock_key_literal_lives_only_in_the_env_package():
    """Constitution §4's observation minimality, enforced statically.

    The realised price shock is published for the identity tests and for nothing
    else. Written as a bare `"walk_bps"` it would be reachable from anywhere by
    anyone who happened to know the string, and un-greppable when M2 wants to
    ask "what can the agent see?". As the single constant
    :data:`~temper.env.execution_env.SHOCK_KEY` it is one name, and this test is
    what keeps it one name: the literal is rejected everywhere under `temper/`
    except the package that defines it.
    """
    from temper.env import SHOCK_KEY

    offenders = [
        source.relative_to(REPO_ROOT)
        for source in _package_sources()
        if source.parent != PACKAGE_ROOT / "env" and SHOCK_KEY in _string_literals(source)
    ]
    assert not offenders, (
        f"{', '.join(str(path) for path in offenders)} spell the shock key as a "
        f"literal {SHOCK_KEY!r}; import temper.env.SHOCK_KEY instead so every read "
        "of the price path is one greppable name"
    )

    # Non-vacuous: the constant really is the key the env publishes under.
    env_literals = set().union(
        *(_string_literals(source) for source in (PACKAGE_ROOT / "env").rglob("*.py"))
    )
    assert SHOCK_KEY in env_literals


def test_no_policy_path_can_reach_the_shock():
    """The agent's own code may not touch the price path by any spelling.

    Stronger than the literal check, and aimed at a different failure: not
    someone hard-coding the string, but a future PPO wrapper importing the
    constant to build a richer observation "just for a debugging run". Phase 1's
    rediscovery claim is only meaningful if the agent could not have learned
    anything else (constitution §7), so the policy packages get no route at all —
    while `temper/eval/rollout.py`, which records the shock so the identity tests
    can subtract it, keeps its access through the named constant.
    """
    from temper.env import SHOCK_KEY

    for package in POLICY_PACKAGES:
        sources = sorted((PACKAGE_ROOT / package).rglob("*.py"))
        assert sources, f"no sources found under temper/{package}"
        for source in sources:
            where = source.relative_to(REPO_ROOT)
            assert SHOCK_KEY not in _string_literals(source), (
                f"{where} names the shock key; a policy may not see the price path"
            )
            assert "SHOCK_KEY" not in _referenced_names(source), (
                f"{where} imports SHOCK_KEY; a policy may not see the price path"
            )

    # The recorder does reach it, through the constant. If that ever reverts to a
    # literal the test above catches it; if it disappears entirely the identity
    # tests lose their input, and this says so first.
    rollout = PACKAGE_ROOT / "eval" / "rollout.py"
    assert "SHOCK_KEY" in _referenced_names(rollout)
    assert SHOCK_KEY not in _string_literals(rollout)


def test_the_env_does_not_import_a_schedule_or_a_closed_form():
    """The differential must not be checking the oracle against itself.

    M1's whole claim is that an independently-written simulator reproduces the
    closed form's moments. If the env reached for `optimal_trajectory` or a
    kappa, the Monte-Carlo tiers would be measuring numpy's arithmetic rather
    than the model.
    """
    forbidden = {
        "ac_kappa",
        "ac_trajectory",
        "cost_moments",
        "linear_cost_moments",
        "objective_curvature",
        "optimal_kappa",
        "optimal_trajectory",
        "schedule_moments",
        "sinh_trajectory",
        "twap_trajectory",
    }
    for source in sorted((PACKAGE_ROOT / "env").rglob("*.py")):
        text = source.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(source))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        offenders = imported & forbidden
        assert not offenders, (
            f"{source.relative_to(REPO_ROOT)} imports {', '.join(sorted(offenders))}; "
            "the env must reach its cost the long way round, bin by bin"
        )


#: The one module under `temper/` allowed to import matplotlib. M2 adds plotting
#: as a pinned dependency; the core stays headless and import-light, so the
#: allow-list is a path rather than a convention.
PLOTTING_MODULE = PACKAGE_ROOT / "eval" / "figures.py"

#: Import roots that pull a rendering stack in.
PLOTTING_MODULES = {"matplotlib", "pylab", "seaborn", "plotly"}


def test_matplotlib_is_confined_to_the_one_plotting_module():
    """Invariant: a figure library may not reach the oracle, the env or the agent.

    A plotting stack on the import path of `temper/` would be dragged behind
    every test, every training run and eventually the M6 Anvil client — and it
    is the sort of dependency that arrives one convenience import at a time.
    The allow-list is exactly one file.
    """
    offenders = [
        source.relative_to(REPO_ROOT)
        for source in _package_sources()
        if source != PLOTTING_MODULE and _imported_roots(source) & PLOTTING_MODULES
    ]
    assert not offenders, (
        f"{', '.join(str(path) for path in offenders)} import a plotting stack; "
        f"only {PLOTTING_MODULE.relative_to(REPO_ROOT)} may"
    )


def test_the_plotting_module_forces_a_headless_backend_before_pyplot():
    """`Agg` is selected before `pyplot` is imported, or a run can block on a display.

    Order matters and is invisible at runtime on a developer's machine, which is
    exactly why it is checked here rather than trusted: `matplotlib.use` after
    `pyplot` has already chosen a backend is a no-op on some versions and a
    warning on others.
    """
    assert PLOTTING_MODULE.exists(), "the plotting module is missing"
    tree = ast.parse(PLOTTING_MODULE.read_text(encoding="utf-8"), filename=str(PLOTTING_MODULE))

    use_line = None
    pyplot_line = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "use"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "Agg"
        ):
            use_line = node.lineno
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "matplotlib.pyplot":
                    pyplot_line = node.lineno

    assert use_line is not None, "the plotting module does not force the Agg backend"
    assert pyplot_line is not None, "the plotting module does not import pyplot"
    assert use_line < pyplot_line, (
        f"matplotlib.use('Agg') is on line {use_line}, after pyplot on line "
        f"{pyplot_line}; the backend must be chosen first"
    )


#: Names that betray a *running* normaliser. M2 task 4: any reward scaling is a
#: single affine transform with constants in the committed config, applied
#: identically at train and eval. Running statistics make the reward
#: non-stationary and seed-dependent, which is objective drift by the back door
#: (constitution invariant 7) — and CleanRL's own wrappers are the most likely
#: route in, so they are named.
RUNNING_NORMALISERS = {
    "NormalizeObservation",
    "NormalizeReward",
    "RunningMeanStd",
    "VecNormalize",
}


def test_no_running_normaliser_anywhere_in_the_package():
    """The scaling is affine and committed, or it is not scaling — it is drift.

    Structural rather than behavioural because the failure is invisible in any
    single episode: a running normaliser produces perfectly reasonable rewards
    and changes what "the objective" means between one seed and the next.
    """
    for source in _package_sources():
        names = _referenced_names(source) | _string_literals(source)
        offenders = sorted(names & RUNNING_NORMALISERS)
        assert not offenders, (
            f"{source.relative_to(REPO_ROOT)} reaches for {', '.join(offenders)}; "
            "M2 task 4 allows one committed affine constant and no running "
            "statistics (see temper/agents/execution.py:RewardScale)"
        )


def test_the_policy_packages_do_not_import_the_control_variate():
    """The estimator seam stays a seam.

    :mod:`temper.eval.variate` and :mod:`temper.eval.antithetic` both read the
    env's published price shock, which is why they live under `eval/` and are
    handed to the training loop as a parameter by the experiment driver. A
    policy package importing either — even without ever naming the shock key —
    would put the price path one attribute access away from an observation, and
    would make the static guards above true but uninformative.
    """
    for package in POLICY_PACKAGES:
        for source in sorted((PACKAGE_ROOT / package).rglob("*.py")):
            text = source.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=str(source))
            modules = {
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module
            } | {
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            }
            assert not any(
                name.endswith("variate") or name.endswith("antithetic")
                for name in modules
            ), (
                f"{source.relative_to(REPO_ROOT)} imports an estimator (the "
                "control variate or the antithetic pair); the driver composes "
                "it, a policy package never sees it"
            )


def _target_recipe(makefile: str, target: str) -> list[str]:
    """The non-comment recipe lines of one make target."""
    lines: list[str] = []
    inside = False
    for line in makefile.splitlines():
        if line.startswith(f"{target}:"):
            inside = True
            continue
        if inside:
            if not line.startswith("	"):
                break
            if not line.strip().startswith("#"):
                lines.append(line.strip())
    return lines


def test_the_sweep_target_runs_both_estimators_and_states_each_expectation():
    """`make sweep` must reach the second config, and must not do so by muting the first.

    The defect this pins was real: the driver exits non-zero on a MISS verdict,
    the sampled-reward sweep is M2's *recorded miss*, so `make sweep` aborted
    after two hours having produced exactly one of the two artefacts it exists to
    produce. The obvious fix — make's `-` ignore-errors prefix — would have been
    worse than the bug, because it also silences the one outcome worth hearing
    about: a recorded miss that starts passing invalidates the amendment resting
    on it. `--expect` makes the exit status mean "did this reach the verdict it
    was supposed to?", which is both a working target and a live check.
    """
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    lines = _target_recipe(makefile, "sweep")
    assert len(lines) == 2, f"expected two sweep invocations, found {len(lines)}"
    assert all("tools/train.py" in line for line in lines)
    assert any("--expect miss" in line for line in lines), (
        "the sampled-reward sweep must declare that it is expected to miss"
    )
    assert any("--expect pass" in line for line in lines), (
        "the control-variate sweep must declare that it is expected to pass"
    )
    assert not any(line.startswith("-") for line in lines), (
        "a sweep line is prefixed with `-`, which ignores its exit status; use "
        "--expect so a surprising verdict still stops the target"
    )


def test_the_oracle_computes_with_sockets_disabled():
    """The static check with a runtime backstop: no network in the test path."""

    def refuse(*args, **kwargs):
        raise AssertionError("the oracle attempted to open a socket")

    original = socket.socket
    socket.socket = refuse
    try:
        market = Market.for_horizon(
            SymbolParams(adv=6e7, sigma=0.0155, half_spread=0.3, eta=0.142, gamma=0.314),
            6.5,
        )
        moments = cost_moments(ac_trajectory(market, 100_000.0, 1e-5), market)
    finally:
        socket.socket = original

    assert np.isfinite(moments.expected)


def test_every_package_source_would_survive_a_clone():
    """No source under `temper/` may be excluded by .gitignore.

    Not hypothetical: `.gitignore`'s virtualenv entry was `env/`, and a bare
    directory pattern matches at *any* depth, so the whole of `temper/env/` was
    invisible to git the moment M1 created it. `make test` would have stayed green
    here and failed from a clean clone — the one gate every milestone is
    measured by.
    """
    git = shutil.which("git")
    if git is None or not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout with git available")

    sources = _package_sources()
    result = subprocess.run(
        [git, "check-ignore", "--stdin"],
        input="\n".join(str(path) for path in sources),
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    ignored = sorted(
        Path(line.strip().strip('"')).name for line in result.stdout.splitlines() if line.strip()
    )
    assert not ignored, (
        "these package sources are excluded by .gitignore and would be missing "
        f"from a clone: {', '.join(ignored)}"
    )


def test_the_golden_fixture_is_vendored_not_generated_here():
    """The fixture must credit FrontierView, or the differential is a tautology.

    The M0 brief's one hard stop: goldens synthesised from this repo's own AC
    implementation would make every downstream claim circular.
    """
    from .conftest import GOLDEN_DOCUMENT, GOLDEN_PATH

    assert GOLDEN_PATH.parent.name == "vendor"
    assert GOLDEN_DOCUMENT["provenance"]["source"] == "FrontierView"
    assert GOLDEN_DOCUMENT["cases"], "fixture contains no cases"


@pytest.mark.parametrize("required", ["numpy", "pytest"])
def test_the_test_path_needs_only_cpu_dependencies(required):
    """No GPU anywhere: the whole M0 suite runs on numpy and pytest."""
    __import__(required)
