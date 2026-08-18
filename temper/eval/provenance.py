"""What a reported artefact has to carry to be regenerable (invariant 1).

Every entry in ``results/`` and every figure records the config that produced it
— by content hash, not by name, because a file called ``m2_ppo.yaml`` is not the
same evidence as *that* ``m2_ppo.yaml`` — together with the git revision and
whether the tree was dirty when it ran. A dirty tree is recorded rather than
refused: a session mid-milestone will regenerate figures from an uncommitted
state constantly, and a stamp that says so is more useful than one that lies or
one that blocks.

No network, and none of the imports here can reach one (invariant 8): the git
revision comes from ``subprocess``, and if git is unavailable the stamp says
``"unknown"`` rather than failing the run.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

#: Characters of the config digest carried on a figure. The full digest is in the
#: JSON; a chart caption needs enough to identify, not enough to verify.
SHORT_DIGEST = 12

UNKNOWN_REV = "unknown"


def config_digest(path: str | Path) -> str:
    """SHA-256 of the config file's bytes.

    Of the *bytes*, not of the parsed document: a comment explaining why a
    threshold is what it is, is part of the pre-statement, and a stamp that
    ignored it would call two materially different files the same experiment.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(repo_root: Path, *args: str) -> str | None:
    """Run git and return its stdout **verbatim**, or ``None`` if it failed.

    Verbatim matters. ``git status --porcelain`` encodes a file's state in the
    first two columns, and an unstaged modification is a *leading space* (``" M
    path"``). Stripping the output removed that space from the first line only,
    which shifted every subsequent index by one and made
    :func:`_source_is_dirty` read the path as ``"sults/..."`` — so a tree whose
    only change was a regenerated file under ``results/`` reported dirty, but
    only when that file happened to sort first. Callers strip what they need.
    """
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


#: Paths whose modification does not make a run's provenance dirty. Exactly one
#: entry, and the reasoning matters: ``dirty`` exists to answer *"does the
#: recorded revision contain the code that produced this artefact?"*. A results
#: file that has just been regenerated is the artefact, not the code — and
#: without this, a sweep that writes ``results/a.json`` would make the very next
#: sweep report ``git_dirty: true`` for a source tree nobody touched, which is a
#: false alarm that teaches readers to ignore the flag.
PROVENANCE_IGNORED_PREFIXES = ("results/",)


def _source_is_dirty(status: str) -> bool:
    """True when anything outside :data:`PROVENANCE_IGNORED_PREFIXES` differs.

    ``git status --porcelain`` lines are ``XY <path>``, with renames written
    ``XY <old> -> <new>``; both sides are checked, so moving a source file *into*
    ``results/`` still reads as dirty.

    The path is taken as ``line[2:].lstrip()`` rather than ``line[3:]``: the two
    agree on well-formed porcelain, and the former also survives a line whose
    leading status space has been trimmed by something upstream — which is
    exactly the defect that once made a clean tree report dirty.
    """
    for line in status.splitlines():
        if not line.strip():
            continue
        paths = line[2:].lstrip().split(" -> ")
        if any(
            not path.strip().strip('"').startswith(PROVENANCE_IGNORED_PREFIXES)
            for path in paths
        ):
            return True
    return False


def git_revision(repo_root: str | Path) -> tuple[str, bool]:
    """``(revision, source_dirty)`` for the checkout at `repo_root`.

    ``source_dirty`` is the honest form of the question invariant 1 asks. A
    ``True`` here means the recorded revision does **not** contain the code that
    produced the artefact, and every number in it is therefore unreproducible
    from that revision alone.
    """
    root = Path(repo_root)
    revision = _git(root, "rev-parse", "HEAD")
    if revision is None:
        return UNKNOWN_REV, False
    status = _git(root, "status", "--porcelain")
    return revision.strip(), _source_is_dirty(status or "")


@dataclass(frozen=True)
class Provenance:
    """The stamp a result carries. Small, flat, and JSON-safe by construction."""

    config: str
    config_sha256: str
    git_rev: str
    git_dirty: bool
    python: str

    @property
    def short(self) -> str:
        """``cfg <digest> · rev <rev>`` — the one line a figure footer carries."""
        rev = self.git_rev
        marker = "-dirty" if self.git_dirty else ""
        head = rev[:SHORT_DIGEST] if rev != UNKNOWN_REV else rev
        return (
            f"config {self.config_sha256[:SHORT_DIGEST]} · git {head}{marker}"
        )

    def as_dict(self) -> dict:
        return {
            "config": self.config,
            "config_sha256": self.config_sha256,
            "git_rev": self.git_rev,
            "git_dirty": self.git_dirty,
            "python": self.python,
        }


def stamp(config_path: str | Path, repo_root: str | Path | None = None) -> Provenance:
    """Build the provenance stamp for a run driven by `config_path`."""
    path = Path(config_path)
    root = Path(repo_root) if repo_root is not None else path.resolve().parent.parent
    revision, dirty = git_revision(root)
    return Provenance(
        config=path.name,
        config_sha256=config_digest(path),
        git_rev=revision,
        git_dirty=dirty,
        python=sys.version.split()[0],
    )
