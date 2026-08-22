"""The vendored Anvil contract is the one it says it is, and it has not been edited.

`docs/vendor/anvil-protocol.md` is a verbatim snapshot of Anvil's `PROTOCOL.md`
at a named commit, with a Temper-side header prepended. It is the *spec* M6's
client is written against, which makes an in-place edit of it a quiet change to
the contract — the same failure mode as editing a golden fixture by hand, and
harder to notice because prose does not go red on its own.

So the header records the source digest and this re-computes it. Re-vendoring
stays a deliberate act with a fresh digest; a correction typed into the snapshot
goes red here.

Line endings are normalised to LF before hashing. The digest is of the *source*
document as Anvil holds it, and a Windows checkout with `core.autocrlf` will have
CRLF in the working copy — hashing raw bytes would make the check pass on one
host and fail on another, which is a test about the checkout rather than about
the contract.

There is deliberately no check against a live Anvil tree. Temper does not depend
on an Anvil checkout existing, and a test that reached for one would fail on
every clone that has no sibling repo — the vendored artefact is the boundary, as
it is for the FrontierView goldens.
"""

from __future__ import annotations

import hashlib
import re

import pytest

from .conftest import REPO_ROOT

VENDORED = REPO_ROOT / "docs" / "vendor" / "anvil-protocol.md"

#: Everything after this marker is Anvil's document, byte for byte.
BODY_MARKER = "Do not edit below this line. -->"

#: The wire version M6 was written against. A bump upstream is a breaking change
#: by Anvil's own definition, and the client refuses to start against a server
#: reporting anything else — see the vendored §2 `GET /api/health`.
WIRE_VERSION = 1


def _document() -> str:
    if not VENDORED.exists():  # pragma: no cover - the artefact is committed
        pytest.fail(
            f"{VENDORED.relative_to(REPO_ROOT)} is missing. It is a verbatim "
            "snapshot of Anvil's PROTOCOL.md; re-vendor it from an Anvil "
            "checkout as the file's own header describes."
        )
    return VENDORED.read_text(encoding="utf-8")


DOCUMENT = _document()
HEADER, BODY = (part for part in DOCUMENT.split(BODY_MARKER, 1))
BODY = BODY.lstrip("\n")


def _lf(text: str) -> str:
    return text.replace("\r\n", "\n")


def test_the_body_matches_the_digest_the_header_records():
    """The snapshot is the document it claims to be."""
    match = re.search(r"\*\*Source SHA-256\*\*.*?`([0-9a-f]{64})`", HEADER)
    assert match, "the header must record the source document's SHA-256"
    digest = hashlib.sha256(_lf(BODY).encode("utf-8")).hexdigest()
    assert digest == match.group(1), (
        "the vendored snapshot no longer hashes to the digest its header "
        "records. Either it was edited in place — which it must not be — or it "
        "was re-vendored without updating the header."
    )


def test_the_recorded_byte_count_matches_the_body():
    """A second, independent statement of the same fact, cheap and unambiguous."""
    match = re.search(r"\*\*Source SHA-256\*\*.*?—\s*([\d,]+)\s*bytes", HEADER)
    assert match, "the header must record the source document's size"
    assert len(_lf(BODY).encode("utf-8")) == int(match.group(1).replace(",", ""))


def test_the_source_commit_is_named_and_the_tree_was_clean():
    """Provenance, in the form invariant 1 asks for on every other artefact."""
    match = re.search(r"\*\*Source commit\*\*\s*\|\s*`([0-9a-f]{40})`(.*)", HEADER)
    assert match, "the header must name the Anvil commit this was taken from"
    assert "clean working tree" in match.group(2), (
        "a snapshot taken from a dirty Anvil checkout names a revision that does "
        "not contain the document it holds"
    )


def test_the_header_and_the_body_agree_about_the_wire_version():
    """Two statements of the version, from two authors, checked against each other.

    Anvil states it in its own document; the Temper header states what the client
    was written against. They are written at different times by different hands,
    so requiring them to agree is what turns "vendored at wire version 1" from a
    claim into a check — and it is the number `GET /api/health` is compared with
    before the client sends anything.
    """
    upstream = re.search(r"\*\*Wire version:\*\*\s*`(\d+)`", BODY)
    assert upstream, "Anvil's document states its wire version in its own header"
    vendored = re.search(r"\*\*Wire version\*\*\s*\|\s*`(\d+)`", HEADER)
    assert vendored, "the vendor header states the version M6 was written against"
    assert int(upstream.group(1)) == int(vendored.group(1)) == WIRE_VERSION


def test_the_traps_the_client_must_not_fall_into_are_recorded_here():
    """The header's prohibition table names the three plausible-wrong-number routes.

    Not a style check. M6's brief lists three ways to produce a confident wrong
    number rather than a crash — a limit that rests instead of crossing, a
    silently dropped trade frame, and an arrival price read from `summary.last`
    — and each defence in `client/` is written against a line in this table. If
    the table is re-vendored away, the defences lose the thing they cite.
    """
    for phrase in (
        "summary.last",
        "rests",
        "not signalled on the wire",
        "sparse subsequence",
    ):
        assert phrase in HEADER, (
            f"the vendor header no longer records {phrase!r}; the client cites "
            "this table instead of restating the schema, so it must stay"
        )


def test_the_snapshot_is_not_paraphrased_anywhere_in_client_code():
    """`client/` cites sections; it does not carry a second copy of the schemas.

    A copy drifts, and a drifted copy reads as authority. The cheapest
    enforceable version of that rule: no source file outside `docs/vendor/` may
    contain one of the wire's JSON frame literals.
    """
    fingerprints = ('"type":"snapshot"', '"type":"book"', '"type":"trade"')
    offenders = []
    for path in sorted(REPO_ROOT.glob("client/**/*.py")):
        text = path.read_text(encoding="utf-8")
        for fingerprint in fingerprints:
            if fingerprint in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {fingerprint}")
    assert not offenders, (
        "these files restate a wire frame rather than citing the vendored "
        "protocol: " + ", ".join(offenders)
    )
