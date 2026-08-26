"""No live file cites a title that has been retired.

``docs/house-notes.md`` opens with the rule: *a title that changes is a title
every citation of it has to change with*. It has said so since M4b, and M4b's own
rename repointed four citations and missed three — the README's list of current
notes, and two ``tools/train.py`` docstrings. Both of the docstring citations were
**truncated** at "fabricated data" and **wrapped mid-title**, so a plain search for
the retired title found neither of them — including the search that repointed the
README's copy, which came back with a single hit and that hit was the rename record
itself. Two commits later they were still there.

That is *A guard that takes its context as an argument is only as strong as its
call sites* one level up, with a rule in place of a function: the discipline was
correct, written where it would be read, and enforced by nothing. The register
below is short, it grows once per rename, and it converts something somebody has
to remember into something that fails.

**What counts as live, and why briefs do not.** A brief is a dated document —
pre-registration for one milestone, written when the retired title was the current
one — so citing it there is *correct*, and this check must never see it. The scan
set is therefore built from the places where a citation is a live pointer:
``temper/``, ``tools/``, ``tests/``, the three root documents, and the *notes* of
``docs/house-notes.md``. ``docs/briefs/`` is simply not in the set; neither is the
preamble that holds the rename record, nor this module, which is the one file
whose job is to name retired titles. All three exclusions are the shape of the
set rather than a list of known failures, which is the difference between a check
that stays true and one that acquires an exception per defect.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from .conftest import REPO_ROOT


@dataclass(frozen=True)
class RetiredTitle:
    """One rename: what a citation used to say, and what it has to say now."""

    retired: str
    now: str
    #: The document the title belongs to. `now` must still be a heading there.
    document: str
    #: What renamed it and why, for the failure message to hand a reader.
    renamed_by: str
    #: The substring a *citation* of the retired title contains, which is not the
    #: whole title: both citations this check was written for stopped at
    #: "fabricated data". Matched against normalised text, so it also survives the
    #: line break that hid them.
    fingerprint: str


RETIRED_TITLES: tuple[RetiredTitle, ...] = (
    RetiredTitle(
        retired="The artefact writer is tested on fabricated data, not on the run",
        now="No code path may be reachable only at the end of a long run",
        document="docs/house-notes.md",
        renamed_by=(
            "M4b, 2026-08-24 — naming the writer had protected the writer and "
            "nothing else, so the note was restated as the property it is"
        ),
        fingerprint="artefact writer is tested on fabricated data",
    ),
)

#: `ARCHITECTURE.md` §9 has retired no title across thirty entries, and its own
#: preamble is why: an entry there is *superseded* by a successor that names it,
#: and both keep their titles and their rows — the log is a record, not a current
#: statement, so nothing in it goes stale by being outgrown. So the register above
#: is all house notes today. It takes a §9 entry in the same shape on the day one
#: is renamed, which is the day this comment stops being true.

#: The trees whose Python holds live pointers. `docs/` is absent, which is how
#: `docs/briefs/` is excluded — by not being named rather than by being skipped.
LIVE_TREES = ("temper", "tools", "tests")
LIVE_DOCUMENTS = ("README.md", "ARCHITECTURE.md", "ROADMAP.md")
HOUSE_NOTES = "docs/house-notes.md"


def as_a_citation_reads(text: str) -> str:
    """The text a matcher must see: one line, no markup, case-folded.

    Three things a citation carries that a title does not, and all three hid the
    two ``tools/train.py`` defects from a plain search. It **wraps**, so the title
    arrives split across lines. It may sit **inside a comment**, so continuation
    lines start with ``#``. And it is **marked up**, in backticks or italics that
    a source file breaks at different points than the document does. All three
    come out before matching; what is left is the words in order.
    """
    text = re.sub(r"(?m)^[ \t]*#+[ \t]?", " ", text)
    text = re.sub(r"[*_`]", "", text)
    return re.sub(r"\s+", " ", text).casefold()


def house_notes_body() -> str:
    """The notes themselves, without the preamble that records the renames.

    The rename record is a *statement about* a retired title and the one place in
    a live file where naming one is right. It sits above the first note heading,
    beside the rule it illustrates, so dropping everything before that heading
    excludes it by the shape of the file rather than by a line number that the
    next edit moves.
    """
    text = (REPO_ROOT / HOUSE_NOTES).read_text(encoding="utf-8")
    _preamble, marker, body = text.partition("\n## ")
    assert marker, f"{HOUSE_NOTES} has no note headings; the split is not splitting"
    return marker + body


def live_sources():
    """Every ``(label, text)`` in which a citation has to be current."""
    register = Path(__file__).resolve()
    for tree in LIVE_TREES:
        for path in sorted((REPO_ROOT / tree).rglob("*.py")):
            if path.resolve() == register:
                continue  # the register names retired titles; it does not cite them
            yield path.relative_to(REPO_ROOT).as_posix(), path.read_text(
                encoding="utf-8"
            )
    for name in LIVE_DOCUMENTS:
        yield name, (REPO_ROOT / name).read_text(encoding="utf-8")
    yield f"{HOUSE_NOTES} (the notes)", house_notes_body()


@pytest.mark.parametrize("title", RETIRED_TITLES, ids=lambda t: t.retired[:32])
def test_no_live_file_cites_a_retired_title(title):
    """The rule ``docs/house-notes.md`` states, as a thing that fails.

    A citation by a retired title is not a broken link that announces itself. It
    resolves to nothing, quietly, for a reader who searches the notes file for the
    words the docstring gave them and concludes the note was deleted.
    """
    fingerprint = as_a_citation_reads(title.fingerprint)
    offenders = [
        label for label, text in live_sources() if fingerprint in as_a_citation_reads(text)
    ]
    assert not offenders, (
        f"a retired title is cited in {', '.join(offenders)}: *{title.retired}* "
        f"was renamed to *{title.now}* ({title.renamed_by}). Repoint every "
        "citation to the current title. "
        f"If the citation is *about* the rename, it belongs in {HOUSE_NOTES}'s "
        "preamble or in a brief, which are the two places this check does not read."
    )


def test_every_replacement_title_is_still_a_live_heading():
    """A register whose replacement has itself been renamed points at nothing.

    Two renames of one note is not a hypothetical — this note has already been
    restated once, and the argument that produced the restatement (name the
    property, not the function) is exactly the kind that gets applied again.
    """
    for title in RETIRED_TITLES:
        document = as_a_citation_reads(
            (REPO_ROOT / title.document).read_text(encoding="utf-8")
        )
        assert as_a_citation_reads(title.now) in document, (
            f"the register replaces *{title.retired}* with *{title.now}*, which is "
            f"not in {title.document}: either the replacement was renamed again "
            "and this row is stale, or the entry is gone and the row should be too"
        )


def test_the_rename_record_is_on_the_excluded_side_of_the_split():
    """The exclusion is load-bearing, so it is asserted rather than assumed.

    ``docs/house-notes.md`` *contains* the retired title — recording the rename is
    the whole point of the line that does it. The check above passes only because
    the split at the first note heading leaves that line out. If the record ever
    moved into a note, or the split stopped splitting, this names which of the two
    happened instead of leaving the failure to be diagnosed from a red assertion
    about a file that is behaving correctly.
    """
    whole = as_a_citation_reads((REPO_ROOT / HOUSE_NOTES).read_text(encoding="utf-8"))
    body = as_a_citation_reads(house_notes_body())
    for title in RETIRED_TITLES:
        if title.document != HOUSE_NOTES:
            continue
        fingerprint = as_a_citation_reads(title.fingerprint)
        assert fingerprint in whole, (
            f"the rename of *{title.retired}* is no longer recorded in "
            f"{HOUSE_NOTES}; without it a reader meeting the old title in a brief "
            "has nowhere to resolve it, and the register above is the only "
            "surviving copy of a fact the document is supposed to carry"
        )
        assert fingerprint not in body, (
            f"the rename record for *{title.retired}* has moved below the first "
            "note heading, so the notes now cite a retired title and the check "
            "above cannot see it"
        )


def test_the_matcher_reads_a_citation_the_way_the_two_defects_were_written():
    """A check that cannot fire is a check nobody is keeping.

    Both ``tools/train.py`` citations were truncated and wrapped mid-title inside
    a docstring, which is precisely why ``grep`` for the retired title returned
    neither of them across two commits that were looking for exactly this. Those
    are the shapes, reproduced here: if the matcher stops seeing them, the test at
    the top of this module goes green by being blind rather than by being clean.
    """
    title = RETIRED_TITLES[0]
    fingerprint = as_a_citation_reads(title.fingerprint)

    wrapped_in_a_docstring = (
        "    class of defect ``docs/house-notes.md``'s *The artefact writer is tested on\n"
        "    fabricated data* exists for — so the branch is here, in a pure function.\n"
    )
    assert fingerprint in as_a_citation_reads(wrapped_in_a_docstring)

    wrapped_in_a_comment = (
        "    # the reason `docs/house-notes.md`'s *The artefact writer is tested\n"
        "    # on fabricated data* gives, one seam along.\n"
    )
    assert fingerprint in as_a_citation_reads(wrapped_in_a_comment)

    # And it does not fire on the current title, which shares no words with the
    # fingerprint — a matcher that hit both would make the check unfixable.
    assert fingerprint not in as_a_citation_reads(title.now)
