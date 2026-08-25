"""The repo's markdown tables render every cell they were written with.

A one-test module about a rendering rule, because the rule bites silently and the
column it eats is the valuable one.

GitHub-flavoured markdown splits a table row on every unescaped ``|``, and it does
so **regardless of code spans**. So ``E[cost | s]`` inside a cell opens a fourth
column, and GFM drops every cell past the header's count. §9's amendment log is
three columns wide, ``Date | Change | Why``, and five of its rows had lost their
entire *Why* column on the rendered page — including the entry recording M5's
pairing reversal, which is the single most consequential thing the milestone
found. The markdown source was complete the whole time; nothing was wrong except
what a reader saw.

That is the shape of defect this repo spends most of its effort on: a claim that
is present in the source, absent from the artefact a reader actually consumes,
and silent in between. It cost nothing to find once looked for, and nothing at all
would have found it, because no test in the suite had ever read a document as
markdown rather than as prose.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import REPO_ROOT

#: The documents whose tables carry claims. Briefs are excluded deliberately: they
#: are pre-registration, superseded by their milestone's row here, and a brief
#: whose table renders short is a historical record rather than a live one.
DOCUMENTS = ("ARCHITECTURE.md", "ROADMAP.md", "README.md", "docs/house-notes.md")


def cells(line: str) -> list[str]:
    """The cells GFM would see: split on every ``|`` that is not backslash-escaped.

    Written out rather than regexed so the escaping rule is visible: a backslash
    consumes the character after it, and only an unescaped pipe divides. Code
    spans are deliberately NOT honoured, because GFM does not honour them either
    and the point of this function is to see the row the way the renderer does.
    """
    out: list[str] = []
    current: list[str] = []
    index = 0
    while index < len(line):
        char = line[index]
        if char == "\\" and index + 1 < len(line):
            current.append(line[index + 1])
            index += 2
            continue
        if char == "|":
            out.append("".join(current))
            current = []
            index += 1
            continue
        current.append(char)
        index += 1
    out.append("".join(current))
    return out[1:-1]


def tables(text: str):
    """Every markdown table in a document, as (line number, header, rows)."""
    lines = text.split("\n")
    found = []
    for index in range(len(lines) - 1):
        header, ruler = lines[index], lines[index + 1]
        if not header.startswith("|") or not ruler.startswith("|"):
            continue
        if set(ruler.replace("|", "").replace(" ", "")) - set("-:"):
            continue
        rows = []
        cursor = index + 2
        while cursor < len(lines) and lines[cursor].startswith("|"):
            rows.append((cursor + 1, lines[cursor]))
            cursor += 1
        found.append((index + 1, header, rows))
    return found


@pytest.mark.parametrize("name", DOCUMENTS)
def test_no_table_row_loses_cells_to_an_unescaped_pipe(name):
    """Every row has exactly as many cells as its header, or a reader loses text.

    The failure is one-directional and that is why it is worth a test rather than a
    review: a row with too FEW cells renders as a short row and looks wrong, so
    somebody fixes it. A row with too many renders as a complete-looking row with
    its tail silently discarded, and looks right.
    """
    path = REPO_ROOT / name
    if not path.exists():
        pytest.skip(f"{name} is not in this tree")
    text = path.read_text(encoding="utf-8")

    problems = []
    for line_no, header, rows in tables(text):
        width = len(cells(header))
        for row_line, row in rows:
            found = len(cells(row))
            if found > width:
                dropped = cells(row)[width]
                problems.append(
                    f"{name}:{row_line} has {found} cells against a {width}-column "
                    f"header (table at line {line_no}); GFM drops everything from "
                    f"cell {width + 1}, which begins {dropped[:70]!r}. Escape the "
                    "pipe as \\| — a code span does not protect it."
                )
    assert not problems, "\n".join(problems)
