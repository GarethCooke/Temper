"""M5 wrap-up — recompute the artefact's summary block, as an audit not a patch.

    python tools/rebuild_m5_summary.py                  # audit only, writes nothing
    python tools/rebuild_m5_summary.py --write          # audit, then correct in place

`results/m5_alpha.json` was written by a passing ten-seed sweep at rev 5f7c675.
Four of its summary fields are wrong, because :func:`summarise` calls ``worst``
the maximum and M5 is the first milestone to report *captures*, where larger is
better. The most visible symptom is ``alpha_capture.worst = 1.109916``: more than
the alpha the reference has available, and therefore impossible on its face. It
alarms a reader who checks it and flatters one who does not.

Every per-seed value in the file is correct. Only the aggregates taken over them
are wrong, so the correction needs no training — which is exactly why it has to
be an audit rather than an edit. This tool recomputes the WHOLE summary from the
file's own per-seed values through the same code path that wrote it
(``ALPHA_DIRECTIONS``, :func:`summarise`, :func:`invert_summary`) and requires
every field other than the four to come back bit-identical. A fifth field that
moves is a fifth defect, and it surfaces here rather than in whatever quotes this
artefact next.

The correction is recorded INSIDE the file, under ``summary_correction``, and
``provenance`` gains ``unmodified_run_output: false`` pointing at it. A file that
has been touched after its run must say so where a reader looks, not in a commit
message they would have to go find. Re-running is idempotent: a corrected file
audits clean and is left alone.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from temper.eval.grading import summarise  # noqa: E402
from temper.eval.sweep import (  # noqa: E402
    ALPHA_CAPTURE_BAR,
    ALPHA_DERIVED,
    ALPHA_DIRECTIONS,
    PREMIUM_RATIO_BAR,
    invert_summary,
)

ARTEFACT = REPO_ROOT / "results" / "m5_alpha.json"

#: The fields the direction defect touched, named in advance. Anything else that
#: moves is not a correction, it is a finding.
EXPECTED_MOVERS = ("alpha_bps", "alpha_capture", "advantage_fraction", "net_capture")

#: Scalars carried by every summary block. ``values`` is handled separately: it is
#: per-seed data, and this tool exists on the promise that it does not touch it.
SCALARS = ("median", "q1", "q3", "iqr", "worst")


def rebuild_summary(summary: dict) -> dict:
    """The summary the corrected code produces from these same per-seed values.

    Deliberately built from ``summary[name]["values"]`` rather than from grades:
    the claim being made is that the aggregates are wrong and the per-seed values
    are right, and reading the per-seed values back out of the file is what makes
    that claim checkable by anyone holding the file.
    """
    rebuilt = {
        name: summarise(name, summary[name]["values"], direction=direction).as_dict()
        for name, direction in ALPHA_DIRECTIONS.items()
    }
    for derived, source in ALPHA_DERIVED.items():
        rebuilt[derived] = invert_summary(derived, rebuilt[source])
    return rebuilt


def audit(summary: dict, rebuilt: dict) -> dict:
    """What moved, field by field, to the last bit.

    Float equality throughout, not a tolerance. The recomputation runs the same
    operations on the same inputs, so anything that differs at all differs because
    the code changed, and "how much" is a question about the defect rather than
    about the comparison.
    """
    if set(summary) != set(rebuilt):
        raise AssertionError(
            f"the file reports {sorted(set(summary) - set(rebuilt))} which the "
            f"corrected code does not, and omits {sorted(set(rebuilt) - set(summary))}"
        )
    moved: dict[str, dict] = {}
    for name in sorted(summary):
        was, now = summary[name], rebuilt[name]
        if list(was["values"]) != list(now["values"]):
            raise AssertionError(
                f"{name}: a PER-SEED value changed. This tool corrects aggregates "
                "taken over the seeds and nothing else; a per-seed value that moves "
                "means the recomputation is not reading what the run wrote"
            )
        changed = {k: (was[k], now[k]) for k in SCALARS if was[k] != now[k]}
        if changed:
            moved[name] = changed
    return moved


def verdict_audit(document: dict, rebuilt: dict) -> dict:
    """Everything the verdict reads off the summary, recomputed from the new one.

    "The verdict is unaffected" is the sentence that decides whether this file can
    be corrected at all rather than re-run, so it is measured here rather than
    reasoned about. Both bar checks read medians and no direction touches a median,
    which is *why* it holds — but that is an argument, and an argument is what the
    original defect survived.
    """
    tol = document["config"]["tolerances"]
    verdict = document["verdict"]
    graded = rebuilt[tol["graded_attribute"]]
    recomputed = {
        "epsilon_met": bool(graded["median"] <= tol["epsilon_fraction"]),
        "per_seed_met": bool(graded["worst"] <= tol["per_seed_fraction"]),
        "alpha_capture_met": bool(
            rebuilt["alpha_capture"]["median"] >= ALPHA_CAPTURE_BAR
        ),
        "premium_ratio_met": bool(
            rebuilt["premium_ratio"]["median"] <= PREMIUM_RATIO_BAR
        ),
        "median_excess_bps": rebuilt["excess_bps"]["median"],
    }
    changed = {k: (verdict[k], v) for k, v in recomputed.items() if verdict[k] != v}
    for key, was in verdict["headline"].items():
        if not key.endswith("_median"):
            continue
        now = rebuilt[key[: -len("_median")]]["median"]
        if was != now:
            changed[f"headline.{key}"] = (was, now)
    return changed


def _rev() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _dirty() -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )


def render(moved: dict) -> str:
    """Enough digits to see the move, including when the move is one ulp.

    Three of the eight moves here are ``advantage_fraction``'s quartiles shifting
    in the last bit, because it is now summarised and inverted into ``net_capture``
    rather than the other way round, and ``1 - (1 - a)`` is not always ``a``.
    Printed at six places they would read as identical numbers in a list of
    changes, which is the sort of thing a reader is right to distrust.
    """
    lines = ["  field                          was                now"]
    for name, changed in moved.items():
        for key, (was, now) in changed.items():
            label = name if key == "worst" else f"{name}.{key}"
            wide = abs(now - was) < 5e-7
            fmt = ".17g" if wide else "14.6f"
            lines.append(f"  {label:26s} {was:{fmt}}  {now:{fmt}}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artefact", type=Path, default=ARTEFACT)
    parser.add_argument(
        "--write",
        action="store_true",
        help="correct the file in place; without it the audit runs and writes nothing",
    )
    args = parser.parse_args(argv)

    document = json.loads(args.artefact.read_text(encoding="utf-8"))
    rebuilt = rebuild_summary(document["summary"])
    moved = audit(document["summary"], rebuilt)
    verdict_moved = verdict_audit(document, rebuilt)

    print(f"auditing {args.artefact.relative_to(REPO_ROOT).as_posix()}")
    print(f"  per-seed values produced at   {document['provenance']['git_rev'][:7]}")
    print(f"  fields audited                {len(document['summary'])}")
    print(f"  per-seed values changed       0 (asserted, not assumed)")
    print(
        f"  verdict fields changed        {len(verdict_moved)}"
        " (recomputed, not argued)"
    )
    if verdict_moved:
        raise SystemExit(
            f"\nSTOP. The verdict moves: {verdict_moved}. A correction that changes "
            "what the sweep concluded is not a correction, it is a different result, "
            "and it needs the run rather than this tool."
        )

    if not moved:
        print("  fields moved                  none — the file already agrees")
        return 0

    unexpected = sorted(set(moved) - set(EXPECTED_MOVERS))
    print(f"  fields moved                  {len(moved)}")
    print(render(moved))
    if unexpected:
        raise SystemExit(
            f"\nSTOP. {unexpected} moved and was not among the four fields the "
            "direction defect touched. That is a defect this correction did not "
            "know about, and it needs a diagnosis rather than an overwrite."
        )

    if not args.write:
        print("\n  audit only — pass --write to correct the file")
        return 0

    document["summary"] = rebuilt
    document["summary_correction"] = {
        "why": (
            "summarise() reports `worst` as the maximum, which is right for a cost "
            "and backwards for a benefit. M5 is the first milestone to report "
            "capture fractions, and four aggregates were taken in the wrong "
            "direction, reporting the sweep's BEST seed as its worst."
        ),
        "produced_at_rev": document["provenance"]["git_rev"],
        "recomputed_at_rev": _rev(),
        "recomputed_from": "this file's own per-seed values",
        "per_seed_values_changed": 0,
        "verdict_changed": False,
        "fields": {
            name: {
                key: {"was": was, "now": now}
                for key, (was, now) in changed.items()
            }
            for name, changed in moved.items()
        },
        "fields_audited_unchanged": sorted(set(document["summary"]) - set(moved)),
        "tool": "tools/rebuild_m5_summary.py",
    }
    document["provenance"]["unmodified_run_output"] = False
    document["provenance"]["see"] = "summary_correction"
    # Preserve the line ending the run wrote. `results/` carries no .gitattributes
    # rule, so these files are stamped with whatever the producing checkout used
    # (CRLF here); rewriting them LF would bury four corrected numbers under
    # fifteen thousand lines of ending churn and make the diff unreadable, which
    # defeats the point of correcting them in a way a reader can check.
    newline = "\r\n" if b"\r\n" in args.artefact.read_bytes() else "\n"
    args.artefact.write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8", newline=newline
    )
    print(f"\n  corrected, and the file now says so under `summary_correction`")
    print(f"  recomputed at rev             {_rev()[:7]} (dirty: {_dirty()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
