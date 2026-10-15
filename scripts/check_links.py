#!/usr/bin/env python3
"""Fail if a relative link in a tracked Markdown file points at a missing path.

Standard library only, so that CI can run it without the devShell:

    python3 scripts/check_links.py

Links with a URL scheme (https:, mailto:, ...) and same-file anchors (#...) are
not checked. For a relative link, the anchor is removed and the path is checked
relative to the file that has the link.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FENCE = re.compile(r"^(```|~~~).*?^\1", re.MULTILINE | re.DOTALL)
INLINE_CODE = re.compile(r"`[^`\n]*`")
# [text](target "title") and ![alt](target); the text can continue on the next line.
INLINE_LINK = re.compile(r"\]\(\s*<?([^)\s>]+)>?(?:\s+[\"'(][^)]*)?\)")
# [label]: target
REFERENCE_LINK = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*<?(\S+?)>?(?:\s|$)", re.MULTILINE)
SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def tracked_markdown() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "*.md"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    return [ROOT / name for name in out.split("\0") if name]


def broken_links(path: Path) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8")
    # Blank code, keeping the newlines so that line numbers stay correct.
    text = FENCE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)
    text = INLINE_CODE.sub(lambda m: " " * len(m.group(0)), text)
    broken = []
    for pattern in (INLINE_LINK, REFERENCE_LINK):
        for match in pattern.finditer(text):
            target = match.group(1)
            if target.startswith("#") or SCHEME.match(target):
                continue
            target_path = target.split("#", 1)[0]
            if not (path.parent / target_path).exists():
                line = text.count("\n", 0, match.start(1)) + 1
                broken.append((line, target))
    return broken


def main() -> int:
    failures = 0
    for path in tracked_markdown():
        for line, target in broken_links(path):
            print(f"{path.relative_to(ROOT)}:{line}: broken link: {target}")
            failures += 1
    if failures:
        print(f"{failures} broken link(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
