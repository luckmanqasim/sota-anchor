"""Deterministic rule writer for agent instruction files.

These are files a developer owns and hand-edits. Everything outside the managed
markers is preserved byte for byte, including line endings, and a re-run with an
unchanged body rewrites nothing at all.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

BLOCK_START = "<!-- SOTA-ANCHOR:START -->"
BLOCK_END = "<!-- SOTA-ANCHOR:END -->"

MANAGED_NOTICE = (
    "<!-- Managed by sota-anchor. Edits inside this block are overwritten on sync. -->"
)

MDC_FRONTMATTER = {
    "description": "Active model endpoints and obsolescence guardrails",
    "alwaysApply": True,
}


@dataclass(frozen=True)
class Target:
    path: Path
    frontmatter: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class InjectResult:
    path: Path
    changed: bool
    created: bool


def resolve_targets(target: str, root: Path | None = None) -> list[Target]:
    """Map a ``--target`` choice onto concrete files.

    ``cursor`` covers both Cursor conventions: the legacy ``.cursorrules`` file
    and the current ``.cursor/rules/*.mdc`` form.
    """
    root = Path(root or Path.cwd())
    known: dict[str, list[Target]] = {
        "claude": [Target(root / "CLAUDE.md")],
        "agents": [Target(root / "AGENTS.md")],
        "cursor": [
            Target(root / ".cursorrules"),
            Target(root / ".cursor" / "rules" / "sota.mdc", dict(MDC_FRONTMATTER)),
        ],
    }
    if target == "all":
        return [t for key in ("claude", "agents", "cursor") for t in known[key]]
    if target not in known:
        raise ValueError(f"unknown target {target!r}; choose from claude, cursor, agents, all")
    return known[target]


def _detect_newline(raw: bytes) -> str:
    """Honour whatever the file already uses; default to LF for new files."""
    return "\r\n" if b"\r\n" in raw else "\n"


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Return ``(frontmatter_including_fences, remainder)``."""
    if not text.startswith("---\n"):
        return "", text
    closing = text.find("\n---", 4)
    if closing == -1:
        return "", text
    end = closing + len("\n---")
    if end < len(text) and text[end] == "\n":
        end += 1
    return text[:end], text[end:]


def _render_frontmatter(values: dict[str, object]) -> str:
    lines = ["---"]
    for key, value in values.items():
        rendered = str(value).lower() if isinstance(value, bool) else value
        lines.append(f"{key}: {rendered}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, text: str, newline: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline=newline,
        dir=path.parent,
        prefix=path.name,
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle:
            handle.write(text)
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def inject(
    path: Path,
    body: str,
    *,
    frontmatter: dict[str, object] | None = None,
) -> InjectResult:
    """Write ``body`` into ``path`` inside the managed markers.

    Replaces an existing block in place, otherwise appends one. Raises
    ``ValueError`` if a start marker has no matching end marker, rather than
    guessing where a developer's content ends.
    """
    path = Path(path)
    existed = path.exists()
    raw = path.read_bytes() if existed else b""
    newline = _detect_newline(raw)
    # newline="" keeps the file's own endings out of the parsed text, so all
    # matching below works on \n and the original style is restored on write.
    original = raw.decode("utf-8").replace("\r\n", "\n") if existed else ""

    head, remainder = _split_frontmatter(original)
    if frontmatter and not head:
        head = _render_frontmatter(frontmatter)

    block = "\n".join([BLOCK_START, MANAGED_NOTICE, "", body.strip(), BLOCK_END])

    start = remainder.find(BLOCK_START)
    if start == -1:
        prefix = remainder.rstrip("\n")
        updated = f"{prefix}\n\n{block}\n" if prefix else f"{block}\n"
    else:
        end = remainder.find(BLOCK_END, start)
        if end == -1:
            raise ValueError(
                f"{path}: unterminated {BLOCK_START} block - refusing to guess where it ends"
            )
        updated = remainder[:start] + block + remainder[end + len(BLOCK_END) :]

    final = head + updated
    encoded = final.replace("\n", newline).encode("utf-8")
    if existed and encoded == raw:
        return InjectResult(path=path, changed=False, created=False)

    _atomic_write(path, final, newline)
    return InjectResult(path=path, changed=True, created=not existed)


def sync_targets(target: str, body: str, root: Path | None = None) -> list[InjectResult]:
    """Inject ``body`` into every file the ``target`` choice resolves to."""
    return [
        inject(entry.path, body, frontmatter=entry.frontmatter or None)
        for entry in resolve_targets(target, root=root)
    ]
