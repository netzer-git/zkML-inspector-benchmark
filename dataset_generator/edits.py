"""Edit-operation engine for applying artifact edits to in-memory file trees.

Operates on a Codebase (dict mapping relative path → list of lines).
Each line does NOT include a trailing newline.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import PurePosixPath
from typing import Any


class EditError(Exception):
    """Raised when an edit operation fails (hash mismatch, missing file, etc.)."""


# Type alias: {relative_path: [line, ...]} — lines without trailing newline.
Codebase = dict[str, list[str]]


def _validate_path(path: str) -> None:
    """Reject paths with '..' components or absolute paths."""
    if ".." in PurePosixPath(path).parts:
        raise EditError(f"Path traversal rejected: {path}")
    if PurePosixPath(path).is_absolute():
        raise EditError(f"Absolute path rejected: {path}")


def _sha256_lines(lines: list[str]) -> str:
    """SHA-256 of lines joined by '\\n' with a trailing '\\n'."""
    content = "\n".join(lines) + "\n"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _resolve_anchor_line_range(
    anchor: dict[str, Any], lines: list[str], file_path: str
) -> tuple[int, int]:
    """Resolve a line_range anchor to 0-based (start, end_exclusive) indices.

    Validates expect_sha256 if present.
    """
    start_1 = anchor["start"]  # 1-based inclusive
    end_1 = anchor["end"]  # 1-based inclusive
    start_0 = start_1 - 1
    end_0 = end_1  # exclusive

    if start_0 < 0 or end_0 > len(lines):
        raise EditError(
            f"{file_path}: line_range {start_1}-{end_1} out of bounds "
            f"(file has {len(lines)} lines)"
        )

    sha = anchor.get("expect_sha256")
    if sha:
        actual = _sha256_lines(lines[start_0:end_0])
        if actual != sha:
            raise EditError(
                f"{file_path}:{start_1}-{end_1}: SHA-256 mismatch. "
                f"Expected {sha}, got {actual}"
            )

    return start_0, end_0


def _resolve_anchor_unique_string(
    anchor: dict[str, Any], lines: list[str], file_path: str
) -> int:
    """Resolve a unique_string anchor to the 0-based line index containing the text."""
    text = anchor["text"]
    matches = [i for i, line in enumerate(lines) if text in line]
    if len(matches) == 0:
        raise EditError(f"{file_path}: unique_string '{text}' not found")
    if len(matches) > 1:
        raise EditError(
            f"{file_path}: unique_string '{text}' found {len(matches)} times "
            f"(lines {[m+1 for m in matches]}), expected exactly 1"
        )
    return matches[0]


def _resolve_anchor_regex(
    anchor: dict[str, Any], lines: list[str], file_path: str
) -> int:
    """Resolve a regex anchor to the 0-based line index of the Nth match."""
    pattern = re.compile(anchor["pattern"])
    match_index = anchor.get("match_index", 0)
    matches = [i for i, line in enumerate(lines) if pattern.search(line)]
    if match_index >= len(matches):
        raise EditError(
            f"{file_path}: regex '{anchor['pattern']}' match_index={match_index} "
            f"but only {len(matches)} match(es) found"
        )
    return matches[match_index]


def apply_edit(codebase: Codebase, edit: dict[str, Any]) -> None:
    """Apply a single edit operation to the in-memory codebase.

    Raises EditError on failure. Mutates codebase in place.
    """
    file_path = edit["file"]
    _validate_path(file_path)
    op = edit["op"]
    anchor = edit.get("anchor", {})
    new_content = edit.get("new_content", "")

    if op == "create_file":
        if file_path in codebase:
            raise EditError(f"create_file: '{file_path}' already exists")
        codebase[file_path] = new_content.split("\n") if new_content else [""]
        return

    if file_path not in codebase:
        raise EditError(f"{op}: file '{file_path}' not found in codebase")

    lines = codebase[file_path]
    kind = anchor.get("kind", "line_range")

    if op == "replace_block":
        start, end = _resolve_anchor_line_range(anchor, lines, file_path)
        replacement = new_content.split("\n") if new_content else []
        lines[start:end] = replacement

    elif op == "delete_block":
        start, end = _resolve_anchor_line_range(anchor, lines, file_path)
        del lines[start:end]

    elif op == "insert_after":
        if kind == "line_range":
            _, end = _resolve_anchor_line_range(anchor, lines, file_path)
            insert_at = end
        elif kind == "unique_string":
            idx = _resolve_anchor_unique_string(anchor, lines, file_path)
            insert_at = idx + 1
        elif kind == "regex":
            idx = _resolve_anchor_regex(anchor, lines, file_path)
            insert_at = idx + 1
        else:
            raise EditError(f"insert_after: unsupported anchor kind '{kind}'")
        new_lines = new_content.split("\n") if new_content else []
        lines[insert_at:insert_at] = new_lines

    elif op == "insert_before":
        if kind == "line_range":
            start, _ = _resolve_anchor_line_range(anchor, lines, file_path)
            insert_at = start
        elif kind == "unique_string":
            insert_at = _resolve_anchor_unique_string(anchor, lines, file_path)
        elif kind == "regex":
            insert_at = _resolve_anchor_regex(anchor, lines, file_path)
        else:
            raise EditError(f"insert_before: unsupported anchor kind '{kind}'")
        new_lines = new_content.split("\n") if new_content else []
        lines[insert_at:insert_at] = new_lines

    elif op == "replace_regex":
        regex_pattern = edit.get("regex_pattern")
        expected_count = edit.get("expected_match_count")
        if not regex_pattern:
            raise EditError("replace_regex: missing regex_pattern")

        full_text = "\n".join(lines)
        actual_count = len(re.findall(regex_pattern, full_text))
        if expected_count is not None and actual_count != expected_count:
            raise EditError(
                f"{file_path}: replace_regex expected {expected_count} matches "
                f"of '{regex_pattern}', found {actual_count}"
            )
        full_text = re.sub(regex_pattern, new_content, full_text)
        codebase[file_path] = full_text.split("\n")

    else:
        raise EditError(f"Unknown edit op: {op}")


def apply_artifact_edits(
    codebase: Codebase, edits: list[dict[str, Any]], artifact_id: str
) -> None:
    """Apply all edits for a single artifact atomically.

    On failure, the codebase is restored to its pre-edit state.
    """
    # Snapshot affected files
    affected_files = {e["file"] for e in edits}
    snapshot: dict[str, list[str] | None] = {}
    for f in affected_files:
        _validate_path(f)
        snapshot[f] = list(codebase[f]) if f in codebase else None

    try:
        for i, edit in enumerate(edits):
            try:
                apply_edit(codebase, edit)
            except EditError as e:
                raise EditError(
                    f"Artifact {artifact_id}, edit #{i}: {e}"
                ) from e
    except EditError:
        # Rollback
        for f, original in snapshot.items():
            if original is None:
                codebase.pop(f, None)
            else:
                codebase[f] = original
        raise
