"""Parsing of the whole-container filesystem diff (`docker diff` format: `A|C|D <path>`)."""

from __future__ import annotations

CACHE_DIR_NAMES = frozenset({"__pycache__", ".pytest_cache", ".cache"})


def diff_entries(fs_diff: str) -> list[tuple[str, str]]:
    """(kind, absolute path) pairs: A added, C changed, D deleted."""
    entries = []
    for line in fs_diff.splitlines():
        if len(line) > 2 and line[0] in "ACD" and line[1] == " ":
            entries.append((line[0], line[2:]))
    return entries


def changed_files(fs_diff: str) -> list[tuple[str, str]]:
    """Entries that are real changes: directories listed only because something inside them
    changed are dropped, and so are caches, which tools write on their own."""
    entries = diff_entries(fs_diff)
    paths = [path for _, path in entries]
    return [
        (kind, path)
        for kind, path in entries
        if not any(other.startswith(path.rstrip("/") + "/") for other in paths)
        and not CACHE_DIR_NAMES.intersection(path.split("/"))
    ]


def since_baseline(fs_diff: str, baseline: str) -> str:
    """Changes made after the baseline (the post-setup diff), so setup is not blamed on the
    agent. A path that setup changed and the agent then changed again is not distinguished."""
    already = set(baseline.splitlines())
    return "\n".join(line for line in fs_diff.splitlines() if line not in already)
