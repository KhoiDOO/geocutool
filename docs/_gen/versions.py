"""Release discovery and checkout for the versioned API reference.

Versions are never written down. The list comes from ``git tag``, the folder
name is derived from the tag, and the manifest the site reads is generated from
whatever was actually built. Adding a release therefore needs no edit here.

A released version is documented from its own source, obtained with
``git worktree`` rather than by mutating the working tree, so a build cannot
disturb uncommitted work.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

#: A tag names a release when it is ``v`` followed by a dotted numeric version.
#: Anything else in the tag list -- a branch snapshot, a release candidate -- is
#: skipped rather than guessed at.
_RELEASE_TAG = re.compile(r"^v(\d+(?:\.\d+)*)$")


@dataclass(frozen=True)
class VersionRef:
    """One documentable release."""

    tag: str  #: e.g. "v0.7.4"
    version: str  #: e.g. "0.7.4"
    slug: str  #: directory name under docs/api/, derived from the tag

    @property
    def key(self) -> tuple:
        """Numeric sort key, so 0.7.10 orders after 0.7.9."""
        return tuple(int(part) for part in self.version.split("."))


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=str(repo_root), capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()[:300]}")
    return result.stdout


def discover(repo_root: Path) -> List[VersionRef]:
    """Every release tag in the repository, newest first.

    Returns:
        List[VersionRef]: Sorted by version number descending. Empty if the
        directory is not a git repository, which is not an error -- the current
        build works without any history.
    """
    try:
        raw = _git(repo_root, "tag", "--list")
    except (RuntimeError, FileNotFoundError):
        return []

    refs = []
    for line in raw.splitlines():
        tag = line.strip()
        match = _RELEASE_TAG.match(tag)
        if match:
            refs.append(VersionRef(tag=tag, version=match.group(1), slug=tag))
    return sorted(refs, key=lambda r: r.key, reverse=True)


def resolve(repo_root: Path, wanted: List[str]) -> List[VersionRef]:
    """Select releases by tag or bare version string.

    Accepts either form ("v0.7.4" or "0.7.4") so the caller does not have to
    know which the repository uses.
    """
    available = {r.tag: r for r in discover(repo_root)}
    available.update({r.version: r for r in available.values()})
    out, missing = [], []
    for name in wanted:
        ref = available.get(name)
        (out.append(ref) if ref else missing.append(name))
    if missing:
        raise SystemExit(f"unknown release(s): {', '.join(missing)}")
    return sorted(set(out), key=lambda r: r.key, reverse=True)


@contextmanager
def worktree(repo_root: Path, ref: VersionRef,
             parent: Optional[Path] = None) -> Iterator[Path]:
    """Check a release out into a throwaway worktree.

    The checkout lives outside the repository so it cannot be picked up by the
    site's own file walks, and it is removed even if the build raises.

    Yields:
        Path: Root of the checked-out source tree.
    """
    base = Path(tempfile.mkdtemp(prefix=f"c3d-{ref.slug}-", dir=parent))
    path = base / "src"
    try:
        _git(repo_root, "worktree", "add", "--detach", "--quiet", str(path), ref.tag)
        yield path
    finally:
        # --force because the build may have left Doxygen scratch behind, and
        # prune in case a previous run was killed before cleaning up.
        subprocess.run(["git", "worktree", "remove", "--force", str(path)],
                       cwd=str(repo_root), capture_output=True, text=True)
        subprocess.run(["git", "worktree", "prune"],
                       cwd=str(repo_root), capture_output=True, text=True)
        shutil.rmtree(base, ignore_errors=True)


def read_version(root: Path) -> str:
    """Version declared by a source tree's own pyproject.toml."""
    try:
        text = (root / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return "0.0.0"
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else "0.0.0"
