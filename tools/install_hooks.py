#!/usr/bin/env python3
"""Put this repository's hooks where git will find them.

The hooks live in `tools/hooks/` so that they are tracked and reviewed with the code
they guard, and this copies them into the directory git actually reads. That
directory is not always `.git/hooks`: in a linked worktree `.git` is a file, and the
hooks belong to the common directory the worktree shares with its siblings. Git is
asked where, rather than guessed at.

    python tools/install_hooks.py            # install
    python tools/install_hooks.py --check    # report whether they are installed

Exit status: 0 when every hook is in place (or was put there); 1 when one is missing
under --check.
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import stat
import subprocess
import sys
from pathlib import Path


def hooks_directory(root: Path) -> Path:
    found = subprocess.run(
        ["git", "rev-parse", "--git-path", "hooks"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if found.returncode != 0:
        raise SystemExit(f"not a git repository: {root}")

    directory = Path(found.stdout.strip())
    if not directory.is_absolute():
        directory = (root / directory).resolve()
    return directory


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report, change nothing")
    options = parser.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    source = root / "tools" / "hooks"
    target = hooks_directory(root)

    if not options.check:
        target.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []

    for hook in sorted(source.iterdir()):
        if not hook.is_file():
            continue

        installed = target / hook.name
        if installed.is_file() and filecmp.cmp(hook, installed, shallow=False):
            print(f"installed  {hook.name}")
            continue

        if options.check:
            missing.append(hook.name)
            print(f"MISSING    {hook.name}")
            continue

        shutil.copyfile(hook, installed)
        installed.chmod(installed.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print(f"installed  {hook.name} -> {installed}")

    if options.check and missing:
        print(
            f"install_hooks: {len(missing)} hook(s) not installed; "
            f"run python tools/install_hooks.py",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
