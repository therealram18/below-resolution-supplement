"""Shared by the tests: numbers as the paper prints them, and small boards written as files."""
import subprocess

import pytest


def printed(*texts):
    """Return comparisons that accept any value which rounds to each printed number."""
    return [pytest.approx(float(text), abs=0.5 * 10 ** -len(text.partition(".")[2])) for text in texts]


def write(root, files):
    """Write {path: text or bytes} under `root`."""
    for path, content in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(content if isinstance(content, bytes) else content.encode())


def commit(root, files, date):
    """Write `files` under the git repository `root`, commit them at `date` and return the commit."""
    def git(*args):
        return subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                              check=True, capture_output=True, env={"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date,
                                                                    "PATH": __import__("os").environ["PATH"]}).stdout
    if not (root / ".git").exists():
        root.mkdir(parents=True, exist_ok=True)
        git("init", "--quiet")
    write(root, files)
    git("add", "--all")
    git("commit", "--quiet", "--message", date)
    return git("rev-parse", "HEAD").decode().strip()
