"""Where a board's files are read from: a folder, or one commit of a git clone that is kept in a cache.

Both give the same three calls, so a board's loader reads a folder, the board's current state and every state
of its history in the same way.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from fnmatch import fnmatchcase
from pathlib import Path
from typing import ClassVar

CACHE = Path(os.environ.get("VERNIER_CACHE") or Path.home() / ".cache" / "vernier")
LFS_POINTER = b"version https://git-lfs"  # a git blob that stands for a large file kept outside git


class BoardError(Exception):
    """A problem with the board's files or the request, stated for the user."""


class Unreachable(BoardError):
    """The board's repository could not be cloned."""


def git(clone, *args, input=None) -> bytes:
    """Return the output of a git command run in `clone`."""
    try:
        done = subprocess.run(["git", "-C", str(clone), *args], input=input, capture_output=True, check=False)
    except FileNotFoundError:
        raise BoardError("git is not installed, and the board's files are read with it") from None
    if done.returncode:
        reason = done.stderr.decode(errors="replace").strip().splitlines()[-1:]
        raise BoardError(f"git {args[0]} failed: {' '.join(reason)}")
    return done.stdout


def clone_of(url) -> Path:
    """Return a bare clone of `url` in the cache: made on first use, brought up to date on every later one."""
    clone = CACHE / re.sub(r"[^A-Za-z0-9.-]+", "_", url.split("://")[-1])
    if not clone.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"Cloning {url}", file=sys.stderr)
        # The clone is made beside its place and moved there whole, so a stopped or second run leaves no half clone.
        making = Path(tempfile.mkdtemp(dir=CACHE))
        try:
            git(CACHE, "clone", "--quiet", "--bare", "--single-branch", "--", url, str(making))
            making.rename(clone)
        except BoardError as error:
            raise Unreachable(f"cannot reach {url}: {error}") from None
        except OSError:  # another run of vernier put its clone there first
            pass
        finally:
            shutil.rmtree(making, ignore_errors=True)
        return clone
    branch = git(clone, "symbolic-ref", "--short", "HEAD").decode().strip()
    try:
        git(clone, "fetch", "--quiet", "origin", f"+{branch}:{branch}")
    except BoardError as error:
        print(f"Could not update {url} ({error}); using the copy in {clone}", file=sys.stderr)
    return clone


def _matches(path, pattern) -> bool:
    """Return whether `path` matches `pattern`, folder by folder, with * for any name."""
    parts, wanted = path.split("/"), pattern.split("/")
    return len(parts) == len(wanted) and all(map(fnmatchcase, parts, wanted))


def read(parse, content, path):
    """Return parse(content), or say which of the board's files cannot be read."""
    try:
        return parse(content)
    except Exception as error:  # noqa: BLE001  a board's file can fail a parser in any way; each way gets this message
        raise BoardError(f"the board's file {path} cannot be read ({type(error).__name__}: {error}); "
                         "--at COMMIT reads the board as of an earlier commit") from None


class Folder:
    """The files under a folder."""

    def __init__(self, root):
        self.root = Path(root)

    def files(self, pattern) -> dict:
        """Return {path: identity} of the files that match `pattern`; a folder's files have no history."""
        return {path.relative_to(self.root).as_posix(): str(path)
                for path in sorted(self.root.glob(pattern)) if path.is_file()}

    def parsed(self, pattern, parse) -> dict:
        """Return {path: parse(content)} of the files that match `pattern`."""
        return {path: read(parse, Path(file).read_bytes(), path) for path, file in self.files(pattern).items()}


class Commit:
    """The files under `path` at one commit of a clone. Each distinct file is read and parsed once."""

    _parsed: ClassVar[dict] = {}  # (blob, parse) -> result; a blob id names one content in every clone

    def __init__(self, clone, commit, path, large=None):
        self.clone, self.commit, self.path, self.large = clone, commit, path, large
        listing = git(clone, "ls-tree", "-r", "-z", commit, "--", path + "/").decode().split("\0")[:-1]
        # Each item is "<mode> blob <id>\t<path>"; -z keeps a path with unusual characters as it is.
        self.blobs = {item.split("\t", 1)[1][len(path) + 1:]: item.split()[2] for item in listing}

    def files(self, pattern) -> dict:
        """Return {path: identity} of the files that match `pattern`; the identity changes with the content."""
        return {path: blob for path, blob in self.blobs.items() if _matches(path, pattern)}

    def _contents(self, files) -> list:
        """Return the content of each (path, blob), read by one git process."""
        output = git(self.clone, "cat-file", "--batch", input="".join(blob + "\n" for _, blob in files).encode())
        contents, start = [], 0
        for _ in files:  # git writes each as "<blob> <type> <size>\n<content>\n"
            header_end = output.index(b"\n", start)
            size = int(output[start:header_end].split()[2])
            contents.append(output[header_end + 1:header_end + 1 + size])
            start = header_end + size + 2
        if not self.large:
            return contents

        def real(file, content):
            return self.large(self.commit, f"{self.path}/{file[0]}") if content.startswith(LFS_POINTER) else content

        with ThreadPoolExecutor(8) as pool:
            return list(pool.map(real, files, contents))

    def parsed(self, pattern, parse) -> dict:
        """Return {path: parse(content)} of the files that match `pattern`."""
        files = self.files(pattern)
        new = [(path, blob) for path, blob in files.items() if (blob, parse) not in self._parsed]
        if new:
            for (path, blob), content in zip(new, self._contents(new)):
                self._parsed[blob, parse] = read(parse, content, path)
        return {path: self._parsed[blob, parse] for path, blob in files.items()}


def hub_file(repo_id, repo_type):
    """Return the function that downloads one large file of a Hugging Face repository at one commit."""
    def fetch(commit, path) -> bytes:
        from huggingface_hub import hf_hub_download  # imported here: only this kind of board needs it
        try:
            return Path(hf_hub_download(repo_id, path, repo_type=repo_type, revision=commit)).read_bytes()
        except Exception as error:  # noqa: BLE001  the library's network errors share no type across its versions
            raise Unreachable(f"cannot fetch {path} of {repo_id} from Hugging Face ({type(error).__name__})") from None
    return fetch


def updates(clone, path, until=None) -> list:
    """Return [commit, date] of each commit that changed `path`, oldest first, up to `until`."""
    log = git(clone, "log", "--reverse", "--format=%H %aI", "--end-of-options", until or "HEAD", "--", path).decode()
    return [line.split() for line in log.splitlines()]
