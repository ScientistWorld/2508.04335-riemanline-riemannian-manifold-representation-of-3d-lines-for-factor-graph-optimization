"""Read-only public workspace and per-call writable directories for a solver.

Linux Landlock confines file access even through absolute paths and symlinks.
The scorer itself is unrestricted.  This is solely an answer-file boundary;
it does not police algorithms, imports, solver settings or search strategies.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path


class Ruleset(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class Beneath(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def restrict_filesystem(read_paths: list[Path], write_paths: list[Path]) -> None:
    """Called in the child just before exec; failure prevents running a solver."""
    libc = ctypes.CDLL(None, use_errno=True)

    def checked(result: int) -> int:
        if result < 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))
        return result

    abi = checked(libc.syscall(444, 0, 0, 1))  # landlock_create_ruleset VERSION
    if abi < 3:
        raise RuntimeError("Hidden-answer isolation requires Landlock ABI >= 3")
    # Execute, read/write, unlink, mkdir, creation, rename/refer and truncate.
    handled = (1 << 15) - 1
    read = (1 << 0) | (1 << 2) | (1 << 3)
    ruleset = Ruleset(handled)
    fd = checked(libc.syscall(444, ctypes.byref(ruleset), ctypes.sizeof(ruleset), 0))
    try:
        for path, writable in [(p, False) for p in read_paths] + [(p, True) for p in write_paths]:
            if not path.exists():
                continue
            parent = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                access = handled if writable else read
                if not path.is_dir():
                    access &= (1 << 0) | (1 << 1) | (1 << 2) | (1 << 14)
                rule = Beneath(access, parent)
                checked(libc.syscall(445, fd, 1, ctypes.byref(rule), 0))
            finally:
                os.close(parent)
        checked(libc.prctl(38, 1, 0, 0, 0))  # PR_SET_NO_NEW_PRIVS
        checked(libc.syscall(446, fd, 0))  # landlock_restrict_self
    finally:
        os.close(fd)


def solver_paths(root: Path, view: Path, inputs: Path, outputs: Path) -> tuple[list[Path], list[Path]]:
    # No ancestor of eval/test, another solver's outputs, or evaluator scratch
    # is granted. Permissions on symlinks resolve to their actual target.
    public = [root / name for name in ("method", "baseline", "data", "scripts", "scoring",
                                        "redteam", "briefing", "paper", "outputs")]
    public += [root / "eval" / "__init__.py", root / "eval" / "train", view]
    public += [Path(p) for p in ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/opt", "/proc", "/dev")]
    # Dataset/model symlinks may point outside the public workspace.
    downloads = root / "data" / "downloads"
    if downloads.is_dir():
        public += [p.resolve() for p in downloads.iterdir()]
    # Do not grant a planted public symlink to the private evaluator, nor a
    # broad ancestor grant through a dataset/model symlink.
    protected = [(root / "eval" / "test").resolve(), Path("/home/user/eval/test").resolve()]
    safe_public = []
    for path in public:
        resolved = path.resolve()
        hidden_component = any(a == "eval" and b == "test" for a, b in zip(resolved.parts, resolved.parts[1:]))
        if hidden_component or any(resolved == p or resolved in p.parents or p in resolved.parents for p in protected):
            continue
        safe_public.append(path)
    writable = [inputs, outputs, view / "tmp", Path("/dev/null")]
    return safe_public, writable
