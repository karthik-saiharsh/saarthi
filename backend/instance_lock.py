"""A lock that allows only one running copy of the application for one set of data.

Two copies on the same database and vector index damage each other: the
start-up recovery of the second copy resets the images that the first copy is
working on, and each copy saves its own vector index over the other.

The lock is an exclusive `flock` on a file. The operating system releases it
when the process ends, also after a crash or `kill -9`. A lock file that is
left behind is therefore never a problem.
"""

import fcntl
import logging
import os
from pathlib import Path
from types import TracebackType
from typing import Self

from backend.errors import BackendError

logger: logging.Logger = logging.getLogger(__name__)

LOCK_FILE_MODE: int = 0o600


class InstanceRunningError(BackendError):
    """Raised when another copy of the application already holds the lock."""


class InstanceLock:
    """An exclusive lock on a file. Use `acquire` and `release`, or a `with` block."""

    def __init__(self, path: Path) -> None:
        assert isinstance(path, Path), "path must be a Path"
        self._path: Path = path
        self._descriptor: int | None = None

    @property
    def is_held(self) -> bool:
        return self._descriptor is not None

    def acquire(self) -> None:
        """Take the lock. Raise `InstanceRunningError` if another process holds it."""
        assert self._descriptor is None, "the lock is already held"
        descriptor: int = os.open(self._path, os.O_RDWR | os.O_CREAT, LOCK_FILE_MODE)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            holder: str = (
                os.pread(descriptor, 32, 0).decode(errors="replace").strip()
                or "unknown"
            )
            os.close(descriptor)
            raise InstanceRunningError(
                f"Another copy of Saarthi is running (process {holder}). It holds {self._path}"
            ) from error
        except OSError:
            os.close(descriptor)
            raise
        # Write the process id, so the message of a second copy can name the first one.
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"{os.getpid()}\n".encode())
        self._descriptor = descriptor
        logger.info("Instance lock taken: %s", self._path)

    def release(self) -> None:
        """Give up the lock. It is safe to call this when the lock is not held."""
        if self._descriptor is None:
            return
        fcntl.flock(self._descriptor, fcntl.LOCK_UN)
        os.close(self._descriptor)
        self._descriptor = None
        logger.info("Instance lock released")

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        self.release()
