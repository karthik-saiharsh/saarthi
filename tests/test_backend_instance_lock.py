import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from backend.errors import BackendError
from backend.instance_lock import InstanceLock, InstanceRunningError

HOLDER: str = """
import sys, time
from pathlib import Path
from backend.instance_lock import InstanceLock
lock = InstanceLock(Path(sys.argv[1]))
lock.acquire()
print("held", flush=True)
time.sleep(60)
"""


def test_the_lock_can_be_taken_and_released_again(tmp_path: Path) -> None:
    lock: InstanceLock = InstanceLock(tmp_path / "x.lock")
    assert not lock.is_held
    lock.acquire()
    assert lock.is_held
    lock.release()
    lock.release()  # Safe when the lock is not held
    assert not lock.is_held
    lock.acquire()  # It can be taken again
    lock.release()


def test_a_second_lock_on_the_same_file_is_refused_and_names_the_holder(
    tmp_path: Path,
) -> None:
    path: Path = tmp_path / "x.lock"
    first: InstanceLock = InstanceLock(path)
    first.acquire()
    try:
        with pytest.raises(InstanceRunningError, match=str(os.getpid())):
            InstanceLock(path).acquire()
    finally:
        first.release()
    InstanceLock(path).acquire()  # Free again after the release


def test_the_lock_file_is_private_and_holds_the_process_id(tmp_path: Path) -> None:
    path: Path = tmp_path / "x.lock"
    with InstanceLock(path) as lock:
        assert lock.is_held
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert path.read_text().strip() == str(os.getpid())
    assert not lock.is_held


def test_the_error_is_a_backend_error() -> None:
    assert issubclass(InstanceRunningError, BackendError)


def test_a_lock_in_another_process_is_respected_and_a_crash_frees_it(
    tmp_path: Path,
) -> None:
    path: Path = tmp_path / "x.lock"
    child: subprocess.Popen[str] = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(path)],
        stdout=subprocess.PIPE,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    try:
        assert child.stdout is not None and child.stdout.readline().strip() == "held"
        with pytest.raises(InstanceRunningError, match=str(child.pid)):
            InstanceLock(path).acquire()
        child.kill()  # A crash: no clean release
        child.wait(timeout=10)
        deadline: float = time.monotonic() + 5
        lock: InstanceLock = InstanceLock(path)
        while True:
            try:
                lock.acquire()
                break
            except InstanceRunningError:
                assert (
                    time.monotonic() < deadline
                ), "The lock was not freed by the crash"
                time.sleep(0.05)
        lock.release()
    finally:
        if child.poll() is None:
            child.kill()


def test_the_lock_checks_its_parameters(tmp_path: Path) -> None:
    with pytest.raises(AssertionError):
        InstanceLock("x.lock")  # type: ignore[arg-type]
    lock: InstanceLock = InstanceLock(tmp_path / "x.lock")
    lock.acquire()
    try:
        with pytest.raises(AssertionError):
            lock.acquire()  # Already held by this object
    finally:
        lock.release()
