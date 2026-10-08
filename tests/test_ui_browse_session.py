"""Tests of the state of the memory page. The session has no NiceGUI code, so no browser is needed."""

import asyncio

import pytest

from ui.browse_session import PAGE_SIZE, BrowseSession
from ui.ports import BrowseError, FrameDetail, FrameInfo, FramesPage


def info(image_id: int, status: str = "done") -> FrameInfo:
    return FrameInfo(image_id, status, None, 60, 80, 60, 1, 0, 2)


def detail(image_id: int) -> FrameDetail:
    return FrameDetail(info(image_id), 100.0, (), (), (), (), ())


class FakeBrowse:
    def __init__(self, total: int = 3) -> None:
        self.total = total
        self.calls: list[tuple[str | None, int, int]] = []
        self.error: Exception | None = None
        self.picture_error: Exception | None = None

    async def list_frames(self, status, offset, limit):
        self.calls.append((status, offset, limit))
        if self.error is not None:
            raise self.error
        ids = range(offset + 1, min(offset + limit, self.total) + 1)
        return FramesPage(
            [info(i) for i in ids], self.total, offset, limit, {"done": self.total}
        )

    async def get_frame(self, image_id):
        if self.error is not None:
            raise self.error
        return detail(image_id)

    async def get_picture(self, image_id, max_side):
        if self.picture_error is not None:
            raise self.picture_error
        return b"jpeg" + bytes([image_id % 256])


def make() -> tuple[BrowseSession, FakeBrowse, list[str]]:
    service = FakeBrowse()
    notices: list[str] = []
    session = BrowseSession(service, on_notice=lambda t, e: notices.append(t))  # type: ignore[arg-type]
    return session, service, notices


def test_load_gets_the_page_and_a_picture_for_each_frame() -> None:
    session, service, _ = make()
    asyncio.run(session.load())
    assert service.calls == [(None, 0, PAGE_SIZE)]
    assert session.page is not None and [f.id for f in session.page.frames] == [1, 2, 3]
    assert set(session.thumbnails) == {1, 2, 3} and not session.busy


def test_a_missing_picture_does_not_hide_the_frame() -> None:
    session, service, _ = make()
    service.picture_error = BrowseError("gone")
    asyncio.run(session.load())
    assert session.page is not None and len(session.page.frames) == 3
    assert session.thumbnails == {}


def test_a_filter_starts_at_the_first_page() -> None:
    session, service, _ = make()
    asyncio.run(session.load())
    asyncio.run(session.set_status("rejected"))
    assert session.status == "rejected" and service.calls[-1] == (
        "rejected",
        0,
        PAGE_SIZE,
    )


def test_the_same_filter_does_not_load_again() -> None:
    session, service, _ = make()
    asyncio.run(session.load())
    asyncio.run(session.set_status(None))
    assert len(service.calls) == 1


def test_the_page_buttons_move_by_one_page() -> None:
    session, service, _ = make()
    service.total = PAGE_SIZE + 5
    asyncio.run(session.load())
    assert session.has_next and not session.has_previous
    asyncio.run(session.next_page())
    assert (
        service.calls[-1] == (None, PAGE_SIZE, PAGE_SIZE)
        and session.has_previous
        and not session.has_next
    )
    asyncio.run(session.next_page())  # No next page: nothing happens
    assert len(service.calls) == 2
    asyncio.run(session.previous_page())
    assert service.calls[-1] == (None, 0, PAGE_SIZE)


def test_open_and_close_a_frame() -> None:
    session, _, _ = make()
    changes: list[int] = []
    session._on_changed = lambda: changes.append(1)
    asyncio.run(session.open(2))
    assert session.detail is not None and session.detail.frame.id == 2
    assert session.detail_picture == b"jpeg\x02"
    session.close()
    assert session.detail is None and session.detail_picture is None
    count = len(changes)
    session.close()  # Already closed: no change event
    assert len(changes) == count


def test_a_frame_without_a_picture_still_opens() -> None:
    session, service, _ = make()
    service.picture_error = BrowseError("gone")
    asyncio.run(session.open(1))
    assert session.detail is not None and session.detail_picture is None


@pytest.mark.parametrize(
    "error",
    [BrowseError("This frame does not exist any more."), RuntimeError("secret")],
)
def test_an_error_gives_a_notice_and_keeps_the_state(error: Exception) -> None:
    session, service, notices = make()
    asyncio.run(session.load())
    service.error = error
    asyncio.run(session.load())
    asyncio.run(session.open(1))
    assert len(notices) == 2 and all("secret" not in n for n in notices)
    assert session.page is not None and session.detail is None and not session.busy
