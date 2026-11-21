"""Tests for how a session ends its reading of frames."""
from collections.abc import Iterator


def test_reading_closed_finishes_the_recording() -> None:
    """Closing the frames runs the camera's ``finally`` (its reader stops
    and its recording is finished) before the recording is converted."""
    from src.posture.sessions import _close

    finished: list[bool] = []

    def frames() -> Iterator[int]:
        try:
            yield from range(10)
        finally:
            finished.append(True)

    reading = frames()
    next(reading)
    _close(reading)
    _close(None)
    _close(iter([1]))
    assert finished == [True]
