"""Tests for the live camera: newest frame only, reconnection, names."""
from pathlib import Path
import threading
import time
from typing import Any

import cv2
import numpy as np
import pytest

from src.exceptions import RagError
from src.posture.sources import Camera, safe_name


class FakeCapture:
    """Stands for ``cv2.VideoCapture``: each opening plays one script.

    A script is a list of frame values (each read gives the next one as a
    16x16 image) ending when it runs out, or ``None`` for a camera that
    does not open.
    """

    scripts: list[list[int] | None] = []
    opened: list[Any] = []
    settings: list[tuple[int, float]] = []
    pause = 0.0

    def __init__(self, device: Any) -> None:
        script = FakeCapture.scripts.pop(0) if FakeCapture.scripts else None
        self.frames = list(script) if script is not None else None
        self.released = False
        FakeCapture.opened.append(device)

    def isOpened(self) -> bool:
        return self.frames is not None

    def read(self) -> tuple[bool, Any]:
        if not self.frames:
            return False, None
        time.sleep(FakeCapture.pause)
        value = self.frames.pop(0)
        return True, np.full((16, 16, 3), value, dtype=np.uint8)

    def release(self) -> None:
        self.released = True

    def set(self, prop: int, value: float) -> bool:
        FakeCapture.settings.append((prop, value))
        return True


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> type[FakeCapture]:
    FakeCapture.scripts = []
    FakeCapture.opened = []
    FakeCapture.settings = []
    FakeCapture.pause = 0.0
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    return FakeCapture


def values(camera: Camera, slow: float = 0.0) -> list[int]:
    found = []
    for frame in camera:
        found.append(int(frame.image[0, 0, 0]))
        time.sleep(slow)
    return found


class TestCamera:
    """Reading a camera in a thread."""

    def test_all_frames_when_fast(self, fake: type[FakeCapture]) -> None:
        """A reader that keeps up gets every frame, in order."""
        fake.scripts = [[1, 2, 3]]
        fake.pause = 0.02
        assert values(Camera(0, reconnects=0)) == [1, 2, 3]

    def test_slow_reader_gets_newest(self, fake: type[FakeCapture]) -> None:
        """Frames read while the analysis works are dropped: it never
        falls behind, and the last one is not lost."""
        fake.scripts = [list(range(1, 41))]
        fake.pause = 0.005
        got = values(Camera(0, reconnects=0), slow=0.05)
        assert len(got) < 40
        assert got == sorted(got)
        assert got[-1] == 40

    def test_reconnects_after_cut(self, fake: type[FakeCapture]) -> None:
        """A camera that stops is opened again and goes on."""
        fake.scripts = [[1, 2], [3, 4]]
        fake.pause = 0.02
        warnings: list[str] = []
        camera = Camera("rtsp://u:p@h/s", reconnects=2, delay=0.01,
                        warn=warnings.append)
        assert values(camera) == [1, 2, 3, 4]
        assert fake.opened == ["rtsp://u:p@h/s"] * 4
        assert "reconectando (1/2)" in warnings[0]
        assert "p@" not in " ".join(warnings)

    def test_gives_up(self, fake: type[FakeCapture]) -> None:
        """After ``reconnects`` failed attempts in a row it ends."""
        fake.scripts = [[1], None, None]
        warnings: list[str] = []
        camera = Camera(0, reconnects=2, delay=0.01, warn=warnings.append)
        assert values(camera) == [1]
        assert "tras 2 intentos" in warnings[-1]

    def test_cannot_open(self, fake: type[FakeCapture]) -> None:
        """A camera that does not open fails at once, without the
        password in the message."""
        fake.scripts = [None]
        with pytest.raises(RagError) as error:
            list(Camera("rtsp://admin:secret@h/s"))
        assert "secret" not in str(error.value)

    def test_every(self, fake: type[FakeCapture]) -> None:
        """Frames closer than ``every`` to the last one are skipped."""
        fake.scripts = [list(range(10))]
        fake.pause = 0.01
        assert len(values(Camera(0, every=0.05, reconnects=0))) <= 4

    def test_stopping_ends_the_thread(self, fake: type[FakeCapture]) -> None:
        """When the analysis stops asking, the reading thread ends."""
        fake.scripts = [list(range(1000))]
        fake.pause = 0.001
        before = threading.active_count()
        for _ in Camera(0, reconnects=0):
            break
        assert threading.active_count() == before

    def test_name_has_no_password(self) -> None:
        assert Camera("rtsp://a:b@h/s").name == "camera rtsp://***@h/s"


class TestSafeName:
    """Credentials never reach a message or a report."""

    @pytest.mark.parametrize(("device", "name"), [
        (0, "camera 0"),
        ("rtsp://admin:se:cr@t@192.168.1.50:554/s1",
         "camera rtsp://***@192.168.1.50:554/s1"),
        ("rtsp://user@h/", "camera rtsp://***@h/"),
        ("http://10.0.0.2:8080/video", "camera http://10.0.0.2:8080/video"),
        ("rtsp://h/cam?channel=1&user=admin&password=x&flag",
         "camera rtsp://h/cam?channel=1&user=***&password=***&flag"),
    ])
    def test_names(self, device: int | str, name: str) -> None:
        assert safe_name(device) == name


class TestRecorder:
    """The original video, at the camera's measured rate."""

    def test_measured_rate(self, tmp_path: Path) -> None:
        import cv2

        from src.posture.sources import Recorder

        path = tmp_path / "v.mp4"
        recorder = Recorder(path)
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        for frame in range(75):  # 3 s at 25 fps
            recorder.add(frame / 25, image)
        recorder.close()
        video = cv2.VideoCapture(str(path))
        assert video.get(cv2.CAP_PROP_FPS) == pytest.approx(25, abs=0.5)
        assert video.get(cv2.CAP_PROP_FRAME_COUNT) == 75
        video.release()

    def test_short_recording_kept(self, tmp_path: Path) -> None:
        from src.posture.sources import Recorder

        recorder = Recorder(tmp_path / "v.mp4")
        recorder.add(0.0, np.zeros((48, 64, 3), dtype=np.uint8))
        recorder.close()
        assert (tmp_path / "v.mp4").stat().st_size > 0

    def test_camera_records(self, fake: type[FakeCapture],
                            tmp_path: Path) -> None:
        fake.scripts = [[1, 2, 3]]
        fake.pause = 0.02
        camera = Camera(0, reconnects=0, record=tmp_path / "c.mp4")
        assert values(camera) == [1, 2, 3]
        assert (tmp_path / "c.mp4").is_file()

    def test_no_record_by_default(self, fake: type[FakeCapture],
                                  tmp_path: Path) -> None:
        fake.scripts = [[1]]
        assert Camera(0, reconnects=0).record is None


class TestCameraSize:
    """A camera of this computer is asked for 1280x720 in MJPG."""

    def test_local_camera(self, fake: type[FakeCapture]) -> None:
        import cv2

        fake.scripts = [[1]]
        values(Camera(0, reconnects=0))
        settings = dict(fake.settings)
        assert settings[cv2.CAP_PROP_FRAME_WIDTH] == 1280
        assert settings[cv2.CAP_PROP_FRAME_HEIGHT] == 720
        assert settings[cv2.CAP_PROP_FOURCC] == cv2.VideoWriter.fourcc(
            *"MJPG")

    def test_stream_left_alone(self, fake: type[FakeCapture]) -> None:
        """An IP camera sends the size it is set to send."""
        fake.scripts = [[1]]
        values(Camera("rtsp://h/s", reconnects=0))
        assert fake.settings == []
