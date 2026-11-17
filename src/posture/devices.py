"""The cameras of this computer, to offer them in the interface.

On Linux every camera is a ``/dev/videoN`` device, but one camera often
has two or more of them: the picture (index 0) and metadata (index 1 and
up), which cannot be opened as a camera. Only the picture devices are
listed. Elsewhere the first camera numbers are tried with OpenCV.
"""
from pathlib import Path
import re

from pydantic import BaseModel

SYSFS = Path("/sys/class/video4linux")
# Camera numbers tried where there is no sysfs (macOS, Windows).
PROBED = 4


class Device(BaseModel):
    """A camera of this computer.

    Attributes:
        number: What OpenCV opens it with (``--camera <number>``).
        name: The name the system gives it.
    """

    number: int
    name: str


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def list_cameras(sysfs: Path = SYSFS) -> list[Device]:
    """The cameras that can be opened, by number.

    Args:
        sysfs: Where Linux lists the video devices (a parameter for the
            tests).

    Returns:
        One device per camera, in number order.
    """
    if sysfs.is_dir():
        found = []
        for entry in sysfs.iterdir():
            match = re.fullmatch(r"video(\d+)", entry.name)
            # Index 0 is the picture; the others are metadata nodes.
            if match and _read(entry / "index") in ("", "0"):
                name = _read(entry / "name") or entry.name
                found.append(Device(number=int(match.group(1)), name=name))
        return sorted(found, key=lambda device: device.number)
    return _probe()


def _probe() -> list[Device]:
    """Try the first camera numbers with OpenCV (no sysfs)."""
    import cv2

    found = []
    for number in range(PROBED):
        capture = cv2.VideoCapture(number)
        if capture.isOpened():
            found.append(Device(number=number, name=f"Cámara {number}"))
        capture.release()
    return found
