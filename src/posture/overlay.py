"""The analysis drawn on the image: each skeleton coloured by its risk,
the score and the main angles, and a warning while a moment of risk
lasts. With several people, each skeleton carries its person's number
and the panel has a line for each.

The same drawing goes to a video file (a recording, to check the
analysis by eye) or to a window on the screen (a live camera).
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from ..exceptions import RagError
from .analysis import Seen
from .angles import Angles
from .lifting import Lift
from .office import (MIN_DISTANCE_CM, NECK_EXTENSION, NECK_FLEXION,
                     NECK_TILT, NECK_TURN, OfficeState)
from .reba import Reba
from .skeleton import MIN_VISIBILITY, Point, Skeleton
from .timeline import Event

# The bones drawn: pairs of MediaPipe landmarks.
BONES = [
    (Point.LEFT_SHOULDER, Point.RIGHT_SHOULDER),
    (Point.LEFT_SHOULDER, Point.LEFT_ELBOW),
    (Point.LEFT_ELBOW, Point.LEFT_WRIST),
    (Point.RIGHT_SHOULDER, Point.RIGHT_ELBOW),
    (Point.RIGHT_ELBOW, Point.RIGHT_WRIST),
    (Point.LEFT_SHOULDER, Point.LEFT_HIP),
    (Point.RIGHT_SHOULDER, Point.RIGHT_HIP),
    (Point.LEFT_HIP, Point.RIGHT_HIP),
    (Point.LEFT_HIP, Point.LEFT_KNEE),
    (Point.LEFT_KNEE, Point.LEFT_ANKLE),
    (Point.RIGHT_HIP, Point.RIGHT_KNEE),
    (Point.RIGHT_KNEE, Point.RIGHT_ANKLE),
    (Point.LEFT_ANKLE, Point.LEFT_FOOT),
    (Point.RIGHT_ANKLE, Point.RIGHT_FOOT),
    (Point.LEFT_EAR, Point.RIGHT_EAR),
]
# The colour of each action level (RGB): no risk to very high.
COLOURS = {0: (46, 204, 113), 1: (46, 204, 113), 2: (241, 196, 15),
           3: (230, 126, 34), 4: (231, 76, 60)}
NO_SCORE = (180, 180, 180)
FONT_FILES = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/TTF/DejaVuSans.ttf",
              "/Library/Fonts/Arial Unicode.ttf"]
WINDOW = "Análisis de posturas (q para salir)"


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """A font with accents and the degree sign, or Pillow's own."""
    for path in FONT_FILES:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def _clock(second: float) -> str:
    """``1:02.3``."""
    return f"{int(second // 60)}:{second % 60:04.1f}"


def _minutes(seconds: float) -> str:
    """``12:05`` (minutes and seconds)."""
    whole = int(max(0.0, seconds))
    return f"{whole // 60}:{whole % 60:02d}"


def _angle(value: float | None) -> str:
    """``45°``, or a dash when not measured."""
    return "–" if value is None else f"{value:.0f}°"


class Overlay:
    """Draws each analysed frame, and writes or shows it.

    Attributes:
        path: Video file to write, or ``None``.
        show: Whether to show a window (a live camera).
        fps: Frames a second of the written video: the analysed ones.
    """

    def __init__(self, path: Path | None = None, show: bool = False,
                 fps: float = 10.0) -> None:
        """Nothing is opened until the first frame.

        Raises:
            RagError: A window is asked for without a screen.
        """
        if show and sys.platform.startswith("linux") and not (
                os.environ.get("DISPLAY") or os.environ.get(
                    "WAYLAND_DISPLAY")):
            raise RagError("There is no screen to show the window on: "
                           "use --overlay to write a video instead.")
        self.path = path
        self.show = show
        self.fps = fps
        self._writer: cv2.VideoWriter | None = None
        self._font = _font(22)
        self._small = _font(18)

    def draw(self, image: NDArray[np.uint8], second: float,
             people: list[Seen]) -> NDArray[np.uint8]:
        """The frame with the analysis of each person on it (RGB).

        At a screen (``office``), the panel shows the screen work
        instead of REBA, and only the upper body is drawn.
        """
        picture = Image.fromarray(image)
        pen = ImageDraw.Draw(picture, "RGBA")
        several = len(people) > 1 or any(p.person > 1 for p in people)
        for seen in people:
            if seen.skeleton.pixels is not None:
                self._skeleton(pen, seen.skeleton, picture.size,
                               self._colour(seen),
                               seen.person if several else None)
        if not people:
            self._panel(pen, second, None, None, NO_SCORE)
        elif len(people) == 1:
            seen = people[0]
            title = f"Persona {seen.person} · " if several else ""
            if seen.office is not None:
                self._office(pen, second, seen.office, self._colour(seen),
                             title)
            else:
                self._panel(pen, second, seen.angles, seen.reba,
                            self._colour(seen), title)
        else:
            self._people(pen, second, people)
        lifts = [(p.lift, p.person) for p in people if p.lift is not None]
        if lifts:
            lift, person = max(
                lifts, key=lambda item: item[0].end.second or 0.0)
            self._lift(pen, lift, picture.size,
                       f"Persona {person} · " if several else "")
        events = [(p.event, p.person) for p in people if p.event is not None]
        if events:
            event, person = max(events, key=lambda item: item[0].level)
            self._warning(pen, event, picture.size,
                          f"Persona {person} · " if several else "")
        return np.asarray(picture, dtype=np.uint8)

    def _colour(self, seen: Seen) -> tuple[int, int, int]:
        """The colour of a person's risk now."""
        if seen.office is not None:
            return self._office_colour(seen.office)
        return COLOURS[seen.reba.level] if seen.reba is not None else NO_SCORE

    def _people(self, pen: ImageDraw.ImageDraw, second: float,
                people: list[Seen]) -> None:
        """A line for each person, top left, coloured by their risk."""
        lines = []
        for seen in people:
            if seen.office is not None:
                face = seen.office.face
                what = ("pantalla, " + (
                    f"distancia {face.distance_cm:.0f} cm, cuello "
                    f"{face.pitch:+.0f}°"
                    if face is not None and face.distance_cm is not None
                    else "cara no visible"))
            elif seen.reba is not None:
                what = (f"REBA {seen.reba.score} · riesgo "
                        f"{seen.reba.risk.lower()}")
            else:
                what = "sin puntuar (cuerpo no visible)"
            lines.append((f"Persona {seen.person}: {what}",
                          self._colour(seen)))
        height = 14 + 30 * (len(lines) + 1)
        pen.rectangle([10, 10, 520, 10 + height], fill=(0, 0, 0, 160))
        for number, (line, colour) in enumerate(lines):
            top = 14 + 30 * number
            pen.rectangle([10, top, 18, top + 30], fill=colour)
            pen.text((28, 18 + 30 * number), line, fill=(255, 255, 255),
                     font=self._small)
        pen.text((28, 18 + 30 * len(lines)), _clock(second),
                 fill=(255, 255, 255), font=self._small)

    def _lift(self, pen: ImageDraw.ImageDraw, lift: Lift,
              size: tuple[int, int], title: str = "") -> None:
        """The last lift, top right: its limit and its index."""
        width, _ = size
        level = (4 if lift.beyond_limits
                 else 0 if lift.index is None or lift.index <= 1
                 else 3 if lift.index <= 3 else 4)
        lines = [f"{title}{lift.kind.capitalize()} de {lift.d:.0f} cm",
                 f"Peso límite (NIOSH): {lift.limit:.1f} kg"]
        if lift.index is not None:
            lines.append(f"Índice: {lift.index:.2f} · {lift.risk.lower()}")
        elif lift.beyond_limits:
            lines.append("Fuera de los límites de la ecuación")
        # As wide as its longest line.
        longest = max(pen.textlength(line, font=self._font if n == 0
                                     else self._small)
                      for n, line in enumerate(lines))
        left = int(width - 10 - longest - 34)
        height = 14 + 30 * len(lines)
        pen.rectangle([left, 10, width - 10, 10 + height],
                      fill=(0, 0, 0, 160))
        pen.rectangle([left, 10, left + 8, 10 + height],
                      fill=COLOURS[level])
        for number, line in enumerate(lines):
            pen.text((left + 18, 18 + 30 * number), line,
                     fill=(255, 255, 255),
                     font=self._font if number == 0 else self._small)

    def _skeleton(self, pen: ImageDraw.ImageDraw, skeleton: Skeleton,
                  size: tuple[int, int], colour: tuple[int, int, int],
                  person: int | None = None) -> None:
        """The bones and joints seen, in the colour of the risk, and the
        person's number over the head (when there are several)."""
        assert skeleton.pixels is not None
        width, height = size
        thick = max(3, width // 250)

        def at(point: Point) -> tuple[float, float]:
            assert skeleton.pixels is not None
            x, y = skeleton.pixels[point]
            return float(x) * width, float(y) * height

        for a, b in BONES:
            if skeleton.seen(a, b):
                pen.line([at(a), at(b)], fill=colour, width=thick)
        for point in {p for bone in BONES for p in bone}:
            if skeleton.seen(point):
                x, y = at(point)
                pen.ellipse([x - thick, y - thick, x + thick, y + thick],
                            fill=(255, 255, 255))
        if person is not None:
            seen = skeleton.pixels[skeleton.visibility >= MIN_VISIBILITY]
            if len(seen):
                x = float(seen[:, 0].mean()) * width
                y = max(0.0, float(seen[:, 1].min()) * height - 44)
                label = str(person)
                pen.rectangle([x - 18, y, x + 18, y + 34], fill=colour)
                pen.text((x - pen.textlength(label, font=self._font) / 2,
                          y + 4), label, fill=(0, 0, 0), font=self._font)

    def _panel(self, pen: ImageDraw.ImageDraw, second: float,
               angles: Angles | None, reba: Reba | None,
               colour: tuple[int, int, int], title: str = "") -> None:
        """The score and the main angles, top left."""
        if reba is None:
            lines = [f"{title}Sin puntuar" if title
                     else "Sin persona detectada"]
        else:
            lines = [f"{title}REBA {reba.score} · riesgo "
                     f"{reba.risk.lower()} (nivel {reba.level})"]
        if angles is not None:
            left = angles.left.upper if angles.left else None
            right = angles.right.upper if angles.right else None
            lines += [f"Tronco {_angle(angles.trunk)}   Cuello "
                      f"{_angle(angles.neck)}   Rodillas "
                      f"{_angle(angles.knee)}",
                      f"Brazo izq. {_angle(left)}   dcho. {_angle(right)}"]
        lines.append(_clock(second))
        height = 14 + 30 * len(lines)
        pen.rectangle([10, 10, 470, 10 + height], fill=(0, 0, 0, 160))
        pen.rectangle([10, 10, 18, 10 + height], fill=colour)
        for number, line in enumerate(lines):
            pen.text((28, 18 + 30 * number), line, fill=(255, 255, 255),
                     font=self._font if number == 0 else self._small)

    @staticmethod
    def _office_colour(office: OfficeState) -> tuple[int, int, int]:
        """Green when the screen work is right now within the guide,
        yellow or orange when something is off."""
        face = office.face
        if face is None:
            return NO_SCORE
        if face.distance_cm is not None and face.distance_cm < MIN_DISTANCE_CM:
            return COLOURS[3]
        if (face.pitch > NECK_FLEXION or face.pitch < -NECK_EXTENSION
                or abs(face.yaw) > NECK_TURN or abs(face.roll) > NECK_TILT):
            return COLOURS[2]
        return COLOURS[1]

    def _office(self, pen: ImageDraw.ImageDraw, second: float,
                office: OfficeState, colour: tuple[int, int, int],
                title: str = "") -> None:
        """The screen work, top left: distance, neck, blinks, rests."""
        face = office.face
        lines = [f"{title}Puesto con pantalla"]
        if face is None:
            lines.append("Cara no visible (mirando a otro lado)")
        else:
            distance = (f"{face.distance_cm:.0f} cm"
                        if face.distance_cm is not None else "–")
            lines += [f"Distancia {distance}   Cuello {face.pitch:+.0f}° "
                      f"giro {face.yaw:+.0f}° incl. {face.roll:+.0f}°"]
        rate = (f"{office.blinks_per_minute:.0f}/min"
                if office.blinks_per_minute is not None else "–")
        lines += [f"Parpadeo {rate}   Sin descanso visual "
                  f"{_minutes(office.since_rest)}",
                  f"Sin pausa {_minutes(office.since_break)}   "
                  f"{_clock(second)}"]
        height = 14 + 30 * len(lines)
        pen.rectangle([10, 10, 520, 10 + height], fill=(0, 0, 0, 160))
        pen.rectangle([10, 10, 18, 10 + height], fill=colour)
        for number, line in enumerate(lines):
            pen.text((28, 18 + 30 * number), line, fill=(255, 255, 255),
                     font=self._font if number == 0 else self._small)

    def _warning(self, pen: ImageDraw.ImageDraw, event: Event,
                 size: tuple[int, int], title: str = "") -> None:
        """A band at the bottom while a moment of risk lasts (the worst,
        with several people)."""
        width, height = size
        text = (f"{title}Riesgo {event.risk.lower()} desde "
                f"{_clock(event.start)}: "
                + (", ".join(event.causes) or "postura"))
        pen.rectangle([0, height - 50, width, height],
                      fill=(*COLOURS[event.level], 220))
        pen.text((16, height - 40), text[:110], fill=(0, 0, 0),
                 font=self._small)

    def write(self, image: NDArray[np.uint8]) -> bool:
        """Write and/or show a drawn frame.

        Returns:
            False when the window was closed with ``q``: the analysis
            should stop.
        """
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if self.path is not None:
            if self._writer is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                height, width = bgr.shape[:2]
                self._writer = cv2.VideoWriter(
                    str(self.path), cv2.VideoWriter.fourcc(*"mp4v"),
                    self.fps, (width, height))
            self._writer.write(bgr)
        if self.show:
            cv2.imshow(WINDOW, bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False
        return True

    def close(self) -> None:
        """Finish the video file and close the window.

        OpenCV writes MPEG-4 part 2, which many players and browsers do
        not open; with ffmpeg installed the file is turned into H.264,
        which any of them plays.
        """
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            if self.path is not None:
                _to_h264(self.path)
        if self.show:
            cv2.destroyAllWindows()


def _to_h264(path: Path, preset: str = "medium") -> None:
    """Re-encode a video as H.264 in place, if ffmpeg is there.

    Args:
        path: The video.
        preset: ffmpeg's speed for the encoding: ``ultrafast`` is several
            times faster and gives a bigger file of the same picture (for
            long camera recordings); ``medium`` is ffmpeg's default.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return
    converted = path.with_name(f"{path.stem}.h264{path.suffix}")
    done = subprocess.run(
        [ffmpeg, "-loglevel", "error", "-y", "-i", str(path), "-c:v",
         "libx264", "-preset", preset, "-pix_fmt", "yuv420p", "-crf", "23",
         "-movflags", "+faststart", str(converted)], check=False)
    if done.returncode == 0:
        converted.replace(path)
    else:
        converted.unlink(missing_ok=True)
