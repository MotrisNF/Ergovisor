"""The skeletons of a video, computed once and replayed.

The pose models take minutes on a long video; the rest of the analysis
(people, REBA, lifts, pushes, screen work) takes seconds. So each video
is run through the models once and every frame's skeletons and faces
are kept in a cache; ``replay`` then runs the whole analysis on them as
often as needed, to try a change on every calibration video at once.

The cache holds the skeletons of the people filmed: it stays on this
computer (``tools/calibration/cache/`` is not committed).

    uv run python -m tools.calibration.cache B5 B7      # build them
    uv run python -m tools.calibration.cache --all
"""
import argparse
from collections.abc import Callable, Iterator
from pathlib import Path
import pickle
from typing import Any

import numpy as np

from src.posture.analysis import Analysis, analyse
from src.posture.face import FaceEstimator
from src.posture.lifting import Lifting
from src.posture.office import Office
from src.posture.pose import PoseEstimator
from src.posture.pushing import Pushing
from src.posture.sources import VideoFile

# Where the caches are kept (not committed: skeletons of real people).
CACHE = Path("tools/calibration/cache")
# The calibration videos, by the name ``calibracion_videos.md`` gives
# them (they are not committed either: they show people).
VIDEOS: dict[str, str] = {
    **{f"B{n}": f"data/uploads/B{n}.mp4" for n in range(1, 8)},
    **{f"C{n}": f"data/uploads/C{n}.mp4" for n in range(1, 6)},
    "sample": "data/video/Como levantar peso correctamente y prevenir el "
              "dolor de espalda(1).mp4",
    "E1": "data/uploads/grabacion_2026-11-18_13-15-28.mp4",
    "E2": "data/uploads/grabacion_2026-11-18_14-38-04.mp4",
    "D1": "data/uploads/grabacion_2026-11-18_15-02-54.mp4",
    "F2": "data/uploads/grabacion_2026-11-20_2.mp4",
    "F3": "data/uploads/grabacion_2026-11-20_3.mp4",
}
# Seconds between two analysed frames, as the application.
EVERY = 0.1
# The most people followed, as the application.
PEOPLE = 3

# One frame's record: when, the image's size, the skeletons, the faces
# (``None``: the face model was not asked).
Record = tuple[float, tuple[int, int], list[Any], list[Any] | None]


class _Frame:
    """A frame without its picture: its time and its size are all the
    analysis needs once the skeletons are known."""

    def __init__(self, timestamp: float, shape: tuple[int, int]) -> None:
        self.timestamp = timestamp
        self.image = np.zeros((shape[0], shape[1], 1), np.uint8)


def path_of(name: str) -> Path:
    """The cache of a calibration video."""
    return CACHE / f"{name}.pkl"


def build(name: str, video: str | None = None) -> Path:
    """Run the pose and face models over a video and keep what they
    found in each frame.

    Args:
        name: The video's name (a key of ``VIDEOS``, or any name with
            ``video``).
        video: Its file, if not one of ``VIDEOS``.

    Returns:
        The cache file.
    """
    source = video or VIDEOS[name]
    records: list[Record] = []
    pose = PoseEstimator("full", people=PEOPLE)
    faces = FaceEstimator(people=PEOPLE)

    class Recording:
        """The models, keeping what they find."""

        def find_all(self, frame: Any) -> list[Any]:
            found = pose.find_all(frame)
            records.append((frame.timestamp, frame.image.shape[:2], found,
                            None))
            return found

    class Faces:
        def find_all(self, frame: Any) -> list[Any]:
            found = faces.find_all(frame)
            timestamp, shape, skeletons, _ = records[-1]
            records[-1] = (timestamp, shape, skeletons, found)
            return found

    try:
        analyse(VideoFile(source, EVERY), Recording(),  # type: ignore
                source, lifting=Lifting(), office=Office(),
                faces=Faces(), pushing=Pushing(),  # type: ignore
                max_people=PEOPLE)
    finally:
        pose.close()
        faces.close()
    CACHE.mkdir(parents=True, exist_ok=True)
    path = path_of(name)
    path.write_bytes(pickle.dumps(records))
    return path


def replay(name: str, lifting: Lifting | None = None,
           on_result: Callable[[Any, Any], Any] | None = None,
           **kwargs: Any) -> Analysis:
    """The whole analysis of a cached video, as the application does it
    (lifting, pushing and screen work, up to ``PEOPLE`` people).

    Args:
        name: The video's name.
        lifting: The lifting set-up (by default, no load given).
        on_result: Called with each frame and what it shows.
        **kwargs: More arguments for ``analyse`` (``on_lift``...).

    Raises:
        FileNotFoundError: The video has no cache yet (``build`` it).
    """
    records: list[Record] = pickle.loads(path_of(name).read_bytes())
    skeletons: Iterator[list[Any]] = iter([r[2] for r in records])
    faces = {r[0]: r[3] or [] for r in records}

    class Pose:
        def find_all(self, frame: Any) -> list[Any]:
            return next(skeletons)

    class Faces:
        def find_all(self, frame: Any) -> list[Any]:
            return faces[frame.timestamp]

    frames = [_Frame(r[0], r[1]) for r in records]
    return analyse(frames, Pose(), name,  # type: ignore[arg-type]
                   lifting=lifting or Lifting(), office=Office(),
                   faces=Faces(), pushing=Pushing(),  # type: ignore
                   on_result=on_result, max_people=PEOPLE, **kwargs)


def records(name: str) -> list[Record]:
    """The raw records of a cached video (the skeletons of each frame)."""
    loaded: list[Record] = pickle.loads(path_of(name).read_bytes())
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("names", nargs="*", help="videos (keys of VIDEOS)")
    parser.add_argument("--all", action="store_true",
                        help="every calibration video found")
    parser.add_argument("--again", action="store_true",
                        help="build the caches that exist too")
    args = parser.parse_args()
    names = ([n for n, v in VIDEOS.items() if Path(v).is_file()]
             if args.all else args.names)
    for name in names:
        if path_of(name).exists() and not args.again:
            print(f"{name}: cached")
            continue
        if not Path(VIDEOS[name]).is_file():
            print(f"{name}: no video at {VIDEOS[name]}")
            continue
        print(f"{name}: {build(name)}", flush=True)


if __name__ == "__main__":
    main()
