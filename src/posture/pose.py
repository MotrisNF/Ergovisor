"""The skeleton of the person in each frame, with MediaPipe Pose.

MediaPipe's "VIDEO" mode reads one frame at a time with its time and
uses the previous frames to follow the person, which is exactly what a
live camera gives: the same estimator works for a file and a camera.

The model (a ``.task`` file of 6 to 31 MB) is downloaded from Google the
first time and kept in a cache folder; after that everything runs
offline, on the CPU, and no image leaves the computer.
"""
from collections.abc import Iterator
from contextlib import contextmanager
import os
from pathlib import Path
import sys
from typing import Literal
import urllib.request

import numpy as np

from ..exceptions import RagError
from .skeleton import Skeleton
from .sources import Frame

Model = Literal["lite", "full", "heavy"]
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "pose_landmarker/pose_landmarker_{0}/float16/latest/"
             "pose_landmarker_{0}.task")
CACHE = Path.home() / ".cache" / "rag-posture"
# Set to see MediaPipe's own log lines (written by its C++ part, which no
# Python setting silences).
VERBOSE_ENV = "RAG_POSE_VERBOSE"


@contextmanager
def _quiet() -> Iterator[None]:
    """Hide what MediaPipe's C++ part writes to stderr meanwhile.

    It prints warnings about its model ("Sparsity tensors...") that say
    nothing to the person. Errors still arrive as Python exceptions.
    """
    if os.environ.get(VERBOSE_ENV):
        yield
        return
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def model_path(model: Model = "full", cache: Path = CACHE) -> Path:
    """The model file, downloaded the first time.

    Args:
        model: ``lite`` is fastest, ``heavy`` most precise.
        cache: Folder where it is kept.

    Raises:
        RagError: It is not there and cannot be downloaded.
    """
    path = cache / f"pose_landmarker_{model}.task"
    if path.is_file():
        return path
    cache.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    try:
        urllib.request.urlretrieve(MODEL_URL.format(model), partial)
    except OSError as e:
        partial.unlink(missing_ok=True)
        raise RagError(f"Cannot download the pose model '{model}': "
                       f"{type(e).__name__}") from e
    partial.rename(path)
    return path


class PoseEstimator:
    """Finds one person's skeleton in each frame, in order.

    The frames must come with increasing times (as a video or a camera
    gives them): the model follows the person from one to the next.
    """

    def __init__(self, model: Model = "full",
                 cache: Path = CACHE) -> None:
        """Load the model (about a second).

        Raises:
            RagError: The model cannot be downloaded or loaded.
        """
        from mediapipe.tasks.python import BaseOptions, vision

        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(model_path(model, cache))),
            running_mode=vision.RunningMode.VIDEO, num_poses=1)
        try:
            with _quiet():
                self._landmarker = (
                    vision.PoseLandmarker.create_from_options(options))
        except (RuntimeError, ValueError) as e:
            raise RagError(f"Cannot load the pose model: {e}") from e
        self._last = -1

    def find(self, frame: Frame) -> Skeleton | None:
        """The skeleton in a frame, or ``None`` without a person.

        A frame whose time is not after the previous one is skipped:
        MediaPipe needs strictly increasing times.
        """
        import mediapipe as mp

        millisecond = int(frame.timestamp * 1000)
        if millisecond <= self._last:
            return None
        self._last = millisecond
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(frame.image))
        with _quiet():
            found = self._landmarker.detect_for_video(image, millisecond)
        if not found.pose_world_landmarks:
            return None
        world = found.pose_world_landmarks[0]
        points = np.array([[p.x, p.y, p.z] for p in world], dtype=float)
        visibility = np.array([p.visibility or 0.0 for p in world],
                              dtype=float)
        return Skeleton(frame.timestamp, points, visibility)

    def close(self) -> None:
        """Free the model."""
        with _quiet():
            self._landmarker.close()
