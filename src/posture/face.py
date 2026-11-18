"""The face of a person at a screen, with MediaPipe Face Landmarker.

Only used when the camera is the screen's own (a person seated in front
of it): the face then fills enough of the image to measure where the
head points, how far the eyes are and when they blink. Like the pose
model, it runs in "VIDEO" mode, on the CPU, and no image leaves the
computer. Nothing identifies the person: only angles, a distance and
the blinks are kept.
"""
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..exceptions import RagError
from .pose import CACHE, _quiet, download
from .sources import Frame

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "face_landmarker/face_landmarker/float16/latest/"
             "face_landmarker.task")
# The iris is about 11.7 mm wide in almost every adult: its size in the
# image gives the distance to the camera.
IRIS_MM = 11.7
# Horizontal field of view of a usual laptop camera (degrees), for the
# focal length: the cameras do not say theirs. A wider camera makes the
# distances come out shorter than they are.
FIELD_OF_VIEW = 65.0
# Landmarks of the iris edges (left and right of each iris), in the
# image, from MediaPipe's 478-point face mesh.
IRISES = ((469, 471), (474, 476))


@dataclass(frozen=True)
class Face:
    """What the face shows in one frame.

    Attributes:
        timestamp: Seconds since the start.
        pitch: Head turned down (positive) or up (negative), degrees,
            seen from the camera.
        yaw: Head turned to one side, degrees (sign: the side).
        roll: Head tilted to one side, degrees.
        distance_cm: Eyes to camera, estimated from the iris; ``None``
            when the irises are not measured.
        blink: How closed the eyes are, from 0 (open) to 1 (closed).
        centre: Where the face is in the image (x and y from 0 to 1), to
            tell whose it is; ``None`` when unknown.
    """

    timestamp: float
    pitch: float
    yaw: float
    roll: float
    distance_cm: float | None
    blink: float
    centre: tuple[float, float] | None = None


def head_angles(matrix: NDArray[np.float64]) -> tuple[float, float, float]:
    """Pitch, yaw and roll (degrees) of the face's transformation matrix.

    MediaPipe gives the face's pose as a 4x4 matrix in the camera's
    space (x right, y up, z towards the camera): its rotation is
    decomposed as yaw (around y), pitch (around x) and roll (around z).
    The sign of the pitch was checked on a recording: it is positive
    when the head bends down (about +10° looking at a laptop's screen
    from a seat, +24° looking at the keyboard, 0° looking straight at the
    camera).
    """
    rotation = np.asarray(matrix, dtype=float)[:3, :3]
    pitch = math.degrees(math.atan2(-rotation[1, 2], rotation[2, 2]))
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, rotation[0, 2]))))
    roll = math.degrees(math.atan2(-rotation[0, 1], rotation[0, 0]))
    return pitch, yaw, roll


def eye_distance_cm(points: NDArray[np.float64], width: int,
                    field_of_view: float = FIELD_OF_VIEW) -> float | None:
    """Distance from the eyes to the camera, from the irises' width.

    Args:
        points: The face landmarks, x and y from 0 to 1 of the image.
        width: The image's width in pixels.
        field_of_view: The camera's horizontal field of view, degrees.

    Returns:
        Centimetres, or ``None`` without the iris landmarks.
    """
    if len(points) <= max(max(pair) for pair in IRISES):
        return None
    focal = (width / 2) / math.tan(math.radians(field_of_view) / 2)
    sizes = [math.dist(points[a][:2], points[b][:2]) * width
             for a, b in IRISES]
    iris = max(sizes)
    if iris <= 0:
        return None
    return focal * IRIS_MM / iris / 10


class FaceEstimator:
    """Finds the faces in each frame, in order (like ``PoseEstimator``)."""

    def __init__(self, cache: Path = CACHE, people: int = 1) -> None:
        """Load the model (a few MB, downloaded the first time).

        Args:
            cache: Folder where it is kept.
            people: The most faces found in a frame.

        Raises:
            RagError: The model cannot be downloaded or loaded.
        """
        from mediapipe.tasks.python import BaseOptions, vision

        path = download(MODEL_URL, cache / "face_landmarker.task",
                        "face")
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO, num_faces=people,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True)
        try:
            with _quiet():
                self._landmarker = (
                    vision.FaceLandmarker.create_from_options(options))
        except (RuntimeError, ValueError) as e:
            raise RagError(f"Cannot load the face model: {e}") from e
        self._last = -1

    def find(self, frame: Frame) -> Face | None:
        """The (first) face in a frame, or ``None`` without one."""
        found = self.find_all(frame)
        return found[0] if found else None

    def find_all(self, frame: Frame) -> list[Face]:
        """The faces in a frame, in no particular order."""
        import mediapipe as mp

        millisecond = int(frame.timestamp * 1000)
        if millisecond <= self._last:
            return []
        self._last = millisecond
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(frame.image))
        with _quiet():
            found = self._landmarker.detect_for_video(image, millisecond)
        faces = []
        for n, landmarks in enumerate(found.face_landmarks):
            if n >= len(found.facial_transformation_matrixes):
                break
            points = np.array([[p.x, p.y, p.z] for p in landmarks],
                              dtype=float)
            pitch, yaw, roll = head_angles(
                np.asarray(found.facial_transformation_matrixes[n]))
            shapes = ({s.category_name: s.score
                       for s in found.face_blendshapes[n]}
                      if n < len(found.face_blendshapes) else {})
            blink = max(shapes.get("eyeBlinkLeft", 0.0),
                        shapes.get("eyeBlinkRight", 0.0))
            centre = points[:, :2].mean(axis=0)
            faces.append(Face(frame.timestamp, pitch, yaw, roll,
                              eye_distance_cm(points, frame.image.shape[1]),
                              blink, (float(centre[0]), float(centre[1]))))
        return faces

    def close(self) -> None:
        """Free the model."""
        with _quiet():
            self._landmarker.close()
