"""Where the people are in a frame, with YOLOX-Tiny.

MediaPipe's pose model is made for one person in the picture: asked for
several, it found both people of a video with two in 27 of 447 frames,
and its own object detector (EfficientDet) found a single person in half
of them. YOLOX-Tiny (Megvii, Apache-2.0 licence, trained on COCO) found
both in 445 of those 447 frames and one person in 266 of the 267 of a
video with one, in 21 ms a frame on the CPU (calibrated on 2026-11-18).
The pose model then measures each person in their own part of the
image.

It runs with OpenCV (no other library). The model (20 MB) is downloaded
the first time, like the pose model, and nothing leaves the computer.
"""
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..exceptions import RagError
from .pose import CACHE, download

MODEL_URL = ("https://github.com/Megvii-BaseDetection/YOLOX/releases/"
             "download/0.1.1rc0/yolox_tiny.onnx")
# The model's square input (pixels).
SIZE = 416
# The model's grids: one cell every this many pixels.
STRIDES = (8, 16, 32)
# Ours: a person is found from this score (0-1); YOLOX-Nano at this score
# took other things for people, Tiny did not (2026-11-18).
MIN_SCORE = 0.5
# Two boxes overlapping more than this are one person (YOLOX's own).
NMS_OVERLAP = 0.45
# Ours: people shorter than this share of the image's height are too
# far to measure: their skeleton is made up (someone passing at the end
# of the corridor in C4, a tenth of the image, gave four lifts that never
# happened; the people measured fill about 0.8 of it, 2026-11-18).
MIN_HEIGHT = 0.25

# x0, y0, x1, y1 in the image, from 0 to 1.
Box = tuple[float, float, float, float]


def _grid() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The centre of each cell of the model's grids, and its stride."""
    cells, strides = [], []
    for stride in STRIDES:
        n = SIZE // stride
        y, x = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        cells.append(np.stack((x, y), 2).reshape(-1, 2))
        strides.append(np.full((n * n, 1), stride))
    return (np.concatenate(cells).astype(float),
            np.concatenate(strides).astype(float))


class PersonDetector:
    """Finds the boxes of the people in a frame."""

    def __init__(self, cache: Path = CACHE) -> None:
        """Load the model (downloaded the first time).

        Raises:
            RagError: It cannot be downloaded or loaded.
        """
        import cv2

        path = download(MODEL_URL, cache / "yolox_tiny.onnx",
                        "people detector")
        try:
            self._net = cv2.dnn.readNetFromONNX(str(path))
        except cv2.error as e:
            raise RagError(f"Cannot load the people detector: {e}") from e
        self._cells, self._strides = _grid()

    def find(self, image: NDArray[np.uint8],
             min_score: float = MIN_SCORE,
             min_height: float = MIN_HEIGHT) -> list[tuple[Box, float]]:
        """The people in an RGB image near enough to measure them: their
        boxes and scores, best first."""
        import cv2

        height, width = image.shape[:2]
        if not height or not width:
            return []
        ratio = min(SIZE / height, SIZE / width)
        inner = (max(1, int(width * ratio)), max(1, int(height * ratio)))
        canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
        canvas[:inner[1], :inner[0]] = cv2.resize(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR), inner)
        self._net.setInput(canvas.transpose(2, 0, 1)[None].astype(
            np.float32))
        out = np.asarray(self._net.forward()[0], dtype=float)
        centre = (out[:, :2] + self._cells) * self._strides
        size = np.exp(out[:, 2:4]) * self._strides
        # Person is COCO's first class.
        score = out[:, 4] * out[:, 5]
        keep = score >= min_score
        if not keep.any():
            return []
        centre, size, score = centre[keep], size[keep], score[keep]
        corners = np.column_stack((centre - size / 2, size)) / ratio
        chosen = cv2.dnn.NMSBoxes(corners.tolist(), score.tolist(),
                                  min_score, NMS_OVERLAP)
        found = []
        for n in np.array(chosen, dtype=int).flatten():
            x, y, w, h = corners[n]
            if h / height < min_height:
                continue
            found.append(((max(0.0, x / width), max(0.0, y / height),
                           min(1.0, (x + w) / width),
                           min(1.0, (y + h) / height)), float(score[n])))
        return sorted(found, key=lambda item: -item[1])
