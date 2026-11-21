"""Frames of a calibration video at given seconds, with the cached
skeleton drawn (green: seen; red: made up by the model), in one image.

    uv run python -m tools.calibration.sheet F3 out.jpg 10 20 30 45
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from tools.calibration.cache import VIDEOS, records

# The bones drawn (MediaPipe's point numbers).
BONES = [(11, 12), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
         (24, 26), (26, 28), (11, 13), (13, 15), (12, 14), (14, 16),
         (0, 11), (0, 12)]
TILE = (400, 225)
PER_ROW = 4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("name", help="a cached video")
    parser.add_argument("output", type=Path, help="the image to write")
    parser.add_argument("seconds", type=float, nargs="+")
    args = parser.parse_args()
    cached = records(args.name)
    capture = cv2.VideoCapture(VIDEOS[args.name])
    tiles = []
    for second in args.seconds:
        capture.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        ok, image = capture.read()
        if not ok:
            continue
        _, _, skeletons, _ = min(cached, key=lambda r: abs(r[0] - second))
        height, width = image.shape[:2]
        for skeleton in skeletons:
            pixels = skeleton.pixels
            for a, b in BONES:
                colour = ((0, 255, 0) if skeleton.visibility[b] > 0.5
                          else (0, 0, 255))
                cv2.line(image, (int(pixels[a][0] * width),
                                 int(pixels[a][1] * height)),
                         (int(pixels[b][0] * width),
                          int(pixels[b][1] * height)), colour, 4)
        cv2.putText(image, f"{second:.1f}s", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
        tiles.append(cv2.resize(image, TILE))
    capture.release()
    if not tiles:
        raise SystemExit("No frame read at those seconds.")
    blank = np.zeros_like(tiles[0])
    rows = [np.hstack(tiles[i:i + PER_ROW]
                      + [blank] * (PER_ROW - len(tiles[i:i + PER_ROW])))
            for i in range(0, len(tiles), PER_ROW)]
    cv2.imwrite(str(args.output), np.vstack(rows))
    print(args.output)


if __name__ == "__main__":
    main()
