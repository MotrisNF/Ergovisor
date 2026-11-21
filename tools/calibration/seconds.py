"""Second by second, what the analysis measures of the first person of a
cached video: to see when and why it scores what it scores.

    uv run python -m tools.calibration.seconds F3
    uv run python -m tools.calibration.seconds F3 --lifts

Each line: the view (``F`` facing the camera, ``S`` side-on), the trunk
bend before and after the facing-camera rule (``raw``/``trunk``), the
neck, the knees, the higher arm, the highest REBA of that second and
``EV`` within a moment of risk. ``--lifts`` lists each lift found with
its time (shown here only: the reports keep no times).
"""
import argparse
from collections import defaultdict
from typing import Any

import numpy as np

from src.posture.angles import SIDE_SEEN, UP, _signed, body_axes
from src.posture.lifting import Lifting
from src.posture.skeleton import Point
from tools.calibration.cache import replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("name", help="a cached video")
    parser.add_argument("--lifts", action="store_true",
                        help="also list the lifts with their times")
    parser.add_argument("--load", type=float, default=0.0,
                        help="load of the lifts, kg (0: not given)")
    args = parser.parse_args()
    rows: dict[int, list[tuple[float, ...]]] = defaultdict(list)
    lifts: list[Any] = []

    def on_result(frame: Any, shown: Any) -> None:
        for seen in shown[:1]:
            skeleton = seen.skeleton
            _, lateral, forward = body_axes(skeleton)
            hips = skeleton.middle(Point.LEFT_HIP, Point.RIGHT_HIP)
            shoulders = skeleton.middle(Point.LEFT_SHOULDER,
                                        Point.RIGHT_SHOULDER)
            raw = _signed(shoulders - hips, UP, forward)
            angles = seen.angles
            arm = max([abs(a.upper) for a in (angles.left, angles.right)
                       if a is not None], default=0.0) if angles else 0.0
            rows[int(frame.timestamp)].append((
                float(abs(lateral[0]) >= SIDE_SEEN), raw,
                angles.trunk if angles else np.nan,
                angles.neck if angles else np.nan,
                angles.knee if angles else np.nan, arm,
                seen.reba.score if seen.reba else 0.0,
                float(seen.event is not None)))

    replay(args.name, lifting=Lifting(load_kg=args.load or None),
           on_result=on_result,
           on_lift=lambda person, lift: lifts.append(lift))
    for second in sorted(rows):
        v = np.array(rows[second], dtype=float)
        if np.isnan(v[:, 2]).all():
            print(f"{second:4d}s not scored (body not whole)")
            continue
        print(f"{second:4d}s {'F' if v[:, 0].mean() > 0.5 else 'S'} "
              f"raw {np.nanmedian(v[:, 1]):4.0f} trunk "
              f"{np.nanmedian(v[:, 2]):4.0f} neck {np.nanmedian(v[:, 3]):4.0f}"
              f" knee {np.nanmedian(v[:, 4]):4.0f} arm "
              f"{np.nanmedian(v[:, 5]):4.0f} REBA {int(v[:, 6].max()):2d}"
              f"{' EV' if v[:, 7].any() else ''}")
    if args.lifts:
        for lift in lifts:
            print(f"{lift.start.second:6.1f}-{lift.end.second:6.1f}s "
                  f"{lift.kind} {lift.start.v:.0f}->{lift.end.v:.0f} cm, "
                  f"H {lift.start.h:.0f}/{lift.end.h:.0f} cm, limit "
                  f"{lift.limit:.1f} kg, index {lift.index}")


if __name__ == "__main__":
    main()
