"""What the analysis finds in each calibration video, against what it
found when it was calibrated: run after any change to the analysis.

    uv run python -m tools.calibration.compare            # every cached
    uv run python -m tools.calibration.compare B5 C2 --load 5

Each line gives the people, the lifts (with their lifting index, or
"0kg" for a limit of 0 kg), the pushes (``e``) and pulls (``a``), the
moments of risk and the highest and mean REBA, and ``OK`` or what
differs from ``EXPECTED``.
"""
import argparse
from typing import NamedTuple

from src.posture.lifting import Lifting
from tools.calibration.cache import VIDEOS, path_of, replay


class Expected(NamedTuple):
    """What the analysis found in a video when calibrated (and, in the
    comment, what really happened, where it differs)."""

    people: int
    lifts: int
    pushes: str


# 2026-11-21, with the code of that day (see calibracion_videos.md).
EXPECTED: dict[str, Expected] = {
    "B1": Expected(1, 0, ""),
    "B2": Expected(1, 0, ""),
    "B3": Expected(1, 0, ""),       # arms raised, no load: no lift
    "B4": Expected(1, 1, ""),       # squats: 1 false lift
    "B5": Expected(1, 4, ""),       # 4 real
    "B6": Expected(1, 8, ""),       # 6-9 real
    "B7": Expected(1, 8, ""),       # 8 real
    "C1": Expected(1, 0, "ee"),     # 3 real pushes
    "C2": Expected(1, 0, "eee"),
    "C3": Expected(1, 0, "aaa"),
    "C4": Expected(1, 0, "a"),      # 3 real pulls (one arm)
    "C5": Expected(1, 0, ""),       # nothing to find
    "sample": Expected(1, 3, ""),   # 3 real
    "E1": Expected(3, 7, ""),       # three people moving things freely
    "D1": Expected(3, 4, ""),       # 35 min at a laptop, people passing:
                                    # the 4 lifts are of passers-by
    "E2": Expected(1, 0, ""),       # standing at a laptop
    "F2": Expected(1, 1, ""),       # standing, squat, bend, arms up
    "F3": Expected(1, 11, ""),      # 12 real with a heater: 9 found
}


def summary(name: str, load_kg: float | None) -> tuple[str, Expected]:
    """One line on a video, and what it found."""
    result = replay(name, lifting=Lifting(load_kg=load_kg))
    parts = []
    lifts = pushes = 0
    kinds = ""
    for person in result.people:
        found = person.lifting.lifts if person.lifting else []
        moved = person.pushing.pushes if person.pushing else []
        lifts += len(found)
        pushes += len(moved)
        kinds += "".join("e" if p.kind == "empuje" else "a" for p in moved)
        indexes = ["0kg" if lift.beyond_limits else
                   f"{lift.index:.2f}" if lift.index is not None else "-"
                   for lift in found]
        s = person.summary
        parts.append(
            f"P{person.person} {person.seconds_seen:.0f}s lifts "
            f"[{' '.join(indexes)}] pushes {kinds or '-'} moments "
            f"{person.moments} REBA max {s.max_score} mean "
            f"{s.mean_score}")
    line = f"{name:7} {len(result.people)} people | " + " | ".join(parts)
    return line, Expected(len(result.people), lifts, kinds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("names", nargs="*", help="videos (default: all)")
    parser.add_argument("--load", type=float, default=0.0,
                        help="load of the lifts, kg (0: not given)")
    args = parser.parse_args()
    names = args.names or [n for n in VIDEOS if path_of(n).exists()]
    if not names:
        print("No cache yet: uv run python -m tools.calibration.cache "
              "--all")
    differ = 0
    for name in names:
        line, found = summary(name, args.load or None)
        expected = EXPECTED.get(name)
        if expected is None:
            check = "(no expected result)"
        elif found == expected:
            check = "OK"
        else:
            differ += 1
            check = f"DIFFERS: expected {tuple(expected)}, got {tuple(found)}"
        print(f"{line}\n        {check}", flush=True)
    if differ:
        print(f"{differ} video(s) differ from the calibration")


if __name__ == "__main__":
    main()
