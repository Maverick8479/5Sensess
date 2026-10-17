"""Checks for the turn reminder logic and the camera-quality guard.
Uses made-up distances and frames: this tests the logic, not real streets."""
import random

import numpy as np

from path_core import camera_text, view_problem
from path_geometry import GroundCamera
from turn_reminder import TurnReminder


def walk(rem, turn, dists, status=None, cam_d=None):
    out = []
    for d in dists:
        m = rem.update(turn, d, status, cam_d)
        if m:
            out.append((d, m))
    return out


def main():
    # 1. clean approach 25 m -> 0 m, then walking past the corner
    r = TurnReminder()
    got = walk(r, "LEFT", [25 - i for i in range(26)] + list(range(1, 10)))
    for d, m in got:
        print(f"  d={d:>4}: {m}")
    msgs = [m for _, m in got]
    assert msgs == [
        "In about 20 meters, turn left.",
        "In about 10 meters, turn left.",
        "In 5 meters, turn left.",
        "In 3 meters, turn left.",
        "Turn left now.",
        "Turn point passed.",
    ], msgs

    # 2. noisy GPS (+/- 2.5 m): nothing is ever said twice
    random.seed(1)
    r = TurnReminder()
    noisy = [max(0.0, (25 - i) + random.uniform(-2.5, 2.5)) for i in range(26)]
    got = walk(r, "RIGHT", noisy)
    msgs = [m for _, m in got]
    print("noisy:", msgs)
    assert len(msgs) == len(set(msgs)), "repeated message"

    # 3. starting late (4 m away) gives one prompt, not a burst
    r = TurnReminder()
    first = r.update("LEFT", 4.0)
    assert first == "In 4 meters, turn left.", first
    assert r.update("LEFT", 3.9) is None

    # 4. camera agrees -> says so in the same prompt
    r = TurnReminder()
    assert r.update("LEFT", 3.0, "MATCH", 3.0) == "In 3 meters, turn left. Opening seen on the left."
    r = TurnReminder()                       # camera a bit later than GPS: the camera's number is used
    assert r.update("LEFT", 3.0, "MATCH", 3.6) == "In 4 meters, turn left. Opening seen on the left."

    # 5. camera disagrees: trusts the camera's distance (GPS 3 m, camera sees it at 5 m)
    r = TurnReminder()
    assert r.update("LEFT", 3.0, "LATER", 5.0) == "In 5 meters, turn left. Opening seen on the left."

    # 6. opening only becomes visible after the 5 m prompt -> one extra message, once
    r = TurnReminder()
    assert r.update("LEFT", 5.0, "NOT_VISIBLE") == "In 5 meters, turn left."
    assert r.update("LEFT", 4.0, "MATCH", 4.0) == "Opening seen on the left."
    assert r.update("LEFT", 3.9, "MATCH", 4.0) is None

    # 7. opening not seen: says so, still never claims safety
    r = TurnReminder()
    m = r.update("LEFT", 3.0, "NOT_SEEN")
    assert m == "In 3 meters, turn left. No left opening seen yet.", m
    m = r.update("LEFT", 1.5, "NOT_SEEN")
    assert m == "Turn left now. Left opening not confirmed.", m

    # 8. poor camera view is announced once
    r = TurnReminder()
    r.update("LEFT", 8.0, "VIEW_POOR")
    assert r.update("LEFT", 7.9, "VIEW_POOR") == "Camera view poor. Use your cane."
    assert r.update("LEFT", 7.8, "VIEW_POOR") is None

    # 9. changing the instruction starts fresh; straight gives no reminders
    r = TurnReminder()
    r.update("LEFT", 3.0)
    assert r.update("RIGHT", 3.0) == "In 3 meters, turn right."
    assert r.update("STRAIGHT", 3.0) is None

    # 10. camera guard: a covered / dark / implausible frame is not trusted
    cam = GroundCamera(640, 480, 1.4, 10, 70)
    rng = np.random.default_rng(0)
    textured = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    horizon = cam.distance_to_row(1e6)
    below = np.zeros((480, 640), bool)
    below[horizon + 10:, :] = True
    assert view_problem(textured, cam, below) is None
    assert view_problem(np.full((480, 640, 3), 120, np.uint8), cam, below) == "blurred or covered"
    assert view_problem(np.full((480, 640, 3), 5, np.uint8), cam, below) == "too dark"
    everywhere = np.ones((480, 640), bool)          # the "hand over the lens" case
    assert view_problem(textured, cam, everywhere) == "ground not plausible"
    res = {"poor": "blurred or covered"}
    assert camera_text(res, "LEFT", 3.0, {}, None, None)[0] == "VIEW_POOR"

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
