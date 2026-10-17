"""Approximate monocular ground geometry for the hackathon prototype.

Assumptions: pinhole camera, flat level ground, fixed height and downward pitch.
These estimates are NOT validated for mobility/safety decisions.
"""
import math
import numpy as np


class GroundCamera:
    def __init__(self, w, h, height=1.3, pitch=10.0, hfov=55.0):
        self.w, self.h = int(w), int(h)
        self.height = float(height)
        self.hfov = float(hfov)
        self.f = (self.w / 2) / math.tan(math.radians(self.hfov) / 2)
        self.vfov = math.degrees(2 * math.atan(self.h / (2 * self.f)))
        self.set_pitch(pitch)

    def set_pitch(self, pitch):
        self.pitch = float(pitch)

    def distance_to_row(self, distance):
        """Image y coordinate of ground point at forward distance (metres)."""
        d = max(float(distance), 1e-6)
        angle = math.atan2(self.height, d) - math.radians(self.pitch)
        return int(round(self.h / 2 + self.f * math.tan(angle)))

    def row_to_distance(self, row):
        a = math.radians(self.pitch) + math.atan2(float(row) - self.h / 2, self.f)
        return self.height / math.tan(a) if a > 1e-6 else math.inf

    def free_distance(self, ground, max_d=10.0):
        """Conservative center-column estimate; no claim that the area is safe."""
        ground = np.asarray(ground, dtype=bool)
        x0, x1 = int(self.w * .42), int(self.w * .58)
        # Check distances outward from the camera, until center corridor disappears.
        distances = np.arange(.5, max_d + .001, .25)
        last_good = 0.0
        stop_y = None
        for d in distances:
            y = self.distance_to_row(d)
            if not 0 <= y < self.h:
                break
            y0, y1 = max(0, y - 2), min(self.h, y + 3)
            patch = ground[y0:y1, x0:x1]
            if patch.size and float(patch.mean()) >= .60:
                last_good = float(d)
            else:
                stop_y = y
                break
        return {"d": last_good, "stop_y": stop_y}

    def scan_side(self, ground, side, d_max=10.0):
        """Samples ground fraction in an outer image strip at several distances."""
        ground = np.asarray(ground, dtype=bool)
        if side < 0:
            x0, x1 = int(.04 * self.w), int(.34 * self.w)
        else:
            x0, x1 = int(.66 * self.w), int(.96 * self.w)
        profile = []
        for d in np.arange(.5, d_max + .001, .5):
            y = self.distance_to_row(d)
            if not 0 <= y < self.h:
                profile.append((float(d), None))
                continue
            patch = ground[max(0, y - 3):min(self.h, y + 4), x0:x1]
            profile.append((float(d), float(patch.mean()) if patch.size else None))
        return profile


def find_opening(profile, threshold=.45):
    """Detect a side ground-like patch, not a confirmed pedestrian opening."""
    visible = [(d, v) for d, v in profile if v is not None]
    if not visible:
        return None
    if visible[0][1] >= threshold:
        return {"d": visible[0][0], "already_open": True}
    # Require two adjacent positive samples to reduce single-frame noise.
    for i in range(1, len(visible) - 1):
        if visible[i][1] >= threshold and visible[i + 1][1] >= threshold:
            return {"d": visible[i][0], "already_open": False}
    return None


def stop_reason(cam, walls, obst, stop_y):
    if stop_y is None:
        return "no end detected"
    y0, y1 = max(0, int(stop_y) - 8), min(cam.h, int(stop_y) + 9)
    x0, x1 = int(cam.w * .35), int(cam.w * .65)
    if np.asarray(obst)[y0:y1, x0:x1].mean() > .20:
        return "object detected"
    if np.asarray(walls)[y0:y1, x0:x1].mean() > .20:
        return "wall/building detected"
    return "ground region ends"


def compare_with_gps(turn, gps_d, free, left, right, reason, visible_from=math.inf):
    """Heuristic camera/GPS comparison; never confirms safe passage."""
    if turn == "STRAIGHT":
        return "STRAIGHT", "Continue straight; camera view is advisory only"
    if turn not in ("LEFT", "RIGHT"):
        return "UNKNOWN", "No turn instruction"
    op = left if turn == "LEFT" else right
    side = turn.lower()
    if op is None:
        if gps_d < visible_from:
            return "NOT_VISIBLE", f"{side.capitalize()} side not yet visible"
        return "NOT_SEEN", f"No {side} ground-like opening detected"
    if op.get("already_open"):
        return "OPEN_NOW", f"Ground-like area visible on {side}"
    d = op.get("d")
    if d is None:
        return "NOT_SEEN", f"No {side} opening detected"
    gap = float(d) - float(gps_d)
    if abs(gap) <= 2.0:
        return "MATCH", f"Possible {side} opening near GPS turn"
    if gap > 2.0:
        return "LATER", f"Possible {side} opening farther than GPS turn"
    return "EARLIER", f"Possible {side} opening before GPS turn"
