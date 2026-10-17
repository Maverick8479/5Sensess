"""Turn reminders: turns a (noisy) distance-to-turn plus what the camera sees
into short prompts, each said once.

Follows the project's rules: silence by default, speak only when it changes
what the user does next, report what is seen and never say a way is "safe".

Typical use, once per frame:
    msg = reminder.update(turn, gps_distance_m, status, cam_opening_m)
    if msg: speak(msg)
Call reminder.reset() when a new route instruction starts.
"""

SEEN = ("MATCH", "LATER", "EARLIER", "OPEN_NOW")   # camera sees an opening on the turn side


class TurnReminder:
    STAGES = (20, 10, 5, 3)   # metres: announce once when the distance first drops to each
    NOW_M = 2.0               # "turn now" at or below this
    PASSED_M = 6.0            # this far past the corner (twice in a row) -> "passed"
    REARM_M = 30.0            # after "passed", ready again once this far away

    def __init__(self):
        self.reset(None)

    def reset(self, turn=None):
        self.turn = turn
        self.fired = set()
        self.confirmed = False
        self.poor_said = False
        self.done = False
        self.away = 0

    @staticmethod
    def effective(gps_d, status, cam_d, max_gap=6.0):
        """Phone GPS is only good to a few metres. When the camera sees the
        opening and roughly agrees, trust its distance."""
        if status in ("MATCH", "LATER", "EARLIER") and cam_d is not None and abs(cam_d - gps_d) <= max_gap:
            return cam_d
        return gps_d

    def update(self, turn, gps_d, status=None, cam_d=None):
        """Returns a message to say now, or None."""
        if turn not in ("LEFT", "RIGHT"):
            if self.turn is not None:
                self.reset(None)
            return None
        if turn != self.turn:
            self.reset(turn)
        side = turn.lower()
        d = self.effective(gps_d, status, cam_d)

        if self.done:
            if gps_d > self.REARM_M:
                self.reset(turn)
            return None

        if "now" in self.fired:                      # at the corner: only watch for "passed"
            self.away = self.away + 1 if d >= self.PASSED_M else 0
            if self.away >= 2:
                self.done = True
                return "Turn point passed."
            return None

        if d <= self.NOW_M:
            self.fired.update(self.STAGES)
            self.fired.add("now")
            msg = f"Turn {side} now."
            if status == "NOT_SEEN":
                msg += f" {side.capitalize()} opening not confirmed."
            return msg

        hits = [s for s in self.STAGES if d <= s]
        if hits:
            s = min(hits)                            # a late start says one prompt, not all of them
            if s not in self.fired:
                self.fired.update(x for x in self.STAGES if x >= s)
                if d >= 10:
                    n, about = int(round(d / 5) * 5), "about "
                else:
                    n, about = int(round(d)), ""
                msg = f"In {about}{n} meters, turn {side}."
                if d <= 10:
                    if status in SEEN:
                        self.confirmed = True
                        msg += f" Opening seen on the {side}."
                    elif status == "NOT_SEEN" and d <= 5:     # farther out it is just noise
                        msg += f" No {side} opening seen yet."
                return msg

        if self.fired and d <= 10:
            if status == "VIEW_POOR" and not self.poor_said:
                self.poor_said = True
                return "Camera view poor. Use your cane."
            if status in SEEN and not self.confirmed and d <= 8:
                self.confirmed = True
                return f"Opening seen on the {side}."
        return None
