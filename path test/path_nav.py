"""Desktop webcam version with turn reminders.

The camera checks where the path continues and where side openings are; the
reminder module turns the distance to the turn into short prompts.

Needs path_core.py, path_geometry.py and turn_reminder.py in the same folder.
Optional voice: pip install pyttsx3  (without it the prompts are printed and
shown on the video).

Keys:  a=LEFT  d=RIGHT  w=STRAIGHT   u/j = distance +/-0.5 m
       s = simulate walking (distance counts down by itself)
       r = restart (distance back to 25 m)   q = quit
"""
import threading
import time

import cv2

from path_core import Segmenter, Tracker, analyse, camera_text, draw_overlay
from path_geometry import GroundCamera
from turn_reminder import TurnReminder

# ---------------- calibrate these on your own device ----------------
CAM_HEIGHT_M = 1.4     # lens height above the ground
PITCH_DEG = 10.0       # downward tilt
HFOV_DEG = 70.0        # horizontal field of view of a landscape webcam
# --------------------------------------------------------------------
START_DIST_M = 25.0
WALK_SPEED_MPS = 1.2   # used by the simulated walk (key s)


class GPSState:
    """Stand-in for the real navigation feed. Replace with turn + metres to turn."""

    def __init__(self):
        self.turn, self.distance_m = "LEFT", START_DIST_M
        self.simulate = False


def say(msg):
    print("REMINDER:", msg, flush=True)

    def run():
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(msg)
            engine.runAndWait()
        except Exception:
            pass                      # no voice installed: the printed line and banner remain
    threading.Thread(target=run, daemon=True).start()


def main():
    seg = Segmenter()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    print(__doc__.split("Keys:")[1].strip())

    gps, track, reminder, cam = GPSState(), Tracker(), TurnReminder(), None
    frame_count, last, banner, banner_until = 0, None, "", 0.0
    t_prev = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if cam is None:
                cam = GroundCamera(w, h, CAM_HEIGHT_M, PITCH_DEG, HFOV_DEG)
            now = time.time()
            if gps.simulate:
                gps.distance_m = max(0.0, gps.distance_m - WALK_SPEED_MPS * (now - t_prev))
            t_prev = now

            if frame_count % 5 == 0:            # model every 5th frame to save CPU
                # cam.set_pitch(imu_pitch_deg)  # <- when you have the IMU
                mask = seg(frame)
                res = analyse(cam, mask, frame)
                free, left, right = track.update(res)
                last = (mask, res, free, left, right)

            display = frame
            if last is not None:
                mask, res, free, left, right = last
                status, text, cam_d = camera_text(res, gps.turn, gps.distance_m, free, left, right)
                msg = reminder.update(gps.turn, gps.distance_m, status, cam_d)
                if msg:
                    say(msg)
                    banner, banner_until = msg, now + 4
                display = draw_overlay(frame, mask, res, cam, gps.turn, gps.distance_m,
                                       free, text, banner if now < banner_until else None)

            cv2.imshow("Path distance", display)
            frame_count += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("a"):
                gps.turn = "LEFT"
            elif key == ord("d"):
                gps.turn = "RIGHT"
            elif key == ord("w"):
                gps.turn = "STRAIGHT"
            elif key == ord("u"):
                gps.distance_m = min(60.0, gps.distance_m + 0.5)
            elif key == ord("j"):
                gps.distance_m = max(0.0, gps.distance_m - 0.5)
            elif key == ord("s"):
                gps.simulate = not gps.simulate
                print("simulated walk:", "ON" if gps.simulate else "OFF", flush=True)
            elif key == ord("r"):
                gps.distance_m = START_DIST_M
                reminder.reset(gps.turn)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
