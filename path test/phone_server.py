"""Phone test server: the phone is the camera and screen, this computer does the work.

Setup (once):
    pip install flask cryptography opencv-python torch transformers pillow numpy

Run:
    python phone_server.py            # real model
    python phone_server.py --fake     # no model, made-up scene, only checks the plumbing

Then on the phone open the https://<computer-ip>:8443 address it prints (same
network), or any https tunnel address that points at this port.
The browser warns about the certificate (it is self-signed): choose Advanced -> Proceed.
Camera access only works over https, which is why the warning page is needed.

Frames are processed in memory and thrown away; nothing is saved to disk.
Anyone who can reach this port can send it pictures while it runs, so use a network you trust.
"""
import argparse
import base64
import os
import socket
import threading
import time

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from path_core import Segmenter, Tracker, analyse, camera_text, draw_overlay, fake_mask
from path_geometry import GroundCamera
from turn_reminder import TurnReminder

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
LOCK = threading.Lock()          # one frame at a time: model and smoothing state are shared
TRACK = Tracker()
REMINDER = TurnReminder()
SEG = None                       # set in main(); None means --fake


def num(name, default, lo, hi):
    try:
        return min(hi, max(lo, float(request.args.get(name, default))))
    except ValueError:
        return default


@app.get("/")
def index():
    resp = send_from_directory(HERE, "phone_test.html")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.post("/analyze")
def analyze():
    t0 = time.time()
    frame = cv2.imdecode(np.frombuffer(request.get_data(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify(error="could not decode image"), 400

    turn = request.args.get("turn", "LEFT")
    if turn not in ("LEFT", "RIGHT", "STRAIGHT"):
        return jsonify(error="bad turn"), 400
    gps_d = num("dist", 25.0, 0.0, 60.0)
    pitch = num("pitch", 10.0, -20.0, 70.0)
    height = num("height", 1.3, 0.5, 2.2)
    hfov = num("hfov", 55.0, 30.0, 120.0)

    h, w = frame.shape[:2]
    cam = GroundCamera(w, h, height, pitch, hfov)

    with LOCK:
        if request.args.get("reset") == "1":
            REMINDER.reset(None)
        mask = SEG(frame) if SEG else fake_mask(h, w)
        res = analyse(cam, mask, frame)
        free, left, right = TRACK.update(res)
        status, text, cam_d = camera_text(res, turn, gps_d, free, left, right)
        reminder = REMINDER.update(turn, gps_d, status, cam_d)
        img = draw_overlay(frame, mask, res, cam, turn, gps_d, free, text)

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return jsonify(
        status=status, text=text, reminder=reminder,
        free=round(free["d"], 1), reason=res["reason"],
        image="data:image/jpeg;base64," + base64.b64encode(buf).decode(),
        ms=int((time.time() - t0) * 1000),
    )


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "localhost"
    finally:
        s.close()


def main():
    global SEG
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--fake", action="store_true", help="skip the model, use a made-up scene")
    args = ap.parse_args()

    SEG = None if args.fake else Segmenter()
    print(f"\nOpen on your phone:  https://{lan_ip()}:{args.port}\n")
    app.run(host="0.0.0.0", port=args.port, ssl_context="adhoc", threaded=True)


if __name__ == "__main__":
    main()
