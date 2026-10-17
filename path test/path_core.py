"""Shared pipeline pieces (used by the phone test server).

segment -> analyse -> smooth -> draw. Kept separate from the camera/web
code so the same logic can later be ported to the on-device app.
"""
import math
from collections import deque
from statistics import median

import cv2
import numpy as np

from path_geometry import compare_with_gps, find_opening, stop_reason

# Cityscapes classes
ROAD, SIDEWALK = 0, 1
WALLS = [2, 3, 4]                                  # building, wall, fence
OBST = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]   # pole, signs, people, vehicles
MAX_D = 10.0


def _round32(x):
    return max(32, int(round(x / 32)) * 32)


class Segmenter:
    """SegFormer wrapper. Keeps the aspect ratio, so portrait phone frames
    are not stretched into the model's 512x1024 landscape default."""
    MODEL = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"

    def __init__(self, long_side=640):
        import torch
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        print("Loading segmentation model...")
        self.torch = torch
        self.processor = SegformerImageProcessor.from_pretrained(self.MODEL)
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.MODEL).eval()
        self.long_side = long_side

    def __call__(self, bgr):
        from PIL import Image
        h, w = bgr.shape[:2]
        s = self.long_side / max(h, w)
        size = {"height": _round32(h * s), "width": _round32(w * s)}
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=Image.fromarray(rgb), size=size, return_tensors="pt")
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits
            logits = self.torch.nn.functional.interpolate(
                logits, size=(h, w), mode="bilinear", align_corners=False)
            return logits.argmax(dim=1)[0].cpu().numpy()


def fake_mask(h, w):
    """Made-up scene (road ahead, wall on both sides, a gap on the left) for
    checking the plumbing without the model. NOT a real prediction."""
    m = np.full((h, w), 2, np.uint8)
    road = np.array([[int(.2 * w), h - 1], [int(.8 * w), h - 1],
                     [int(.56 * w), int(.45 * h)], [int(.44 * w), int(.45 * h)]], np.int32)
    cv2.fillPoly(m, [road], 0)
    cv2.rectangle(m, (0, int(.6 * h)), (int(.3 * w), int(.75 * h)), 0, -1)
    return m


def connected_ground_mask(mask):
    """Keep only road/sidewalk that touches the area right in front of the user."""
    surface = np.uint8((mask == ROAD) | (mask == SIDEWALK))
    _, labels, _, _ = cv2.connectedComponentsWithStats(surface, connectivity=8)
    h, w = surface.shape
    ids = set(np.unique(labels[int(h * 0.95):, int(w * 0.35):int(w * 0.65)]))
    ids.discard(0)
    return np.isin(labels, list(ids)) if ids else np.zeros_like(surface, dtype=bool)


def view_problem(frame, cam, ground):
    """Why the numbers can't be trusted right now, or None.

    A hand over the lens gives a dark, blurry frame that the model may label
    as 'road' everywhere; the old code then reported the maximum distance.
    Flat ground can never appear above the horizon, so a lot of 'ground'
    up there means the picture is not what the geometry assumes."""
    if frame is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.mean() < 25:
            return "too dark"
        if cv2.Laplacian(gray, cv2.CV_64F).var() < 15:
            return "blurred or covered"
    total = int(ground.sum())
    if total:
        horizon = int(np.clip(cam.distance_to_row(1e6), 0, cam.h))
        if ground[:horizon].sum() / total > 0.3:
            return "ground not plausible"
    return None


def analyse(cam, mask, frame=None):
    ground = connected_ground_mask(mask)
    walls = np.isin(mask, WALLS)
    obst = np.isin(mask, OBST)
    free = cam.free_distance(ground, max_d=MAX_D)
    lp = cam.scan_side(ground, -1, d_max=MAX_D)
    rp = cam.scan_side(ground, +1, d_max=MAX_D)

    def first_visible(profile):      # nearest distance where that side is in the image at all
        return next((d for d, r in profile if r is not None), math.inf)

    return {
        "ground": ground, "walls": walls, "free": free,
        "reason": stop_reason(cam, walls, obst, free["stop_y"]),
        "left": find_opening(lp), "right": find_opening(rp),
        "left_vis": first_visible(lp), "right_vis": first_visible(rp),
        "poor": view_problem(frame, cam, ground),
    }


def camera_text(res, turn, gps_d, free, left, right):
    """What the camera can say about the turn. Returns (status, text, opening_m).

    opening_m is where the camera sees the opening on the turn side
    (None if not seen, or if the side is open all along)."""
    if res["poor"]:
        return "VIEW_POOR", f"Camera view poor ({res['poor']})", None
    vis = {"LEFT": res["left_vis"], "RIGHT": res["right_vis"]}.get(turn, 0.0)
    status, text = compare_with_gps(turn, gps_d, free, left, right, res["reason"], visible_from=vis)
    op = {"LEFT": left, "RIGHT": right}.get(turn)
    cam_d = op["d"] if op and not op["already_open"] else None
    return status, text, cam_d


class Smoother:
    """Median of the last n values; None if too few of them saw anything."""

    def __init__(self, n=5, need=3):
        self.buf, self.need = deque(maxlen=n), need

    def push(self, v):
        self.buf.append(v)
        vals = [x for x in self.buf if x is not None]
        return median(vals) if len(vals) >= self.need else None


class Tracker:
    """Smooths free distance and side openings over the last few frames."""

    def __init__(self):
        self.free, self.left, self.right = Smoother(need=1), Smoother(), Smoother()

    def update(self, res):
        free = dict(res["free"], d=self.free.push(res["free"]["d"]))
        out = []
        for sm, op in ((self.left, res["left"]), (self.right, res["right"])):
            d = sm.push(op["d"] if op else None)
            out.append({"d": d, "already_open": bool(op and op["already_open"])} if d is not None else None)
        return free, out[0], out[1]


def _hline(img, y, label, color, fs):
    y = int(np.clip(y, 0, img.shape[0] - 1))
    cv2.line(img, (0, y), (img.shape[1], y), color, 1)
    cv2.putText(img, label, (5, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8, color, 1)


def draw_overlay(frame, mask, res, cam, turn, gps_d, free, text, banner=None):
    ground, walls = res["ground"], res["walls"]
    overlay = np.zeros_like(frame)
    overlay[(mask == ROAD) & ground] = (0, 255, 0)
    overlay[(mask == SIDEWALK) & ground] = (255, 0, 0)
    overlay[walls] = (0, 0, 255)
    out = frame.copy()
    shown = ground | walls
    out[shown] = cv2.addWeighted(frame, 0.55, overlay, 0.45, 0)[shown]

    h, w = out.shape[:2]
    fs = max(0.45, w / 800)
    for d in (1, 2, 3, 5, 8):               # ruler: compare with tape marks on the floor
        _hline(out, cam.distance_to_row(d), f"{d}m", (200, 200, 200), fs)
    _hline(out, cam.distance_to_row(gps_d), "GPS", (0, 255, 255), fs)
    if free["stop_y"] is not None:
        _hline(out, free["stop_y"], f"end {free['d']:.1f}m", (255, 0, 255), fs)

    cv2.putText(out, f"GPS: {turn} in {gps_d:.1f} m", (10, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 255), 2)
    cv2.putText(out, text, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8, (0, 255, 255), 2)
    cv2.putText(out, f"ahead {free['d']:.1f} m ({res['reason']})", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2)
    if banner:                               # latest reminder, large
        sc = fs * 1.1
        (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, sc, 3)
        if tw > w - 20:
            sc *= (w - 20) / tw
            (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, sc, 3)
        x, y = max(5, (w - tw) // 2), int(h * 0.2)
        cv2.rectangle(out, (x - 8, y - th - 10), (x + tw + 8, y + 12), (0, 0, 0), -1)
        cv2.putText(out, banner, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 255, 255), 3)
    return out
