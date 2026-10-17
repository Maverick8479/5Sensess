#!/usr/bin/env python3
"""
walking_navigation.py: Walking navigation prototype (Version 2 feature)
for the Immersive Assistant for Blind Users.

What it does
  1. Map service   - geocodes a destination and fetches a walking route.
                     Valhalla (no key; default), OpenRouteService (key), or a
                     straight-line offline fallback. Pick with --map.
                     Geocoding for Valhalla: Postcodes.io + Nominatim, biased to Manchester.
  2. Directions    - follows the route from GPS fixes and speaks turn-by-turn
                     prompts only when they change what the user does next.
  3. Distance      - distance to the next turn / destination (feet or meters)
                     from GPS, and distance to objects ahead from the camera
                     (monocular pinhole estimate; replace with LiDAR/stereo depth).
  4. Camera        - runs a YOLO detector + ByteTrack on camera frames, raises
                     collision (TTC) and obstacle warnings that interrupt navigation.
  5. Audio arbiter - one message at a time; a higher priority cuts off a lower one.

Design rules from the project plan that this code follows
  - Adds to the cane / guide dog; it never replaces them.
  - Silence by default: speaks only at thresholds, on off-route, or on request.
  - Never says "safe to cross". Crossing prompts say the decision is the user's.
  - Fails loudly: dead camera, missing model, weak GPS -> spoken "alerts off".
  - No video is stored; frames are processed in memory and discarded.

Setup
  pip install requests opencv-python ultralytics pyserial
  # Valhalla needs no key. For OpenRouteService instead:  export ORS_API_KEY=your_key
  # For your own Valhalla server:                          export VALHALLA_URL=http://localhost:8002

Try it without hardware (simulated walker, no camera):
  python walking_navigation.py --origin "53.4808,-2.2426" --dest "M1 1AE" --simulate 3 --no-camera

Real run (USB/GPS puck on a serial port, webcam 0):
  python walking_navigation.py --nmea /dev/ttyUSB0 --dest "Manchester Piccadilly Station" --units feet

Console commands while running:  d = distance/status,  r = repeat last,  q = quit
This is a Python prototype; production code would be Swift/Kotlin, but the logic ports directly.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterator, Optional

WALK_SPEED_MPS = 1.3  # used for ETA only

# --------------------------------------------------------------------------- #
# Geometry and formatting
# --------------------------------------------------------------------------- #
EARTH_R = 6_371_000.0


@dataclass(frozen=True)
class LatLon:
    lat: float
    lon: float


def haversine(a: LatLon, b: LatLon) -> float:
    p1, p2 = math.radians(a.lat), math.radians(b.lat)
    dp, dl = p2 - p1, math.radians(b.lon - a.lon)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(h))


def bearing(a: LatLon, b: LatLon) -> float:
    p1, p2 = math.radians(a.lat), math.radians(b.lat)
    dl = math.radians(b.lon - a.lon)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def cardinal(deg: float) -> str:
    names = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]
    return names[int((deg + 22.5) // 45) % 8]


def relative_direction(heading: float, target: float) -> str:
    diff = (target - heading + 540) % 360 - 180  # -180..180, positive = right
    if abs(diff) <= 30:
        return "The route starts ahead of you."
    if abs(diff) >= 150:
        return "The route starts behind you. Turn around."
    return "The route starts to your right." if diff > 0 else "The route starts to your left."


def _xy(origin: LatLon, p: LatLon) -> tuple[float, float]:
    """Local flat-earth metres (east, north) of p relative to origin; fine for short distances."""
    return ((p.lon - origin.lon) * math.cos(math.radians(origin.lat)) * 111_320.0,
            (p.lat - origin.lat) * 110_540.0)


def point_segment(p: LatLon, a: LatLon, b: LatLon) -> tuple[float, float]:
    """Distance (m) from p to segment ab and the fraction t (0..1) along ab of the closest point."""
    bx, by = _xy(a, b)
    px, py = _xy(a, p)
    l2 = bx * bx + by * by
    t = 0.0 if l2 == 0 else max(0.0, min(1.0, (px * bx + py * by) / l2))
    return math.hypot(px - t * bx, py - t * by), t


def _fmt(n: float, one: str, many: str) -> str:
    return f"{n:g} {one if n == 1 else many}"


def fmt_distance(m: float, units: str = "feet") -> str:
    """Speakable distance, rounded the way a person would say it."""
    if units == "feet":
        ft = m * 3.28084
        if ft >= 1000:
            return _fmt(round(ft / 5280, 1), "mile", "miles")
        step = 1 if ft < 20 else 5 if ft < 100 else 10
        return _fmt(max(1, int(round(ft / step) * step)), "foot", "feet")
    if m >= 1000:
        return _fmt(round(m / 1000, 1), "kilometer", "kilometers")
    step = 1 if m < 10 else 5 if m < 100 else 10
    return _fmt(max(1, int(round(m / step) * step)), "meter", "meters")


def parse_latlon(text: str) -> Optional[LatLon]:
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$", text or "")
    return LatLon(float(m.group(1)), float(m.group(2))) if m else None


# --------------------------------------------------------------------------- #
# Route model
# --------------------------------------------------------------------------- #
@dataclass
class Step:
    instruction: str
    distance_m: float
    maneuver: int      # ORS type; 10 = arrive, 11 = depart
    start_idx: int     # index into Route.points where this step begins
    end_idx: int
    name: str = ""


@dataclass
class Route:
    points: list[LatLon]
    steps: list[Step]
    seg_len: list[float] = field(init=False)
    cum: list[float] = field(init=False)   # cum[k] = metres from start to vertex k
    total_m: float = field(init=False)

    def __post_init__(self) -> None:
        self.seg_len = [haversine(a, b) for a, b in zip(self.points, self.points[1:])]
        self.cum = [0.0]
        for s in self.seg_len:
            self.cum.append(self.cum[-1] + s)
        self.total_m = self.cum[-1]

    def snap(self, p: LatLon, start_idx: int, lookahead_m: float = 200.0) -> tuple[float, int, float]:
        """Closest point on the route, searched forward from start_idx.
        Returns (distance_from_route_m, segment_index, fraction_along_segment)."""
        best = (float("inf"), start_idx, 0.0)
        i = max(0, start_idx - 2)
        limit = self.cum[min(start_idx, len(self.seg_len))] + lookahead_m
        while i < len(self.seg_len) and self.cum[i] <= limit:
            d, t = point_segment(p, self.points[i], self.points[i + 1])
            if d < best[0]:
                best = (d, i, t)
            i += 1
        return best

    def step_at(self, seg_idx: int) -> int:
        k = 0
        for n, s in enumerate(self.steps):
            if s.start_idx <= seg_idx:
                k = n
            else:
                break
        return k

    def point_at(self, dist_m: float) -> tuple[LatLon, float]:
        dist_m = max(0.0, min(dist_m, self.total_m))
        i = max(0, min(len(self.seg_len) - 1, next((n for n in range(len(self.seg_len)) if self.cum[n + 1] >= dist_m), len(self.seg_len) - 1)))
        t = 0.0 if self.seg_len[i] == 0 else (dist_m - self.cum[i]) / self.seg_len[i]
        a, b = self.points[i], self.points[i + 1]
        return LatLon(a.lat + (b.lat - a.lat) * t, a.lon + (b.lon - a.lon) * t), bearing(a, b)


# --------------------------------------------------------------------------- #
# Map services
# --------------------------------------------------------------------------- #
class MapService(ABC):
    @abstractmethod
    def geocode(self, text: str, near: Optional[LatLon] = None) -> LatLon: ...

    @abstractmethod
    def route(self, origin: LatLon, dest: LatLon) -> Route: ...


class ORSMapService(MapService):
    """OpenRouteService, 'foot-walking' profile (OpenStreetMap data, sidewalks and footpaths)."""
    BASE = "https://api.openrouteservice.org"

    def __init__(self, api_key: str):
        import requests  # imported here so the file still loads without it
        self._http = requests
        self.key = api_key

    def geocode(self, text: str, near: Optional[LatLon] = None) -> LatLon:
        params = {"api_key": self.key, "text": text, "size": 1}
        if near:
            params.update({"focus.point.lat": near.lat, "focus.point.lon": near.lon})
        r = self._http.get(f"{self.BASE}/geocode/search", params=params, timeout=10)
        r.raise_for_status()
        feats = r.json().get("features", [])
        if not feats:
            raise ValueError(f"No place found for '{text}'")
        lon, lat = feats[0]["geometry"]["coordinates"][:2]
        return LatLon(lat, lon)

    def route(self, origin: LatLon, dest: LatLon) -> Route:
        params = {"api_key": self.key, "start": f"{origin.lon},{origin.lat}", "end": f"{dest.lon},{dest.lat}"}
        r = self._http.get(f"{self.BASE}/v2/directions/foot-walking", params=params, timeout=10)
        r.raise_for_status()
        feat = r.json()["features"][0]
        pts = [LatLon(c[1], c[0]) for c in feat["geometry"]["coordinates"]]
        steps = [Step(s["instruction"], s["distance"], s["type"], s["way_points"][0], s["way_points"][1], s.get("name", ""))
                 for s in feat["properties"]["segments"][0]["steps"]]
        return Route(pts, steps)


def decode_polyline6(encoded: str) -> list[LatLon]:
    """Decode a Valhalla encoded polyline (precision 6) into coordinates."""
    pts, idx, lat, lon = [], 0, 0, 0
    while idx < len(encoded):
        for axis in (0, 1):
            shift = result = 0
            while True:
                b = ord(encoded[idx]) - 63
                idx += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if result & 1 else result >> 1
            if axis == 0:
                lat += delta
            else:
                lon += delta
        pts.append(LatLon(lat / 1e6, lon / 1e6))
    return pts


UK_POSTCODE_RE = re.compile(r"^[A-Za-z]{1,2}\d[A-Za-z\d]?\s*\d[A-Za-z]{2}$")
USER_AGENT = "ImmersiveAssistant/0.1 (replace-with-your-contact-email)"   # Nominatim requires an identifying agent


class OSMGeocoderMixin:
    """UK-friendly geocoding: Postcodes.io for postcodes, Nominatim (OSM) for place names.
    Results are biased to the user's area (Greater Manchester by default)."""
    MANCHESTER_VIEWBOX = "-2.73,53.69,-1.91,53.33"   # left,top,right,bottom (lon,lat)
    _http = None

    def geocode(self, text: str, near: Optional[LatLon] = None) -> LatLon:
        text = text.strip()
        if UK_POSTCODE_RE.match(text):
            r = self._http.get(f"https://api.postcodes.io/postcodes/{text.replace(' ', '')}", timeout=10)
            r.raise_for_status()
            res = r.json()["result"]
            return LatLon(res["latitude"], res["longitude"])
        viewbox = (f"{near.lon - 0.15},{near.lat + 0.1},{near.lon + 0.15},{near.lat - 0.1}"
                   if near else self.MANCHESTER_VIEWBOX)
        r = self._http.get("https://nominatim.openstreetmap.org/search",
                           params={"q": text, "format": "jsonv2", "limit": 1, "countrycodes": "gb",
                                   "viewbox": viewbox, "bounded": 0},
                           headers={"User-Agent": USER_AGENT}, timeout=10)   # policy: max 1 request/second
        r.raise_for_status()
        hits = r.json()
        if not hits:
            raise ValueError(f"No place found for '{text}'")
        return LatLon(float(hits[0]["lat"]), float(hits[0]["lon"]))


class ValhallaMapService(OSMGeocoderMixin, MapService):
    """Valhalla pedestrian routing on OpenStreetMap data. No API key needed.
    The default is the public FOSSGIS server (fine for prototypes, not for production);
    point base_url at your own Valhalla with a Greater Manchester extract for the pilot."""
    PUBLIC = "https://valhalla1.openstreetmap.de"

    def __init__(self, base_url: str = PUBLIC):
        import requests
        self._http = requests
        self.base = base_url.rstrip("/")

    def route(self, origin: LatLon, dest: LatLon) -> Route:
        body = {
            "locations": [{"lat": origin.lat, "lon": origin.lon}, {"lat": dest.lat, "lon": dest.lon}],
            "costing": "pedestrian",
            "directions_options": {"units": "kilometers", "language": "en-GB"},
        }
        r = self._http.get(f"{self.base}/route", params={"json": json.dumps(body)},
                           headers={"User-Agent": USER_AGENT}, timeout=15)
        r.raise_for_status()
        leg = r.json()["trip"]["legs"][0]
        pts = decode_polyline6(leg["shape"])
        steps = []
        for m in leg["maneuvers"]:
            kind = 11 if m["type"] in (1, 2, 3) else 10 if m["type"] in (4, 5, 6) else 6   # depart / arrive / other
            steps.append(Step(m["instruction"].rstrip(". "), m["length"] * 1000.0, kind,
                              m["begin_shape_index"], m["end_shape_index"],
                              ", ".join(m.get("street_names", []))))
        return Route(pts, steps)


class StraightLineMapService(MapService):
    """Offline fallback: one straight leg. No street-level turns, so it says so."""

    def geocode(self, text: str, near: Optional[LatLon] = None) -> LatLon:
        p = parse_latlon(text)
        if not p:
            raise ValueError("Without ORS_API_KEY the destination must be 'lat,lon'.")
        return p

    def route(self, origin: LatLon, dest: LatLon) -> Route:
        b = bearing(origin, dest)
        return Route([origin, dest], [
            Step(f"Head {cardinal(b)}. Street-level directions are unavailable", haversine(origin, dest), 11, 0, 1),
            Step("You have arrived", 0, 10, 1, 1),
        ])


# --------------------------------------------------------------------------- #
# Audio: text-to-speech and the priority arbiter
# --------------------------------------------------------------------------- #
class Prio(IntEnum):          # lower number = more important (matches the plan's audio priority)
    COLLISION = 0
    OBSTACLE = 1
    SIGNAL = 2
    NAVIGATION = 3
    PEOPLE = 4
    DESCRIPTION = 5


MAX_AGE_S = {Prio.COLLISION: 1.0, Prio.OBSTACLE: 2.0}   # stale warnings are dropped, not spoken late
DEFAULT_MAX_AGE_S = 10.0


class SystemTTS:
    """Speaks through the OS voice in a subprocess so it can be cut off mid-sentence."""

    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen] = None
        self.cmd: Optional[list[str]] = None
        if shutil.which("say"):
            self.cmd = ["say"]
        elif shutil.which("espeak-ng"):
            self.cmd = ["espeak-ng"]
        elif shutil.which("espeak"):
            self.cmd = ["espeak"]
        elif os.name == "nt":
            self.cmd = ["powershell", "-Command"]

    def speak(self, text: str) -> None:
        print(f"[SPEAK] {text}", flush=True)
        if not self.cmd:
            time.sleep(min(3.0, 0.06 * len(text)))   # no voice available: simulate duration
            return
        if self.cmd[0] == "powershell":
            safe = text.replace("'", " ").replace('"', " ")
            args = self.cmd + [f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{safe}')"]
        else:
            args = self.cmd + [text]
        self.proc = subprocess.Popen(args)
        self.proc.wait()

    def stop(self) -> None:
        p = self.proc
        if p and p.poll() is None:
            p.terminate()


class AudioArbiter(threading.Thread):
    """One message at a time. A higher-priority message cuts off a lower one mid-sentence."""

    def __init__(self, tts: SystemTTS):
        super().__init__(daemon=True)
        self.tts = tts
        self.q: queue.PriorityQueue = queue.PriorityQueue()
        self.seq = itertools.count()
        self.current: Optional[int] = None
        self.lock = threading.Lock()

    def say(self, text: str, prio: Prio) -> None:
        with self.lock:
            cur = self.current
        self.q.put((int(prio), next(self.seq), time.time(), text))
        if cur is not None and int(prio) < cur:
            self.tts.stop()

    def run(self) -> None:
        while True:
            prio, _, ts, text = self.q.get()
            if text is None:
                return
            if time.time() - ts > MAX_AGE_S.get(Prio(prio), DEFAULT_MAX_AGE_S):
                continue
            with self.lock:
                self.current = prio
            try:
                self.tts.speak(text)
            finally:
                with self.lock:
                    self.current = None

    def wait_idle(self, timeout: float = 8.0) -> None:
        end = time.time() + timeout
        while time.time() < end and (not self.q.empty() or self.current is not None):
            time.sleep(0.1)

    def close(self) -> None:
        self.q.put((-1, next(self.seq), time.time(), None))


# --------------------------------------------------------------------------- #
# GPS sources
# --------------------------------------------------------------------------- #
@dataclass
class Fix:
    pos: LatLon
    t: float
    accuracy_m: Optional[float] = None
    heading: Optional[float] = None   # degrees; from compass/IMU or GPS course while moving


class SimulatedGPS:
    """Walks along a route with GPS-like noise, for testing without hardware."""

    def __init__(self, route: Route, noise_m: float = 2.0):
        self.route, self.noise = route, noise_m

    def fixes(self, speedup: float = 1.0, tick_s: float = 0.5) -> Iterator[Fix]:
        d = 0.0
        while True:
            p, hdg = self.route.point_at(d)
            n, e = random.gauss(0, self.noise), random.gauss(0, self.noise)
            noisy = LatLon(p.lat + n / 110_540.0, p.lon + e / (111_320.0 * math.cos(math.radians(p.lat))))
            yield Fix(noisy, time.time(), accuracy_m=self.noise * 2, heading=hdg)
            if d >= self.route.total_m:
                return
            d += 1.4 * tick_s * speedup
            time.sleep(tick_s)


class NMEAGPS:
    """Reads a USB/serial GPS receiver (RMC for position/course, GGA for HDOP)."""

    def __init__(self, port: str, baud: int = 9600):
        import serial  # pyserial
        self.ser = serial.Serial(port, baud, timeout=2)

    @staticmethod
    def _deg(value: str, hemi: str) -> float:
        v = float(value)
        d = int(v / 100)
        deg = d + (v - d * 100) / 60
        return -deg if hemi in ("S", "W") else deg

    def fixes(self) -> Iterator[Fix]:
        hdop: Optional[float] = None
        while True:
            line = self.ser.readline().decode("ascii", errors="ignore").strip()
            f = line.split(",")
            try:
                if line[3:6] == "GGA" and len(f) > 8 and f[8]:
                    hdop = float(f[8])
                elif line[3:6] == "RMC" and len(f) > 8 and f[2] == "A":
                    pos = LatLon(self._deg(f[3], f[4]), self._deg(f[5], f[6]))
                    knots = float(f[7] or 0)
                    course = float(f[8]) if f[8] and knots * 0.514 > 0.5 else None  # course is noise when standing still
                    yield Fix(pos, time.time(), accuracy_m=(hdop * 5 if hdop else None), heading=course)
            except (ValueError, IndexError):
                continue


# --------------------------------------------------------------------------- #
# Navigator: turn-by-turn from GPS fixes
# --------------------------------------------------------------------------- #
CROSSING_RE = re.compile(r"\b(cross|crossing|crosswalk)\b", re.I)


class Navigator:
    PRE_M = 50.0               # "In 150 feet, turn left"
    NOW_M = 10.0               # "Turn left"
    ARRIVE_M = 15.0
    OFF_ROUTE_M = 25.0
    OFF_ROUTE_FIXES = 3        # consecutive bad fixes before rerouting
    REROUTE_COOLDOWN_S = 20.0
    MAX_ACCURACY_M = 30.0

    def __init__(self, route: Route, mapsvc: MapService, dest: LatLon, audio: AudioArbiter, units: str):
        self.map, self.dest, self.audio, self.units = mapsvc, dest, audio, units
        self.last_msg = ""
        self._last_reroute = 0.0
        self._last_weak_gps = 0.0
        self._load(route)

    def _load(self, route: Route) -> None:
        self.route = route
        self._idx = 0
        self._step = -1
        self._announced: set[tuple[int, str]] = set()
        self._off = 0
        self.remaining_m = route.total_m
        self.dist_to_next_m: Optional[float] = None
        self.next_text = ""

    def _say(self, text: str, prio: Prio = Prio.NAVIGATION) -> None:
        self.last_msg = text
        self.audio.say(text, prio)

    @staticmethod
    def _lower(s: str) -> str:
        return s[:1].lower() + s[1:]

    def begin(self) -> None:
        r = self.route
        mins = max(1, round(r.total_m / WALK_SPEED_MPS / 60))
        self._say(f"Route ready. {fmt_distance(r.total_m, self.units)}, about {mins} minutes. {r.steps[0].instruction}.")

    def status(self) -> str:
        mins = max(1, round(self.remaining_m / WALK_SPEED_MPS / 60))
        s = f"{fmt_distance(self.remaining_m, self.units)} to go, about {mins} minutes."
        if self.dist_to_next_m is not None and self.next_text:
            s += f" In {fmt_distance(self.dist_to_next_m, self.units)}, {self._lower(self.next_text)}."
        return s

    def request_status(self) -> None:
        self._say(self.status())

    def repeat(self) -> None:
        if self.last_msg:
            self.audio.say(self.last_msg, Prio.NAVIGATION)

    def update(self, fix: Fix) -> bool:
        """Feed one GPS fix. Returns True when the destination has been reached."""
        now = time.time()
        if fix.accuracy_m and fix.accuracy_m > self.MAX_ACCURACY_M:       # fail loudly, but not constantly
            if now - self._last_weak_gps > 60:
                self._last_weak_gps = now
                self._say("GPS signal weak. Directions paused.")
            return False

        r = self.route
        if haversine(fix.pos, r.points[-1]) <= self.ARRIVE_M:
            self._say("You have arrived at your destination.")
            return True

        off_m, i, t = r.snap(fix.pos, self._idx)
        if off_m > self.OFF_ROUTE_M:
            self._off += 1
            if self._off >= self.OFF_ROUTE_FIXES and now - self._last_reroute > self.REROUTE_COOLDOWN_S:
                self._reroute(fix)
            return False
        self._off = 0
        self._idx = i

        pos_m = r.cum[i] + t * r.seg_len[i]
        self.remaining_m = max(0.0, r.total_m - pos_m)
        k = max(r.step_at(i), self._step)        # never step backwards because of GPS jitter at a corner

        if k != self._step:                      # entered a new step
            self._step = k
            step = r.steps[k]
            if k > 0 and step.distance_m > 80 and step.maneuver != 10:
                self._say(f"Continue {fmt_distance(step.distance_m, self.units)}.")

        nxt = r.steps[k + 1] if k + 1 < len(r.steps) else None
        if nxt is None:
            self.dist_to_next_m, self.next_text = None, ""
            return False

        d_man = max(0.0, r.cum[nxt.start_idx] - pos_m)
        self.dist_to_next_m, self.next_text = d_man, nxt.instruction
        caution = " Crossing ahead. The decision to cross is yours; listen for traffic." if CROSSING_RE.search(nxt.instruction) else ""
        arriving = nxt.maneuver == 10            # arrival itself is announced by the distance check above

        if d_man <= self.NOW_M and not arriving and (k, "now") not in self._announced:
            self._announced |= {(k, "pre"), (k, "now")}
            self._say(f"{nxt.instruction} now.{caution}")
        elif d_man <= self.PRE_M and (k, "pre") not in self._announced:
            self._announced.add((k, "pre"))
            if arriving:
                self._say(f"Destination in {fmt_distance(d_man, self.units)}.")
            else:
                self._say(f"In {fmt_distance(d_man, self.units)}, {self._lower(nxt.instruction)}.{caution}")
        return False

    def _reroute(self, fix: Fix) -> None:
        self._last_reroute = time.time()
        self._say("Off route. Recalculating.")
        try:
            new = self.map.route(fix.pos, self.dest)
        except Exception as e:                    # network down, API error, etc.
            print(f"[reroute failed] {e}", file=sys.stderr)
            self._say("Cannot recalculate. Check your connection. Directions paused.")
            return
        self._load(new)
        msg = f"New route, {fmt_distance(new.total_m, self.units)}."
        if fix.heading is not None and len(new.points) > 1:
            msg += " " + relative_direction(fix.heading, bearing(new.points[0], new.points[1]))
        self._say(msg)


# --------------------------------------------------------------------------- #
# Camera: distance estimate, obstacle and collision warnings
# --------------------------------------------------------------------------- #
# Approximate real-world heights (m) used for a monocular distance estimate.
REAL_HEIGHT_M = {
    "person": 1.70, "bicycle": 1.10, "motorcycle": 1.20, "car": 1.50, "bus": 3.20,
    "truck": 3.00, "bench": 0.85, "chair": 0.90, "fire hydrant": 0.60, "parking meter": 1.20,
}
MOVERS = {"person", "bicycle", "motorcycle", "car", "bus", "truck"}
SPOKEN_NAME = {"bicycle": "bike", "motorcycle": "motorbike"}


def estimate_distance(box_h_px: float, real_h_m: float, focal_px: float) -> float:
    """Pinhole model: distance = focal_px * real_height / pixel_height."""
    return focal_px * real_h_m / max(1.0, box_h_px)


class CameraMonitor(threading.Thread):
    """Runs on every camera frame. Collision uses time-to-collision; obstacle uses a distance threshold."""

    def __init__(self, audio: AudioArbiter, units: str, source=0, model_path: str = "yolo26n.pt",
                 hfov_deg: float = 70.0, conf: float = 0.4, ttc_threshold_s: float = 3.0,
                 obstacle_m: float = 2.0, max_fps: float = 15.0):
        super().__init__(daemon=True)
        self.audio, self.units, self.source, self.model_path = audio, units, source, model_path
        self.hfov, self.conf, self.ttc_thr, self.obstacle_m = hfov_deg, conf, ttc_threshold_s, obstacle_m
        self.min_dt = 1.0 / max_fps
        self.stop_event = threading.Event()
        self.history: dict[int, deque] = {}       # track id -> recent (time, distance)
        self.last_alert: dict[object, float] = {}

    def _alert(self, key, cooldown: float, text: str, prio: Prio) -> None:
        now = time.time()
        if now - self.last_alert.get(key, 0) >= cooldown:
            self.last_alert[key] = now
            self.audio.say(text, prio)

    def run(self) -> None:
        try:
            import cv2
            from ultralytics import YOLO
            model = YOLO(self.model_path)
        except Exception as e:
            print(f"[camera] unavailable: {e}", file=sys.stderr)
            self.audio.say("Safety alerts off. Obstacle detection unavailable.", Prio.OBSTACLE)
            return
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.audio.say("Camera not found. Safety alerts off.", Prio.OBSTACLE)
            return

        fails = poor = 0
        while not self.stop_event.is_set():
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                fails += 1
                if fails == 10:
                    self.audio.say("Camera blocked. Safety alerts off.", Prio.OBSTACLE)
                time.sleep(0.1)
                continue
            fails = 0

            h, w = frame.shape[:2]
            poor = poor + 1 if frame.std() < 8 else 0           # lens covered or too dark
            if poor >= 15:
                self._alert("poor_view", 30, "Camera view poor. Safety alerts reduced.", Prio.OBSTACLE)

            focal_px = (w / 2) / math.tan(math.radians(self.hfov) / 2)
            res = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=self.conf, verbose=False)[0]
            now = time.time()
            if res.boxes is not None and len(res.boxes):
                xyxy = res.boxes.xyxy.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy().astype(int)
                ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else [None] * len(cls)
                for (x1, y1, x2, y2), c, tid in zip(xyxy, cls, ids):
                    self._process(res.names[c], x1, y1, x2, y2, tid, w, h, focal_px, now)

            for k in [k for k, v in self.history.items() if now - v[-1][0] > 3]:   # forget lost tracks
                del self.history[k]

            time.sleep(max(0.0, self.min_dt - (time.time() - t0)))
        cap.release()

    def _process(self, label, x1, y1, x2, y2, tid, w, h, focal_px, now) -> None:
        real_h = REAL_HEIGHT_M.get(label)
        if real_h is None:
            return
        d = estimate_distance(y2 - y1, real_h, focal_px)
        if y1 <= 2 and y2 >= h - 2:        # box fills the frame vertically -> truncated, so the estimate is too far
            d = min(d, 1.0)
        cx = (x1 + x2) / 2 / w
        side = "left" if cx < 0.35 else "right" if cx > 0.65 else "ahead"
        in_path = x2 > 0.3 * w and x1 < 0.7 * w
        name = SPOKEN_NAME.get(label, label)

        ttc, closing = None, 0.0
        if tid is not None:
            hist = self.history.setdefault(int(tid), deque(maxlen=10))
            hist.append((now, d))
            old = next((s for s in hist if now - s[0] >= 0.4), None)
            if old:
                closing = (old[1] - d) / (now - old[0])      # m/s toward the user (includes the user's own walking)
                if closing > 0.3:
                    ttc = d / closing

        key = tid if tid is not None else label
        if label in MOVERS and ttc is not None and ttc < self.ttc_thr and d < 15:
            self._alert(("c", key), 2.0, f"{name}, {side}" + (", fast" if closing > 3 else ""), Prio.COLLISION)
        elif in_path and d <= self.obstacle_m:
            self._alert(("o", key), 4.0, f"{name} {side}, {fmt_distance(d, self.units)}", Prio.OBSTACLE)

    def stop(self) -> None:
        self.stop_event.set()


# --------------------------------------------------------------------------- #
# Console commands (stand-in for the physical button / wake word)
# --------------------------------------------------------------------------- #
def command_listener(nav: Navigator, stop: threading.Event) -> None:
    for line in sys.stdin:
        c = line.strip().lower()
        if c == "d":
            nav.request_status()
        elif c == "r":
            nav.repeat()
        elif c == "q":
            stop.set()
            return


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_map_service(kind: str, valhalla_url: str) -> MapService:
    key = os.getenv("ORS_API_KEY")
    if kind == "auto":
        kind = "ors" if key else "valhalla"
    try:
        if kind == "ors":
            if not key:
                raise RuntimeError("ORS_API_KEY is not set")
            return ORSMapService(key)
        if kind == "valhalla":
            return ValhallaMapService(valhalla_url)
    except Exception as e:                       # e.g. 'requests' not installed
        print(f"[warn] {kind} map service unavailable ({e}); using straight-line fallback.", file=sys.stderr)
    return StraightLineMapService()


def main() -> None:
    ap = argparse.ArgumentParser(description="Walking navigation for the Immersive Assistant")
    ap.add_argument("--dest", required=True, help="'lat,lon', a UK postcode, or a place name")
    ap.add_argument("--map", choices=["auto", "valhalla", "ors", "straight"], default="auto",
                    help="map service: auto = ORS if ORS_API_KEY is set, else Valhalla")
    ap.add_argument("--valhalla-url", default=os.getenv("VALHALLA_URL", ValhallaMapService.PUBLIC))
    ap.add_argument("--origin", help="'lat,lon' start for simulation (omit when using --nmea)")
    ap.add_argument("--nmea", help="serial port of a GPS receiver, e.g. /dev/ttyUSB0 or COM3")
    ap.add_argument("--simulate", type=float, default=1.0, help="simulated walking speed multiplier")
    ap.add_argument("--units", choices=["feet", "meters"], default="feet")
    ap.add_argument("--camera", default="0", help="camera index or video file path")
    ap.add_argument("--no-camera", action="store_true", help="navigation only, skip obstacle detection")
    ap.add_argument("--model", default="yolo26n.pt", help="Ultralytics weights (yolo11n.pt also works)")
    args = ap.parse_args()
    if not args.nmea and not args.origin:
        ap.error("give --origin (simulation) or --nmea (real GPS)")

    audio = AudioArbiter(SystemTTS())
    audio.start()

    mapsvc = build_map_service(args.map, args.valhalla_url)

    gps_iter: Optional[Iterator[Fix]] = None
    if args.nmea:
        gps_iter = NMEAGPS(args.nmea).fixes()
        audio.say("Waiting for GPS.", Prio.NAVIGATION)
        first = next(gps_iter)
        origin = first.pos
    else:
        origin = parse_latlon(args.origin)
        if not origin:
            ap.error("--origin must be 'lat,lon'")

    try:
        dest = parse_latlon(args.dest) or mapsvc.geocode(args.dest, near=origin)
        route = mapsvc.route(origin, dest)
    except Exception as e:
        audio.say("Could not get directions. Check your connection.", Prio.NAVIGATION)
        print(f"[error] {e}", file=sys.stderr)
        audio.wait_idle()
        return

    if gps_iter is None:
        gps_iter = SimulatedGPS(route).fixes(speedup=args.simulate)

    nav = Navigator(route, mapsvc, dest, audio, args.units)
    nav.begin()

    stop = threading.Event()
    threading.Thread(target=command_listener, args=(nav, stop), daemon=True).start()

    cam: Optional[CameraMonitor] = None
    if not args.no_camera:
        src = int(args.camera) if args.camera.isdigit() else args.camera
        cam = CameraMonitor(audio, args.units, source=src, model_path=args.model)
        cam.start()

    try:
        for fix in gps_iter:
            if stop.is_set() or nav.update(fix):
                break
    except KeyboardInterrupt:
        pass
    finally:
        if cam:
            cam.stop()
        audio.wait_idle()
        audio.close()


if __name__ == "__main__":
    main()
