from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from time import time
from typing import Any, Iterable

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon, box

_logger = logging.getLogger(__name__)

# ── COCO-80 class ID → human-readable name ──────────────────────────
COCO_NAMES: dict[int, str] = {
    0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle", 4: "Airplane",
    5: "Bus", 6: "Train", 7: "Truck", 8: "Boat", 9: "Traffic Light",
    10: "Fire Hydrant", 11: "Stop Sign", 12: "Parking Meter", 13: "Bench",
    14: "Bird", 15: "Cat", 16: "Dog", 17: "Horse", 18: "Sheep", 19: "Cow",
    20: "Elephant", 21: "Bear", 22: "Zebra", 23: "Giraffe", 24: "Backpack",
    25: "Umbrella", 26: "Handbag", 27: "Tie", 28: "Suitcase",
    29: "Frisbee", 30: "Skis", 31: "Snowboard", 32: "Sports Ball",
    33: "Kite", 34: "Baseball Bat", 35: "Baseball Glove", 36: "Skateboard",
    37: "Surfboard", 38: "Tennis Racket", 39: "Bottle", 40: "Wine Glass",
    41: "Cup", 42: "Fork", 43: "Knife", 44: "Spoon", 45: "Bowl",
    46: "Banana", 47: "Apple", 48: "Sandwich", 49: "Orange", 50: "Broccoli",
    51: "Carrot", 52: "Hot Dog", 53: "Pizza", 54: "Donut", 55: "Cake",
    56: "Chair", 57: "Couch", 58: "Potted Plant", 59: "Bed",
    60: "Dining Table", 61: "Toilet", 62: "TV", 63: "Laptop",
    64: "Mouse", 65: "Remote", 66: "Keyboard", 67: "Cell Phone",
    68: "Microwave", 69: "Oven", 70: "Toaster", 71: "Sink",
    72: "Refrigerator", 73: "Book", 74: "Clock", 75: "Vase",
    76: "Scissors", 77: "Teddy Bear", 78: "Hair Drier", 79: "Toothbrush",
}


def _resolve_class_name(class_id: int, class_name: str | None) -> str:
    """Return a human-readable name: prefer YOLO-supplied name, fall back
    to COCO lookup, last resort 'Object-<id>'."""
    if class_name:
        return class_name.title()
    return COCO_NAMES.get(class_id, f"Object-{class_id}")


@dataclass
class ObjectState:
    track_id: int
    class_id: int
    class_name: str  # always resolved to human-readable
    confidence: float
    bbox: np.ndarray  # [x1, y1, x2, y2]
    center: np.ndarray  # [cx, cy]
    speed: float
    zone: str | None
    loiter_seconds: float
    erratic: bool
    close_to: list[str] = field(default_factory=list)  # e.g. ["Person#2", "Handbag#5"]
    keypoints: list | None = field(default=None)
    theft_stage: str = field(default="NONE")
    last_kinematic_time: float = field(default=0.0)
    last_keypoints: list | None = field(default=None)
    kp_ttl: int = field(default=0)


class AdvancedLogicEngine:
    def __init__(self, zones: dict[str, Iterable[Iterable[float]]]) -> None:
        self._zones = {name: Polygon(points) for name, points in zones.items()}
        self._history: dict[int, dict[str, Any]] = {}
        self._frame_index = 0
        self._last_timestamp = None

        # Behavior tuning (pixels per second unless calibrated)
        self._fps = 30.0
        self._max_missing_frames = 15
        self._hysteresis_frames = 3
        self._running_speed = 250.0
        self._proximity_px = 120.0
        self._erratic_std = 120.0

        # Event hysteresis counters
        self._event_counts = defaultdict(int)

        # Zone hysteresis: (track_id, zone_name) -> consecutive frames inside
        self._zone_frame_counts: dict[tuple[int, str], int] = {}

        # Interaction thresholds
        self._interaction_px = 20.0  # bbox gap < 20px = "touching/near"

        # Perspective scaling (updated each frame from YOLO result)
        self._frame_height: float = 720.0
        self._frame_width: float = 1280.0

        # Ghost state: tracks that disappeared but may reappear (anti-jitter)
        # track_id -> {"hist": <history dict>, "ttl": int, "entered_frame": int}
        self._ghost_states: dict[int, dict[str, Any]] = {}

        # Object class semantics
        self._person_class_ids = {0}
        self._weapon_class_names = {"knife", "gun", "weapon", "pistol", "rifle"}
        self._weapon_class_ids: set[int] = set()
        # COCO stealable objects: handbag(26), backpack(24), suitcase(28),
        # laptop(63), cell phone(67), book(73)
        self._stealable_class_ids = {24, 26, 28, 63, 67, 73}
        self._stealable_class_names = {
            "handbag", "backpack", "suitcase", "bag", "laptop",
            "cell phone", "phone", "book", "purse", "wallet",
        }

        self.theft_detection_enabled: bool = True
        self.pose_theft_mode: bool = False

        # Event log — rolling buffer of alerts for the LLM
        self._event_log: deque[str] = deque(maxlen=20)

        # Timeline — sliding window of the last N significant state snapshots
        self._timeline: deque[dict[str, Any]] = deque(maxlen=5)

        self._scene_state: dict[str, Any] = {
            "objects": [],
            "scene_text": "",
            "frame_index": 0,
            "timestamp": 0.0,
        }

    def reset(self) -> None:
        """Clear all state for a fresh job start."""
        self._history.clear()
        self._event_log.clear()
        self._timeline.clear()
        self._event_counts.clear()
        self._zone_frame_counts.clear()
        self._ghost_states.clear()
        self._frame_index = 0
        self._last_timestamp = None
        self._scene_state = {
            "objects": [],
            "scene_text": "",
            "frame_index": 0,
            "timestamp": 0.0,
        }

    def set_zones(self, zones: dict[str, Iterable[Iterable[float]]]) -> None:
        """Replace active zones at runtime (e.g. when a new job starts)."""
        self._zones = {name: Polygon(points) for name, points in zones.items()}

    def update(self, yolo_results: dict[str, Any] | list[Any]) -> None:
        if isinstance(yolo_results, dict):
            if "frame_height" in yolo_results:
                self._frame_height = float(yolo_results["frame_height"])
            if "frame_width" in yolo_results:
                self._frame_width = float(yolo_results["frame_width"])

        detections = self._normalize_detections(yolo_results)
        if not detections:
            self._advance_time(yolo_results)
            self._scene_state["scene_text"] = "No detections."
            return

        keypoints_map: dict[int, list] = {}
        if isinstance(yolo_results, dict):
            keypoints_map = yolo_results.get("keypoints_map", {}) or {}

        frame_index, timestamp = self._advance_time(yolo_results)
        states = self._update_object_history(detections, frame_index, timestamp, keypoints_map)
        self._record_timeline(states, timestamp)
        scene_text = self._build_scene_text(states)
        self._scene_state = {
            "objects": [s.__dict__ for s in states],
            "scene_text": scene_text,
            "frame_index": frame_index,
            "timestamp": timestamp,
        }

    def _advance_time(self, yolo_results: dict[str, Any] | list[Any]) -> tuple[int, float]:
        self._frame_index += 1
        if isinstance(yolo_results, dict):
            frame_index = int(yolo_results.get("frame_index", self._frame_index))
            timestamp = float(yolo_results.get("timestamp", time()))
        else:
            frame_index = self._frame_index
            timestamp = time()
        self._last_timestamp = timestamp
        return frame_index, timestamp

    def _normalize_detections(self, yolo_results: dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
        if isinstance(yolo_results, dict):
            if "detections" in yolo_results:
                raw = yolo_results["detections"]
            elif "vectors" in yolo_results:
                raw = yolo_results["vectors"]
            else:
                raw = []
        else:
            raw = yolo_results

        detections: list[dict[str, Any]] = []
        for item in raw:
            if isinstance(item, dict):
                detections.append(item)
                continue
            if isinstance(item, (list, tuple)) and len(item) >= 6:
                # [class_id, conf, x1, y1, x2, y2, track_id?]
                track_id = int(item[6]) if len(item) >= 7 and item[6] is not None else -1
                detections.append(
                    {
                        "track_id": track_id,
                        "class_id": int(item[0]),
                        "confidence": float(item[1]),
                        "bbox": [float(x) for x in item[2:6]],
                    }
                )
        return detections

    def _update_object_history(
        self, detections: list[dict[str, Any]], frame_index: int, timestamp: float,
        keypoints_map: dict[int, list] | None = None,
    ) -> list[ObjectState]:
        states: list[ObjectState] = []
        seen_ids = set()

        for det in detections:
            track_id = int(det.get("track_id", -1))
            class_id = int(det.get("class_id", -1))
            class_name = _resolve_class_name(class_id, det.get("class_name"))
            conf = float(det.get("confidence", 0.0))
            bbox = np.array(det.get("bbox", det.get("xyxy", [0, 0, 0, 0])), dtype=float)
            center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            zone_point = (
                np.array([(bbox[0] + bbox[2]) / 2, bbox[3]])
                if class_id in self._person_class_ids
                else center
            )

            if track_id in self._ghost_states:
                ghost = self._ghost_states.pop(track_id)
                self._history[track_id] = ghost["hist"]
                _logger.debug("ghost_restored track_id=%d after %d frames",
                              track_id, self._max_missing_frames - ghost["ttl"] + 1)

            if track_id not in self._history:
                self._history[track_id] = {
                    "last_center": center,
                    "last_seen_frame": frame_index,
                    "last_seen_time": timestamp,
                    "speed": 0.0,
                    "speed_hist": deque(maxlen=10),
                    "zone_durations": defaultdict(float),
                    "class_id": class_id,
                    "class_name": class_name,
                    "last_close_to": [],
                    "theft_stage": "NONE",
                    "last_kinematic_time": 0.0,
                    "last_keypoints": None,
                    "kp_ttl": 0,
                }
            hist = self._history[track_id]
            dt = max(timestamp - hist["last_seen_time"], 1.0 / self._fps)
            raw_speed = float(np.linalg.norm(center - hist["last_center"]) / dt)
            depth_scale = max(bbox[3] / self._frame_height, 0.1)
            speed = raw_speed / depth_scale
            hist["speed"] = speed
            hist["speed_hist"].append(speed)
            hist["last_center"] = center
            hist["last_seen_frame"] = frame_index
            hist["last_seen_time"] = timestamp
            hist["class_id"] = class_id
            hist["class_name"] = class_name
            seen_ids.add(track_id)

            zone = self._find_zone(zone_point)
            if class_id in self._person_class_ids and self._zones and frame_index % 20 == 0:
                _logger.info(
                    "zone_check Person#%d bbox=[%.0f,%.0f,%.0f,%.0f] zone_point=(%.0f,%.0f) zones=%s result=%s",
                    track_id, bbox[0], bbox[1], bbox[2], bbox[3],
                    zone_point[0], zone_point[1],
                    {n: [(int(c[0]), int(c[1])) for c in list(p.exterior.coords)[:3]] for n, p in self._zones.items()},
                    zone,
                )
            prev_zone = hist.get("current_zone")
            hist["current_zone"] = zone

            if zone:
                hist["zone_durations"][zone] += dt
                self._zone_frame_counts[(track_id, zone)] = (
                    self._zone_frame_counts.get((track_id, zone), 0) + 1
                )
                for key in [k for k in self._zone_frame_counts if k[0] == track_id and k[1] != zone]:
                    del self._zone_frame_counts[key]
                if self._zone_frame_counts[(track_id, zone)] == self._hysteresis_frames:
                    self._event_log.append(
                        f"ZONE_INTRUSION: {class_name}#{track_id} entered zone '{zone}'."
                    )
            else:
                for key in [k for k in self._zone_frame_counts if k[0] == track_id]:
                    del self._zone_frame_counts[key]

            speed_hist = np.array(hist["speed_hist"], dtype=float)
            erratic = bool(speed_hist.size >= 3 and speed_hist.std() > self._erratic_std)

            kp = None
            if self.pose_theft_mode and keypoints_map and track_id in keypoints_map:
                kp = keypoints_map[track_id]

            if kp is not None:
                hist["last_keypoints"] = kp
                hist["kp_ttl"] = 15
            elif hist.get("kp_ttl", 0) > 0:
                kp = hist.get("last_keypoints")
                hist["kp_ttl"] -= 1

            states.append(
                ObjectState(
                    track_id=track_id,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=conf,
                    bbox=bbox,
                    center=center,
                    speed=speed,
                    zone=zone,
                    loiter_seconds=float(hist["zone_durations"].get(zone, 0.0)) if zone else 0.0,
                    erratic=erratic,
                    keypoints=kp,
                    theft_stage=hist.get("theft_stage", "NONE"),
                    last_kinematic_time=hist.get("last_kinematic_time", 0.0),
                    last_keypoints=hist.get("last_keypoints"),
                    kp_ttl=hist.get("kp_ttl", 0),
                )
            )

        # --- Interaction detection ---
        self._detect_interactions(states)

        if self.pose_theft_mode:
            self._detect_pose_theft_heuristics(states)

        # Snapshot close_to and theft state back into history
        for s in states:
            if s.track_id in self._history:
                self._history[s.track_id]["last_close_to"] = list(s.close_to)
                self._history[s.track_id]["theft_stage"] = s.theft_stage
                self._history[s.track_id]["last_kinematic_time"] = s.last_kinematic_time

        # --- Disappearance detection (theft / concealment) ---
        self._detect_disappearances(states, seen_ids, frame_index)

        return states

    # -----------------------------------------------------------------
    # Interaction detection
    # -----------------------------------------------------------------
    def _detect_interactions(self, objects: list[ObjectState]) -> None:
        """Populate close_to for each object by checking bbox proximity.

        Uses a KD-Tree on center points as a broad-phase filter, then
        verifies candidates with exact Shapely bbox distance.
        """
        n = len(objects)
        if n < 2:
            return

        centers = np.array([obj.center for obj in objects])
        fh = self._frame_height

        max_threshold = self._interaction_px * max(
            obj.bbox[3] / fh for obj in objects
        )
        tree = cKDTree(centers)
        candidate_pairs = tree.query_pairs(r=max_threshold + self._interaction_px)

        if not candidate_pairs:
            return

        polys = [box(*obj.bbox.tolist()) for obj in objects]
        for i, j in candidate_pairs:
            avg_scale = (objects[i].bbox[3] + objects[j].bbox[3]) / (2.0 * fh)
            threshold = self._interaction_px * max(avg_scale, 0.1)
            if polys[i].distance(polys[j]) < threshold:
                objects[i].close_to.append(f"{objects[j].class_name}#{objects[j].track_id}")
                objects[j].close_to.append(f"{objects[i].class_name}#{objects[i].track_id}")

    # -----------------------------------------------------------------
    # Pose-based theft heuristics (only when pose_theft_mode is True)
    # -----------------------------------------------------------------
    _SHOULDER_INDICES = (5, 6)  # COCO: left_shoulder, right_shoulder
    _WRIST_INDICES = (9, 10)   # COCO: left_wrist, right_wrist
    _HIP_INDICES = (11, 12)    # COCO: left_hip, right_hip
    _KP_MIN_CONF = 0.3
    _WRIST_HIP_PX = 100.0      # base proximity threshold (scaled by perspective)
    _BROWSING_TOUCH_PX = 80.0   # wrist-to-item base distance (scaled by perspective)

    _STAGE_TIMEOUT = 15.0       # max seconds between BROWSING -> CONCEALING
    _BROWSING_IDLE_TIMEOUT = 30.0  # seconds without interaction before reset to NONE
    _CONCEALING_TIMEOUT = 20.0  # seconds in CONCEALING before reset if no FLIGHT
    _FLIGHT_SPEED = 60.0        # perspective-normalized speed threshold for FLIGHT

    def _detect_pose_theft_heuristics(self, objects: list[ObjectState]) -> None:
        """Kinematic checks with deterministic state machine transitions.

        State machine: NONE -> BROWSING -> CONCEALING -> FLIGHT
        BROWSING: wrist touches a stealable item (perspective-scaled distance)
        CONCEALING: wrist moves AWAY from item and INTO hip/torso area
        FLIGHT: person starts moving after concealment, OR item disappears
        """
        persons = [o for o in objects if o.class_id in self._person_class_ids and o.keypoints]
        if not persons:
            return

        stealables = [
            o for o in objects
            if o.class_id in self._stealable_class_ids
            or o.class_name.lower() in self._stealable_class_names
        ]
        stealable_polys = [(s, box(*s.bbox.tolist())) for s in stealables] if stealables else []

        fh = self._frame_height
        now = self._last_timestamp or 0.0

        for person in persons:
            kp = person.keypoints
            if not kp or len(kp) < 13:
                continue

            wrists: list[tuple[float, float]] = []
            for idx in self._WRIST_INDICES:
                pt = kp[idx]
                if len(pt) >= 3 and pt[2] >= self._KP_MIN_CONF:
                    wrists.append((pt[0], pt[1]))
                elif len(pt) >= 2 and pt[0] > 0 and pt[1] > 0:
                    wrists.append((pt[0], pt[1]))

            if not wrists:
                continue

            depth_scale = max(person.bbox[3] / fh, 0.1)
            item_threshold = self._BROWSING_TOUCH_PX * depth_scale

            # --- Wrist-to-Item (BROWSING) ---
            item_touched = False
            if stealable_polys:
                for wx, wy in wrists:
                    wrist_pt = Point(wx, wy)
                    for item, item_poly in stealable_polys:
                        if item_poly.distance(wrist_pt) < item_threshold:
                            self._event_log.append(
                                f"POSE_KINEMATIC: Person#{person.track_id} wrist "
                                f"interacted with {item.class_name}#{item.track_id}"
                            )
                            item_touched = True
                            break
                    if item_touched:
                        break

            if item_touched:
                person.last_kinematic_time = now
                if person.theft_stage == "NONE":
                    person.theft_stage = "BROWSING"
                    self._event_log.append(
                        f"STATE_CHANGE: Person#{person.track_id} entered BROWSING stage"
                    )

            # --- Idle timeout: BROWSING -> NONE ---
            if (
                not item_touched
                and person.theft_stage == "BROWSING"
                and (now - person.last_kinematic_time) > self._BROWSING_IDLE_TIMEOUT
            ):
                person.theft_stage = "NONE"
                self._event_log.append(
                    f"STATE_CHANGE: Person#{person.track_id} returned to NONE (idle timeout)"
                )

            # --- Idle timeout: CONCEALING -> NONE ---
            if (
                person.theft_stage == "CONCEALING"
                and (now - person.last_kinematic_time) > self._CONCEALING_TIMEOUT
            ):
                person.theft_stage = "NONE"
                self._event_log.append(
                    f"STATE_CHANGE: Person#{person.track_id} returned to NONE (concealment timeout)"
                )

            # --- Wrist-to-Hip / Wrist-to-Torso (CONCEALING) ---
            # Only check when wrist is NOT currently touching an item.
            # Physically: person grabbed item, then moved hand to pocket/torso.
            concealment_threshold = self._WRIST_HIP_PX * depth_scale
            concealment_detected = False

            if not item_touched:
                hips: list[tuple[float, float]] = []
                for idx in self._HIP_INDICES:
                    pt = kp[idx]
                    if len(pt) >= 3 and pt[2] >= self._KP_MIN_CONF:
                        hips.append((pt[0], pt[1]))
                    elif len(pt) >= 2 and pt[0] > 0 and pt[1] > 0:
                        hips.append((pt[0], pt[1]))

                if hips:
                    for wx, wy in wrists:
                        for hx, hy in hips:
                            if float(np.hypot(wx - hx, wy - hy)) < concealment_threshold:
                                self._event_log.append(
                                    f"POSE_KINEMATIC: Person#{person.track_id} "
                                    f"wrist moved to hip/pocket area"
                                )
                                concealment_detected = True
                                break
                        if concealment_detected:
                            break

                if not concealment_detected:
                    torso_pts: list[tuple[float, float]] = []
                    for idx in self._SHOULDER_INDICES + self._HIP_INDICES:
                        pt = kp[idx]
                        if len(pt) >= 3 and pt[2] >= self._KP_MIN_CONF:
                            torso_pts.append((pt[0], pt[1]))
                        elif len(pt) >= 2 and pt[0] > 0 and pt[1] > 0:
                            torso_pts.append((pt[0], pt[1]))
                    if len(torso_pts) >= 2:
                        tcx = sum(p[0] for p in torso_pts) / len(torso_pts)
                        tcy = sum(p[1] for p in torso_pts) / len(torso_pts)
                        for wx, wy in wrists:
                            if float(np.hypot(wx - tcx, wy - tcy)) < concealment_threshold:
                                self._event_log.append(
                                    f"POSE_KINEMATIC: Person#{person.track_id} "
                                    f"wrist moved to inner jacket/torso area"
                                )
                                concealment_detected = True
                                break

            if (
                concealment_detected
                and person.theft_stage == "BROWSING"
                and (now - person.last_kinematic_time) < self._STAGE_TIMEOUT
            ):
                person.theft_stage = "CONCEALING"
                person.last_kinematic_time = now
                self._event_log.append(
                    f"STATE_CHANGE: Person#{person.track_id} entered CONCEALING stage"
                )

            # --- Speed-based FLIGHT: person concealed item and is now moving ---
            if (
                person.theft_stage == "CONCEALING"
                and person.speed >= self._FLIGHT_SPEED
                and (now - person.last_kinematic_time) > 1.0
            ):
                person.theft_stage = "FLIGHT"
                self._event_log.append(
                    f"STATE_CHANGE: Person#{person.track_id} entered FLIGHT stage "
                    f"— moving at {person.speed:.0f}px/s after concealment"
                )

    # -----------------------------------------------------------------
    # Disappearance / concealment detection
    # -----------------------------------------------------------------
    def _detect_disappearances(
        self,
        current_states: list[ObjectState],
        seen_ids: set[int],
        frame_index: int,
    ) -> None:
        """Move stale tracks into a ghost state with a TTL before alerting.

        If a track reappears before its TTL expires, it is restored in
        _update_object_history.  Only when TTL reaches 0 do we evaluate
        theft/concealment alerts."""
        _GHOST_TTL = self._max_missing_frames

        for track_id, hist in list(self._history.items()):
            if track_id in seen_ids:
                continue
            frames_missing = frame_index - hist["last_seen_frame"]
            if frames_missing <= self._max_missing_frames:
                continue

            if track_id not in self._ghost_states:
                ttl = _GHOST_TTL
                if self.pose_theft_mode:
                    for nb in hist.get("last_close_to", []):
                        if nb.lower().startswith("person"):
                            import re as _re
                            m = _re.search(r"#(\d+)", nb)
                            if m:
                                ptid = int(m.group(1))
                                phist = self._history.get(ptid)
                                if phist and phist.get("theft_stage") in ("BROWSING", "CONCEALING"):
                                    ttl = 60  # Extended TTL for concealment sequences

                self._ghost_states[track_id] = {
                    "hist": dict(hist),
                    "ttl": ttl,
                    "entered_frame": frame_index,
                }
                del self._history[track_id]
                continue

        for track_id, ghost in list(self._ghost_states.items()):
            if track_id in seen_ids:
                continue
            ghost["ttl"] -= 1
            if ghost["ttl"] > 0:
                continue

            hist = ghost["hist"]
            last_class_id = hist.get("class_id", -1)
            last_class_name = hist.get("class_name") or _resolve_class_name(last_class_id, None)
            last_close_to = hist.get("last_close_to", [])

            if self.theft_detection_enabled:
                is_stealable = (
                    last_class_id in self._stealable_class_ids
                    or last_class_name.lower() in self._stealable_class_names
                )
                if is_stealable and last_close_to:
                    person_neighbors = [
                        nb for nb in last_close_to
                        if nb.lower().startswith("person")
                    ]
                    if person_neighbors:
                        self._event_log.append(
                            f"ALERT: {last_class_name}#{track_id} DISAPPEARED while "
                            f"close to {', '.join(person_neighbors)}. "
                            f"Possible theft/concealment."
                        )
                        if self.pose_theft_mode:
                            import re as _re
                            for nb in person_neighbors:
                                m = _re.search(r"#(\d+)", nb)
                                if not m:
                                    continue
                                ptid = int(m.group(1))
                                phist = self._history.get(ptid)
                                if phist and phist.get("theft_stage") == "CONCEALING":
                                    phist["theft_stage"] = "FLIGHT"
                                    self._event_log.append(
                                        f"STATE_CHANGE: Person#{ptid} entered FLIGHT stage "
                                        f"— {last_class_name}#{track_id} confirmed stolen"
                                    )

            del self._ghost_states[track_id]

    # -----------------------------------------------------------------
    # Timeline — structured per-object snapshots for delta computation
    # -----------------------------------------------------------------
    def _record_timeline(self, states: list[ObjectState], timestamp: float) -> None:
        snapshot: dict[str, Any] = {"timestamp": timestamp, "objects": {}}
        for obj in states:
            cx = float(obj.center[0]) if obj.center is not None else 0.0
            cy = float(obj.center[1]) if obj.center is not None else 0.0
            snapshot["objects"][obj.track_id] = {
                "class_name": obj.class_name,
                "class_id": obj.class_id,
                "speed": obj.speed,
                "center": (cx, cy),
                "grid": self._get_grid_position(cx, cy),
                "erratic": obj.erratic,
                "close_to": list(obj.close_to),
                "zone": obj.zone,
                "is_static": self._is_static(obj),
            }
        self._timeline.append(snapshot)

    # -----------------------------------------------------------------
    # Spatial grid helper
    # -----------------------------------------------------------------
    _GRID_COLS = ("Left", "Center", "Right")
    _GRID_ROWS = ("Top", "Center", "Bottom")

    def _get_grid_position(self, cx: float, cy: float) -> str:
        col_idx = min(int(cx / (self._frame_width / 3.0)), 2)
        row_idx = min(int(cy / (self._frame_height / 3.0)), 2)
        row = self._GRID_ROWS[row_idx]
        col = self._GRID_COLS[col_idx]
        if row == "Center" and col == "Center":
            return "Center"
        return f"{row}-{col}"

    # -----------------------------------------------------------------
    # Semantic scene text — helpers
    # -----------------------------------------------------------------
    _WALK_SPEED = 80.0
    _FAST_SPEED = 200.0

    _PLURALS: dict[str, str] = {
        "person": "people", "mouse": "mice", "knife": "knives",
    }

    @staticmethod
    def _is_static(obj: ObjectState) -> bool:
        if obj.class_id == 0:
            return False
        return obj.speed < 15 and not obj.erratic

    def _classify_motion(self, speed: float, is_person: bool) -> str:
        if is_person:
            if speed < 5:
                return "Stationary"
            if speed < self._WALK_SPEED:
                return "Walking"
            if speed < self._FAST_SPEED:
                return "Fast movement"
            return "Running"
        return "Stationary" if speed < 15 else "DISPLACED"

    @classmethod
    def _pluralize(cls, name: str, count: int) -> str:
        if count == 1:
            return f"1 {name.lower()}"
        plural = cls._PLURALS.get(name.lower(), f"{name.lower()}s")
        return f"{count} {plural}"

    def _describe_direction(self, track_id: int, cx: float, cy: float) -> str:
        if len(self._timeline) < 2:
            return ""
        prev = self._timeline[0].get("objects", {}).get(track_id)
        if not prev:
            return ""
        dx = cx - prev["center"][0]
        dy = cy - prev["center"][1]
        if abs(dx) < 15 and abs(dy) < 15:
            return ""
        parts: list[str] = []
        if dy < -15:
            parts.append("upward")
        elif dy > 15:
            parts.append("downward")
        if dx < -15:
            parts.append("left")
        elif dx > 15:
            parts.append("right")
        return f"heading {'-'.join(parts)}" if parts else ""

    # -----------------------------------------------------------------
    # Semantic scene text — section builders
    # -----------------------------------------------------------------

    def _build_scene_line(self, objects: list[ObjectState]) -> str:
        type_groups: dict[str, dict[str, int]] = {}
        for obj in objects:
            name = obj.class_name
            if name not in type_groups:
                type_groups[name] = {"total": 0, "stationary": 0, "active": 0}
            type_groups[name]["total"] += 1
            if self._is_static(obj):
                type_groups[name]["stationary"] += 1
            else:
                type_groups[name]["active"] += 1

        parts: list[str] = []
        for name in sorted(
            type_groups,
            key=lambda n: (0 if n == "Person" else 1, -type_groups[n]["total"]),
        ):
            g = type_groups[name]
            count = g["total"]
            is_person = name == "Person"
            label = self._pluralize(name, count)

            if g["stationary"] == count:
                parts.append(f"{label} (stationary)")
            elif g["active"] == count:
                parts.append(label if is_person else f"{label} (displaced)")
            else:
                if is_person:
                    parts.append(label)
                else:
                    parts.append(
                        f"{label} ({g['stationary']} stationary, {g['active']} displaced)"
                    )

        return f"SCENE: {', '.join(parts)}" if parts else "SCENE: Empty"

    def _build_behavior_section(self, objects: list[ObjectState]) -> list[str]:
        lines: list[str] = []
        oldest = self._timeline[0] if len(self._timeline) >= 2 else None

        for obj in objects:
            if self._is_static(obj):
                continue

            is_person = obj.class_id in self._person_class_ids
            current_motion = self._classify_motion(obj.speed, is_person)

            prev_motion = None
            was_static = False
            if oldest:
                prev = oldest.get("objects", {}).get(obj.track_id)
                if prev:
                    prev_motion = self._classify_motion(prev["speed"], is_person)
                    was_static = prev["is_static"]

            cx = float(obj.center[0]) if obj.center is not None else 0.0
            cy = float(obj.center[1]) if obj.center is not None else 0.0
            grid = self._get_grid_position(cx, cy)
            direction = self._describe_direction(obj.track_id, cx, cy)

            desc: list[str] = []

            if is_person:
                if prev_motion and prev_motion != current_motion:
                    desc.append(f"Changed from {prev_motion} to {current_motion}")
                else:
                    desc.append(current_motion)
                if obj.erratic:
                    desc.append("erratic path")
                if direction:
                    desc.append(direction)
                desc.append(f"area: {grid}")
                if obj.zone:
                    desc.append(f"inside {obj.zone}")
                if obj.loiter_seconds > 5:
                    desc.append(f"loitering {obj.loiter_seconds:.0f}s")
                if obj.theft_stage not in ("NONE", ""):
                    desc.append(f"theft stage: {obj.theft_stage}")
            else:
                carrier = ""
                if obj.close_to:
                    persons_near = [n for n in obj.close_to if n.startswith("Person")]
                    if persons_near:
                        carrier = f" by {persons_near[0]}"

                if was_static:
                    desc.append(
                        f"DISPLACED{carrier} — was stationary, now being moved"
                    )
                elif current_motion == "DISPLACED":
                    desc.append(f"DISPLACED{carrier} — being moved")
                else:
                    desc.append(current_motion)

                if direction:
                    desc.append(direction)
                desc.append(f"area: {grid}")
                if obj.zone:
                    desc.append(f"entered {obj.zone}")
                if obj.erratic:
                    desc.append("erratic movement")

            lines.append(f"- {obj.class_name}#{obj.track_id}: {', '.join(desc)}")

        if oldest:
            for obj in objects:
                if not self._is_static(obj):
                    continue
                prev = oldest.get("objects", {}).get(obj.track_id)
                if prev and not prev["is_static"]:
                    lines.append(
                        f"- {obj.class_name}#{obj.track_id}: Stopped moving (now stationary)"
                    )

        return lines

    def _build_proximity_section(self, objects: list[ObjectState]) -> list[str]:
        active = [o for o in objects if not self._is_static(o)]
        if len(active) < 2:
            return []
        lines: list[str] = []
        seen_pairs: set[frozenset[str]] = set()
        for obj in active:
            for neighbor in obj.close_to:
                pair = frozenset(
                    [f"{obj.class_name}#{obj.track_id}", neighbor]
                )
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    lines.append(
                        f"- {obj.class_name}#{obj.track_id} is NEAR {neighbor}"
                    )
        return lines

    def _build_alerts_section(self, objects: list[ObjectState]) -> list[str]:
        alerts: list[str] = []
        for obj in objects:
            is_person = obj.class_id in self._person_class_ids

            if not is_person and obj.speed >= 15:
                alerts.append(
                    f"- {obj.class_name}#{obj.track_id} is an item — "
                    f"items cannot self-propel. Displacement indicates human interaction."
                )

            if self._is_weapon(obj):
                alerts.append(
                    f"- WEAPON DETECTED: {obj.class_name}#{obj.track_id}"
                )

            if obj.theft_stage == "BROWSING":
                alerts.append(
                    f"- Person#{obj.track_id}: Interacting with merchandise (BROWSING)"
                )
            elif obj.theft_stage == "CONCEALING":
                alerts.append(
                    f"- Person#{obj.track_id}: CONCEALMENT — "
                    f"hand moved from item to body"
                )
            elif obj.theft_stage == "FLIGHT":
                alerts.append(
                    f"- Person#{obj.track_id}: FLIGHT — "
                    f"moving away after concealing item. Theft confirmed."
                )

            if obj.loiter_seconds > 60:
                alerts.append(
                    f"- {obj.class_name}#{obj.track_id}: Extended loitering "
                    f"in {obj.zone} for {obj.loiter_seconds:.0f}s"
                )

        return alerts

    # -----------------------------------------------------------------
    # Build final semantic scene text for the LLM
    # -----------------------------------------------------------------

    def _build_scene_text(self, objects: list[ObjectState]) -> str:
        if not objects:
            return "Scene empty."

        parts: list[str] = [self._build_scene_line(objects)]

        if self._event_log:
            parts.append("")
            parts.append("EVENTS:")
            for evt in self._event_log:
                parts.append(f"  - {evt}")
            self._event_log.clear()

        behavior = self._build_behavior_section(objects)
        if behavior:
            time_window = ""
            if len(self._timeline) >= 2:
                dt = self._timeline[-1]["timestamp"] - self._timeline[0]["timestamp"]
                time_window = f" (last {dt:.1f}s)"
            parts.append("")
            parts.append(f"BEHAVIOR{time_window}:")
            parts.extend(behavior)

        proximity = self._build_proximity_section(objects)
        if proximity:
            parts.append("")
            parts.append("PROXIMITY:")
            parts.extend(proximity)

        alerts = self._build_alerts_section(objects)
        if alerts:
            parts.append("")
            parts.append("ALERTS:")
            parts.extend(alerts)

        return "\n".join(parts)

    def _is_weapon(self, obj: ObjectState) -> bool:
        if obj.class_id in self._weapon_class_ids:
            return True
        if obj.class_name and obj.class_name.lower() in self._weapon_class_names:
            return True
        return False

    def _find_zone(self, center: np.ndarray) -> str | None:
        point = Point(float(center[0]), float(center[1]))
        for name, polygon in self._zones.items():
            if polygon.contains(point):
                return name
        return None

    def generate_scene_summary(self) -> dict[str, Any]:
        state = dict(self._scene_state)
        objects = []
        for obj in state.get("objects", []):
            if isinstance(obj, dict):
                cleaned = dict(obj)
                bbox = cleaned.get("bbox")
                center = cleaned.get("center")
                if isinstance(bbox, np.ndarray):
                    cleaned["bbox"] = bbox.tolist()
                if isinstance(center, np.ndarray):
                    cleaned["center"] = center.tolist()
                objects.append(cleaned)
            else:
                objects.append(obj)
        state["objects"] = objects
        return state
