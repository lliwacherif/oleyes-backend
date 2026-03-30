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
        if isinstance(yolo_results, dict) and "frame_height" in yolo_results:
            self._frame_height = float(yolo_results["frame_height"])

        detections = self._normalize_detections(yolo_results)
        if not detections:
            self._advance_time(yolo_results)
            self._scene_state["summary_text"] = "No detections."
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
                )
            )

        # --- Interaction detection ---
        self._detect_interactions(states)

        if self.pose_theft_mode:
            self._detect_pose_theft_heuristics(states)

        # Snapshot close_to back into history for disappearance checks
        for s in states:
            if s.track_id in self._history:
                self._history[s.track_id]["last_close_to"] = list(s.close_to)

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
    _WRIST_INDICES = (9, 10)   # COCO: left_wrist, right_wrist
    _HIP_INDICES = (11, 12)    # COCO: left_hip, right_hip
    _KP_MIN_CONF = 0.3
    _WRIST_HIP_PX = 30.0      # base proximity threshold (scaled by perspective)

    def _detect_pose_theft_heuristics(self, objects: list[ObjectState]) -> None:
        """Check wrist-to-item and wrist-to-pocket kinematics for persons."""
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

            for wx, wy in wrists:
                wrist_pt = Point(wx, wy)
                for item, item_poly in stealable_polys:
                    if item_poly.distance(wrist_pt) < self._interaction_px:
                        self._event_log.append(
                            f"POSE_KINEMATIC: Person#{person.track_id} wrist "
                            f"interacted with {item.class_name}#{item.track_id}"
                        )
                        break

            hips: list[tuple[float, float]] = []
            for idx in self._HIP_INDICES:
                pt = kp[idx]
                if len(pt) >= 3 and pt[2] >= self._KP_MIN_CONF:
                    hips.append((pt[0], pt[1]))
                elif len(pt) >= 2 and pt[0] > 0 and pt[1] > 0:
                    hips.append((pt[0], pt[1]))

            if hips:
                depth_scale = max(person.bbox[3] / fh, 0.1)
                threshold = self._WRIST_HIP_PX * depth_scale
                for wx, wy in wrists:
                    for hx, hy in hips:
                        dist = float(np.hypot(wx - hx, wy - hy))
                        if dist < threshold:
                            self._event_log.append(
                                f"POSE_KINEMATIC: Person#{person.track_id} "
                                f"wrist moved to hip/pocket area"
                            )
                            break
                    else:
                        continue
                    break

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
                self._ghost_states[track_id] = {
                    "hist": dict(hist),
                    "ttl": _GHOST_TTL,
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

            del self._ghost_states[track_id]

    # -----------------------------------------------------------------
    # Timeline — sliding window of significant snapshots
    # -----------------------------------------------------------------
    def _record_timeline(self, states: list[ObjectState], timestamp: float) -> None:
        """Capture a short summary of the current frame for the sliding window."""
        entries: list[str] = []
        for obj in states:
            parts_obj: list[str] = []
            # Only record noteworthy info
            if obj.speed >= self._running_speed:
                parts_obj.append(f"FAST ({obj.speed:.0f}px/s)")
            elif obj.speed < 5:
                parts_obj.append("Stationary")
            else:
                parts_obj.append(f"Moving ({obj.speed:.0f}px/s)")
            if obj.erratic:
                parts_obj.append("Erratic")
            if obj.close_to:
                parts_obj.append(f"near [{', '.join(obj.close_to)}]")
            entries.append(f"{obj.class_name}#{obj.track_id}: {', '.join(parts_obj)}")

        self._timeline.append({
            "timestamp": timestamp,
            "summary": "; ".join(entries) if entries else "No objects",
        })

    # -----------------------------------------------------------------
    # Build final scene text for the LLM
    # -----------------------------------------------------------------
    def _build_scene_text(
        self, objects: list[ObjectState]
    ) -> str:
        if not objects:
            return "Scene empty."

        parts: list[str] = []

        # --- Section 1: Events (disappearances, zone intrusions) ---
        if self._event_log:
            parts.append("EVENTS:")
            for evt in self._event_log:
                parts.append(f"  - {evt}")
            parts.append("")
            self._event_log.clear()

        # --- Section 2: Timeline (sliding window of recent states) ---
        if len(self._timeline) > 1:
            parts.append("TIMELINE (recent history):")
            now = self._timeline[-1]["timestamp"]
            # Show all except the very last (that's the "current" state)
            for entry in list(self._timeline)[:-1]:
                age = now - entry["timestamp"]
                parts.append(f"  T-{age:.1f}s: {entry['summary']}")
            parts.append("")

        # --- Section 3: Current object states ---
        parts.append("CURRENT STATE:")
        for obj in objects:
            cname = obj.class_name
            # Speed label
            if obj.speed < 5:
                motion = "Stationary"
            elif obj.speed < self._running_speed:
                motion = f"Moving ({obj.speed:.0f}px/s)"
            else:
                motion = f"FAST/Running ({obj.speed:.0f}px/s)"

            erratic_tag = ", Erratic" if obj.erratic else ""
            zone_tag = f", Zone: {obj.zone}" if obj.zone else ""
            loiter_tag = f", Loitering: {obj.loiter_seconds:.0f}s" if obj.loiter_seconds > 2 else ""
            nearby = f", Nearby: [{', '.join(obj.close_to)}]" if obj.close_to else ""
            weapon_tag = " [WEAPON]" if self._is_weapon(obj) else ""

            parts.append(
                f"  {cname}#{obj.track_id}: {motion}{erratic_tag}"
                f"{zone_tag}{loiter_tag}{nearby}{weapon_tag}"
            )

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
