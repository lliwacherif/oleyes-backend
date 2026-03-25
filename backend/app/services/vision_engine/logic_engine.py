from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from time import time
from typing import Any, Iterable

import numpy as np
from shapely.geometry import Point, Polygon, box

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

        # Interaction thresholds
        self._interaction_px = 20.0  # bbox gap < 20px = "touching/near"

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

    def set_zones(self, zones: dict[str, Iterable[Iterable[float]]]) -> None:
        """Replace active zones at runtime (e.g. when a new job starts)."""
        self._zones = {name: Polygon(points) for name, points in zones.items()}

    def update(self, yolo_results: dict[str, Any] | list[Any]) -> None:
        detections = self._normalize_detections(yolo_results)
        if not detections:
            self._advance_time(yolo_results)
            self._scene_state["summary_text"] = "No detections."
            return

        frame_index, timestamp = self._advance_time(yolo_results)
        states = self._update_object_history(detections, frame_index, timestamp)
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
        self, detections: list[dict[str, Any]], frame_index: int, timestamp: float
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
            speed = float(np.linalg.norm(center - hist["last_center"]) / dt)
            hist["speed"] = speed
            hist["speed_hist"].append(speed)
            hist["last_center"] = center
            hist["last_seen_frame"] = frame_index
            hist["last_seen_time"] = timestamp
            hist["class_id"] = class_id
            hist["class_name"] = class_name
            seen_ids.add(track_id)

            zone = self._find_zone(zone_point)
            prev_zone = hist.get("current_zone")
            hist["current_zone"] = zone
            if zone:
                hist["zone_durations"][zone] += dt
                if zone != prev_zone:
                    self._event_log.append(
                        f"ZONE_INTRUSION: {class_name}#{track_id} entered zone '{zone}'."
                    )

            speed_hist = np.array(hist["speed_hist"], dtype=float)
            erratic = bool(speed_hist.size >= 3 and speed_hist.std() > self._erratic_std)

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
                )
            )

        # --- Interaction detection (N^2 proximity check) ---
        self._detect_interactions(states)

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
        """Populate close_to for each object by checking bbox proximity."""
        n = len(objects)
        if n < 2:
            return
        # Pre-build shapely boxes
        polys = [box(*obj.bbox.tolist()) for obj in objects]
        for i in range(n):
            for j in range(i + 1, n):
                dist = polys[i].distance(polys[j])
                if dist < self._interaction_px:
                    objects[i].close_to.append(f"{objects[j].class_name}#{objects[j].track_id}")
                    objects[j].close_to.append(f"{objects[i].class_name}#{objects[i].track_id}")

    # -----------------------------------------------------------------
    # Disappearance / concealment detection
    # -----------------------------------------------------------------
    def _detect_disappearances(
        self,
        current_states: list[ObjectState],
        seen_ids: set[int],
        frame_index: int,
    ) -> None:
        """Check stale tracks. If a stealable object vanished while near a
        person, log a concealment alert instead of silently deleting it."""
        for track_id, hist in list(self._history.items()):
            if track_id in seen_ids:
                continue
            frames_missing = frame_index - hist["last_seen_frame"]
            if frames_missing <= self._max_missing_frames:
                continue

            # Track is dead — check if it was a stealable object
            last_class_id = hist.get("class_id", -1)
            last_class_name = hist.get("class_name") or _resolve_class_name(last_class_id, None)
            last_close_to = hist.get("last_close_to", [])

            is_stealable = (
                last_class_id in self._stealable_class_ids
                or last_class_name.lower() in self._stealable_class_names
            )

            if is_stealable and last_close_to:
                # Check if any neighbor was a person
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

            del self._history[track_id]

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

        # --- Section 1: Events (disappearances, concealment alerts) ---
        if self._event_log:
            parts.append("EVENTS:")
            for evt in self._event_log:
                parts.append(f"  - {evt}")
            parts.append("")

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
