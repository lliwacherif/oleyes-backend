# OLEYES — Pose Theft Mode (Dual-Pass Architecture)

Complete technical documentation covering the end-to-end flow from the frontend toggle to the LLM risk assessment.

---

## 1. Overview

Pose Theft Mode is an optional operational mode that adds **skeleton keypoint tracking** on top of standard YOLO object detection. When activated, the system can detect **concealment kinematics** — the physical motion of a person's hand reaching for an item, then moving to their pocket or hip area.

**Standard Mode** can only detect theft by noticing an object disappear from the frame while a person was nearby. **Pose Theft Mode** can detect the act of theft *as it happens* by watching the person's wrist interact with the item and then move to their body.

---

## 2. Activation Flow

### Frontend (React)

| Component | File | Role |
|-----------|------|------|
| Settings Context | `src/contexts/SettingsContext.tsx` | Stores `poseTheftMode` boolean, persisted to `localStorage` |
| Settings Page | `src/pages/Settings.tsx` | Toggle button flips `setPoseTheftMode(!poseTheftMode)` |
| API Types | `src/api/types.ts` | All three request interfaces include `pose_theft_mode?: boolean` |
| API Client | `src/api/client.ts` | Maps camelCase `poseTheftMode` to snake_case `pose_theft_mode` in POST body |
| Detector | `src/components/YoloDetector.tsx` | Reads context, passes flag through to the API call |

### The API Call

When the user clicks EXECUTE on a camera with the toggle ON, the frontend sends:

```json
POST /api/v1/vision/detect-rtsp
{
    "rtsp_url": "rtsp://camera.ip:554/stream",
    "camera_id": "abc123",
    "pose_theft_mode": true
}
```

The same field is supported on all three endpoints:
- `POST /api/v1/vision/detect-rtsp`
- `POST /api/v1/vision/detect-rtmp`
- `POST /api/v1/vision/detect-youtube`

---

## 3. Backend — Registration Phase

**File:** `backend/app/api/v1/endpoints/vision.py`

The Pydantic request models (`VisionRtspRequest`, `VisionRtmpRequest`, `VisionYoutubeRequest`) all include:

```
pose_theft_mode: bool = False
```

The endpoint handler passes it through to the service:

```
_service.register_job(..., pose_theft_mode=request.pose_theft_mode)
```

**File:** `backend/app/services/vision_engine/yolo26_service.py` → `register_job()`

The service stores the flag in two places:
1. **Job dict:** `job["pose_theft_mode"] = True` — used by the inference loop and LLM prompt builder
2. **Logic engine:** `self._logic.pose_theft_mode = True` — used by the kinematic heuristics

---

## 4. Backend — Dual-Pass Inference Loop

This is the core architectural change. The system does NOT swap models. It runs **both models on every frame**.

### Model Inventory

| Model | File | Config Key | Detects |
|-------|------|------------|---------|
| Standard | `yolo26n.pt` | `YOLO_MODEL` | All 80 COCO classes (person, bag, phone, laptop, car, etc.) + ByteTrack tracking IDs |
| Pose | `yolo26n-pose.pt` | `YOLO_POSE_MODEL` | Person skeletons only (17 COCO keypoints per person) |

Both models are lazily loaded and kept in memory simultaneously:
- `self._model` / `self._get_model()` — standard model (always loaded)
- `self._pose_model` / `self._get_pose_model()` — pose model (loaded on first pose job)

### Per-Frame Processing

**File:** `yolo26_service.py` → `_run_stream_job()` and `_run_rtmp_job()`

```
For each video frame:

  PASS 1 — Standard Model (always runs)
  ┌─────────────────────────────────────────────┐
  │ std_result = model.track(frame)             │
  │ → Detects ALL objects (persons + items)     │
  │ → Maintains persistent ByteTrack IDs        │
  │ → Returns: boxes, class_ids, conf, track_ids│
  └─────────────────────────────────────────────┘

  PASS 2 — Pose Model (only if pose_theft_mode=True)
  ┌─────────────────────────────────────────────┐
  │ pose_result = pose_model.predict(frame)     │
  │ → Detects person skeletons only             │
  │ → No tracking (stateless prediction)        │
  │ → Returns: person boxes + 17 keypoints each │
  └─────────────────────────────────────────────┘

  MERGE → _process_result(std_result, pose_result)
```

### Why Dual-Pass?

The pose model (`yolo26n-pose.pt`) only detects class 0 (Person). If you use it as the sole model, **all objects disappear** — no bags, phones, or laptops are detected. Theft detection breaks because there are no stealable items to track.

The dual-pass architecture ensures the standard model always provides the full scene (persons + items + tracking), and the pose model only adds skeleton data on top.

---

## 5. Keypoint Merging — Nearest-Center Matching

**File:** `yolo26_service.py` → `_merge_pose_keypoints()`

The two models run independently and produce separate result objects. The merger maps pose keypoints onto the standard model's tracked persons.

### Algorithm

```
For each person box in std_result (class_id == 0, has track_id):
    Compute its center (cx, cy)

    For each person box in pose_result:
        Compute its center (pcx, pcy)
        Calculate Euclidean distance to (cx, cy)

    Find the closest pose box
    If distance < 50 pixels:
        Map that pose box's keypoints → std track_id
        Mark that pose box as used (1:1 matching)

Result: keypoints_map = {track_id: [[x,y,conf], ...17 keypoints...]}
```

### Matching Threshold

`_POSE_MATCH_PX = 50.0` pixels. Both models see the same frame, so person boxes from the standard and pose models should overlap almost exactly. The 50px tolerance handles minor bbox size differences between the two model architectures.

---

## 6. Logic Engine — Kinematic Analysis

**File:** `backend/app/services/vision_engine/logic_engine.py`

### Data Flow

The `keypoints_map` is passed into the logic engine via the update dict:

```
self._logic.update({
    "frame_index": ...,
    "vectors": detections,        ← all objects from standard model
    "frame_height": ...,
    "keypoints_map": keypoints_map ← skeleton data mapped to track IDs
})
```

Inside the engine:
1. `update()` extracts `keypoints_map` and passes it to `_update_object_history()`
2. Each `ObjectState` for a person gets its `keypoints` field populated if found in the map
3. After interaction detection, `_detect_pose_theft_heuristics(states)` runs

### ObjectState Dataclass

```
@dataclass
class ObjectState:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: ndarray         # [x1, y1, x2, y2]
    center: ndarray       # [cx, cy]
    speed: float
    zone: str | None
    loiter_seconds: float
    erratic: bool
    close_to: list[str]
    keypoints: list | None  ← 17 COCO keypoints when pose mode active
```

### COCO Keypoint Indices (17 total)

```
 0  nose
 1  left_eye        2  right_eye
 3  left_ear        4  right_ear
 5  left_shoulder   6  right_shoulder
 7  left_elbow      8  right_elbow
 9  left_wrist     10  right_wrist      ← USED
11  left_hip       12  right_hip        ← USED
13  left_knee      14  right_knee
15  left_ankle     16  right_ankle
```

Each keypoint is `[x, y, confidence]`.

### Heuristic 1: Wrist-to-Item (Browsing / Grabbing)

**Method:** `_detect_pose_theft_heuristics()`

```
For each Person with keypoints:
    Extract wrist positions from indices 9 and 10
    Skip if confidence < 0.3

    For each stealable item in the scene:
        Build a Shapely polygon from the item's bounding box
        Check: distance(wrist_point, item_polygon) < interaction_threshold

    If wrist is inside or touching the item's box:
        → Log: "POSE_KINEMATIC: Person#3 wrist interacted with Handbag#7"
```

**Stealable item classes:** Backpack (24), Handbag (26), Suitcase (28), Laptop (63), Cell Phone (67), Book (73)

### Heuristic 2: Wrist-to-Hip (Concealment / Pocketing)

```
For each Person with keypoints:
    Extract wrist positions from indices 9, 10
    Extract hip positions from indices 11, 12
    Skip if confidence < 0.3

    For each wrist-hip pair on the SAME person:
        distance = euclidean(wrist, hip)
        threshold = 30px × (person_bbox_y2 / frame_height)  ← perspective scaled

    If any wrist is within threshold of any hip:
        → Log: "POSE_KINEMATIC: Person#3 wrist moved to hip/pocket area"
```

The perspective scaling ensures the threshold adapts to camera depth — a person close to the camera has a larger threshold than a person far away.

---

## 7. Scene Text Generation

**File:** `logic_engine.py` → `_build_scene_text()`

The kinematic events are appended to `_event_log`, which is drained into the EVENTS section of the scene text automatically. No special handling is needed.

Example output sent to the LLM:

```
EVENTS:
  - POSE_KINEMATIC: Person#3 wrist interacted with Handbag#7
  - POSE_KINEMATIC: Person#3 wrist moved to hip/pocket area

TIMELINE (recent history):
  T-1.2s: Person#3: Moving (85px/s), near [Handbag#7]
  T-2.4s: Person#3: Stationary, near [Handbag#7]

CURRENT STATE:
  Person#3: FAST/Running (310px/s), Erratic
  Handbag#7: Stationary
```

---

## 8. LLM Prompt Override

**File:** `yolo26_service.py` → `_build_dynamic_prompt()`

When `pose_theft_mode` is True AND `theft_detection` is enabled, the standard theft rule is replaced with a specialized prompt:

### Standard Theft Rule (pose OFF)
> "THEFT: If a Person overlaps an Item and the Item disappears while the Person moves away, that is HIGH risk theft. CRITICAL TEMPORAL ANALYSIS: Compare the two frame batches. If a stealable object appears in the older frame but is missing in the newest frame, check the TIMELINE for Person proximity. If yes, and the person is moving quickly or erratically, this is HIGH risk theft."

### Pose Theft Rule (pose ON)
> "ADVANCED POSE THEFT MODE: Analyze the kinematic events in the text. Look for the Pre-Crime Sequence: 1. Browsing (wrist interacts with item). 2. Concealment (wrist moves to hip/pocket/backpack). 3. Flight (item disappears, person speeds up or leaves). If you see Concealment followed by the item disappearing, flag as HIGH RISK THEFT immediately."

The LLM receives the kinematic events in the scene text and the specialized prompt telling it how to interpret the Browse → Conceal → Flight sequence.

---

## 9. SSE Output

The SSE stream to the frontend is **unchanged**. The `analysis` dict always contains:

```json
{
    "risk_score": 0-100,
    "risk_level": "LOW" | "MEDIUM" | "HIGH",
    "label": "2-5 word title",
    "explanation": "1 sentence",
    "risk_score_raw": 0-100,
    "text": "raw LLM output"
}
```

The `logic.objects` array now includes a `keypoints` field on person objects when pose mode is active, but the frontend SSE parser does not need to read it — it's there for future visualization (skeleton overlay).

---

## 10. Configuration

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `YOLO_MODEL` | `yolo26n.pt` | Standard detection model (80 classes) |
| `YOLO_POSE_MODEL` | `yolo26n-pose.pt` | Pose estimation model (persons + 17 keypoints) |
| `SCALWAY_ANALYSIS_MAX_TOKENS` | `512` | Must be ≥512 to fit chain-of-thought JSON |

### Tunable Thresholds (in code)

| Constant | Value | Location | Purpose |
|----------|-------|----------|---------|
| `_POSE_MATCH_PX` | 50.0 | `yolo26_service.py` | Max distance (px) to match a pose box to a standard box |
| `_KP_MIN_CONF` | 0.3 | `logic_engine.py` | Min keypoint confidence to consider a wrist/hip valid |
| `_WRIST_HIP_PX` | 30.0 | `logic_engine.py` | Base wrist-to-hip proximity (scaled by perspective) |
| `_interaction_px` | 20.0 | `logic_engine.py` | Base bbox gap threshold for wrist-to-item check |
| Pose predict conf | 0.5 | `yolo26_service.py` | Min confidence for pose model person detection |

---

## 11. Pre-Crime Detection Sequence

The complete sequence the system looks for:

```
Phase 1 — BROWSING
  Person's wrist enters the bounding box of a stealable item
  → Event: "POSE_KINEMATIC: Person#3 wrist interacted with Handbag#7"

Phase 2 — CONCEALMENT
  Person's wrist moves to their hip/pocket area
  → Event: "POSE_KINEMATIC: Person#3 wrist moved to hip/pocket area"

Phase 3 — FLIGHT
  The stealable item disappears from detections (ghost state → TTL expires)
  Person speed increases or becomes erratic
  → Event: "ALERT: Handbag#7 DISAPPEARED while close to Person#3"

LLM sees all three events in sequence → HIGH RISK THEFT
```

---

## 12. Files Modified (Summary)

| File | Changes |
|------|---------|
| `backend/app/api/v1/endpoints/vision.py` | `pose_theft_mode` field on 3 request models + 3 `register_job` calls |
| `backend/app/services/vision_engine/yolo26_service.py` | Pose model slot, dual-pass inference, `_merge_pose_keypoints`, `_process_result` merge, LLM prompt override |
| `backend/app/services/vision_engine/logic_engine.py` | `ObjectState.keypoints`, `pose_theft_mode` flag, `_detect_pose_theft_heuristics()` |
