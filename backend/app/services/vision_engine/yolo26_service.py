from __future__ import annotations

import json
import logging
import os
from collections import deque
import queue
from threading import Event, Lock, Thread
from time import time

import cv2
import httpx
from ultralytics import YOLO
from yt_dlp import YoutubeDL

from app.core import config
from app.services.vision_engine.logic_engine import AdvancedLogicEngine

_RTMP_RECONNECT_DELAY = 2.0
_RTMP_MAX_RECONNECTS = 10

logger = logging.getLogger(__name__)

# FFmpeg capture options (global, applied to all cv2.VideoCapture calls):
#   HTTP  — timeout 120s, auto-reconnect (for YouTube streams)
#   RTSP  — stimeout 10s socket timeout (for cameras)
# NOTE: rtsp_transport is NOT forced here so cameras auto-negotiate UDP/TCP.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "timeout;120000000|reconnect;1|reconnect_streamed;1|reconnect_delay_max;30"
    "|stimeout;10000000",
)


class Yolo26Service:
    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, object]] = {}
        self._model: YOLO | None = None
        self._model_lock = Lock()
        self._pose_model: YOLO | None = None
        self._pose_model_lock = Lock()
        self._logic = AdvancedLogicEngine(zones={})
        self._analysis_client = httpx.Client(
            base_url=config.SCALWAY_BASE_URL,
            headers={"Authorization": f"Bearer {config.SCALWAY_API_KEY}"},
            timeout=config.SCALWAY_TIMEOUT,
        )
        self._analysis_queue: queue.Queue = queue.Queue(maxsize=4)
        self._analysis_worker: Thread | None = None
        self._worker_lock = Lock()

    def _get_model(self) -> YOLO:
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is None:
                self._model = YOLO(config.YOLO_MODEL)
        return self._model

    def _get_pose_model(self) -> YOLO:
        if self._pose_model is not None:
            return self._pose_model
        with self._pose_model_lock:
            if self._pose_model is None:
                self._pose_model = YOLO(config.YOLO_POSE_MODEL)
                logger.info("pose_model_loaded model=%s", config.YOLO_POSE_MODEL)
        return self._pose_model

    def stop_all_running(self) -> int:
        """Signal every running/queued job to stop. Returns how many were stopped."""
        count = 0
        for jid, job in list(self._jobs.items()):
            if job.get("status") in ("running", "queued"):
                stop_event = job.get("stop_event")
                if isinstance(stop_event, Event):
                    stop_event.set()
                job["status"] = "stopped"
                job["finished_at"] = time()
                count += 1
                logger.info("auto_stopped job=%s (new job incoming)", jid)
        self._shutdown_analysis_worker()
        return count

    def _reset_tracker(self) -> None:
        """Clear the model's internal tracker state so new jobs start fresh."""
        model = self._model
        if model is None:
            return
        predictor = getattr(model, "predictor", None)
        if predictor is None:
            return
        trackers = getattr(predictor, "trackers", None)
        if trackers:
            for t in trackers:
                if hasattr(t, "reset"):
                    t.reset()
            logger.info("tracker_state_reset")

    def register_job(
        self,
        *,
        job_id: str,
        source_url: str,
        scene_context: str | None = None,
        stream_protocol: str = "RTSP",
        zones: dict[str, list[list[float]]] | None = None,
        zone_instructions: dict[str, str] | None = None,
        security_priorities: dict[str, bool] | None = None,
        pose_theft_mode: bool = False,
    ) -> None:
        self.stop_all_running()
        self._reset_tracker()

        self._logic.reset()
        if zones:
            self._logic.set_zones(zones)
            logger.info("roi_zones_loaded count=%d names=%s", len(zones), list(zones.keys()))
        else:
            self._logic.set_zones({})

        sp = security_priorities or {}
        theft_on = sp.get("theft_detection", True)
        self._logic.theft_detection_enabled = theft_on
        self._logic.pose_theft_mode = pose_theft_mode
        logger.info("security_priorities theft=%s fire=%s fall=%s violence=%s analytics=%s pose_theft=%s",
                     theft_on, sp.get("fire_detection"), sp.get("person_fall_detection"),
                     sp.get("violence_detection"), sp.get("customer_behavior_analytics"),
                     pose_theft_mode)

        self._jobs[job_id] = {
            "status": "queued",
            "source_url": source_url,
            "scene_context": scene_context,
            "stream_protocol": stream_protocol,
            "pose_theft_mode": pose_theft_mode,
            "zones": zones or {},
            "zone_instructions": zone_instructions or {},
            "security_priorities": sp,
            "result": None,
            "error": None,
            "frames": 0,
            "detections": 0,
            "started_at": None,
            "finished_at": None,
            "last_update": None,
            "event_id": 0,
            "last_event": None,
            "batch": [],
            "analysis_buffer": [],
            "analysis": None,
            "zone_alerts": [],
            "raw_zone_events": [],
            "risk_scores": deque(maxlen=5),
            "stop_event": Event(),
        }
        self._ensure_analysis_worker()
        if scene_context:
            logger.info("Job queued id=%s proto=%s scene_context=\"%s\"", job_id, stream_protocol, scene_context)
        else:
            logger.info("Job queued id=%s proto=%s (no scene context)", job_id, stream_protocol)

    def _resolve_source(self, source_url: str) -> str:
        if "youtube.com" in source_url or "youtu.be" in source_url:
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "format": "best[ext=mp4]/best",
            }
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(source_url, download=False)
            stream_url = info.get("url")
            if not stream_url:
                raise ValueError("Unable to resolve YouTube stream URL")
            return stream_url
        return source_url

    def start_job(self, *, job_id: str, source_url: str) -> None:
        job = self._jobs.get(job_id)
        if not job:
            logger.warning("yolo_job_missing id=%s", job_id)
            return
        job["status"] = "running"
        job["started_at"] = time()
        logger.info("yolo_job_started id=%s url=%s", job_id, source_url)

        proto = str(job.get("stream_protocol", "RTSP")).upper()
        if proto == "RTMP":
            self._run_rtmp_job(job_id, job, source_url)
        else:
            self._run_stream_job(job_id, job, source_url)

    # ------------------------------------------------------------------
    # RTSP / YouTube job — uses model.track() which manages its own capture
    # ------------------------------------------------------------------

    def _run_stream_job(self, job_id: str, job: dict, source_url: str) -> None:
        try:
            if self._is_stopped(job):
                job["status"] = "stopped"
                job["finished_at"] = time()
                return
            resolved_source = self._resolve_source(source_url)
            logger.info("yolo_source_resolved id=%s", job_id)
            self._reset_tracker()
            model = self._get_model()
            pose_mode = bool(job.get("pose_theft_mode"))
            pose_model = self._get_pose_model() if pose_mode else None
            results = model.track(
                source=resolved_source,
                stream=True,
                conf=config.YOLO_CONF,
                device=config.YOLO_DEVICE,
                verbose=False,
                max_det=config.YOLO_MAX_DETECTIONS,
                persist=True,
                tracker="bytetrack.yaml",
                vid_stride=config.YOLO_VID_STRIDE,
                stream_buffer=config.YOLO_STREAM_BUFFER,
            )
            for result in results:
                if self._is_stopped(job):
                    job["status"] = "stopped"
                    job["finished_at"] = time()
                    break
                pose_result = None
                if pose_mode and pose_model is not None and self._should_run_pose(result):
                    frame_img = getattr(result, "orig_img", None)
                    if frame_img is not None:
                        pose_results = pose_model.predict(
                            frame_img, conf=0.5,
                            device=config.YOLO_DEVICE, verbose=False,
                        )
                        if pose_results:
                            pose_result = pose_results[0]
                self._process_result(job_id, job, result, pose_result)
            self._finalize_job(job_id, job)
        except Exception as exc:
            job["status"] = "error"
            job["error"] = str(exc)
            job["finished_at"] = time()
            logger.exception("yolo_job_error id=%s error=%s", job_id, exc)

    # ------------------------------------------------------------------
    # RTMP job — manual cv2.VideoCapture loop with reconnection
    # ------------------------------------------------------------------

    def _run_rtmp_job(self, job_id: str, job: dict, source_url: str) -> None:
        import time as _time
        reconnects = 0
        cap: cv2.VideoCapture | None = None

        try:
            while not self._is_stopped(job):
                if cap is None or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    if reconnects > 0:
                        logger.info(
                            "rtmp_reconnecting attempt=%d/%d url=%s",
                            reconnects, _RTMP_MAX_RECONNECTS, source_url,
                        )
                        if reconnects > _RTMP_MAX_RECONNECTS:
                            logger.warning("rtmp_max_reconnects url=%s", source_url)
                            break
                        _time.sleep(_RTMP_RECONNECT_DELAY)

                    saved_opts = os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
                    try:
                        cap = cv2.VideoCapture(source_url, cv2.CAP_FFMPEG)
                    finally:
                        if saved_opts is not None:
                            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = saved_opts
                    if not cap.isOpened():
                        cap.release()
                        cap = None
                        reconnects += 1
                        logger.warning("rtmp_open_failed url=%s", source_url)
                        continue
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    logger.info("rtmp_connected url=%s", source_url)
                    self._reset_tracker()

                ret, frame = cap.read()
                if not ret:
                    reconnects += 1
                    if cap is not None:
                        cap.release()
                    cap = None
                    continue

                reconnects = 0
                model = self._get_model()
                results = model.track(
                    source=frame,
                    conf=config.YOLO_CONF,
                    device=config.YOLO_DEVICE,
                    verbose=False,
                    max_det=config.YOLO_MAX_DETECTIONS,
                    persist=True,
                    tracker="bytetrack.yaml",
                )
                if results:
                    pose_result = None
                    if job.get("pose_theft_mode") and self._should_run_pose(results[0]):
                        pose_results = self._get_pose_model().predict(
                            frame, conf=0.5,
                            device=config.YOLO_DEVICE, verbose=False,
                        )
                        if pose_results:
                            pose_result = pose_results[0]
                    self._process_result(job_id, job, results[0], pose_result)

            self._finalize_job(job_id, job)
        except Exception as exc:
            job["status"] = "error"
            job["error"] = str(exc)
            job["finished_at"] = time()
            logger.exception("rtmp_job_error id=%s error=%s", job_id, exc)
        finally:
            if cap is not None:
                cap.release()

    # ------------------------------------------------------------------
    # Shared helpers for frame processing
    # ------------------------------------------------------------------

    _STEALABLE_CLS = {24, 26, 28, 63, 67, 73}
    _POSE_GATE_PX = 150.0

    @staticmethod
    def _should_run_pose(std_result) -> bool:
        """Return True only if the frame has a person within proximity of a stealable item."""
        import numpy as _np
        boxes = getattr(std_result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return False
        person_centers = []
        item_centers = []
        for b in boxes:
            cls_id = int(b.cls.item()) if b.cls is not None else -1
            xyxy = b.xyxy.tolist()[0]
            cx = (xyxy[0] + xyxy[2]) / 2.0
            cy = (xyxy[1] + xyxy[3]) / 2.0
            if cls_id == 0:
                person_centers.append((cx, cy))
            elif cls_id in Yolo26Service._STEALABLE_CLS:
                item_centers.append((cx, cy))
        if not person_centers or not item_centers:
            return False
        for pcx, pcy in person_centers:
            for icx, icy in item_centers:
                if _np.hypot(pcx - icx, pcy - icy) < Yolo26Service._POSE_GATE_PX:
                    return True
        return False

    _POSE_IOU_THRESH = 0.60

    @staticmethod
    def _iou(a: list[float], b: list[float]) -> float:
        """Intersection over Union of two [x1, y1, x2, y2] boxes."""
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _merge_pose_keypoints(std_boxes, pose_result) -> dict[int, list]:
        """Map keypoints from a separate pose prediction to the tracked
        person boxes from the standard model using IoU-based matching."""
        if std_boxes is None or std_boxes.xyxy is None:
            return {}

        pose_boxes = getattr(pose_result, "boxes", None)
        pose_kpts = getattr(pose_result, "keypoints", None)
        if pose_boxes is None or pose_kpts is None or pose_kpts.data is None:
            return {}

        pose_bboxes = []
        pose_kpts_data = pose_kpts.data
        for i, pb in enumerate(pose_boxes):
            if i >= len(pose_kpts_data):
                break
            pose_bboxes.append((pb.xyxy.tolist()[0], i))

        if not pose_bboxes:
            return {}

        keypoints_map: dict[int, list] = {}
        used_pose: set[int] = set()

        for sb in std_boxes:
            cls_id = int(sb.cls.item()) if sb.cls is not None else -1
            if cls_id != 0:
                continue
            track_id = int(sb.id.item()) if getattr(sb, "id", None) is not None else -1
            if track_id == -1:
                continue
            sxyxy = sb.xyxy.tolist()[0]

            best_iou = 0.0
            best_idx = -1
            for pxyxy, pidx in pose_bboxes:
                if pidx in used_pose:
                    continue
                score = Yolo26Service._iou(sxyxy, pxyxy)
                if score > best_iou:
                    best_iou = score
                    best_idx = pidx

            if best_idx >= 0 and best_iou >= Yolo26Service._POSE_IOU_THRESH:
                keypoints_map[track_id] = pose_kpts_data[best_idx].tolist()
                used_pose.add(best_idx)

        return keypoints_map

    def _process_result(self, job_id: str, job: dict, result, pose_result=None) -> None:
        """Extract detections from the standard YOLO result and, when in
        pose_theft_mode, merge skeleton keypoints from a separate pose
        prediction by nearest-center matching."""
        job["frames"] = int(job.get("frames", 0)) + 1
        frame_index = int(job["frames"])

        boxes = getattr(result, "boxes", None)
        detections = []
        if boxes is not None and boxes.xyxy is not None:
            for box in boxes:
                cls_id = int(box.cls.item()) if box.cls is not None else -1
                conf = float(box.conf.item()) if box.conf is not None else 0.0
                xyxy = [float(x) for x in box.xyxy.tolist()[0]]
                track_id = int(box.id.item()) if getattr(box, "id", None) is not None else -1
                detections.append([cls_id, conf, *xyxy, track_id])
            job["detections"] = int(job.get("detections", 0)) + len(detections)

        keypoints_map: dict[int, list] = {}
        if pose_result is not None:
            keypoints_map = self._merge_pose_keypoints(boxes, pose_result)

        event = {
            "frame_index": frame_index,
            "timestamp": time(),
            "vectors": detections,
        }
        orig_shape = getattr(result, "orig_shape", (720, 1280))
        self._logic.update(
            {
                "frame_index": frame_index,
                "timestamp": time(),
                "vectors": detections,
                "frame_height": float(orig_shape[0]),
                "frame_width": float(orig_shape[1]),
                "keypoints_map": keypoints_map,
            }
        )
        logic_summary = self._logic.generate_scene_summary()
        job["logic"] = logic_summary
        event["scene_text"] = logic_summary.get("scene_text", "")

        scene_text = logic_summary.get("scene_text", "")
        if "ZONE_INTRUSION" in scene_text:
            raw_buf = job.get("raw_zone_events")
            if isinstance(raw_buf, list):
                for line in scene_text.split("\n"):
                    if "ZONE_INTRUSION" in line:
                        raw_buf.append({
                            "message": line.strip().lstrip("- "),
                            "frame": frame_index,
                            "timestamp": time(),
                        })
                        if len(raw_buf) > 30:
                            raw_buf.pop(0)
                self._judge_zone_events(job)

        batch = job.get("batch")
        if isinstance(batch, list):
            batch.append(event)
        stream_every = max(int(config.YOLO_STREAM_EVERY), 1)
        if frame_index % stream_every == 0:
            job["event_id"] = int(job.get("event_id", 0)) + 1
            job["last_event"] = {"batch": list(batch or [])}
            self._maybe_analyze(job)
            if isinstance(batch, list):
                batch.clear()
            logger.info(
                "frame=%s  detections=%d  batch_sent",
                frame_index,
                len(detections),
            )
        log_every = max(int(config.YOLO_LOG_EVERY), 1)
        if frame_index % log_every == 0:
            logger.debug("yolo_frame=%s vectors=%s", frame_index, detections)
        job["last_update"] = time()

    def _finalize_job(self, job_id: str, job: dict) -> None:
        """Mark a job as done and flush any remaining batch."""
        if job["status"] not in ("stopped", "error"):
            job["status"] = "done"
        job["finished_at"] = time()
        job["result"] = {
            "frames": job["frames"],
            "detections": job["detections"],
        }
        batch = job.get("batch")
        if isinstance(batch, list) and batch:
            job["event_id"] = int(job.get("event_id", 0)) + 1
            job["last_event"] = {"batch": list(batch)}
            self._maybe_analyze(job)
            batch.clear()
        logger.info(
            "yolo_job_done id=%s frames=%s detections=%s",
            job_id, job["frames"], job["detections"],
        )

    def get_status(self, job_id: str) -> str:
        job = self._jobs.get(job_id)
        if not job:
            return "unknown"
        return str(job["status"])

    def stop_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False
        job["status"] = "stopped"
        job["finished_at"] = time()
        stop_event = job.get("stop_event")
        if isinstance(stop_event, Event):
            stop_event.set()
        return True

    def get_result(self, job_id: str) -> dict[str, object] | None:
        job = self._jobs.get(job_id)
        if not job:
            return None
        return {
            "status": job["status"],
            "result": job["result"],
            "error": job["error"],
            "source_url": job["source_url"],
        }

    def get_snapshot(self, job_id: str) -> dict[str, object] | None:
        job = self._jobs.get(job_id)
        if not job:
            return None
        status = job["status"]
        frames = int(job["frames"])
        detail = "OK"
        if status == "running" and frames == 0:
            detail = "Waiting for video stream..."
        return {
            "status": status,
            "frames": frames,
            "detections": job["detections"],
            "started_at": job["started_at"],
            "finished_at": job["finished_at"],
            "last_update": job["last_update"],
            "error": job["error"],
            "event_id": job["event_id"],
            "last_event": job["last_event"],
            "detail": detail,
            "logic": job.get("logic"),
            "analysis": job.get("analysis"),
            "zone_alerts": list(job.get("zone_alerts", [])),
        }
        alerts = job.get("zone_alerts")
        if isinstance(alerts, list) and alerts:
            alerts.clear()

    def _is_stopped(self, job: dict[str, object]) -> bool:
        stop_event = job.get("stop_event")
        return bool(isinstance(stop_event, Event) and stop_event.is_set())

    _CALM_SPEED = 10.0  # px/s
    _SENTINEL = object()

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        """Pull a JSON object out of LLM output that may be wrapped in
        markdown, thinking tags, or surrounding prose.

        Discards the ``chain_of_thought`` key (used only to force the LLM
        to reason before scoring) so the client payload stays unchanged.
        """
        import re

        def _clean(obj: dict) -> dict:
            obj.pop("chain_of_thought", None)
            return obj

        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return _clean(obj)
        except (json.JSONDecodeError, TypeError):
            pass

        fence = re.search(r"```(?:json)?\s*(\{.+?\})\s*```", text, re.DOTALL)
        if fence:
            try:
                obj = json.loads(fence.group(1))
                if isinstance(obj, dict):
                    return _clean(obj)
            except (json.JSONDecodeError, TypeError):
                pass

        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    candidate = text[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict) and ("risk" in str(obj) or obj.get("label")):
                            return _clean(obj)
                    except (json.JSONDecodeError, TypeError):
                        pass
                    start = -1

        for m in reversed(list(re.finditer(r'\{[^{}]*\}', text))):
            try:
                obj = json.loads(m.group())
                if isinstance(obj, dict) and ("risk" in str(obj) or obj.get("label")):
                    return _clean(obj)
            except (json.JSONDecodeError, TypeError):
                continue
        return None

    def _scene_is_calm(self, job: dict[str, object]) -> bool:
        logic = job.get("logic")
        if not isinstance(logic, dict):
            return True
        scene_text = logic.get("scene_text", "")
        if "ALERT" in scene_text or "DISAPPEARED" in scene_text:
            return False
        objects = logic.get("objects", [])
        if not objects:
            return True
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            if obj.get("speed", 0) >= self._CALM_SPEED:
                return False
            if obj.get("erratic", False):
                return False
        return True

    _METADATA_PATTERNS = [
        r"(?im)^.*\bbusiness[_ ]?name\b.*$",
        r"(?im)^.*\bnumber[_ ]?of[_ ]?locations?\b.*$",
        r"(?im)^.*\bbusiness[_ ]?size\b.*$",
        r"(?im)^.*\bcamera[_ ]?type\b.*$",
        r"(?im)^.*\bestimated[_ ]?cameras?\b.*$",
    ]

    @classmethod
    def _sanitize_scene_context(cls, text: str) -> str:
        """Strip corporate metadata lines from a scene_context string."""
        import re
        for pattern in cls._METADATA_PATTERNS:
            text = re.sub(pattern, "", text)
        return "\n".join(line for line in text.splitlines() if line.strip())

    _OBJECT_KEYWORDS: dict[str, list[str]] = {
        "person": ["person", "people", "someone", "anybody", "human", "man", "woman", "child", "enter", "entered"],
        "car": ["car", "vehicle", "automobile"],
        "truck": ["truck"],
        "bicycle": ["bicycle", "bike", "cyclist"],
        "motorcycle": ["motorcycle", "motorbike"],
        "dog": ["dog"],
        "cat": ["cat"],
        "phone": ["phone", "cell phone", "cellphone", "mobile"],
        "laptop": ["laptop", "computer"],
        "backpack": ["backpack", "bag"],
        "handbag": ["handbag", "purse"],
        "suitcase": ["suitcase", "luggage"],
    }

    def _judge_zone_events(self, job: dict) -> None:
        """Filter raw zone events against user instructions via keyword matching.

        Rules:
        - No instructions → pass ALL events (every zone entry is an alert)
        - Has instructions → match the object type in the event against keywords
          extracted from the instruction text
        - "person" keywords also match generic instructions like "alert when entered"
        """
        raw_events = job.get("raw_zone_events")
        if not isinstance(raw_events, list) or not raw_events:
            return

        events_snapshot = list(raw_events)
        raw_events.clear()

        zone_instructions = job.get("zone_instructions") or {}
        has_instructions = isinstance(zone_instructions, dict) and any(
            v.strip() for v in zone_instructions.values()
        )

        alerts = job.get("zone_alerts")
        if not isinstance(alerts, list):
            alerts = []
            job["zone_alerts"] = alerts

        accepted = 0
        for ev in events_snapshot:
            msg = ev.get("message", "")

            if not has_instructions:
                self._append_zone_alert(alerts, ev)
                accepted += 1
                continue

            import re as _re
            zm = _re.search(r"zone '([^']+)'", msg)
            zone_name = zm.group(1) if zm else ""

            instruction = (zone_instructions.get(zone_name, "") or "").lower()
            if not instruction:
                all_instructions = " ".join(v for v in zone_instructions.values() if v)
                instruction = all_instructions.lower()

            if not instruction:
                self._append_zone_alert(alerts, ev)
                accepted += 1
                continue

            obj_in_event = msg.split(":")[0].rsplit(" ", 1)[-1].lower() if ":" in msg else ""
            obj_in_event = obj_in_event.replace("zone_intrusion", "").strip()
            name_part = msg.split("#")[0].strip().lower()
            for part in ["zone_intrusion:", "zone_intrusion"]:
                name_part = name_part.replace(part, "").strip()

            matched = False
            for obj_type, keywords in self._OBJECT_KEYWORDS.items():
                instruction_mentions = any(kw in instruction for kw in keywords)
                event_has_object = (
                    obj_type in name_part
                    or any(kw in name_part for kw in keywords)
                )
                if instruction_mentions and event_has_object:
                    matched = True
                    break

            if not matched:
                generic_triggers = ["any", "all", "everything", "anything"]
                if any(t in instruction for t in generic_triggers):
                    matched = True

            if matched:
                self._append_zone_alert(alerts, ev)
                accepted += 1

        if accepted:
            logger.info("zone_filter passed %d/%d events", accepted, len(events_snapshot))

    @staticmethod
    def _append_zone_alert(alerts: list, ev: dict) -> None:
        alerts.append({
            "type": "ZONE_INTRUSION",
            "message": ev.get("message", ""),
            "frame": ev.get("frame", 0),
            "timestamp": ev.get("timestamp", 0),
        })
        if len(alerts) > 50:
            del alerts[:-50]

    def _build_dynamic_prompt(self, job: dict) -> str:
        """Build the LLM system prompt based on the user's security priorities."""
        sp = job.get("security_priorities") or {}

        base = (
            "You are an AI CCTV analyst. Analyze the scene data and respond ONLY with valid JSON.\n"
            "You MUST think step-by-step INSIDE the JSON before scoring. Use this exact schema:\n"
            '{"chain_of_thought": "Briefly summarize object movements and disappearances between the frames", '
            '"risk_score": 0-100, "risk_level": "LOW"|"MEDIUM"|"HIGH", '
            '"label": "2-5 word title", "explanation": "1 short sentence concluding the risk"}\n\n'
        )

        if job.get("pose_theft_mode"):
            base += (
                "RULES:\n"
                "  1) ADVANCED POSE THEFT MODE: The system tracks deterministic theft stages "
                "(BROWSING -> CONCEALING -> FLIGHT). If the text reports "
                "'STATE_CHANGE: Person#X entered FLIGHT stage', this is absolute "
                "mathematical proof of theft. Flag as HIGH RISK THEFT immediately. "
                "Provide a short explanation of the sequence.\n"
                "  2) You are in THEFT-ONLY mode. Ignore violence, falls, fire, "
                "and all other non-theft events. If no theft stages are reported, "
                "respond with LOW risk and a neutral label.\n\n"
                "IMPORTANT: ONLY analyze POSE_KINEMATIC and STATE_CHANGE events. "
                "Normal person movement is LOW risk. Only FLIGHT stage = HIGH risk."
            )
            zone_instructions = job.get("zone_instructions")
            if zone_instructions and isinstance(zone_instructions, dict):
                zone_rules = [
                    f"- Zone \"{n}\": {instr}"
                    for n, instr in zone_instructions.items() if instr
                ]
                if zone_rules:
                    base += (
                        "\n\nROI ZONE RULES:\n"
                        + "\n".join(zone_rules)
                    )
            return base

        rules = []
        if sp.get("theft_detection", False):
            rules.append(
                "THEFT: If a Person overlaps an Item and the Item disappears "
                "while the Person moves away, that is HIGH risk theft. "
                "CRITICAL TEMPORAL ANALYSIS: You are receiving two recent frame batches. "
                "Compare them. If a stealable object (bag, phone, laptop, etc.) appears "
                "in the older frame but is missing in the newest frame, check the TIMELINE "
                "to see if a Person was 'Nearby' or 'close_to' it right before it vanished. "
                "If yes, and the person is now moving quickly or erratically, this is a "
                "HIGH risk theft."
            )
        else:
            rules.append(
                "THEFT: Theft detection is DISABLED. Do NOT flag theft, "
                "concealment, or disappearing objects. Ignore DISAPPEARED alerts."
            )

        if sp.get("violence_detection", False):
            rules.append(
                "VIOLENCE: If two Persons are Nearby and one has high speed, "
                "erratic movement, or fighting indicators = HIGH risk."
            )
        else:
            rules.append(
                "VIOLENCE: Violence detection is DISABLED. Do NOT flag fighting "
                "or aggression."
            )

        if sp.get("person_fall_detection", False):
            rules.append(
                "FALL: If a Person's bounding box suddenly changes from tall/narrow "
                "to wide/short (aspect ratio shift), or speed drops to 0 after fast "
                "movement, flag as possible fall = HIGH risk."
            )

        if sp.get("fire_detection", False):
            rules.append(
                "FIRE: If fire, smoke, or flames are detected in the scene = HIGH risk."
            )

        if sp.get("customer_behavior_analytics", False):
            rules.append(
                "ANALYTICS: Note customer flow patterns, dwell times, and crowding. "
                "Normal customer movement is LOW risk."
            )

        if not any(sp.get(k, False) for k in sp):
            rules.append(
                "GENERAL: Monitor for unusual activity. "
                "Normal movement of people and objects is LOW risk."
            )

        base += "RULES:\n" + "\n".join(f"  {i+1}) {r}" for i, r in enumerate(rules))

        base += (
            "\n\nIMPORTANT: Only flag events that match the ENABLED rules above. "
            "If a rule is DISABLED, you MUST give it LOW risk and a neutral label. "
            "A person simply standing near objects is NORMAL, not theft. "
            "Objects leaving the frame or becoming occluded is NORMAL, not theft."
        )

        scene_context = job.get("scene_context")
        if scene_context:
            clean = self._sanitize_scene_context(str(scene_context))
            if clean:
                base += f"\n\nSCENE CONTEXT:\n{clean}"

        zone_instructions = job.get("zone_instructions")
        if zone_instructions and isinstance(zone_instructions, dict):
            zone_rules = [
                f"- Zone \"{n}\": {instr}"
                for n, instr in zone_instructions.items() if instr
            ]
            if zone_rules:
                base += (
                    "\n\nROI ZONE RULES:\n"
                    + "\n".join(zone_rules)
                )

        return base

    # ------------------------------------------------------------------
    # LLM analysis producer-consumer
    # ------------------------------------------------------------------

    def _ensure_analysis_worker(self) -> None:
        """Start the LLM consumer thread if it is not already running."""
        with self._worker_lock:
            if self._analysis_worker is None or not self._analysis_worker.is_alive():
                self._analysis_worker = Thread(
                    target=self._analysis_consumer,
                    daemon=True,
                    name="llm-consumer",
                )
                self._analysis_worker.start()
                logger.info("analysis_worker_started")

    def _shutdown_analysis_worker(self) -> None:
        """Drain the queue and join the consumer thread."""
        while not self._analysis_queue.empty():
            try:
                self._analysis_queue.get_nowait()
            except queue.Empty:
                break
        with self._worker_lock:
            if self._analysis_worker is not None and self._analysis_worker.is_alive():
                self._analysis_queue.put(self._SENTINEL)
                self._analysis_worker.join(timeout=config.SCALWAY_TIMEOUT + 2)
                logger.info("analysis_worker_joined")
            self._analysis_worker = None

    def _analysis_consumer(self) -> None:
        """Background thread: pull analysis tasks from the queue and call Scaleway."""
        while True:
            item = self._analysis_queue.get()
            if item is self._SENTINEL:
                break
            job, combined, system_prompt = item
            try:
                self._run_llm_analysis(job, combined, system_prompt)
            except Exception as exc:
                logger.warning("analysis_consumer unhandled: %s", exc)

    def _run_llm_analysis(self, job: dict, combined: str, system_prompt: str) -> None:
        """Execute the synchronous Scaleway LLM call (runs in consumer thread)."""
        fallback = {
            "risk_score": 30, "risk_score_raw": 30,
            "risk_level": "MEDIUM", "label": "Activity detected",
            "explanation": "Movement or event detected, AI assessment pending.",
            "text": "",
        }
        try:
            payload = {
                "model": config.SCALWAY_ANALYSIS_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": combined},
                ],
                "max_tokens": config.SCALWAY_ANALYSIS_MAX_TOKENS,
                "temperature": config.SCALWAY_ANALYSIS_TEMPERATURE,
                "top_p": config.SCALWAY_ANALYSIS_TOP_P,
                "presence_penalty": config.SCALWAY_ANALYSIS_PRESENCE_PENALTY,
                "stream": False,
            }
            response = self._analysis_client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            raw_content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content")
            ) or ""

            if not raw_content:
                logger.warning("LLM empty content — using fallback")
                job["analysis"] = fallback
                return

            analysis = self._extract_json(raw_content)
            if not analysis or not analysis.get("label"):
                logger.warning("LLM no JSON found — raw: %s", raw_content[:200])
                job["analysis"] = fallback
                return

            raw_score = int(analysis.get("risk_score", 0))
            risk_scores = job.get("risk_scores")
            if not isinstance(risk_scores, deque):
                risk_scores = deque(maxlen=5)
                job["risk_scores"] = risk_scores
            risk_scores.append(raw_score)

            smoothed_score = int(sum(risk_scores) / len(risk_scores))
            consecutive_high = sum(1 for s in risk_scores if s >= 70)
            if consecutive_high >= 3:
                smoothed_level = "HIGH"
            elif smoothed_score >= 50:
                smoothed_level = "MEDIUM"
            else:
                smoothed_level = "LOW"

            analysis["risk_score_raw"] = raw_score
            analysis["risk_score"] = smoothed_score
            analysis["risk_level"] = smoothed_level
            analysis["text"] = raw_content

            job["analysis"] = analysis
            logger.info(
                "🤖 Analysis: %s | risk=%d (raw=%d) %s",
                analysis.get("label", "?"),
                smoothed_score,
                raw_score,
                smoothed_level,
            )
        except Exception as exc:
            logger.warning("❌ Analysis failed: %s", exc)
            job["analysis"] = fallback

    # ------------------------------------------------------------------
    # Producer: prepare + enqueue (called from detection loop)
    # ------------------------------------------------------------------

    def _maybe_analyze(self, job: dict[str, object]) -> None:
        """Prepare analysis payload and enqueue for the background LLM consumer."""
        batch = job.get("last_event", {})
        if not isinstance(batch, dict):
            return
        frames = batch.get("batch")
        if not isinstance(frames, list) or not frames:
            return
        buffer = job.get("analysis_buffer")
        if not isinstance(buffer, list):
            buffer = []
            job["analysis_buffer"] = buffer
        texts = []
        for i, frame in enumerate(frames):
            if not isinstance(frame, dict):
                continue
            scene_text = frame.get("scene_text")
            if not scene_text:
                continue

            frame_idx = frame.get("frame_index", "?")

            if i < len(frames) - 1:
                if "CURRENT STATE:" in scene_text:
                    _, _, parsed_state = scene_text.partition("CURRENT STATE:")
                    texts.append(f"Frame {frame_idx} CURRENT STATE:\n{parsed_state.strip()}")
                else:
                    texts.append(f"Frame {frame_idx} CURRENT STATE:\n{scene_text.strip()}")
            else:
                texts.append(f"Frame {frame_idx} FULL SUMMARY:\n{scene_text.strip()}")

        if texts:
            buffer.append("\n\n".join(texts))

        if self._scene_is_calm(job):
            job["analysis"] = {
                "risk_score": 0, "risk_score_raw": 0,
                "risk_level": "LOW", "label": "Normal activity",
                "explanation": "Scene is calm, no significant movement detected.",
                "text": "",
            }
            return

        if len(buffer) < 2:
            return

        combined = "\n\n".join(buffer[-2:])
        system_prompt = self._build_dynamic_prompt(job)
        buffer.clear()

        try:
            self._analysis_queue.put_nowait((job, combined, system_prompt))
        except queue.Full:
            try:
                self._analysis_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._analysis_queue.put_nowait((job, combined, system_prompt))
            except queue.Full:
                pass
