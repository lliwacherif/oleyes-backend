from __future__ import annotations

import base64
import json
import logging
import os
from collections import deque
import queue
from threading import Event, Lock, Thread
from time import time

import cv2
import httpx
import numpy as np
from ultralytics import YOLO
from yt_dlp import YoutubeDL

from app.core import config
from app.services.vision_engine.logic_engine import AdvancedLogicEngine
from app.services.vlm_engine.pixtral_client import PixtralClient

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

        # Supreme OLEYES — Pixtral VLM pipeline
        self._vlm_client: PixtralClient | None = None
        self._vlm_client_lock = Lock()
        self._vlm_queue: queue.Queue = queue.Queue(maxsize=2)
        self._vlm_worker: Thread | None = None
        self._vlm_worker_lock = Lock()

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
        self._shutdown_vlm_worker()
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
        supreme_mode: bool = False,
    ) -> None:
        self.stop_all_running()
        self._reset_tracker()

        # Supreme mode takes priority over pose theft mode
        if supreme_mode:
            pose_theft_mode = False

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
        logger.info("security_priorities theft=%s fire=%s fall=%s violence=%s analytics=%s pose_theft=%s supreme=%s",
                     theft_on, sp.get("fire_detection"), sp.get("person_fall_detection"),
                     sp.get("violence_detection"), sp.get("customer_behavior_analytics"),
                     pose_theft_mode, supreme_mode)

        self._jobs[job_id] = {
            "status": "queued",
            "source_url": source_url,
            "scene_context": scene_context,
            "stream_protocol": stream_protocol,
            "pose_theft_mode": pose_theft_mode,
            "supreme_mode": supreme_mode,
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
            # Supreme OLEYES — per-person tracking buffers
            # {track_id: {"frames": [b64], "last_capture": float, "last_call": float, "items": [str]}}
            "supreme_suspects": {},
            "supreme_vlm_pending": False,
        }
        if supreme_mode:
            self._ensure_vlm_worker()
        else:
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
                if pose_mode and pose_model is not None and self._has_person(result):
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

    _RTMP_FFMPEG_OPTS = "fflags;nobuffer|flags;low_delay|analyzeduration;0|probesize;32"

    def _open_rtmp_capture(self, url: str) -> cv2.VideoCapture:
        """Open an RTMP stream with zero-latency FFmpeg options."""
        saved_opts = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = self._RTMP_FFMPEG_OPTS
        try:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        finally:
            if saved_opts is not None:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = saved_opts
            elif "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ:
                del os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
        return cap

    def _run_rtmp_job(self, job_id: str, job: dict, source_url: str) -> None:
        import time as _time
        reconnects = 0
        cap: cv2.VideoCapture | None = None
        frame_idx = 0
        stride = max(int(config.YOLO_VID_STRIDE), 1)

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

                    cap = self._open_rtmp_capture(source_url)
                    if not cap.isOpened():
                        cap.release()
                        cap = None
                        reconnects += 1
                        logger.warning("rtmp_open_failed url=%s", source_url)
                        continue
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    logger.info(
                        "rtmp_connected url=%s ffmpeg_opts=%s stride=%d",
                        source_url, self._RTMP_FFMPEG_OPTS, stride,
                    )
                    self._reset_tracker()

                ret, frame = cap.read()
                if not ret:
                    reconnects += 1
                    if cap is not None:
                        cap.release()
                    cap = None
                    continue

                reconnects = 0
                frame_idx += 1
                if stride > 1 and frame_idx % stride != 0:
                    continue

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
                    if job.get("pose_theft_mode") and self._has_person(results[0]):
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
    def _dist_rects(a: list[float], b: list[float]) -> float:
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])
        if ix1 < ix2 and iy1 < iy2:
            return 0.0
        import numpy as _np
        return float(_np.hypot(max(0.0, ix1 - ix2), max(0.0, iy1 - iy2)))

    @staticmethod
    def _has_person(result) -> bool:
        """Return True if the frame has at least one detected person."""
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return False
        for b in boxes:
            if int(b.cls.item()) if b.cls is not None else -1 == 0:
                return True
        return False

    @staticmethod
    def _should_run_pose(std_result) -> bool:
        """Return True only if the frame has a person within proximity of a stealable item."""
        boxes = getattr(std_result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return False
        person_rects = []
        item_rects = []
        for b in boxes:
            cls_id = int(b.cls.item()) if b.cls is not None else -1
            xyxy = b.xyxy.tolist()[0]
            if cls_id == 0:
                person_rects.append(xyxy)
            elif cls_id in Yolo26Service._STEALABLE_CLS:
                item_rects.append(xyxy)
        if not person_rects or not item_rects:
            return False
        for pr in person_rects:
            for ir in item_rects:
                if Yolo26Service._dist_rects(pr, ir) < Yolo26Service._POSE_GATE_PX:
                    return True
        return False

    _POSE_IOU_THRESH = 0.30
    _POSE_FALLBACK_PX = 50.0

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
        person boxes from the standard model using IoU and Center-Fallback matching."""
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
            scx = (sxyxy[0] + sxyxy[2]) / 2.0
            scy = (sxyxy[1] + sxyxy[3]) / 2.0

            best_iou = 0.0
            best_idx = -1
            best_dist = float('inf')
            best_dist_idx = -1

            for pxyxy, pidx in pose_bboxes:
                if pidx in used_pose:
                    continue
                score = Yolo26Service._iou(sxyxy, pxyxy)
                if score > best_iou:
                    best_iou = score
                    best_idx = pidx

                pcx = (pxyxy[0] + pxyxy[2]) / 2.0
                pcy = (pxyxy[1] + pxyxy[3]) / 2.0
                dist = ((scx - pcx) ** 2 + (scy - pcy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_dist_idx = pidx

            if best_idx >= 0 and best_iou >= Yolo26Service._POSE_IOU_THRESH:
                keypoints_map[track_id] = pose_kpts_data[best_idx].tolist()
                used_pose.add(best_idx)
            elif best_dist_idx >= 0 and best_dist < Yolo26Service._POSE_FALLBACK_PX:
                keypoints_map[track_id] = pose_kpts_data[best_dist_idx].tolist()
                used_pose.add(best_dist_idx)

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

        job["event_id"] = int(job.get("event_id", 0)) + 1
        job["last_event"] = {"batch": [event]}

        if job.get("supreme_mode"):
            # Supreme OLEYES: skip LLM text pipeline, run VLM frame buffer
            self._supreme_buffer_step(job, result, detections)
        else:
            batch = job.get("batch")
            if isinstance(batch, list):
                batch.append(event)
            stream_every = max(int(config.YOLO_STREAM_EVERY), 1)
            if frame_index % stream_every == 0:
                self._maybe_analyze(job, list(batch or []))
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
            self._maybe_analyze(job, list(batch))
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
        if "STATE_CHANGE" in scene_text or "POSE_KINEMATIC" in scene_text:
            return False
        objects = logic.get("objects", [])
        if not objects:
            return True
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            if obj.get("theft_stage", "NONE") not in ("NONE", ""):
                return False
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
            "You are an AI CCTV analyst. You receive pre-interpreted scene data:\n"
            "- SCENE: Object inventory (counts, stationary/displaced status)\n"
            "- EVENTS: Critical alerts — zone intrusions, disappearances, pose kinematics\n"
            "- BEHAVIOR: What each object is doing — speed changes, direction, area.\n"
            "  'DISPLACED' = an item is moving (items can't self-propel, a human moved it)\n"
            "- PROXIMITY: Who is NEAR whom\n"
            "- ALERTS: Pre-flagged suspicious patterns requiring your assessment\n\n"
            "Respond ONLY with valid JSON. Think step-by-step INSIDE the JSON:\n"
            '{"chain_of_thought": "Analyze behavior changes, displacement, and alerts", '
            '"risk_score": 0-100, "risk_level": "LOW"|"MEDIUM"|"HIGH", '
            '"label": "2-5 word title", "explanation": "1 short sentence concluding the risk"}\n\n'
        )

        if job.get("pose_theft_mode"):
            base += (
                "RULES:\n"
                "  1) ADVANCED POSE THEFT MODE: The system tracks deterministic theft stages "
                "(BROWSING -> CONCEALING -> FLIGHT) via skeleton keypoint analysis.\n"
                "  2) STAGE RISK LEVELS:\n"
                "     - BROWSING (wrist near item): LOW risk — informational, person interacting with merchandise.\n"
                "     - CONCEALING (wrist moved from item to hip/torso): MEDIUM risk — possible concealment, "
                "flag as 'Suspicious concealment' with risk_score 50-65.\n"
                "     - FLIGHT (moving after concealment OR item disappeared): HIGH risk — "
                "this is mathematical proof of theft. Flag as HIGH RISK THEFT immediately "
                "with risk_score 85-100.\n"
                "  3) You are in THEFT-ONLY mode. Ignore violence, falls, fire, "
                "and all other non-theft events. If no theft stages are reported, "
                "respond with LOW risk and a neutral label.\n\n"
                "IMPORTANT: Analyze POSE_KINEMATIC and STATE_CHANGE events. "
                "Normal person movement without stage transitions is LOW risk."
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
                "THEFT: Focus on ALERTS for displaced items and BEHAVIOR for person movements. "
                "If an item is DISPLACED and a person who was NEAR it is now moving away, "
                "this is HIGH risk theft. Check EVENTS for DISAPPEARED alerts — "
                "items vanishing while a person was nearby = theft. "
                "A person with erratic movement near displaced items = HIGH risk."
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

    def _maybe_analyze(
        self, job: dict[str, object], batch_frames: list | None = None,
    ) -> None:
        """Prepare analysis payload and enqueue for the background LLM consumer."""
        frames = batch_frames
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
                for marker in ("\nEVENTS:", "\nBEHAVIOR", "\nPROXIMITY", "\nALERTS"):
                    idx = scene_text.find(marker)
                    if idx >= 0:
                        texts.append(f"Frame {frame_idx}:{scene_text[idx:]}")
                        break
                else:
                    texts.append(f"Frame {frame_idx}:\n{scene_text.strip()}")
            else:
                texts.append(f"Frame {frame_idx} FULL:\n{scene_text.strip()}")

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

        if not buffer:
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

    # ------------------------------------------------------------------
    # Supreme OLEYES — Enhanced Pixtral VLM pipeline
    #   - Per-person buffers tracked by ByteTrack ID
    #   - Selective trigger (person + stealable item only)
    #   - Dynamic crop recomputation per frame
    #   - Resize to max 512px, JPEG quality 70
    #   - Hybrid prompt with YOLO context
    #   - Preliminary alert while VLM processes
    #   - Pre-warmed VLM client connection
    # ------------------------------------------------------------------

    _SUPREME_CROP_PAD = 40
    _SUPREME_JPEG_QUALITY = 70
    _SUPREME_MIN_CROP = 80  # skip VLM if crop is smaller than this

    def _get_vlm_client(self) -> PixtralClient:
        if self._vlm_client is not None:
            return self._vlm_client
        with self._vlm_client_lock:
            if self._vlm_client is None:
                self._vlm_client = PixtralClient()
                logger.info("pixtral_vlm_client_initialized")
        return self._vlm_client

    @staticmethod
    def _crop_and_encode(
        frame: np.ndarray,
        boxes: list[list[float]],
        pad: int = 40,
        max_dim: int = 512,
        jpeg_quality: int = 70,
    ) -> str | None:
        """Crop union bbox, resize to *max_dim*, JPEG-encode, return Base64.

        Returns None if the crop is too small to be useful.
        """
        h, w = frame.shape[:2]
        x1 = max(int(min(b[0] for b in boxes)) - pad, 0)
        y1 = max(int(min(b[1] for b in boxes)) - pad, 0)
        x2 = min(int(max(b[2] for b in boxes)) + pad, w)
        y2 = min(int(max(b[3] for b in boxes)) + pad, h)
        cw, ch = x2 - x1, y2 - y1
        if cw < Yolo26Service._SUPREME_MIN_CROP or ch < Yolo26Service._SUPREME_MIN_CROP:
            return None
        cropped = frame[y1:y2, x1:x2]
        longest = max(cw, ch)
        if longest > max_dim:
            scale = max_dim / longest
            cropped = cv2.resize(
                cropped,
                (int(cw * scale), int(ch * scale)),
                interpolation=cv2.INTER_AREA,
            )
        _, buf = cv2.imencode(".jpg", cropped, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        return base64.b64encode(buf).decode("utf-8")

    def _supreme_find_suspects(
        self, detections: list,
    ) -> list[dict]:
        """Identify persons near stealable items. Returns per-person info.

        Each entry: {"track_id": int, "person_box": [x1,y1,x2,y2],
                     "crop_boxes": [[x1,y1,x2,y2], ...], "item_labels": ["Handbag#7"]}
        Only returns persons that are within proximity of a stealable item.
        """
        from app.services.vision_engine.logic_engine import COCO_NAMES

        persons: list[dict] = []
        items: list[dict] = []
        for det in detections:
            cls_id = int(det[0])
            xyxy = det[2:6]
            track_id = int(det[6]) if len(det) >= 7 and det[6] is not None else -1
            name = COCO_NAMES.get(cls_id, f"Object-{cls_id}")
            if cls_id == 0:
                persons.append({"track_id": track_id, "box": xyxy, "name": name})
            elif cls_id in self._STEALABLE_CLS:
                items.append({"track_id": track_id, "box": xyxy, "name": name})

        if not persons or not items:
            return []

        suspects: list[dict] = []
        for p in persons:
            if p["track_id"] == -1:
                continue
            near_items: list[dict] = []
            for it in items:
                if self._dist_rects(p["box"], it["box"]) < self._POSE_GATE_PX:
                    near_items.append(it)
            if near_items:
                crop_boxes = [p["box"]] + [it["box"] for it in near_items]
                item_labels = [
                    f"{it['name']}#{it['track_id']}" for it in near_items
                ]
                suspects.append({
                    "track_id": p["track_id"],
                    "person_box": p["box"],
                    "crop_boxes": crop_boxes,
                    "item_labels": item_labels,
                })
        return suspects

    def _supreme_build_context(self, job: dict, suspect: dict) -> str:
        """Build a YOLO context string for the hybrid VLM prompt."""
        tid = suspect["track_id"]
        items_str = ", ".join(suspect["item_labels"])
        lines = [f"Person #{tid} detected near {items_str}."]

        logic = job.get("logic")
        if isinstance(logic, dict):
            scene_text = logic.get("scene_text", "")
            if "DISAPPEARED" in scene_text:
                for line in scene_text.split("\n"):
                    if "DISAPPEARED" in line:
                        lines.append(line.strip().lstrip("- "))
            if "DISPLACED" in scene_text:
                for line in scene_text.split("\n"):
                    if "DISPLACED" in line and f"#{tid}" in line:
                        lines.append(line.strip().lstrip("- "))
        return "\n".join(lines)

    def _supreme_buffer_step(self, job: dict, result, detections: list) -> None:
        """Per-person sliding window buffer for the Pixtral VLM pipeline.

        For each person near a stealable item:
        1. Start a per-person buffer (first frame encoded immediately)
        2. Capture subsequent frames at SUPREME_FRAME_INTERVAL with fresh crops
        3. When SUPREME_FRAME_COUNT frames collected, enqueue VLM analysis
        """
        now = time()
        frame_img = getattr(result, "orig_img", None)
        if frame_img is None:
            return

        suspects = self._supreme_find_suspects(detections)
        suspect_ids = {s["track_id"] for s in suspects}

        buffers: dict = job.get("supreme_suspects", {})
        interval = config.SUPREME_FRAME_INTERVAL
        needed = config.SUPREME_FRAME_COUNT
        cooldown = config.SUPREME_COOLDOWN
        max_dim = config.SUPREME_CROP_MAX_DIM

        # Prune buffers for persons no longer in the scene
        for tid in list(buffers.keys()):
            if tid not in suspect_ids:
                del buffers[tid]

        for suspect in suspects:
            tid = suspect["track_id"]

            if tid not in buffers:
                buffers[tid] = {
                    "frames": [],
                    "last_capture": 0.0,
                    "last_call": 0.0,
                    "items": suspect["item_labels"],
                }

            sb = buffers[tid]
            sb["items"] = suspect["item_labels"]

            if now - sb["last_call"] < cooldown:
                continue

            # Encode from fresh detections on every capture (dynamic crop)
            if not sb["frames"] or (now - sb["last_capture"]) >= interval:
                b64 = self._crop_and_encode(
                    frame_img,
                    suspect["crop_boxes"],
                    self._SUPREME_CROP_PAD,
                    max_dim,
                    self._SUPREME_JPEG_QUALITY,
                )
                if b64 is None:
                    continue
                sb["frames"].append(b64)
                sb["last_capture"] = now

                if len(sb["frames"]) == 1:
                    logger.info(
                        "supreme_buffer_started person=#%d items=%s frames=1/%d",
                        tid, suspect["item_labels"], needed,
                    )
                else:
                    logger.info(
                        "supreme_buffer_captured person=#%d frames=%d/%d",
                        tid, len(sb["frames"]), needed,
                    )

            if len(sb["frames"]) >= needed:
                frames_to_send = list(sb["frames"][:needed])
                context = self._supreme_build_context(job, suspect)
                sb["frames"].clear()
                sb["last_call"] = now
                logger.info(
                    "supreme_buffer_complete person=#%d — enqueueing VLM",
                    tid,
                )
                self._enqueue_vlm_analysis(job, frames_to_send, context, tid)

        job["supreme_suspects"] = buffers

    def _enqueue_vlm_analysis(
        self,
        job: dict,
        base64_frames: list[str],
        yolo_context: str = "",
        track_id: int = -1,
    ) -> None:
        try:
            self._vlm_queue.put_nowait((job, base64_frames, yolo_context, track_id))
        except queue.Full:
            try:
                self._vlm_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._vlm_queue.put_nowait((job, base64_frames, yolo_context, track_id))
            except queue.Full:
                logger.warning("supreme_vlm_queue_full — dropping analysis")

    # ------------------------------------------------------------------
    # VLM consumer thread (mirrors _analysis_consumer pattern)
    # ------------------------------------------------------------------

    def _ensure_vlm_worker(self) -> None:
        with self._vlm_worker_lock:
            if self._vlm_worker is None or not self._vlm_worker.is_alive():
                self._vlm_worker = Thread(
                    target=self._vlm_consumer,
                    daemon=True,
                    name="vlm-consumer",
                )
                self._vlm_worker.start()
                logger.info("vlm_worker_started")

    def _shutdown_vlm_worker(self) -> None:
        while not self._vlm_queue.empty():
            try:
                self._vlm_queue.get_nowait()
            except queue.Empty:
                break
        with self._vlm_worker_lock:
            if self._vlm_worker is not None and self._vlm_worker.is_alive():
                self._vlm_queue.put(self._SENTINEL)
                self._vlm_worker.join(timeout=60)
                logger.info("vlm_worker_joined")
            self._vlm_worker = None

    def _vlm_consumer(self) -> None:
        """Background thread: pre-warm connection, then pull VLM tasks."""
        client = self._get_vlm_client()
        client.warm_up()
        while True:
            item = self._vlm_queue.get()
            if item is self._SENTINEL:
                break
            job, base64_frames, yolo_context, track_id = item
            try:
                self._run_vlm_analysis(job, base64_frames, yolo_context, track_id)
            except Exception as exc:
                logger.warning("vlm_consumer unhandled: %s", exc)

    def _run_vlm_analysis(
        self,
        job: dict,
        base64_frames: list[str],
        yolo_context: str = "",
        track_id: int = -1,
    ) -> None:
        """Execute the Pixtral VLM call and map result to the analysis dict."""
        fallback = {
            "risk_score": 0,
            "risk_score_raw": 0,
            "risk_level": "LOW",
            "label": "VLM analysis pending",
            "explanation": "Visual analysis could not be completed.",
            "theft_detected": False,
            "confidence_score": 0,
            "mode": "supreme",
        }
        try:
            client = self._get_vlm_client()
            result = client.analyze_frames(base64_frames, yolo_context=yolo_context)

            theft = result.get("theft_detected", False)
            score = int(result.get("confidence_score", 0))
            analysis_text = result.get("analysis", "")

            if theft:
                risk_level = "HIGH"
            elif score >= 50:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            label = "Normal Activity"
            if theft:
                label = f"Theft Detected (Person #{track_id})" if track_id >= 0 else "Theft Detected"

            job["analysis"] = {
                "risk_score": score,
                "risk_score_raw": score,
                "risk_level": risk_level,
                "label": label,
                "explanation": analysis_text,
                "theft_detected": theft,
                "confidence_score": score,
                "mode": "supreme",
                "suspect_track_id": track_id,
            }
            job["supreme_vlm_pending"] = False
            logger.info(
                "supreme_vlm_result person=#%d theft=%s score=%d level=%s analysis=%s",
                track_id, theft, score, risk_level, analysis_text[:100],
            )
        except Exception as exc:
            logger.warning("supreme_vlm_failed: %s", exc)
            job["analysis"] = fallback
            job["supreme_vlm_pending"] = False
