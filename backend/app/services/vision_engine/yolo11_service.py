from __future__ import annotations

import json
import logging
from collections import deque
from threading import Lock
from threading import Event
from time import time

import httpx
from ultralytics import YOLO
from yt_dlp import YoutubeDL

from app.core import config
from app.services.vision_engine.logic_engine import AdvancedLogicEngine

logger = logging.getLogger(__name__)

class Yolo11Service:
    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, object]] = {}
        self._model: YOLO | None = None
        self._model_lock = Lock()
        self._logic = AdvancedLogicEngine(zones={})
        self._analysis_client = httpx.Client(
            base_url=config.SCALWAY_BASE_URL,
            headers={"Authorization": f"Bearer {config.SCALWAY_API_KEY}"},
            timeout=config.SCALWAY_TIMEOUT,
        )

    def _get_model(self) -> YOLO:
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is None:
                self._model = YOLO(config.YOLO_MODEL)
        return self._model

    def register_job(
        self, *, job_id: str, source_url: str, scene_context: str | None = None
    ) -> None:
        self._jobs[job_id] = {
            "status": "queued",
            "source_url": source_url,
            "scene_context": scene_context,
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
            "risk_scores": deque(maxlen=5),  # rolling window for smoothing
            "stop_event": Event(),
        }
        if scene_context:
            logger.info("📋 Job queued id=%s | 🎬 scene_context=\"%s\"", job_id, scene_context)
        else:
            logger.info("📋 Job queued id=%s | ⚠️ no scene context", job_id)

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
        try:
            if self._is_stopped(job):
                job["status"] = "stopped"
                job["finished_at"] = time()
                return
            resolved_source = self._resolve_source(source_url)
            logger.info("yolo_source_resolved id=%s", job_id)
            model = self._get_model()
            results = model.track(
                source=resolved_source,
                stream=True,
                conf=config.YOLO_CONF,
                device=config.YOLO_DEVICE,
                verbose=False,
                max_det=config.YOLO_MAX_DETECTIONS,
                persist=True,
                vid_stride=config.YOLO_VID_STRIDE,
                stream_buffer=config.YOLO_STREAM_BUFFER,
            )
            for result in results:
                if self._is_stopped(job):
                    job["status"] = "stopped"
                    job["finished_at"] = time()
                    break
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

                event = {
                    "frame_index": frame_index,
                    "timestamp": time(),
                    "vectors": detections,
                }
                self._logic.update(
                    {
                        "frame_index": frame_index,
                        "timestamp": time(),
                        "vectors": detections,
                    }
                )
                logic_summary = self._logic.generate_scene_summary()
                job["logic"] = logic_summary
                event["scene_text"] = logic_summary.get("scene_text", "")
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
                        "📹 frame=%s  detections=%d  batch_sent",
                        frame_index,
                        len(detections),
                    )
                log_every = max(int(config.YOLO_LOG_EVERY), 1)
                if frame_index % log_every == 0:
                    logger.debug(
                        "yolo_frame=%s vectors=%s",
                        frame_index,
                        detections,
                    )
                job["last_update"] = time()
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
                job_id,
                job["frames"],
                job["detections"],
            )
        except Exception as exc:  # pragma: no cover - runtime dependency errors
            job["status"] = "error"
            job["error"] = str(exc)
            job["finished_at"] = time()
            logger.exception("yolo_job_error id=%s error=%s", job_id, exc)

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
        }

    def _is_stopped(self, job: dict[str, object]) -> bool:
        stop_event = job.get("stop_event")
        return bool(isinstance(stop_event, Event) and stop_event.is_set())

    def _maybe_analyze(self, job: dict[str, object]) -> None:
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
        for frame in frames:
            if isinstance(frame, dict) and frame.get("scene_text"):
                texts.append(
                    f"Frame {frame.get('frame_index')}: {frame.get('scene_text')}"
                )
        if texts:
            buffer.append("\n".join(texts))

        if len(buffer) < 2:
            return

        combined = "\n\n".join(buffer[-2:])
        system_prompt = config.SCALWAY_SYSTEM_PROMPT or ""
        scene_context = job.get("scene_context")
        if scene_context:
            system_prompt = system_prompt.replace(
                "monitoring a CCTV feed",
                f"monitoring a CCTV feed in: {scene_context}",
            )
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
            "response_format": {"type": "json_object"},
        }
        try:
            response = self._analysis_client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            raw_content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

            # Parse JSON response from LLM
            try:
                analysis = json.loads(raw_content)
            except json.JSONDecodeError:
                # Fallback: treat as plain text if LLM didn't return JSON
                analysis = {
                    "risk_score": 0,
                    "risk_level": "LOW",
                    "label": "Parse Error",
                    "explanation": raw_content[:80],
                }

            # --- Risk score smoothing ---
            raw_score = int(analysis.get("risk_score", 0))
            risk_scores = job.get("risk_scores")
            if not isinstance(risk_scores, deque):
                risk_scores = deque(maxlen=5)
                job["risk_scores"] = risk_scores
            risk_scores.append(raw_score)

            smoothed_score = int(sum(risk_scores) / len(risk_scores))

            # Only escalate to HIGH if score stays >= 70 for 3+ consecutive updates
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
            buffer.clear()
        except Exception as exc:  # pragma: no cover
            logger.warning("❌ Analysis failed: %s", exc)
            job["analysis"] = {"text": "", "error": str(exc)}
