# Smart AI CCTV Orchestrator — Project Guide

This guide documents the backend API and configuration so a React client can integrate cleanly.

## Architecture Overview

- FastAPI backend under `backend/`
- Modular structure:
  - `app/api` → HTTP endpoints
  - `app/services` → core services (LLM, vision, camera manager)
  - `app/core` → configuration and shared utilities

## Project Structure

```
/backend
  /app
    /api
      /v1
        /endpoints
          connect.py
          llm.py
          stream.py
          vision.py
    /core
      config.py
      security.py
    /services
      /vision_engine
        yolo26_service.py
      /llm_engine
        scaleway_client.py
      /camera_manager
  main.py
  requirements.txt
  .env
  .env.example
```

## Environment Configuration

Create `backend/.env` with:

```
SCALWAY_API_KEY=YOUR_KEY_HERE
SCALWAY_BASE_URL=https://api.scaleway.ai/d067acb3-2897-4c85-a126-957eb6768d0b/v1
SCALWAY_MODEL=gpt-oss-120b
SCALWAY_TIMEOUT=30
YOLO_MODEL=yolo26n.pt
YOLO_DEVICE=cpu
YOLO_CONF=0.25
RTSP_URLS=
```

Notes:
- `SCALWAY_TIMEOUT` is in seconds.
- `YOLO_MODEL` defaults to the lightweight CPU model.

## API Base

All endpoints are prefixed with:

```
/api/v1
```

## Endpoints

### Health

```
GET /health
```

Response:
```
{ "status": "ok" }
```

### LLM Chat (Scaleway)

```
POST /api/v1/llm/chat
```

Request body:
```
{
  "messages": [
    { "role": "system", "content": "You are a helpful assistant" },
    { "role": "user", "content": "Hello" }
  ],
  "max_tokens": 512,
  "temperature": 1.0,
  "top_p": 1.0,
  "presence_penalty": 0.0
}
```

Response body:
```
{
  "model": "gpt-oss-120b",
  "content": "..."
}
```

Notes:
- This uses Scaleway’s OpenAI-compatible API.
- Streaming can be added later if needed.

### YOLO26 — YouTube Detection

```
POST /api/v1/vision/detect-youtube
```

Request body:
```
{
  "youtube_url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "callback_url": "https://example.com/webhook"  // optional
}
```

Response body:
```
{
  "job_id": "uuid",
  "status": "queued",
  "detail": "YOLO26 job queued.",
  "result": null,
  "error": null
}
```

### YOLO26 — Job Status

```
GET /api/v1/vision/jobs/{job_id}
```

Response body (completed):
```
{
  "job_id": "uuid",
  "status": "done",
  "detail": "OK",
  "result": {
    "frames": 123,
    "detections": 456
  },
  "error": null
}
```

Response body (running/queued):
```
{
  "job_id": "uuid",
  "status": "running",
  "detail": "OK",
  "result": null,
  "error": null
}
```

## React Integration Notes

- The React app should call the API endpoints directly.
- For YOLO:
  1. POST `/api/v1/vision/detect-youtube` with a YouTube URL.
  2. Poll `/api/v1/vision/jobs/{job_id}` until `status` is `done` or `error`.
- For LLM:
  - POST `/api/v1/llm/chat` with `messages`.

## Running the Backend (Dev)

Start the server from `backend/`:

```
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Dependencies

Backend deps are in `backend/requirements.txt`:
- fastapi
- uvicorn[standard]
- httpx
- ultralytics
- opencv-python-headless
- python-dotenv
- yt-dlp

## Roadmap

- Stream LLM responses to the client
- Return detailed YOLO results (boxes, labels, timestamps)
- Add camera RTSP ingestion and live stream APIs
