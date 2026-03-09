# OLEYES API Endpoints Documentation

> **Backend:** FastAPI `v0.1.0` — Smart AI CCTV Orchestrator  
> **Base URL:** `http://localhost:8000`  
> **API Prefix:** `/api/v1`  
> **Auth:** None (all endpoints are public)  
> **CORS:** All origins allowed

---

## Table of Contents

- [Quick Reference](#quick-reference)
- [Health Checks](#health-checks)
- [LLM — Chat](#llm--chat)
- [Vision — Object Detection](#vision--object-detection)
  - [Start YouTube Detection](#post-apiv1visiondetect-youtube)
  - [Get Job Status](#get-apiv1visionjobsjob_id)
  - [Stream Job Status (SSE)](#get-apiv1visionjobsjob_idstream)
  - [Stop Job](#post-apiv1visionjobsjob_idstop)
- [Data Models](#data-models)
- [Error Handling](#error-handling)
- [SSE (Server-Sent Events) Guide](#sse-server-sent-events-guide)

---

## Quick Reference

| Method | Endpoint                                  | Description                  |
| ------ | ----------------------------------------- | ---------------------------- |
| GET    | `/health`                                 | Root health check            |
| GET    | `/api/v1/connect/health`                  | Connect service health       |
| GET    | `/api/v1/stream/health`                   | Stream service health        |
| POST   | `/api/v1/llm/chat`                        | Send chat messages to LLM    |
| POST   | `/api/v1/vision/detect-youtube`           | Start YOLOv11 detection job  |
| GET    | `/api/v1/vision/jobs/{job_id}`            | Poll job status              |
| GET    | `/api/v1/vision/jobs/{job_id}/stream`     | Stream job status via SSE    |
| POST   | `/api/v1/vision/jobs/{job_id}/stop`       | Stop a running job           |

---

## Health Checks

Three health-check endpoints are available. All return the same response.

### `GET /health`

Root-level health check.

### `GET /api/v1/connect/health`

Connect service health check.

### `GET /api/v1/stream/health`

Stream service health check.

**Response** `200 OK`

```json
{
  "status": "ok"
}
```

---

## LLM — Chat

### `POST /api/v1/llm/chat`

Send a conversation to the Scaleway-hosted LLM and receive an AI response. Optionally provide a `scene_context` to ground the AI's analysis in a specific environment (e.g. a supermarket, a parking lot).

**Request Body** — `application/json`

| Field               | Type              | Required | Default | Description                                                                                         |
| ------------------- | ----------------- | -------- | ------- | --------------------------------------------------------------------------------------------------- |
| `messages`          | `LLMMessage[]`    | Yes      | —       | Array of chat messages (see [LLMMessage](#llmmessage))                                              |
| `scene_context`     | `string \| null`  | No       | `null`  | Optional scene description to ground the AI analysis (e.g. `"Supermarket, busy hour, food aisles"`) |
| `max_tokens`        | `integer \| null` | No       | `512`   | Maximum tokens in the response                                                                      |
| `temperature`       | `float \| null`   | No       | `1.0`   | Sampling temperature (`0.0` — `2.0`)                                                                |
| `top_p`             | `float \| null`   | No       | `1.0`   | Nucleus sampling parameter                                                                          |
| `presence_penalty`  | `float \| null`   | No       | `0.0`   | Penalise tokens that already appeared                                                               |

**Example Request**

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an AI security supervisor monitoring a CCTV feed."
    },
    {
      "role": "user",
      "content": "A person is running through the store carrying a bag. What is the risk level?"
    }
  ],
  "scene_context": "Supermarket, busy hour, primarily food aisles",
  "max_tokens": 256,
  "temperature": 0.7
}
```

**Response** `200 OK` — `LLMChatResponse`

| Field     | Type     | Description                            |
| --------- | -------- | -------------------------------------- |
| `model`   | `string` | Model name used (e.g. `"gpt-oss-120b"`) |
| `content` | `string` | The AI-generated text response         |

**Example Response**

```json
{
  "model": "gpt-oss-120b",
  "content": "{\"risk_score\": 75, \"risk_level\": \"HIGH\", \"label\": \"Possible theft in progress\", \"explanation\": \"Person running with bag suggests potential shoplifting.\"}"
}
```

> **Note:** When `scene_context` is provided, it is automatically injected into the first system message (or a new system message is prepended if none exists).

---

## Vision — Object Detection

The Vision module runs YOLOv11 object detection on YouTube video streams as background jobs. Each job gets a unique `job_id` that you use to poll status, stream events, or stop the job.

### Typical Frontend Flow

```
1. POST /api/v1/vision/detect-youtube   →  get job_id
2. GET  /api/v1/vision/jobs/{job_id}/stream  →  open SSE connection
3. Listen to SSE events until status is "done" or "error"
4. (Optional) POST /api/v1/vision/jobs/{job_id}/stop  →  cancel early
```

---

### `POST /api/v1/vision/detect-youtube`

Start a new YOLOv11 detection job on a YouTube video. The job runs in the background; use the returned `job_id` to track progress.

**Request Body** — `application/json`

| Field           | Type             | Required | Default | Description                                                        |
| --------------- | ---------------- | -------- | ------- | ------------------------------------------------------------------ |
| `youtube_url`   | `string` (URL)   | Yes      | —       | Public YouTube video URL                                           |
| `callback_url`  | `string \| null` (URL) | No | `null`  | Optional webhook URL to receive results when the job completes     |
| `scene_context` | `string \| null` | No       | `null`  | Optional scene description (e.g. `"Supermarket, busy hour"`)      |

**Example Request**

```json
{
  "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "scene_context": "Parking lot, night time, low traffic"
}
```

**Response** `200 OK` — `VisionJobResponse`

```json
{
  "job_id": "a3f1b2c4-5678-9abc-def0-1234567890ab",
  "status": "queued",
  "detail": "YOLOv11 job queued.",
  "result": null,
  "error": null
}
```

---

### `GET /api/v1/vision/jobs/{job_id}`

Poll the current status of a detection job.

**Path Parameters**

| Parameter | Type     | Description        |
| --------- | -------- | ------------------ |
| `job_id`  | `string` | The UUID of the job |

**Response** `200 OK` — `VisionJobResponse`

| Field    | Type              | Description                                                        |
| -------- | ----------------- | ------------------------------------------------------------------ |
| `job_id` | `string`          | The job UUID                                                       |
| `status` | `string`          | One of: `"queued"`, `"running"`, `"done"`, `"error"`, `"unknown"`  |
| `detail` | `string \| null`  | Human-readable status message                                      |
| `result` | `object \| null`  | Detection results (populated when `status` is `"done"`)            |
| `error`  | `string \| null`  | Error message (populated when `status` is `"error"`)               |

**Example — Job Running**

```json
{
  "job_id": "a3f1b2c4-5678-9abc-def0-1234567890ab",
  "status": "running",
  "detail": "OK",
  "result": null,
  "error": null
}
```

**Example — Job Completed**

```json
{
  "job_id": "a3f1b2c4-5678-9abc-def0-1234567890ab",
  "status": "done",
  "detail": "OK",
  "result": {
    "detections": [...],
    "frame_count": 120,
    "summary": "..."
  },
  "error": null
}
```

**Example — Job Not Found**

```json
{
  "job_id": "nonexistent-id",
  "status": "unknown",
  "detail": "Job not found.",
  "result": null,
  "error": null
}
```

---

### `GET /api/v1/vision/jobs/{job_id}/stream`

Open a **Server-Sent Events (SSE)** connection to receive real-time updates for a detection job. The stream sends a JSON payload every ~1 second.

**Path Parameters**

| Parameter | Type     | Description        |
| --------- | -------- | ------------------ |
| `job_id`  | `string` | The UUID of the job |

**Response** `200 OK` — `text/event-stream`

Each SSE message is a `data:` line containing a JSON object:

| Field       | Type      | Description                                             |
| ----------- | --------- | ------------------------------------------------------- |
| `status`    | `string`  | Current job status                                      |
| `event_id`  | `integer` | Incrementing event counter                              |
| `heartbeat` | `boolean` | `true` if no state change since last event              |
| `...`       | `varies`  | Additional snapshot fields (detections, frame data, etc.) |

**Stream Behavior**

- Sends a snapshot every ~1 second
- If nothing changed since the last event, `heartbeat: true` is set
- Stream **automatically closes** when `status` becomes `"done"` or `"error"`
- If the job doesn't exist, a single event is sent and the stream closes:

```
data: {"status": "unknown", "detail": "Job not found."}
```

**Example SSE Events**

```
data: {"status": "running", "event_id": 1, "heartbeat": false, "frame": 10, "detections": [...]}

data: {"status": "running", "event_id": 1, "heartbeat": true, "frame": 10, "detections": [...]}

data: {"status": "running", "event_id": 2, "heartbeat": false, "frame": 20, "detections": [...]}

data: {"status": "done", "event_id": 3, "heartbeat": false, "result": {...}}
```

**Frontend Usage (JavaScript)**

```javascript
const evtSource = new EventSource(
  `http://localhost:8000/api/v1/vision/jobs/${jobId}/stream`
);

evtSource.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.heartbeat) {
    // No new data — skip or show "waiting..." indicator
    return;
  }

  // Update UI with new detection data
  console.log("Status:", data.status);
  console.log("Detections:", data.detections);

  // Close when done or error
  if (data.status === "done" || data.status === "error") {
    evtSource.close();
  }
};

evtSource.onerror = () => {
  evtSource.close();
};
```

---

### `POST /api/v1/vision/jobs/{job_id}/stop`

Stop a running detection job.

**Path Parameters**

| Parameter | Type     | Description        |
| --------- | -------- | ------------------ |
| `job_id`  | `string` | The UUID of the job |

**Response** `200 OK`

**Success — Job Stopped**

```json
{
  "status": "stopped",
  "job_id": "a3f1b2c4-5678-9abc-def0-1234567890ab"
}
```

**Failure — Job Not Found**

```json
{
  "status": "not_found",
  "job_id": "nonexistent-id"
}
```

---

## Data Models

### `LLMMessage`

| Field     | Type     | Required | Description                                    |
| --------- | -------- | -------- | ---------------------------------------------- |
| `role`    | `string` | Yes      | One of: `"system"`, `"user"`, `"assistant"`    |
| `content` | `string` | Yes      | The message text                               |

### `LLMChatRequest`

| Field               | Type              | Required | Default | Description                              |
| ------------------- | ----------------- | -------- | ------- | ---------------------------------------- |
| `messages`          | `LLMMessage[]`    | Yes      | —       | Conversation history                     |
| `scene_context`     | `string \| null`  | No       | `null`  | Scene description for grounding          |
| `max_tokens`        | `integer \| null` | No       | `512`   | Max response tokens                      |
| `temperature`       | `float \| null`   | No       | `1.0`   | Sampling temperature                     |
| `top_p`             | `float \| null`   | No       | `1.0`   | Nucleus sampling                         |
| `presence_penalty`  | `float \| null`   | No       | `0.0`   | Token presence penalty                   |

### `LLMChatResponse`

| Field     | Type     | Description                  |
| --------- | -------- | ---------------------------- |
| `model`   | `string` | Model name used              |
| `content` | `string` | AI-generated response text   |

### `VisionYoutubeRequest`

| Field           | Type             | Required | Default | Description                        |
| --------------- | ---------------- | -------- | ------- | ---------------------------------- |
| `youtube_url`   | `string` (URL)   | Yes      | —       | Public YouTube video URL           |
| `callback_url`  | `string \| null` | No       | `null`  | Webhook for results                |
| `scene_context` | `string \| null` | No       | `null`  | Scene description for grounding    |

### `VisionJobResponse`

| Field    | Type              | Description                                |
| -------- | ----------------- | ------------------------------------------ |
| `job_id` | `string`          | Unique job identifier (UUID)               |
| `status` | `string`          | `"queued"` `"running"` `"done"` `"error"` `"unknown"` |
| `detail` | `string \| null`  | Human-readable message                     |
| `result` | `object \| null`  | Detection results when completed           |
| `error`  | `string \| null`  | Error details if failed                    |

---

## Error Handling

The API uses standard HTTP status codes. FastAPI automatically validates request bodies against the Pydantic models and returns `422 Unprocessable Entity` for invalid input.

**Validation Error Response** `422`

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "messages"],
      "msg": "Field required",
      "input": {}
    }
  ]
}
```

**Internal Server Error** `500`

```json
{
  "detail": "Internal Server Error"
}
```

---

## SSE (Server-Sent Events) Guide

### What is SSE?

Server-Sent Events provide a one-way, real-time stream from server to client over HTTP. Unlike WebSockets, SSE uses a standard HTTP connection and is natively supported by browsers via `EventSource`.

### Connection Lifecycle

```
Frontend                          Backend
   |                                 |
   |  GET /vision/jobs/{id}/stream   |
   | ------------------------------> |
   |                                 |
   |  data: { status: "running" }    |
   | <------------------------------ |
   |                                 |
   |  data: { heartbeat: true }      |
   | <------------------------------ |
   |                                 |
   |  data: { status: "done" }       |
   | <------------------------------ |
   |                                 |
   |  (stream closes)                |
   |                                 |
```

### Tips for Frontend Integration

1. **Always handle `onerror`** — close the connection and show a user-friendly message.
2. **Filter heartbeats** — skip UI updates when `heartbeat: true` to avoid unnecessary re-renders.
3. **Close on completion** — call `evtSource.close()` when `status` is `"done"` or `"error"`.
4. **Reconnection** — `EventSource` auto-reconnects by default. If you don't want that, use `fetch()` with a `ReadableStream` instead.

---

## Interactive API Docs

FastAPI provides auto-generated documentation:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
