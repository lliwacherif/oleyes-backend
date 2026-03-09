# API Testing — Postman-style Collection

Base URL: `http://localhost:8000`

---

## 1. Signup

```
POST http://localhost:8000/api/v1/auth/signup

Headers:
  Content-Type: application/json

Body:
{
  "email": "test@example.com",
  "username": "testuser",
  "password": "123456"
}

Expected: 201
Response:
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer"
}
```

---

## 2. Login

```
POST http://localhost:8000/api/v1/auth/login

Headers:
  Content-Type: application/json

Body:
{
  "email": "test@example.com",
  "password": "123456"
}

Expected: 200
Response:
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer"
}
```

---

## 3. Get Current User

```
GET http://localhost:8000/api/v1/auth/me

Headers:
  Authorization: Bearer <access_token>

Expected: 200
Response:
{
  "id": "uuid",
  "email": "test@example.com",
  "username": "testuser",
  "is_active": true,
  "created_at": "2026-02-05T..."
}
```

---

## 4. Refresh Token

```
POST http://localhost:8000/api/v1/auth/refresh

Headers:
  Content-Type: application/json

Body:
{
  "refresh_token": "<refresh_token from login/signup>"
}

Expected: 200
Response:
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer"
}
```

---

## 5. Create Scene Context

```
POST http://localhost:8000/api/v1/context/

Headers:
  Content-Type: application/json
  Authorization: Bearer <access_token>

Body:
{
  "business_type": "Supermarket",
  "business_name": "Oleyes Supermarket",
  "short_description": "A local grocery store",
  "number_of_locations": "1",
  "estimated_number_of_cameras": "15",
  "business_size": "Medium",
  "camera_type": "IP Cameras",
  "theft_detection": true,
  "suspicious_behavior_detection": false,
  "loitering_detection": true,
  "employee_monitoring": false,
  "customer_behavior_analytics": false
}

Expected: 201
Response:
{
  "id": "uuid",
  "user_id": "uuid",
  "context_text": "{\"business_type\": \"Supermarket\", ...}",
  "environment_type": "Supermarket",
  "context_data": {
    "business_type": "Supermarket",
    "business_name": "Oleyes Supermarket",
    "short_description": "A local grocery store",
    "number_of_locations": "1",
    "estimated_number_of_cameras": "15",
    "business_size": "Medium",
    "camera_type": "IP Cameras",
    "theft_detection": true,
    "suspicious_behavior_detection": false,
    "loitering_detection": true,
    "employee_monitoring": false,
    "customer_behavior_analytics": false
  },
  "created_at": "...",
  "updated_at": "..."
}
```

---

## 6. Get Scene Context

```
GET http://localhost:8000/api/v1/context/

Headers:
  Authorization: Bearer <access_token>

Expected: 200
```

---

## 7. Update Scene Context

```
PUT http://localhost:8000/api/v1/context/

Headers:
  Content-Type: application/json
  Authorization: Bearer <access_token>

Body:
{
  "business_type": "Parking Lot",
  "business_name": "City Parking",
  "short_description": "3-level parking structure",
  "number_of_locations": "1",
  "estimated_number_of_cameras": "8",
  "business_size": "Large",
  "camera_type": "IP Cameras",
  "theft_detection": false,
  "suspicious_behavior_detection": true,
  "loitering_detection": true,
  "employee_monitoring": false,
  "customer_behavior_analytics": false
}

Expected: 200
```

---

## 8. Delete Scene Context

```
DELETE http://localhost:8000/api/v1/context/

Headers:
  Authorization: Bearer <access_token>

Expected: 200
Response:
{
  "status": "deleted",
  "user_id": "uuid"
}
```

Error: 404 if no context exists.

---

## 9. Start YOLO Detection

```
POST http://localhost:8000/api/v1/vision/detect-youtube

Headers:
  Content-Type: application/json
  Authorization: Bearer <access_token>

Body:
{
  "youtube_url": "https://www.youtube.com/watch?v=VIDEO_ID"
}

Expected: 200
Response:
{
  "job_id": "uuid",
  "status": "queued"
}
```

---

## 10. Stream Job Results (SSE)

```
GET http://localhost:8000/api/v1/vision/jobs/<job_id>/stream

Headers:
  Accept: text/event-stream

Note: This is an SSE stream. In Postman, use "Send and Download" or test in browser.
```

---

## 11. Stop Job

```
POST http://localhost:8000/api/v1/vision/jobs/<job_id>/stop

Expected: 200
Response:
{
  "status": "stopped",
  "job_id": "uuid"
}
```

---

## 12. LLM Chat

```
POST http://localhost:8000/api/v1/llm/chat

Headers:
  Content-Type: application/json

Body:
{
  "messages": [
    { "role": "user", "content": "Hello" }
  ]
}

Expected: 200
Response:
{
  "model": "qwen3-235b-a22b-instruct-2507",
  "content": "..."
}
```

---

## 13. Health Check

```
GET http://localhost:8000/health

Expected: 200
Response:
{
  "status": "ok"
}
```
