# OLEYES Authentication API — Frontend Integration Guide

> **Base URL:** `http://localhost:8000`
> **Prefix:** `/api/v1/auth`
> **Content-Type:** `application/json`

---

## Table of Contents

- [Quick Reference](#quick-reference)
- [Authentication Flow](#authentication-flow)
- [Endpoints](#endpoints)
  - [POST /auth/signup](#post-apiv1authsignup)
  - [POST /auth/login](#post-apiv1authlogin)
  - [POST /auth/refresh](#post-apiv1authrefresh)
  - [GET /auth/me](#get-apiv1authme)
- [Token Management](#token-management)
- [Protecting Requests](#protecting-requests)
- [Error Reference](#error-reference)
- [Full Frontend Example](#full-frontend-example)

---

## Quick Reference

| Method | Endpoint              | Auth Required | Description                    |
| ------ | --------------------- | ------------- | ------------------------------ |
| POST   | `/api/v1/auth/signup`   | No            | Create a new account           |
| POST   | `/api/v1/auth/login`    | No            | Log in with email + password   |
| POST   | `/api/v1/auth/refresh`  | No            | Get new tokens via refresh token |
| GET    | `/api/v1/auth/me`       | Yes (Bearer)  | Get current user profile       |

---

## Authentication Flow

```
 SIGNUP / LOGIN
 ==============
 1. Frontend sends credentials to /auth/signup or /auth/login
 2. Backend returns { access_token, refresh_token, token_type }
 3. Frontend stores both tokens

 MAKING REQUESTS
 ===============
 4. Frontend sends access_token in the Authorization header:
    Authorization: Bearer <access_token>

 TOKEN EXPIRED
 =============
 5. Backend returns 401
 6. Frontend sends refresh_token to /auth/refresh
 7. Backend returns a new { access_token, refresh_token }
 8. Frontend stores the new tokens and retries the original request

 LOGOUT
 ======
 9. Frontend deletes both tokens from storage (no backend call needed)
```

---

## Endpoints

### `POST /api/v1/auth/signup`

Create a new user account. Returns tokens immediately (user is logged in after signup).

**Request Body**

| Field      | Type     | Required | Constraints               |
| ---------- | -------- | -------- | ------------------------- |
| `email`    | `string` | Yes      | Valid email format        |
| `username` | `string` | Yes      | 3 - 100 characters        |
| `password` | `string` | Yes      | 6 - 128 characters        |

**Example Request**

```json
{
  "email": "john@example.com",
  "username": "john_doe",
  "password": "securePass123"
}
```

**Response `201 Created`**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

**Possible Errors**

| Status | Detail                       | Cause                        |
| ------ | ---------------------------- | ---------------------------- |
| 409    | `"Email already registered."` | Email is taken               |
| 409    | `"Username already taken."`   | Username is taken            |
| 422    | Validation error             | Missing/invalid fields       |

---

### `POST /api/v1/auth/login`

Authenticate with email and password.

**Request Body**

| Field      | Type     | Required |
| ---------- | -------- | -------- |
| `email`    | `string` | Yes      |
| `password` | `string` | Yes      |

**Example Request**

```json
{
  "email": "john@example.com",
  "password": "securePass123"
}
```

**Response `200 OK`**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

**Possible Errors**

| Status | Detail                           | Cause                    |
| ------ | -------------------------------- | ------------------------ |
| 401    | `"Invalid email or password."`   | Wrong credentials        |
| 401    | `"User account is deactivated."` | Account disabled         |
| 422    | Validation error                 | Missing/invalid fields   |

---

### `POST /api/v1/auth/refresh`

Exchange a valid refresh token for a brand-new token pair. The old refresh token should be discarded.

**Request Body**

| Field           | Type     | Required |
| --------------- | -------- | -------- |
| `refresh_token` | `string` | Yes      |

**Example Request**

```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response `200 OK`**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

**Possible Errors**

| Status | Detail                                   | Cause                      |
| ------ | ---------------------------------------- | -------------------------- |
| 401    | `"Invalid or expired refresh token."`    | Token expired or malformed |
| 401    | `"Token is not a refresh token."`        | Sent an access token       |
| 401    | `"User not found or deactivated."`       | Account deleted/disabled   |

---

### `GET /api/v1/auth/me`

Get the profile of the currently authenticated user. Requires a valid access token.

**Headers**

```
Authorization: Bearer <access_token>
```

**Response `200 OK`**

```json
{
  "id": "a3f1b2c4-5678-9abc-def0-1234567890ab",
  "email": "john@example.com",
  "username": "john_doe",
  "is_active": true,
  "created_at": "2026-02-10T15:30:00+00:00"
}
```

**Possible Errors**

| Status | Detail                             | Cause                    |
| ------ | ---------------------------------- | ------------------------ |
| 401    | `"Invalid or expired token."`      | Token expired/malformed  |
| 401    | `"Invalid token type."`            | Sent a refresh token     |
| 401    | `"User not found."`                | Account deleted          |
| 401    | `"User account is deactivated."`   | Account disabled         |
| 403    | `"Not authenticated"`              | No Authorization header  |

---

## Token Management

### Token Lifetimes

| Token          | Lifetime | Purpose                          |
| -------------- | -------- | -------------------------------- |
| Access Token   | 30 min   | Authenticate API requests        |
| Refresh Token  | 7 days   | Obtain new access tokens silently |

### Where to Store Tokens

**Recommended: `localStorage`** (simplest for SPAs)

```javascript
// After login/signup
localStorage.setItem("access_token", data.access_token);
localStorage.setItem("refresh_token", data.refresh_token);

// On logout
localStorage.removeItem("access_token");
localStorage.removeItem("refresh_token");
```

### Token Payload Structure

The JWT payload contains:

```json
{
  "sub": "user-uuid-here",
  "exp": 1707580200,
  "type": "access"
}
```

- `sub` — the user's UUID
- `exp` — expiration timestamp (Unix)
- `type` — either `"access"` or `"refresh"`

---

## Protecting Requests

Every authenticated request must include the access token in the `Authorization` header.

### Basic Fetch Example

```javascript
const response = await fetch("http://localhost:8000/api/v1/auth/me", {
  method: "GET",
  headers: {
    "Authorization": `Bearer ${localStorage.getItem("access_token")}`,
    "Content-Type": "application/json",
  },
});

if (response.status === 401) {
  // Token expired — try refreshing
  const refreshed = await refreshTokens();
  if (!refreshed) {
    // Refresh also failed — redirect to login
    window.location.href = "/login";
  }
}
```

### Axios Interceptor (Recommended)

```javascript
import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:8000/api/v1",
});

// Attach token to every request
api.interceptors.request.use((config) => {
  const token = localStorage.getItem("access_token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Auto-refresh on 401
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;

      try {
        const refreshToken = localStorage.getItem("refresh_token");
        const { data } = await axios.post(
          "http://localhost:8000/api/v1/auth/refresh",
          { refresh_token: refreshToken }
        );

        // Store new tokens
        localStorage.setItem("access_token", data.access_token);
        localStorage.setItem("refresh_token", data.refresh_token);

        // Retry original request with new token
        originalRequest.headers.Authorization = `Bearer ${data.access_token}`;
        return api(originalRequest);
      } catch (refreshError) {
        // Refresh failed — force logout
        localStorage.removeItem("access_token");
        localStorage.removeItem("refresh_token");
        window.location.href = "/login";
        return Promise.reject(refreshError);
      }
    }

    return Promise.reject(error);
  }
);

export default api;
```

---

## Error Reference

All error responses follow this format:

```json
{
  "detail": "Error message here."
}
```

Validation errors (422) have a different format:

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "email"],
      "msg": "value is not a valid email address",
      "input": "not-an-email"
    }
  ]
}
```

### Status Code Summary

| Code | Meaning                | When                                       |
| ---- | ---------------------- | ------------------------------------------ |
| 200  | OK                     | Login, refresh, me — success               |
| 201  | Created                | Signup — success                            |
| 401  | Unauthorized           | Bad credentials, expired/invalid token      |
| 403  | Forbidden              | Missing Authorization header entirely       |
| 409  | Conflict               | Email or username already taken (signup)    |
| 422  | Unprocessable Entity   | Invalid request body (validation failed)    |
| 500  | Internal Server Error  | Unexpected backend error                    |

---

## Full Frontend Example

### Signup Flow

```javascript
async function signup(email, username, password) {
  const response = await fetch("http://localhost:8000/api/v1/auth/signup", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, username, password }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail);
  }

  const data = await response.json();
  localStorage.setItem("access_token", data.access_token);
  localStorage.setItem("refresh_token", data.refresh_token);
  return data;
}
```

### Login Flow

```javascript
async function login(email, password) {
  const response = await fetch("http://localhost:8000/api/v1/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail);
  }

  const data = await response.json();
  localStorage.setItem("access_token", data.access_token);
  localStorage.setItem("refresh_token", data.refresh_token);
  return data;
}
```

### Get Current User

```javascript
async function getCurrentUser() {
  const response = await fetch("http://localhost:8000/api/v1/auth/me", {
    headers: {
      "Authorization": `Bearer ${localStorage.getItem("access_token")}`,
    },
  });

  if (!response.ok) {
    throw new Error("Not authenticated");
  }

  return await response.json();
}
```

### Logout

```javascript
function logout() {
  localStorage.removeItem("access_token");
  localStorage.removeItem("refresh_token");
  window.location.href = "/login";
}
```

---

## Interactive API Docs

You can test all auth endpoints directly in the browser:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

In Swagger, click the **Authorize** button (lock icon) and paste your access token to test protected endpoints like `/auth/me`.
