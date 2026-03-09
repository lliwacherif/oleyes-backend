# Authentication API — Technical Guide (Flutter)

## Base URL

```
http://localhost:8000
```

## Auth Flow Overview

```
Signup → tokens → store locally → use access_token for all requests
Login  → tokens → store locally → use access_token for all requests
Token expired → call /refresh with refresh_token → new tokens
```

## Token Details

| Token | Type | Lifetime | Usage |
|-------|------|----------|-------|
| `access_token` | JWT (HS256) | 30 minutes | Sent in `Authorization: Bearer <token>` header for every authenticated request |
| `refresh_token` | JWT (HS256) | 7 days | Used only to get a new access/refresh pair via `/auth/refresh` |

JWT payload structure:
```json
{
  "sub": "user-uuid",
  "exp": 1770000000,
  "type": "access"  // or "refresh"
}
```

---

## Endpoints

### 1. Signup

```
POST /api/v1/auth/signup
Content-Type: application/json
```

**Request body:**
```json
{
  "email": "user@example.com",
  "username": "johndoe",
  "password": "mypassword123"
}
```

**Validation rules:**
- `email`: valid email format, unique
- `username`: 3–100 characters, unique
- `password`: 6–128 characters

**Success response** `201 Created`:
```json
{
  "access_token": "eyJhbGciOi...",
  "refresh_token": "eyJhbGciOi...",
  "token_type": "bearer"
}
```

**Error responses:**

| Status | Detail |
|--------|--------|
| `409` | `"Email already registered."` |
| `409` | `"Username already taken."` |
| `422` | Validation error (missing/invalid fields) |

---

### 2. Login

```
POST /api/v1/auth/login
Content-Type: application/json
```

**Request body:**
```json
{
  "email": "user@example.com",
  "password": "mypassword123"
}
```

**Success response** `200 OK`:
```json
{
  "access_token": "eyJhbGciOi...",
  "refresh_token": "eyJhbGciOi...",
  "token_type": "bearer"
}
```

**Error responses:**

| Status | Detail |
|--------|--------|
| `401` | `"Invalid email or password."` |
| `401` | `"User account is deactivated."` |
| `422` | Validation error |

---

### 3. Refresh Token

```
POST /api/v1/auth/refresh
Content-Type: application/json
```

**Request body:**
```json
{
  "refresh_token": "eyJhbGciOi..."
}
```

**Success response** `200 OK`:
```json
{
  "access_token": "eyJhbGciOi...",
  "refresh_token": "eyJhbGciOi...",
  "token_type": "bearer"
}
```

**Error responses:**

| Status | Detail |
|--------|--------|
| `401` | `"Invalid or expired refresh token."` |
| `401` | `"Token is not a refresh token."` |
| `401` | `"User not found or deactivated."` |

---

### 4. Get Current User

```
GET /api/v1/auth/me
Authorization: Bearer <access_token>
```

**Success response** `200 OK`:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "user@example.com",
  "username": "johndoe",
  "is_active": true,
  "created_at": "2026-02-05T12:00:00+00:00"
}
```

**Error responses:**

| Status | Detail |
|--------|--------|
| `401` | `"Invalid or expired token."` |
| `401` | `"User not found."` |
| `401` | `"User account is deactivated."` |

---

## Flutter Implementation Guide

### 1. Store tokens securely

Use `flutter_secure_storage`:

```dart
final storage = FlutterSecureStorage();

Future<void> saveTokens(String access, String refresh) async {
  await storage.write(key: 'access_token', value: access);
  await storage.write(key: 'refresh_token', value: refresh);
}

Future<String?> getAccessToken() async {
  return await storage.read(key: 'access_token');
}

Future<String?> getRefreshToken() async {
  return await storage.read(key: 'refresh_token');
}
```

### 2. Signup

```dart
Future<void> signup(String email, String username, String password) async {
  final response = await http.post(
    Uri.parse('$baseUrl/api/v1/auth/signup'),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode({
      'email': email,
      'username': username,
      'password': password,
    }),
  );

  if (response.statusCode == 201) {
    final data = jsonDecode(response.body);
    await saveTokens(data['access_token'], data['refresh_token']);
  } else {
    final error = jsonDecode(response.body);
    throw Exception(error['detail']);
  }
}
```

### 3. Login

```dart
Future<void> login(String email, String password) async {
  final response = await http.post(
    Uri.parse('$baseUrl/api/v1/auth/login'),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode({
      'email': email,
      'password': password,
    }),
  );

  if (response.statusCode == 200) {
    final data = jsonDecode(response.body);
    await saveTokens(data['access_token'], data['refresh_token']);
  } else {
    final error = jsonDecode(response.body);
    throw Exception(error['detail']);
  }
}
```

### 4. Authenticated requests

```dart
Future<http.Response> authenticatedGet(String path) async {
  final token = await getAccessToken();
  final response = await http.get(
    Uri.parse('$baseUrl$path'),
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer $token',
    },
  );

  if (response.statusCode == 401) {
    final refreshed = await refreshTokens();
    if (refreshed) {
      final newToken = await getAccessToken();
      return await http.get(
        Uri.parse('$baseUrl$path'),
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer $newToken',
        },
      );
    }
  }
  return response;
}
```

### 5. Token refresh

```dart
Future<bool> refreshTokens() async {
  final refreshToken = await getRefreshToken();
  if (refreshToken == null) return false;

  final response = await http.post(
    Uri.parse('$baseUrl/api/v1/auth/refresh'),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode({'refresh_token': refreshToken}),
  );

  if (response.statusCode == 200) {
    final data = jsonDecode(response.body);
    await saveTokens(data['access_token'], data['refresh_token']);
    return true;
  }
  return false;
}
```

### 6. Logout (client-side only)

```dart
Future<void> logout() async {
  await storage.delete(key: 'access_token');
  await storage.delete(key: 'refresh_token');
}
```

---

## Error Handling Summary

All error responses follow this format:
```json
{
  "detail": "Human-readable error message"
}
```

| Status Code | Meaning |
|-------------|---------|
| `200` | Success |
| `201` | Created |
| `401` | Unauthorized (bad credentials or expired token) |
| `409` | Conflict (email/username taken) |
| `422` | Validation error (missing or invalid fields) |

---

## CORS

CORS is fully open in dev (`allow_origins=["*"]`). No preflight issues from any origin.
