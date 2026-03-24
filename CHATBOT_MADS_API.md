# MADS CRM Chatbot API

**Base URL:** `https://api.oleyes.com/api/v1/chatbot-mads`

---

## POST `/chat`

Send a message to the MADS CRM support chatbot and receive an AI-powered reply based on the full MADS CRM application documentation.

### Request

**Headers:**

| Header | Value |
|---|---|
| `Content-Type` | `application/json` |

**Body:**

```json
{
  "message": "How do I create a new client?"
}
```

| Field | Type | Required | Max Length | Description |
|---|---|---|---|---|
| `message` | string | Yes | 2000 chars | The user's question in any language |

### Response (200 OK)

```json
{
  "reply": "Tap the '+ New Client' button at the top of your dashboard. This opens the 6-step client creation wizard."
}
```

| Field | Type | Description |
|---|---|---|
| `reply` | string | The chatbot's answer |

### Error Responses

| Code | Body | When |
|---|---|---|
| `422` | `{"detail": [...]}` | Missing or empty `message` field |
| `502` | `{"detail": "LLM API error: ..."}` | AI service error or empty response |
| `504` | `{"detail": "LLM request timed out."}` | AI request took longer than 60 seconds |

---

## Examples

### cURL

```bash
curl -X POST https://api.oleyes.com/api/v1/chatbot-mads/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I upload signed documents?"}'
```

### JavaScript / TypeScript (Fetch)

```typescript
const response = await fetch("https://api.oleyes.com/api/v1/chatbot-mads/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ message: "How do I change my password?" }),
});

const data = await response.json();
console.log(data.reply);
```

### Python (requests)

```python
import requests

response = requests.post(
    "https://api.oleyes.com/api/v1/chatbot-mads/chat",
    json={"message": "What does Redo status mean?"},
)

print(response.json()["reply"])
```

### Dart / Flutter

```dart
final response = await http.post(
  Uri.parse('https://api.oleyes.com/api/v1/chatbot-mads/chat'),
  headers: {'Content-Type': 'application/json'},
  body: jsonEncode({'message': 'How do I sign a document?'}),
);

final data = jsonDecode(response.body);
print(data['reply']);
```

---

## Behavior

- The chatbot answers **only** based on the MADS CRM application documentation.
- If the answer is not in the documentation, it replies: *"I don't have that information. Please go to Profile > Contact us for support."*
- The chatbot responds in **the same language** the user writes in (Spanish, French, English, Arabic, etc.).
- No authentication is required to use this endpoint.
