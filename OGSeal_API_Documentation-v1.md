# OGSeal API Reference (v1)

This document describes the OGSeal HTTP API used by the Android app (Jetpack Compose client). It uses a Dual-Authentication model: JWT for human account management, and long-lived API keys for machine access to heavy GPU endpoints.

Base URL

- Production: `https://api.ogseal.example.com`
- Local development: `http://localhost:8000`

Authentication Overview

- Human (JWT): `POST /token` with email/password returns a short-lived JWT. Use `Authorization: Bearer <JWT>` for dashboard/account endpoints such as `POST /keys`, `GET /keys`, `DELETE /keys/{key_id}`.
- Machine (API Key): Long-lived raw API keys in the format `<prefix>.<secret>` (e.g., `ogs_3f7a21.K9x...`). Use the raw key in the header `X-API-Key: <RAW_API_KEY>` for core GPU endpoints (`/encode`, `/decode`, and their progress/download endpoints).

Key lifecycle

- Create: `POST /keys` (JWT required) returns the raw API key once.
- List: `GET /keys` (JWT required) returns a JSON array of key metadata. Raw secrets are never returned.
- Revoke: `DELETE /keys/{key_id}` (JWT required) removes the key.

---

User Management Endpoints (Human JWT)

1) Register

- Endpoint: `POST /register`
- Body (application/json): `{ "email": "user@example.com", "password": "hunter2" }`
- Response (201): `{ "id": 1, "email": "user@example.com", "message": "User created." }`

2) Token (Login)

- Endpoint: `POST /token`
- Body (application/json): `{ "email": "user@example.com", "password": "hunter2" }`
- Response (200): `{ "access_token": "<JWT>", "token_type": "bearer", "expires_in": 3600 }`

3) Create API Key

- Endpoint: `POST /keys`
- Headers: `Authorization: Bearer <JWT>`
- Body (optional): `{ "name": "android-worker-1", "scopes": ["encode","decode"] }`
- Response (201): `{ "api_key": "ogs_ab12cd.XyZ...", "key_id": 42, "message": "Store this raw API key securely — it will not be shown again." }`

Notes: Save the `api_key` value securely; it will not be shown again.

---

List API Keys

- Endpoint: `GET /keys`
- Headers: `Authorization: Bearer <JWT>`
- Response (200): JSON array with objects containing metadata for each key. Example:

[
  {
    "key_id": 42,
    "key_prefix": "ogs_ab12cd",
    "name": "android-worker-1",
    "scopes": ["encode","decode"],
    "created_at": "2026-06-13T12:34:56.789012",
    "last_used": "2026-06-13T12:40:00.123456"
  }
]

Notes: The returned objects do NOT include the raw secret. Use `key_id` for revocation.

---

Revoke (Delete) API Key

- Endpoint: `DELETE /keys/{key_id}`
- Headers: `Authorization: Bearer <JWT>`
- Path param: `key_id` (integer)
- Success Response: `204 No Content` (empty body)
- Error (404): `{ "detail": "API key not found" }` if key does not belong to the user or does not exist.

---

Core Processing Endpoints (Machine API Key)

All core endpoints accept only `X-API-Key: <RAW_API_KEY>` in the header. They will return `401 Unauthorized` if missing or invalid.

Encode (Asynchronous, 3 steps)

1) POST /encode
- Headers: `X-API-Key: <RAW_API_KEY>`
- Content-Type: multipart/form-data
- Form fields: `input_file` (file), `algorithm` (string)
- Response (200): `{ "task_id": "uuid-string" }`

2) GET /encode/progress/{task_id}
- Headers: `X-API-Key: <RAW_API_KEY>`
- Response example while processing:
  `{ "task_id": "...", "progress": 42, "status": "Working...", "result": null }`
- On completion `progress == 100`, `result` contains `{ "secret_id": <int> }`.

3) GET /encode/download/{task_id}
- Headers: `X-API-Key: <RAW_API_KEY>`
- Returns: audio/wav binary stream. Response header includes `X-Secret-ID: <secret_id>`.

Decode (Asynchronous, 2 steps)

1) POST /decode
- Headers: `X-API-Key: <RAW_API_KEY>`
- Content-Type: multipart/form-data
- Form fields: `corrupted_file` (file), `secret_id` (int)
- Response (200): `{ "task_id": "uuid-string" }`

2) GET /decode/progress/{task_id}
- Headers: `X-API-Key: <RAW_API_KEY>`
- Response on completion includes `result` with original/decoded secret and accuracy.

---

Error codes

- 400 Bad Request — missing or invalid fields.
- 401 Unauthorized — missing/invalid JWT for account routes or missing/invalid `X-API-Key` for core routes.
- 403 Forbidden — attempting to access or modify another user's resources.
- 404 Not Found — invalid `task_id`, `secret_id`, or `key_id`.
- 500 Internal Server Error — server error; check response `status` field in progress endpoints.

---

Example curl commands

GET /keys (list keys)

```bash
curl -s -X GET "http://localhost:8000/keys" \
  -H "Authorization: Bearer <JWT_TOKEN>"
# Response: JSON array of key metadata (see List API Keys section)
```

DELETE /keys/{key_id} (revoke)

```bash
curl -s -X DELETE "http://localhost:8000/keys/42" \
  -H "Authorization: Bearer <JWT_TOKEN>" -v
# Success: HTTP 204 No Content
```

Notes:
- Replace `http://localhost:8000` with your production base URL.
- Replace `<JWT_TOKEN>` with the `access_token` value from `POST /token`.
- The Android client should call `GET /keys` to display key metadata and call `DELETE /keys/{key_id}` to revoke a selected key.
