# OGSeal API Reference (v1) - PowerShell Edition

This document describes the complete OGSeal HTTP API. It uses a **Dual-Authentication** model:

1. **JWT (JSON Web Tokens)** for human account management.
2. **Long-lived API Keys** for machine access to heavy GPU endpoints.

**Base URL**

- Production: `https://api.ogseal.example.com`
- Local development: `http://localhost:8000`

---

## Part 1: User Management (JWT Required)

These endpoints manage accounts and API keys. They require a standard JSON payload and return JSON.

### 1. Register a New Account

Create a new user account to access the dashboard and generate API keys. Requires full name.

- **Endpoint:** `POST /register`
- **Content-Type:** `application/json`

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/register" `
    -Method Post `
    -Headers @{"Content-Type"="application/json"} `
    -Body '{"full_name": "Abd Alrhman", "email": "user@example.com", "password": "hunter2"}'
```

### 2. Login (Get JWT Token)

Exchange your credentials for a short-lived JSON Web Token.

- **Endpoint:** `POST /token`
- **Content-Type:** `application/json`

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/token" `
    -Method Post `
    -Headers @{"Content-Type"="application/json"} `
    -Body '{"email": "user@example.com", "password": "hunter2"}'
```

_(Save the `access_token` from the response. You will use it as ` $JWT_TOKEN` below)._

### 3. Create an API Key

Generate a long-lived API key for machine-to-machine communication.

- **Endpoint:** `POST /keys`
- **Headers:** `Authorization: Bearer  $JWT_TOKEN`
- **Content-Type:** `application/json`

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/keys" `
    -Method Post `
    -Headers @{
        "Authorization"="Bearer  $JWT_TOKEN"
        "Content-Type"="application/json"
    } `
    -Body '{"name": "android-worker-1", "scopes": ["encode", "decode", "corrupt"]}'
```

**CRITICAL:** Save the `api_key` string returned here (e.g., `ogs_ab12cd.XyZ...`). It is shown only once and will be used as ` $RAW_API_KEY` for all core endpoints.

### 4. List API Keys

View metadata for all active API keys.

- **Endpoint:** `GET /keys`
- **Headers:** `Authorization: Bearer  $JWT_TOKEN`

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/keys" `
    -Method Get `
    -Headers @{"Authorization"="Bearer  $JWT_TOKEN"}
```

### 5. Revoke an API Key

Delete an API key immediately.

- **Endpoint:** `DELETE /keys/{key_id}`
- **Headers:** `Authorization: Bearer  $JWT_TOKEN`

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/keys/42" `
    -Method Delete `
    -Headers @{"Authorization"="Bearer  $JWT_TOKEN"}
```

---

## Part 2: Core Processing (Machine API Key Required)

These endpoints handle heavy audio processing. They use `multipart/form-data` for file uploads and require your raw API key in the `X-API-Key` header.

**Watermarks statelessly carry the User ID directly within the audio payload.**

### 6. Encode Watermark (Step 1)

Upload an audio file and assign a worker thread to embed the digital signature.
_Valid algorithms: `wavmark`, `audioseal`, `qim`, `dwt_svd`, `echo`, `lsb`, `phase`, `hybrid`_

- **Endpoint:** `POST /encode`
- **Headers:** `X-API-Key:  $RAW_API_KEY`
- **Content-Type:** `multipart/form-data`

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/encode" `
    -Method Post `
    -Headers @{"X-API-Key"=" $RAW_API_KEY"} `
    -Form @{
        input_file = Get-Item -Path ".udio.wav"
        algorithm = "wavmark"
    }
```

_(Save the `task_id` from the response)._

### 7. Check Encode Progress (Step 2)

Poll this endpoint to track background encoding.

- **Endpoint:** `GET /encode/progress/{task_id}`
- **Headers:** `X-API-Key:  $RAW_API_KEY`

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/encode/progress/<TASK_ID>" `
    -Method Get `
    -Headers @{"X-API-Key"=" $RAW_API_KEY"}
```

### 8. Download Encoded Audio (Step 3)

Retrieve the final watermarked binary `.wav` file once progress hits 100%.

- **Endpoint:** `GET /encode/download/{task_id}`
- **Headers:** `X-API-Key:  $RAW_API_KEY`

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/encode/download/<TASK_ID>" `
    -Method Get `
    -Headers @{"X-API-Key"=" $RAW_API_KEY"} `
    -OutFile "watermarked.wav"
```

---

### 9. Corrupt Audio (Attack Pipeline)

Apply synchronous signal degradation to test watermark resilience.

- **Endpoint:** `POST /corrupt`
- **Headers:** `X-API-Key:  $RAW_API_KEY`
- **Content-Type:** `multipart/form-data`
- **Available Flags:** `attack_mp3`, `mp3_bitrate`, `attack_noise`, `noise_level`, `simulate_platform` (`youtube`, `instagram`, `whatsapp`, `tiktok`, `none`), etc.

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/corrupt" `
    -Method Post `
    -Headers @{"X-API-Key"=" $RAW_API_KEY"} `
    -Form @{
        watermarked_file = Get-Item -Path ".\watermarked.wav"
        simulate_platform = "instagram"
        attack_noise = "true"
        noise_level = "0.02"
    } `
    -OutFile "corrupted.wav"
```

---

### 10. Decode Watermark (Step 1)

Upload a potentially damaged audio file to extract the ownership ID statelessly.

- **Endpoint:** `POST /decode`
- **Headers:** `X-API-Key:  $RAW_API_KEY`
- **Content-Type:** `multipart/form-data`

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/decode" `
    -Method Post `
    -Headers @{"X-API-Key"=" $RAW_API_KEY"} `
    -Form @{
        corrupted_file = Get-Item -Path ".\corrupted.wav"
        algorithm = "wavmark"
    }
```

_(Save the `task_id` from the response)._

### 11. Check Decode Progress & Result (Step 2)

Poll this endpoint to extract operational telemetry and the final decoded owner name.

- **Endpoint:** `GET /decode/progress/{task_id}`
- **Headers:** `X-API-Key:  $RAW_API_KEY`

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/decode/progress/<TASK_ID>" `
    -Method Get `
    -Headers @{"X-API-Key"=" $RAW_API_KEY"}
```

**Example Success Response:**

```json
{
  "progress": 100,
  "status": "Completed",
  "result": {
    "owner_id": "Abd Alrhman"
  }
}
```

---

## Error Codes

- **400 Bad Request:** Missing or invalid fields, invalid file formats.
- **401 Unauthorized:** Missing or invalid JWT (for account routes) or `X-API-Key` (for core routes).
- **403 Forbidden:** Attempting to access or modify another user's resources.
- **404 Not Found:** Invalid `task_id` or `key_id`.
- **413 Payload Too Large:** Audio file exceeds the 50MB limit.
- **500 Internal Server Error:** Server crash or FFmpeg pipeline failure.
