# WatermarkAPI

## Overview

WatermarkAPI is a FastAPI-based audio watermarking service that provides endpoints for embedding, corrupting, and decoding audio watermarks. It supports multiple watermarking algorithms and attack transformations, with optional user-based data isolation and audio file downloads.

**Key Features:**
- Embed watermarks into audio files using WavMark or AudioSeal algorithms
- Apply attack transformations (MP3, cutting, noise, speed, volume, low-pass filter)
- Decode and verify watermarks from original or corrupted audio
- User-based data isolation (store per-user files separately)
- Return audio files directly in HTTP responses
- Interactive API documentation at `/docs`

## Installation

### Prerequisites
- Python 3.9 or higher
- PyTorch and torchaudio (CUDA 12.6 or CPU variant)

### Setup

1. **Install dependencies** from `requirements.txt`:
```bash
pip install -r requirements.txt
```

**Main dependencies:**
- `fastapi==0.136.1` — Web framework
- `uvicorn==0.47.0` — ASGI server
- `pydantic==2.13.4` — Request/response validation
- `torch==2.12.0+cu126` — ML framework for watermarking
- `torchaudio==2.11.0+cu126` — Audio processing
- `audioseal==0.2.0` — AudioSeal watermarking algorithm
- `wavmark==0.0.3` — WavMark watermarking algorithm
- `python-dotenv==1.2.2` — Environment configuration

2. **Configure environment variables** in `.env`:
```bash
API_KEY=your_hugging_face_token_here
```

The `API_KEY` is required for accessing HuggingFace models used by the watermarking algorithms.

## Running the Server

### Option 1: Direct Python execution
```bash
python master.py
```

### Option 2: Using Uvicorn directly
```bash
uvicorn master:app --reload --host 0.0.0.0 --port 8000
```

### Option 3: With custom settings
```bash
uvicorn master:app --host 0.0.0.0 --port 8000 --workers 4
```

The server will start on `http://127.0.0.1:8000`

**Interactive API documentation:** `http://127.0.0.1:8000/docs`

## API Endpoints

### Health Check

#### `GET /health`
Returns a simple health check.

**Response:**
```json
{
  "status": "ok"
}
```

**Example:**
```bash
curl http://127.0.0.1:8000/health
```

---

### Configuration Management

#### `GET /config`
Retrieve the current runtime configuration.

**Response:**
```json
{
  "config": {
    "input_file": "input.wav",
    "watermarked_file": "watermarked.wav",
    "corrupted_file": "corrupted.wav",
    "secret_file": "secret.npy",
    "algorithm": "wavmark",
    "attack_mp3": false,
    "mp3_bitrate": 128.0,
    "attack_cut": false,
    "cut_duration": 4.0,
    "attack_speed": false,
    "speed_factor": 1.15,
    "attack_noise": false,
    "noise_level": 0.02,
    "attack_lpf": false,
    "lpf_cutoff": 4000.0,
    "attack_volume": false,
    "volume_factor": 0.5,
    "simulate_platform": "none"
  }
}
```

**Example:**
```bash
curl http://127.0.0.1:8000/config
```

#### `POST /config`
Update runtime configuration. Only supplied keys are modified.

**Request Body:**
```json
{
  "algorithm": "audioseal",
  "attack_mp3": true,
  "mp3_bitrate": 64.0,
  "user_id": "user123"
}
```

**Supported config keys:**
- `algorithm` — `"wavmark"` or `"audioseal"`
- `attack_mp3` — Enable MP3 compression attack (boolean)
- `mp3_bitrate` — MP3 bitrate in kbps (float)
- `attack_cut` — Enable audio cutting attack (boolean)
- `cut_duration` — Cut duration in seconds (float)
- `attack_speed` — Enable speed change attack (boolean)
- `speed_factor` — Speed multiplication factor (float)
- `attack_noise` — Enable noise injection (boolean)
- `noise_level` — Noise amplitude (float)
- `attack_lpf` — Enable low-pass filter attack (boolean)
- `lpf_cutoff` — LPF cutoff frequency in Hz (float)
- `attack_volume` — Enable volume change (boolean)
- `volume_factor` — Volume multiplication factor (float)
- `simulate_platform` — Simulate platform compression: `"none"`, `"youtube"`, `"instagram"`, or `"whatsapp"`

**Example:**
```bash
curl.exe -X POST "http://127.0.0.1:8000/config" \
  -H "Content-Type: application/json" \
  -d '{"algorithm":"audioseal","attack_mp3":true,"mp3_bitrate":64}'
```

---

### Watermark Encoding

#### `POST /encode`
Embed a watermark into an audio file.

**Request Body:**
```json
{
  "user_id": "user123",
  "return_file": true,
  "input_file": "input.wav",
  "watermarked_file": "output.wav",
  "secret_file": "secret.npy",
  "algorithm": "wavmark"
}
```

**Parameters:**
- `user_id` (optional) — User identifier for data isolation
- `return_file` (optional) — Return WAV file directly in response
- `input_file` (optional) — Path to input audio (default: `input.wav`)
- `watermarked_file` (optional) — Output file path (default: `watermarked.wav`)
- `secret_file` (optional) — Secret key file path (default: `secret.npy`)
- `algorithm` (optional) — `"wavmark"` or `"audioseal"`

**Response (with return_file=true):**
Binary WAV audio file

**Response (with return_file=false):**
```json
{
  "success": true,
  "config": {
    "input_file": "input.wav",
    "watermarked_file": "watermarked.wav",
    ...
  }
}
```

**Examples:**

*Upload and encode audio, return file:*
```bash
curl.exe -X POST "http://127.0.0.1:8000/encode" \
  -F "input_file=@input.wav" \
  -d user_id=user123 \
  -d return_file=true \
  --output watermarked.wav
```

*Encode existing file, return JSON:*
```bash
curl.exe -X POST "http://127.0.0.1:8000/encode" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user123","algorithm":"audioseal"}' \
  | jq
```

---

### Audio Corruption

#### `POST /corrupt`
Apply attack transformations to a watermarked audio file.

**Request Body:**
```json
{
  "user_id": "user123",
  "return_file": true,
  "watermarked_file": "watermarked.wav",
  "corrupted_file": "corrupted.wav",
  "attack_mp3": true,
  "mp3_bitrate": 128,
  "attack_noise": true,
  "noise_level": 0.05
}
```

**Parameters:**
- Same as config POST, plus:
- `watermarked_file` (optional) — Input watermarked file (default: `watermarked.wav`)
- `corrupted_file` (optional) — Output file path (default: `corrupted.wav`)

**Response:** Binary WAV file (if `return_file=true`) or JSON success response

**Examples:**

*Apply MP3 compression and noise:*
```bash
curl.exe -X POST "http://127.0.0.1:8000/corrupt" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id":"user123",
    "return_file":true,
    "attack_mp3":true,
    "mp3_bitrate":64,
    "attack_noise":true,
    "noise_level":0.05
  }' \
  --output corrupted.wav
```

*Upload watermarked file and corrupt:*
```bash
curl.exe -X POST "http://127.0.0.1:8000/corrupt" \
  -F "watermarked_file=@watermarked.wav" \
  -d user_id=user123 \
  -d attack_mp3=true \
  -d return_file=true \
  --output corrupted.wav
```

---

### Watermark Decoding

#### `POST /decode`
Decode watermark from a corrupted audio file.

**Request Body:**
```json
{
  "user_id": "user123",
  "corrupted_file": "corrupted.wav",
  "secret_file": "secret.npy",
  "algorithm": "wavmark"
}
```

**Parameters:**
- `user_id` (optional) — User identifier
- `corrupted_file` (optional) — File to decode (default: `corrupted.wav`)
- `target_file` (optional) — Alternative file path to decode
- `secret_file` (optional) — Secret key file path (default: `secret.npy`)
- `algorithm` (optional) — `"wavmark"` or `"audioseal"`

**Response:**
```json
{
  "success": true,
  "result": {
    "detected": true,
    "confidence": 0.95,
    "bit_error_rate": 0.02
  }
}
```

**Examples:**

*Decode from stored file:*
```bash
curl.exe -X POST "http://127.0.0.1:8000/decode" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user123"}' | jq
```

*Upload corrupted file and decode:*
```bash
curl.exe -X POST "http://127.0.0.1:8000/decode" \
  -F "corrupted_file=@corrupted.wav" \
  -d user_id=user123 | jq
```

#### `POST /decode_original`
Decode watermark from the original watermarked file (without corruption).

**Request Body:**
```json
{
  "user_id": "user123",
  "watermarked_file": "watermarked.wav",
  "secret_file": "secret.npy",
  "algorithm": "wavmark"
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "detected": true,
    "confidence": 1.0,
    "bit_error_rate": 0.0
  }
}
```

**Example:**
```bash
curl.exe -X POST "http://127.0.0.1:8000/decode_original" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user123"}' | jq
```

---

## File Structure

```
.
├── master.py                    # Entry point, runs FastAPI server
├── requirements.txt             # Python dependencies
├── .env                         # Environment config (API_KEY required)
├── README.md                    # This file
│
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app definition & routes
│   ├── models/
│   │   └── schemas.py           # Pydantic request/response schemas
│   └── services/
│       ├── encode_service.py    # Watermark embedding logic
│       ├── corrupt_service.py   # Attack transformation logic
│       ├── decode_service.py    # Watermark extraction logic
│       └── user_service.py      # User data management
│
├── user_data/                   # User-specific files (created at runtime)
│   ├── anonymous/
│   │   ├── watermarked.wav
│   │   ├── corrupted.wav
│   │   └── secret.npy
│   └── user123/
│       ├── watermarked.wav
│       └── ...
```

## User-Based Data Isolation

All endpoints support optional `user_id` parameter for isolating user data:

```bash
# Request for user123 — files stored in user_data/user123/
curl.exe -X POST "http://127.0.0.1:8000/encode" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user123","return_file":true}' \
  --output watermarked.wav

# Request without user_id — files stored in user_data/anonymous/
curl.exe -X POST "http://127.0.0.1:8000/encode" \
  -H "Content-Type: application/json" \
  -d '{"return_file":true}' \
  --output watermarked.wav
```

Files are automatically isolated in `user_data/{user_id}/` subdirectories.

## Complete Workflow Example

### 1. Encode watermark into audio
```bash
curl.exe -X POST "http://127.0.0.1:8000/encode" \
  -F "input_file=@myaudio.wav" \
  -d user_id=john \
  -d algorithm=audioseal \
  -d return_file=true \
  --output watermarked.wav
```

### 2. Check configuration
```bash
curl http://127.0.0.1:8000/config | jq
```

### 3. Apply attacks (MP3 + noise)
```bash
curl.exe -X POST "http://127.0.0.1:8000/config" \
  -H "Content-Type: application/json" \
  -d '{"attack_mp3":true,"mp3_bitrate":64,"attack_noise":true,"noise_level":0.05}'
```

### 4. Corrupt the watermarked file
```bash
curl.exe -X POST "http://127.0.0.1:8000/corrupt" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"john","return_file":true}' \
  --output corrupted.wav
```

### 5. Verify watermark in corrupted file
```bash
curl.exe -X POST "http://127.0.0.1:8000/decode" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"john"}' | jq '.result'
```

### 6. Verify watermark in original
```bash
curl.exe -X POST "http://127.0.0.1:8000/decode_original" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"john"}' | jq '.result'
```

## Windows PowerShell Notes

PowerShell aliases `curl` to `Invoke-WebRequest`. For consistent behavior, use `curl.exe`:

```powershell
# Good — uses actual curl
curl.exe -X POST "http://127.0.0.1:8000/encode" \
  -H "Content-Type: application/json" \
  -d '{"return_file":true}' \
  --output watermarked.wav

# Alternative — use Invoke-WebRequest directly
$body = @{return_file=$true} | ConvertTo-Json
Invoke-WebRequest `
  -Uri "http://127.0.0.1:8000/encode" `
  -Method POST `
  -Body $body `
  -ContentType "application/json" `
  -OutFile watermarked.wav
```

## Troubleshooting

### API won't start — "API_KEY is missing"
- Create `.env` file with `API_KEY=your_token`
- Get a HuggingFace API key from https://huggingface.co/settings/tokens

### JSON validation errors
- Ensure all property names use double quotes: `{"key":"value"}`
- Use `curl.exe` (not `curl`) on Windows PowerShell
- Check request body syntax with `jq`

### Downloaded file is tiny/corrupt
- File may contain an error message instead of audio
- Check response with `curl` without `--output` to see error
- Verify `return_file=true` in request

### Watermark detection fails
- Ensure same algorithm used for encode and decode
- Verify `secret.npy` file exists for the user
- Try on `/decode_original` first (should have high confidence)
- Check corruption settings aren't too aggressive

### Files stored in wrong location
- Use `user_id` parameter for per-user isolation
- Check file paths with `GET /config`
- Verify `user_data/` directory permissions

## Development & Live Reload

Run with auto-reload on code changes:
```bash
uvicorn master:app --reload --host 0.0.0.0 --port 8000
```

Interactive API docs (SwaggerUI):
```
http://127.0.0.1:8000/docs
```

Alternative ReDoc documentation:
```
http://127.0.0.1:8000/redoc
```
