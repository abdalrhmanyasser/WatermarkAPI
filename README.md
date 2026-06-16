# OGSeal Audio Watermarking API

## Overview

OGSeal is a high-performance, FastAPI-based audio watermarking service providing endpoints for embedding, attacking (corrupting), and decoding cryptographic digital signatures inside audio files. It natively wraps heavy DSP routines and Deep Learning neural frameworks into safe background workers to avoid HTTP connection timeouts.

**Key Features:**
- **Dual-Authentication Architecture:** JWT-based human account management alongside highly secure, bcrypt-hashed long-lived API keys for machine access.
- **Asynchronous GPU-Accelerated Embedding:** Uses multiple digital signature algorithms (WavMark, AudioSeal, QIM, Echo, Phase, Hybrid, etc.).
- **Robust Database Tracking:** Persists cryptographic secrets and API key metadata securely per registered user.
- **Audio Attack Pipeline:** Supports up to 11 distinct signal degradations or simulated social media compression parameters (e.g., MP3 compression, clipping, echo, pitch shifts, TikTok/WhatsApp simulation).
- **Active GPU Out-of-Memory Mitigation:** Utilizes global execution locks during intensive model inference passes.

---

## Installation

### Prerequisites
- Python 3.9 or higher
- FFmpeg installed and added to your system's `PATH` variable

### Setup

1. **Install dependencies** from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables** in a `.env` file at the project root:
   ```env
   API_KEY=your_hugging_face_token_here
   DATABASE_URL=sqlite:///watermarks.db
   ```
   *Note: `API_KEY` maps to your HuggingFace token, which is essential to fetch and stream core model weights for neural algorithms.*

3. **Initialize an Administrator / User Account:**
   Use the CLI script to create your first human account (which will let you generate API keys later):
   ```bash
   python create_user.py your_email@example.com your_secure_password
   ```

---

## Running the Server

### Development Mode (With Auto-Reload)
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Execution
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

Interactive API documentation will spin up at: `http://127.0.0.1:8000/docs`

---

## Authentication Overview

OGSeal utilizes a **Dual-Authentication** model separating user management from heavy machine workloads:

1. **Human Management (JWT):**
   Used for dashboard/account endpoints (`/register`, `/token`, `/keys`). 
   - Get a token via `POST /token`.
   - Pass it via header: `Authorization: Bearer <JWT>`

2. **Machine Processing (API Key):**
   Used for core GPU endpoints (`/encode`, `/decode`, `/corrupt`).
   - Generate via `POST /keys` (requires JWT).
   - Pass it via header: `X-API-Key: <RAW_API_KEY>`

---

## Core API Endpoints

### 1. User Management (JWT Required)

- **`POST /register`**: Register a new user account (Requires `email` and `password` in JSON body).
- **`POST /token`**: Exchange email/password for a short-lived JWT.
- **`POST /keys`**: Create a new API key for machine access. Returns the raw API key *once*.
- **`GET /keys`**: List metadata for all active API keys.
- **`DELETE /keys/{key_id}`**: Revoke a specific API key.

### 2. Watermark Encoding (API Key Required)

#### `POST /encode`
Uploads a high-resolution file and spins up a background thread worker to stamp the digital payload.
- **Content-Type:** `multipart/form-data`
- **Form Parameters:** `input_file` (Binary), `algorithm` (String: `wavmark`, `audioseal`, `qim`, `hybrid`, etc.)
- **Response:** Returns a `task_id`.

#### `GET /encode/progress/{task_id}`
Poll this endpoint to track encoding progress. Returns `secret_id` upon 100% completion.

#### `GET /encode/download/{task_id}`
Triggers binary output transfer.
- **Returns:** Native `audio/wav` binary file stream.
- **Headers:** Includes `X-Secret-ID`. Save this locally; it is mandatory for extraction/decoding later.

### 3. Audio Corruption Pipeline (API Key Required)

#### `POST /corrupt`
Applies signal degradation filters using underlying FFmpeg processes. Synchronous action.
- **Content-Type:** `multipart/form-data`
- **Form Parameters:** `watermarked_file` (Binary), plus various attack flags (`attack_mp3`, `attack_noise`, `attack_echo`, `simulate_platform`, etc.).
- **Response:** Binary `.wav` audio output.

### 4. Watermark Decoding (API Key Required)

#### `POST /decode`
Upload damaged audio files along with their matching database `secret_id` to trace signal verification metrics.
- **Content-Type:** `multipart/form-data`
- **Form Parameters:** `corrupted_file` (Binary), `secret_id` (Integer)
- **Response:** Returns a `task_id`.

#### `GET /decode/progress/{task_id}`
Poll this path to extract operational telemetry or complete payloads.
- **Completed Response:** Includes decoded secret, accuracy metrics, and confidence scores.

---

## Repository Structure

```
.
├── main.py                      # API definition, routers, validation hooks
├── auth.py                      # Dual JWT and API Key authentication middleware
├── database.py                  # SQLAlchemy engine initialization maps
├── models.py                    # Database schema mapping models (User, ApiKey, WatermarkSecret)
├── create_user.py               # Token deployment pipeline CLI script
├── app/
│   ├── models/
│   │   └── schemas.py           # Fallback Pydantic definitions
│   └── services/
│       ├── encode_service.py    # Multi-algorithm target encoder handlers
│       ├── corrupt_service.py   # FFmpeg signal degradation filters
│       ├── decode_service.py    # Matrix distance calculations & trace model decoders
│       └── progress_manager.py  # Thread-safe thread tracking memory maps
├── requirements.txt             # Project dependencies 
└── .env                         # Local deployment configuration parameters
```
