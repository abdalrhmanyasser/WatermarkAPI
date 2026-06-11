# Watermark API Reference

Welcome to the WatermarkPro API documentation. This API provides asynchronous, GPU-accelerated endpoints for embedding and extracting cryptographic watermarks in audio files.

## Base URL
For local Android Emulator testing, use:
`http://10.0.2.2:8000`

## Authentication
All API endpoints require an API key to be passed in the headers.

**Header Key:** `X-API-Key`  
**Value:** `<Your_Raw_API_Key>`

---

## 1. The Asynchronous Task Pattern
Because audio processing relies on heavy DSP and neural networks, the `/encode` and `/decode` endpoints operate asynchronously to prevent HTTP timeouts. 

**The standard workflow is:**
1. **Initiate:** Send a `POST` request with the audio file. The server immediately returns a `task_id`.
2. **Track (Poll):** Set up a coroutine loop to send a `GET` request to the `/progress/{task_id}` endpoint every **1500ms**. Update your UI with the returned `progress` integer and `status` string.
3. **Complete:** Once `progress == 100`, retrieve the final JSON payload or trigger the file download.

---

## 2. Encode Workflow (3-Step Pattern)

### A. Initiate Encode
Uploads a high-resolution WAV file and spins up a background GPU worker to embed the payload.

* **Endpoint:** `POST /encode`
* **Content-Type:** `multipart/form-data`
* **Form Data:**
  * `input_file` (File): The `.wav` audio file.
  * `algorithm` (String): e.g., "wavmark", "audioseal", "qim", "hybrid".

**Success Response (200 OK):**
{
  "task_id": "uuid-string-here"
}

### B. Track Encode Progress
Used to power the frontend loading animation.

* **Endpoint:** `GET /encode/progress/{task_id}`

**Response (200 OK):**
{
  "progress": 85,
  "status": "Mixing and exporting studio quality audio...",
  "result": null
}

*Note: When `progress == 100`, the `result` object will populate with `{"secret_id": <int>}`.*

### C. Download Watermarked Audio
Once progress hits 100%, call this to download the resulting file. **The server will permanently delete the temporary file immediately after this download finishes.**

* **Endpoint:** `GET /encode/download/{task_id}`
* **Returns:** Binary `audio/wav` file stream.
* **Important Headers:** The response headers will include `X-Secret-ID`. You must save this ID locally; it is required for decoding.

---

## 3. Decode Workflow (2-Step Pattern)

### A. Initiate Decode
Uploads a corrupted audio file and the Secret ID to begin extraction.

* **Endpoint:** `POST /decode`
* **Content-Type:** `multipart/form-data`
* **Form Data:**
  * `corrupted_file` (File): The `.wav` file to analyze.
  * `secret_id` (Integer): The ID obtained during the encode phase.

**Success Response (200 OK):**
{
  "task_id": "uuid-string-here"
}

### B. Track Decode & Get Results
Poll this endpoint every 1500ms.

* **Endpoint:** `GET /decode/progress/{task_id}`

**Response while processing (200 OK):**
{
  "progress": 40,
  "status": "Running neural network pass...",
  "result": null
}

**Response upon completion (200 OK):**
{
  "progress": 100,
  "status": "Completed",
  "result": {
    "algorithm": "hybrid",
    "original_secret": [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1],
    "decoded_secret": [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1],
    "accuracy": 100.0,
    "is_watermarked": true,
    "confidence": "BOTH DETECTED (Sync: 100.0%, WavMark: 100.0%)"
  }
}

---

## Error Handling
The API uses standard HTTP status codes:
* `400 Bad Request`: Missing parameters or trying to download an incomplete task.
* `401 Unauthorized`: Missing or invalid `X-API-Key`.
* `404 Not Found`: Task ID expired, file lost, or Secret ID does not belong to the user.
* `500 Internal Server Error`: Background worker crashed (check the `status` field in the progress endpoint for the exception trace).
