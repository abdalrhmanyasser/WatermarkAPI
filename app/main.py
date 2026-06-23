import asyncio
import json
import os
import shutil
import subprocess
import uuid
import numpy as np
from typing import Any, Dict, Literal
import torch

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile, status, Body, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.services.corrupt_service import corrupt_audio
from app.services.decode_service import test_watermark
from app.services.encode_service import encode_audio
from app.services.progress_manager import progress_tracker
from auth import get_current_user_from_jwt, verify_api_key, create_access_token, get_password_hash, verify_password
from database import get_db, init_db, SessionLocal
from models import User, WatermarkSecret, ApiKey
from sqlalchemy.orm import Session
import secrets


load_dotenv()
# os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["HF_TOKEN"] = os.getenv("API_KEY")
if not os.getenv("API_KEY"):
    print("HuggingFace API_KEY is missing from the environment!")

# Initialize Database on Startup
init_db()

app = FastAPI(title="Stateless Audio Watermark API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Lock to prevent GPU Out-of-Memory crashes
gpu_lock = asyncio.Lock()

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB limit
ALLOWED_AUDIO_TYPES = ["audio/wav", "audio/x-wav", "audio/vnd.wave", "audio/mpeg", "audio/mp3"]
ALLOWED_EXTENSIONS = (".wav", ".mp3")


def cleanup_temp_dir(dir_path: str):
    """Deletes temporary files after the HTTP response is sent."""
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path, ignore_errors=True)
        print(f"Cleaned up temporary workspace: {dir_path}")


def convert_to_wav(input_path: str, output_path: str) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-ac",
        "1",
        "-ar",
        "48000",
        "-c:a",
        "pcm_s16le",
        output_path,
    ]
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if res.returncode != 0:
        err = res.stderr.decode('utf-8', errors='replace')
        print(f"[decode] ffmpeg conversion failed: {err}")
        return False
    return True


async def validate_and_save_upload(upload_file: UploadFile, dest_path: str):
    """Validates the file type and size, then saves it to disk."""
    # Check both the MIME type and the file extension
    is_valid_mime = upload_file.content_type in ALLOWED_AUDIO_TYPES
    is_valid_ext = upload_file.filename.lower().endswith(ALLOWED_EXTENSIONS)
    
    if not (is_valid_mime or is_valid_ext):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Client sent: '{upload_file.content_type}'. Only WAV/MP3 allowed."
        )
    
    file_bytes = await upload_file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 50MB.")
        
    with open(dest_path, "wb") as f:
        f.write(file_bytes)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/register", status_code=201)
def register(payload: dict = Body(...), db: Session = Depends(get_db)):
    """Create a new user account. Body: {"email":..., "password":...}"""
    email = payload.get("email")
    password = payload.get("password")
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")
    hashed = get_password_hash(password)
    user = User(email=email, hashed_password=hashed)
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"id": user.id, "email": user.email, "message": "User created."}


@app.post("/token")
def token(payload: dict = Body(...), db: Session = Depends(get_db)):
    """Exchange email/password for a JWT. Body: {"email":..., "password":...}"""
    email = payload.get("email")
    password = payload.get("password")
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": str(user.id)})
    return {"access_token": token, "token_type": "bearer", "expires_in": 3600}


@app.post("/keys", status_code=201)
def create_api_key(body: dict = Body(None), current_user: User = Depends(get_current_user_from_jwt), db: Session = Depends(get_db)):
    """Create a new API key for machine access. Requires JWT in Authorization header."""
    name = None
    scopes = None
    if body:
        name = body.get("name")
        scopes = body.get("scopes")
    # Key format: <prefix>.<secret>
    prefix = f"ogs_{secrets.token_hex(6)}"
    secret = secrets.token_urlsafe(32)
    raw = f"{prefix}.{secret}"
    hashed = get_password_hash(secret)
    scopes_json = json.dumps(scopes) if scopes else None
    key = ApiKey(user_id=current_user.id, key_prefix=prefix, hashed_secret=hashed, name=name, scopes=scopes_json)
    db.add(key)
    db.commit()
    db.refresh(key)
    return {"api_key": raw, "key_id": key.id, "message": "Store this raw API key securely — it will not be shown again."}


@app.get("/keys")
def list_api_keys(current_user: User = Depends(get_current_user_from_jwt), db: Session = Depends(get_db)):
    """List API keys belonging to the authenticated user (JWT required).

    Response JSON array of objects:
    [ {"key_id": int, "key_prefix": str, "name": str|null, "scopes": list|null, "created_at": iso, "last_used": iso|null } ]
    """
    keys = db.query(ApiKey).filter(ApiKey.user_id == current_user.id).all()
    out = []
    for k in keys:
        try:
            scopes = json.loads(k.scopes) if k.scopes else None
        except Exception:
            scopes = None
        out.append({
            "key_id": k.id,
            "key_prefix": k.key_prefix,
            "name": k.name,
            "scopes": scopes,
            "created_at": k.created_at.isoformat() if k.created_at else None,
            "last_used": k.last_used.isoformat() if k.last_used else None,
        })
    return out


@app.delete("/keys/{key_id}", status_code=204)
def revoke_api_key(key_id: int, current_user: User = Depends(get_current_user_from_jwt), db: Session = Depends(get_db)):
    """Revoke (delete) an API key owned by the authenticated user. Returns 204 No Content on success."""
    key = db.query(ApiKey).filter(ApiKey.id == key_id, ApiKey.user_id == current_user.id).first()
    if not key:
        raise HTTPException(status_code=404, detail="API key not found")
    db.delete(key)
    db.commit()
    return Response(status_code=204)

def run_decode_worker(task_id: str, temp_dir: str, target_path: str, secret_path: str, algorithm: str):
    try:
        # The lock protects the GPU from Out Of Memory errors during background tasks
        with torch.no_grad(): 
            result = test_watermark(target_path, secret_path, algorithm, task_id=task_id)
        
        if result:
            progress_tracker.complete_task(task_id, result)
        else:
            progress_tracker.update_progress(task_id, -1, "Decode failed internally.")
    except Exception as e:
        progress_tracker.update_progress(task_id, -1, f"Crash: {str(e)}")
    finally:
        # Guarantee cleanup only happens AFTER the decoding is fully complete or crashes
        cleanup_temp_dir(temp_dir)

# NOTE: Duplicate /decode endpoint removed. The active decode endpoint
# lives further down in this file (it includes proper cleanup).

@app.post("/corrupt")
async def corrupt(
    background_tasks: BackgroundTasks,
    watermarked_file: UploadFile = File(...),
    attack_mp3: bool = Form(False),
    mp3_bitrate: float = Form(128.0),
    attack_cut: bool = Form(False),
    cut_duration: float = Form(4.0),
    attack_speed: bool = Form(False),
    speed_factor: float = Form(1.15),
    attack_noise: bool = Form(False),
    noise_level: float = Form(0.02),
    attack_lpf: bool = Form(False),
    lpf_cutoff: float = Form(4000.0),
    attack_volume: bool = Form(False),
    volume_factor: float = Form(0.5),
    attack_hpf: bool = Form(False),
    hpf_cutoff: float = Form(500.0),
    attack_clipping: bool = Form(False),
    clip_threshold: float = Form(0.5),
    attack_resample: bool = Form(False),
    downsample_sr: int = Form(8000),
    attack_echo: bool = Form(False),
    echo_delay_ms: int = Form(200),
    echo_decay: float = Form(0.5),
    attack_pitch: bool = Form(False),
    pitch_steps: int = Form(2),
    simulate_platform: Literal["none", "youtube", "instagram", "whatsapp", "tiktok"] = Form("none"),
    current_user: User = Depends(get_current_user_from_jwt),
) -> Any:
    # 1. Setup secure temporary workspace
    req_id = str(uuid.uuid4())
    temp_dir = os.path.join(os.getcwd(), "temp_workspaces", req_id)
    os.makedirs(temp_dir, exist_ok=True)
    background_tasks.add_task(cleanup_temp_dir, temp_dir)

    in_path = os.path.join(temp_dir, "watermarked.wav")
    out_path = os.path.join(temp_dir, "corrupted.wav")

    # 2. Validate and process file
    await validate_and_save_upload(watermarked_file, in_path)

    # 3. Process on GPU securely using Lock and Threadpool
    async with gpu_lock:
        success = await run_in_threadpool(
            corrupt_audio,
            input_file=in_path,
            output_file=out_path,
            attack_mp3=attack_mp3,
            mp3_bitrate=mp3_bitrate,
            attack_cut=attack_cut,
            cut_duration=cut_duration,
            attack_speed=attack_speed,
            speed_factor=speed_factor,
            attack_noise=attack_noise,
            noise_level=noise_level,
            attack_lpf=attack_lpf,
            lpf_cutoff=lpf_cutoff,
            attack_volume=attack_volume,
            volume_factor=volume_factor,
            attack_hpf=attack_hpf,
            hpf_cutoff=hpf_cutoff,
            attack_clipping=attack_clipping,
            clip_threshold=clip_threshold,
            attack_resample=attack_resample,
            downsample_sr=downsample_sr,
            attack_echo=attack_echo,
            echo_delay_ms=echo_delay_ms,
            echo_decay=echo_decay,
            attack_pitch=attack_pitch,
            pitch_steps=pitch_steps,
            simulate_platform=simulate_platform,
        )
        
    if not success:
        raise HTTPException(status_code=500, detail="Corruption failed")

    return FileResponse(out_path, media_type="audio/wav", filename="corrupted.wav")

def run_encode_worker(task_id: str, temp_dir: str, in_path: str, out_path: str, secret_path: str, algorithm: str, user_id: int):
    success = False
    try:
        with torch.no_grad():
            success = encode_audio(
                input_file=in_path,
                output_file=out_path,
                secret_file=secret_path,
                algorithm=algorithm,
                task_id=task_id
            )
            
        if success:
            progress_tracker.update_progress(task_id, 90, "Saving cryptographic secret to database...")
            db = SessionLocal()
            try:
                payload_np = np.load(secret_path)
                payload_json = json.dumps(payload_np.tolist())
                
                secret_record = WatermarkSecret(
                    user_id=user_id,
                    algorithm=algorithm,
                    payload=payload_json,
                )
                db.add(secret_record)
                db.commit()
                db.refresh(secret_record)
                
                progress_tracker.complete_task(task_id, {"secret_id": secret_record.id})
                
            finally:
                db.close()
        else:
            progress_tracker.update_progress(task_id, -1, "Encoding failed internally.")
            # If the process fails organically, we clean up the failed files.
            cleanup_temp_dir(temp_dir)
            
    except Exception as e:
        progress_tracker.update_progress(task_id, -1, f"Crash: {str(e)}")
        # If the process crashes violently, we also clean up.
        cleanup_temp_dir(temp_dir)


@app.post("/encode")
async def start_encode(
    background_tasks: BackgroundTasks,
    input_file: UploadFile = File(...),
    algorithm: Literal["wavmark", "audioseal", "qim", "dwt_svd", "echo", "lsb", "phase", "hybrid"] = Form("wavmark"),
    current_user: User = Depends(verify_api_key),
):
    task_id = str(uuid.uuid4())
    temp_dir = os.path.join(os.getcwd(), "temp_workspaces", task_id)
    os.makedirs(temp_dir, exist_ok=True)

    in_path = os.path.join(temp_dir, "input.wav")
    out_path = os.path.join(temp_dir, "watermarked.wav")
    secret_path = os.path.join(temp_dir, "secret.npy")

    await validate_and_save_upload(input_file, in_path)

    progress_tracker.update_progress(task_id, 0, "Audio queued for encoding...")
    
    # We pass the temp_dir so the worker can manage its own lifecycle based on success or failure
    background_tasks.add_task(
        run_encode_worker,
        task_id=task_id,
        temp_dir=temp_dir,
        in_path=in_path,
        out_path=out_path,
        secret_path=secret_path,
        algorithm=algorithm,
        user_id=current_user.id
    )
    
    return {"task_id": task_id}

@app.get("/encode/progress/{task_id}")
def check_encode_progress(task_id: str, current_user: User = Depends(verify_api_key)):
    task_state = progress_tracker.get_status(task_id)
    
    if task_state["status"] == "Not Found":
        raise HTTPException(status_code=404, detail="Task not found or expired.")
        
    if task_state["progress"] == -1:
        raise HTTPException(status_code=500, detail=task_state["status"])
        
    return task_state

@app.get("/encode/download/{task_id}")
def download_encoded(task_id: str, background_tasks: BackgroundTasks, current_user: User = Depends(verify_api_key)):
    task_state = progress_tracker.get_status(task_id)
    
    if task_state["status"] != "Completed":
        raise HTTPException(status_code=400, detail="Task not ready or failed.")
        
    secret_id = task_state["result"]["secret_id"]
    temp_dir = os.path.join(os.getcwd(), "temp_workspaces", task_id)
    out_path = os.path.join(temp_dir, "watermarked.wav")
    
    if not os.path.exists(out_path):
        raise HTTPException(status_code=404, detail="File lost or expired.")
        
    # FastAPI's background_tasks in a GET request execute strictly AFTER the HTTP response finishes sending.
    # This securely deletes the temp folder immediately after the Android app successfully finishes downloading the WAV.
    background_tasks.add_task(cleanup_temp_dir, temp_dir)
    
    return FileResponse(
        out_path, 
        media_type="audio/wav", 
        filename=f"watermarked_{secret_id}.wav",
        headers={"X-Secret-ID": str(secret_id)}
    )
@app.post("/decode")
async def start_decode(
    background_tasks: BackgroundTasks,
    secret_id: int = Form(...),
    corrupted_file: UploadFile = File(...),
    current_user: User = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    # 1. Verify user owns this secret
    secret_record = db.query(WatermarkSecret).filter(
        WatermarkSecret.id == secret_id,
        WatermarkSecret.user_id == current_user.id
    ).first()
    
    if not secret_record:
        raise HTTPException(status_code=404, detail="Secret ID not found or access denied.")

    # 2. Setup secure workspace and ID
    task_id = str(uuid.uuid4())
    temp_dir = os.path.join(os.getcwd(), "temp_workspaces", task_id)
    os.makedirs(temp_dir, exist_ok=True)

    raw_target_path = os.path.join(temp_dir, "target.raw")
    target_path = os.path.join(temp_dir, "target.wav")
    secret_path = os.path.join(temp_dir, "secret.npy")

    # 3. Rebuild secret.npy
    payload_list = json.loads(secret_record.payload)
    payload_np = np.array(payload_list, dtype=np.int32)
    np.save(secret_path, payload_np)

    # 4. Save upload and normalize to WAV
    await validate_and_save_upload(corrupted_file, raw_target_path)
    if not convert_to_wav(raw_target_path, target_path):
        raise HTTPException(status_code=500, detail="Could not convert uploaded audio to WAV for decoding.")
    if os.path.exists(raw_target_path):
        os.remove(raw_target_path)

    # 5. Initialize tracking and start background GPU task
    progress_tracker.update_progress(task_id, 0, "Audio queued for processing...")
    
    background_tasks.add_task(
        run_decode_worker, 
        task_id=task_id, 
        temp_dir=temp_dir,
        target_path=target_path, 
        secret_path=secret_path, 
        algorithm=secret_record.algorithm
    )
    
    # 6. Instantly return task_id to Android app
    return {"task_id": task_id}

# --- ADD THIS NEW ENDPOINT ---
@app.get("/decode/progress/{task_id}")
def check_decode_progress(task_id: str, current_user: User = Depends(verify_api_key)):
    task_state = progress_tracker.get_status(task_id)
    
    if task_state["status"] == "Not Found":
        raise HTTPException(status_code=404, detail="Task not found or expired.")
        
    if task_state["progress"] == -1:
        raise HTTPException(status_code=500, detail=task_state["status"])
        
    return task_state


if __name__ == "__main__":
    import uvicorn
    # Tell Uvicorn to ignore the temp folder when looking for code changes
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        reload_excludes=["temp_workspaces/*", "*.wav", "*.npy"]
    )