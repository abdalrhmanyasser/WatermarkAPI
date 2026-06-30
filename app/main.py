import asyncio
import json
import os
import shutil
import subprocess
import uuid
import numpy as np
from typing import Any, Dict, Literal
from datetime import datetime
import torch

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile, status, Body, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from bson import ObjectId
import secrets

from app.services.corrupt_service import corrupt_audio
from app.services.decode_service import test_watermark
from app.services.encode_service import encode_audio
from app.services.progress_manager import progress_tracker
from auth import get_current_user_from_jwt, verify_api_key, create_access_token, get_password_hash, verify_password
from database import get_db

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("API_KEY")

app = FastAPI(title="Stateless Audio Watermark API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

gpu_lock = asyncio.Lock()
MAX_FILE_SIZE = 50 * 1024 * 1024 
ALLOWED_AUDIO_TYPES = ["audio/wav", "audio/x-wav", "audio/vnd.wave", "audio/mpeg", "audio/mp3"]
ALLOWED_EXTENSIONS = (".wav", ".mp3")

def cleanup_temp_dir(dir_path: str):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path, ignore_errors=True)

def convert_to_wav(input_path: str, output_path: str) -> bool:
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", output_path]
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE).returncode == 0

async def validate_and_save_upload(upload_file: UploadFile, dest_path: str):
    if not (upload_file.content_type in ALLOWED_AUDIO_TYPES or upload_file.filename.lower().endswith(ALLOWED_EXTENSIONS)):
        raise HTTPException(status_code=400, detail="Invalid file type.")
    file_bytes = await upload_file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large.")
    with open(dest_path, "wb") as f:
        f.write(file_bytes)

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/register", status_code=201)
def register(payload: dict = Body(...), db = Depends(get_db)):
    full_name = payload.get("full_name")
    email = payload.get("email")
    password = payload.get("password")
    
    # Require the new full_name field
    if not email or not password or not full_name: 
        raise HTTPException(400, "Full name, email, and password required")
    if db.users.find_one({"email": email}): 
        raise HTTPException(409, "Email exists")
    
    user_doc = {
        "full_name": full_name,
        "email": email, 
        "hashed_password": get_password_hash(password)
    }
    res = db.users.insert_one(user_doc)
    return {
        "id": str(res.inserted_id), 
        "full_name": full_name, 
        "email": email, 
        "message": "User created."
    }

@app.post("/token")
def token(payload: dict = Body(...), db = Depends(get_db)):
    user = db.users.find_one({"email": payload.get("email")})
    if not user or not verify_password(payload.get("password"), user["hashed_password"]):
        raise HTTPException(401, "Invalid credentials")
    return {"access_token": create_access_token({"sub": str(user["_id"])}), "token_type": "bearer", "expires_in": 3600}

@app.post("/keys", status_code=201)
def create_api_key(body: dict = Body(None), current_user: dict = Depends(get_current_user_from_jwt), db = Depends(get_db)):
    name, scopes = (body.get("name"), body.get("scopes")) if body else (None, None)
    prefix, secret = f"ogs_{secrets.token_hex(6)}", secrets.token_urlsafe(32)
    
    key_doc = {
        "user_id": current_user["_id"],
        "key_prefix": prefix,
        "hashed_secret": get_password_hash(secret),
        "name": name,
        "scopes": scopes,
        "created_at": datetime.utcnow(),
        "last_used": None
    }
    res = db.api_keys.insert_one(key_doc)
    return {"api_key": f"{prefix}.{secret}", "key_id": str(res.inserted_id), "message": "Store securely"}

@app.get("/keys")
def list_api_keys(current_user: dict = Depends(get_current_user_from_jwt), db = Depends(get_db)):
    keys = db.api_keys.find({"user_id": current_user["_id"]})
    return [{
        "key_id": str(k["_id"]), 
        "key_prefix": k["key_prefix"], 
        "name": k.get("name"), 
        "scopes": k.get("scopes"), 
        "created_at": k["created_at"].isoformat() if k.get("created_at") else None, 
        "last_used": k["last_used"].isoformat() if k.get("last_used") else None
    } for k in keys]

@app.delete("/keys/{key_id}", status_code=204)
def revoke_api_key(key_id: str, current_user: dict = Depends(get_current_user_from_jwt), db = Depends(get_db)):
    try:
        obj_id = ObjectId(key_id)
    except:
        raise HTTPException(400, "Invalid Key ID format")
        
    result = db.api_keys.delete_one({"_id": obj_id, "user_id": current_user["_id"]})
    if result.deleted_count == 0: 
        raise HTTPException(404, "API key not found")
    return Response(status_code=204)

@app.post("/corrupt")
async def corrupt(background_tasks: BackgroundTasks, watermarked_file: UploadFile = File(...), attack_mp3: bool = Form(False), mp3_bitrate: float = Form(128.0), attack_cut: bool = Form(False), cut_duration: float = Form(4.0), attack_speed: bool = Form(False), speed_factor: float = Form(1.15), attack_noise: bool = Form(False), noise_level: float = Form(0.02), attack_lpf: bool = Form(False), lpf_cutoff: float = Form(4000.0), attack_volume: bool = Form(False), volume_factor: float = Form(0.5), attack_hpf: bool = Form(False), hpf_cutoff: float = Form(500.0), attack_clipping: bool = Form(False), clip_threshold: float = Form(0.5), attack_resample: bool = Form(False), downsample_sr: int = Form(8000), attack_echo: bool = Form(False), echo_delay_ms: int = Form(200), echo_decay: float = Form(0.5), attack_pitch: bool = Form(False), pitch_steps: int = Form(2), simulate_platform: Literal["none", "youtube", "instagram", "whatsapp", "tiktok"] = Form("none"), current_user: dict = Depends(get_current_user_from_jwt)):
    req_id = str(uuid.uuid4())
    temp_dir = os.path.join(os.getcwd(), "temp_workspaces", req_id)
    os.makedirs(temp_dir, exist_ok=True)
    background_tasks.add_task(cleanup_temp_dir, temp_dir)
    in_path, out_path = os.path.join(temp_dir, "watermarked.wav"), os.path.join(temp_dir, "corrupted.wav")
    await validate_and_save_upload(watermarked_file, in_path)
    
    async with gpu_lock:
        if not await run_in_threadpool(corrupt_audio, input_file=in_path, output_file=out_path, attack_mp3=attack_mp3, mp3_bitrate=mp3_bitrate, attack_cut=attack_cut, cut_duration=cut_duration, attack_speed=attack_speed, speed_factor=speed_factor, attack_noise=attack_noise, noise_level=noise_level, attack_lpf=attack_lpf, lpf_cutoff=lpf_cutoff, attack_volume=attack_volume, volume_factor=volume_factor, attack_hpf=attack_hpf, hpf_cutoff=hpf_cutoff, attack_clipping=attack_clipping, clip_threshold=clip_threshold, attack_resample=attack_resample, downsample_sr=downsample_sr, attack_echo=attack_echo, echo_delay_ms=echo_delay_ms, echo_decay=echo_decay, attack_pitch=attack_pitch, pitch_steps=pitch_steps, simulate_platform=simulate_platform):
            raise HTTPException(500, "Corruption failed")
    return FileResponse(out_path, media_type="audio/wav", filename="corrupted.wav")

def run_encode_worker(task_id: str, temp_dir: str, in_path: str, out_path: str, algorithm: str, user_id: str):
    success = False
    try:
        with torch.no_grad():
            # Pass a numeric hash of the ObjectId for the payload embedding
            numeric_user_id = int(str(user_id)[:8], 16) % 65535 
            success = encode_audio(in_path, out_path, algorithm, numeric_user_id, task_id)
        if success:
            progress_tracker.update_progress(task_id, 100, "Encoding Successful")
            progress_tracker.complete_task(task_id, {"status": "Ready"})
        else:
            progress_tracker.update_progress(task_id, -1, "Encoding failed")
    except Exception as e:
        progress_tracker.update_progress(task_id, -1, f"Crash: {e}")
    finally:
        if not success: cleanup_temp_dir(temp_dir)

@app.post("/encode")
async def start_encode(background_tasks: BackgroundTasks, input_file: UploadFile = File(...), algorithm: Literal["wavmark", "audioseal", "qim", "dwt_svd", "echo", "lsb", "phase", "hybrid"] = Form("wavmark"), current_user: dict = Depends(verify_api_key)):
    task_id = str(uuid.uuid4())
    temp_dir = os.path.join(os.getcwd(), "temp_workspaces", task_id)
    os.makedirs(temp_dir, exist_ok=True)
    in_path, out_path = os.path.join(temp_dir, "input.wav"), os.path.join(temp_dir, "watermarked.wav")
    await validate_and_save_upload(input_file, in_path)
    progress_tracker.update_progress(task_id, 0, "Queued")
    background_tasks.add_task(run_encode_worker, task_id, temp_dir, in_path, out_path, algorithm, current_user["_id"])
    return {"task_id": task_id}

@app.get("/encode/progress/{task_id}")
def check_encode_progress(task_id: str, current_user: dict = Depends(verify_api_key)):
    state = progress_tracker.get_status(task_id)
    if state["status"] == "Not Found": raise HTTPException(404, "Not found")
    if state["progress"] == -1: raise HTTPException(500, state["status"])
    return state

@app.get("/encode/download/{task_id}")
def download_encoded(task_id: str, background_tasks: BackgroundTasks, current_user: dict = Depends(verify_api_key)):
    if progress_tracker.get_status(task_id)["status"] != "Completed": raise HTTPException(400, "Not ready")
    temp_dir = os.path.join(os.getcwd(), "temp_workspaces", task_id)
    out_path = os.path.join(temp_dir, "watermarked.wav")
    if not os.path.exists(out_path): raise HTTPException(404, "Lost")
    background_tasks.add_task(cleanup_temp_dir, temp_dir)
    return FileResponse(out_path, media_type="audio/wav", filename=f"watermarked.wav")

def run_decode_worker(task_id: str, temp_dir: str, target_path: str, algorithm: str):
    try:
        with torch.no_grad(): 
            result = test_watermark(target_path, algorithm, task_id)
            
        if result: 
            db = next(get_db())
            owner_name = "Unknown User"
            
            # Find the user whose ObjectId hashes to the decoded 16-bit integer
            for user in db.users.find():
                numeric_hash = int(str(user["_id"])[:8], 16) % 65535
                if numeric_hash == result["owner_id"]:
                    # Grab the full_name directly from the database record
                    owner_name = user.get("full_name", "Unknown User")
                    break
                    
            # Swap out the raw integer for the user's actual Full Name
            result["owner_id"] = owner_name 
            
            progress_tracker.complete_task(task_id, result)
        else: 
            progress_tracker.update_progress(task_id, -1, "Decode failed")
    except Exception as e:
        progress_tracker.update_progress(task_id, -1, f"Crash: {e}")
    finally:
        cleanup_temp_dir(temp_dir)

@app.post("/decode")
async def start_decode(background_tasks: BackgroundTasks, algorithm: Literal["wavmark", "audioseal", "qim", "dwt_svd", "echo", "lsb", "phase", "hybrid"] = Form(...), corrupted_file: UploadFile = File(...), current_user: dict = Depends(verify_api_key)):
    task_id = str(uuid.uuid4())
    temp_dir = os.path.join(os.getcwd(), "temp_workspaces", task_id)
    os.makedirs(temp_dir, exist_ok=True)
    raw_path, target_path = os.path.join(temp_dir, "target.raw"), os.path.join(temp_dir, "target.wav")
    await validate_and_save_upload(corrupted_file, raw_path)
    if not convert_to_wav(raw_path, target_path): raise HTTPException(500, "Conversion fail")
    if os.path.exists(raw_path): os.remove(raw_path)
    progress_tracker.update_progress(task_id, 0, "Queued")
    background_tasks.add_task(run_decode_worker, task_id, temp_dir, target_path, algorithm)
    return {"task_id": task_id}

@app.get("/decode/progress/{task_id}")
def check_decode_progress(task_id: str, current_user: dict = Depends(verify_api_key)):
    state = progress_tracker.get_status(task_id)
    if state["status"] == "Not Found": raise HTTPException(404, "Not found")
    if state["progress"] == -1: raise HTTPException(500, state["status"])
    return state

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, reload_excludes=["temp_workspaces/*", "*.wav", "*.npy"])