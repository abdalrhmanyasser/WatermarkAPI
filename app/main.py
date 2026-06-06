import asyncio
import json
import os
import shutil
import uuid
import numpy as np
from typing import Any, Dict, Literal

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.services.corrupt_service import corrupt_audio
from app.services.decode_service import test_watermark
from app.services.encode_service import encode_audio
from auth import get_current_user
from database import get_db, init_db
from models import User, WatermarkSecret


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

@app.post("/encode")
async def encode(
    background_tasks: BackgroundTasks,
    input_file: UploadFile = File(...),
    # Added all 6 new algorithms to the type hint:
    algorithm: Literal["wavmark", "audioseal", "qim", "dwt_svd", "echo", "lsb", "phase", "hybrid"] = Form("wavmark"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    # 1. Setup secure temporary workspace
    req_id = str(uuid.uuid4())
    temp_dir = os.path.join(os.getcwd(), "temp_workspaces", req_id)
    os.makedirs(temp_dir, exist_ok=True)
    background_tasks.add_task(cleanup_temp_dir, temp_dir)

    in_path = os.path.join(temp_dir, "input.wav")
    out_path = os.path.join(temp_dir, "watermarked.wav")
    secret_path = os.path.join(temp_dir, "secret.npy")

    # 2. Validate and process file
    await validate_and_save_upload(input_file, in_path)

    # 3. Process on GPU securely using Lock and Threadpool
    async with gpu_lock:
        success = await run_in_threadpool(
            encode_audio,
            input_file=in_path,
            output_file=out_path,
            secret_file=secret_path,
            algorithm=algorithm,
        )
        
    if not success:
        raise HTTPException(status_code=500, detail="Encoding failed")

    # 4. Read numpy secret array and save to SQLite
    payload_np = np.load(secret_path)
    payload_json = json.dumps(payload_np.tolist())

    secret_record = WatermarkSecret(
        user_id=current_user.id,
        algorithm=algorithm,
        payload=payload_json,
    )
    db.add(secret_record)
    db.commit()
    db.refresh(secret_record)

    # 5. Return the processed file directly, attaching the Secret ID to the headers
    return FileResponse(
        out_path, 
        media_type="audio/wav", 
        filename=f"watermarked_{secret_record.id}.wav",
        headers={"X-Secret-ID": str(secret_record.id)}
    )

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
    # --- NEW FORM FIELDS ADDED ---
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
    current_user: User = Depends(get_current_user),
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


@app.post("/decode")
async def decode(
    background_tasks: BackgroundTasks,
    secret_id: int = Form(...),
    corrupted_file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    
    # 1. Verify user owns this secret
    secret_record = db.query(WatermarkSecret).filter(
        WatermarkSecret.id == secret_id,
        WatermarkSecret.user_id == current_user.id
    ).first()
    
    if not secret_record:
        raise HTTPException(status_code=404, detail="Secret ID not found or access denied.")

    # 2. Setup secure temporary workspace
    req_id = str(uuid.uuid4())
    temp_dir = os.path.join(os.getcwd(), "temp_workspaces", req_id)
    os.makedirs(temp_dir, exist_ok=True)
    background_tasks.add_task(cleanup_temp_dir, temp_dir)

    target_path = os.path.join(temp_dir, "target.wav")
    secret_path = os.path.join(temp_dir, "secret.npy")

    # 3. Rebuild secret.npy from the Database JSON
    payload_list = json.loads(secret_record.payload)
    payload_np = np.array(payload_list, dtype=np.int32)
    np.save(secret_path, payload_np)

    # 4. Validate and process file
    await validate_and_save_upload(corrupted_file, target_path)

    # 5. Process on GPU securely using Lock and Threadpool
    async with gpu_lock:
        result = await run_in_threadpool(
            test_watermark, 
            target_path, 
            secret_path, 
            secret_record.algorithm
        )
        
    if result is None:
        raise HTTPException(status_code=500, detail="Decode failed")
        
    return {"success": True, "result": result}


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