import os
from dotenv import load_dotenv
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import Any, Dict, Optional

from app.models.schemas import (
    ConfigRequest,
    CorruptRequest,
    DecodeOriginalRequest,
    DecodeRequest,
    EncodeRequest,
)
from app.services.corrupt_service import corrupt_audio
from app.services.decode_service import test_watermark
from app.services.encode_service import encode_audio
from app.services.user_service import append_user_secret, ensure_data_root

# Loads the variables from the .env file into your environment
load_dotenv()

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["HF_TOKEN"] = os.getenv("API_KEY")
if not os.getenv("API_KEY"):
    raise ValueError("API_KEY is missing from the environment!")

print("Token loaded successfully!")

CONFIG: Dict[str, Any] = {
    "input_file": "input.wav",
    "watermarked_file": "watermarked.wav",
    "corrupted_file": "corrupted.wav",
    "secret_file": "secret.npy",
    "algorithm": "wavmark",
    "attack_mp3": False,
    "mp3_bitrate": 128.0,
    "attack_cut": False,
    "cut_duration": 4.0,
    "attack_speed": False,
    "speed_factor": 1.15,
    "attack_noise": False,
    "noise_level": 0.02,
    "attack_lpf": False,
    "lpf_cutoff": 4000.0,
    "attack_volume": False,
    "volume_factor": 0.5,
    "simulate_platform": "none",
}

VALID_CONFIG_KEYS = set(CONFIG.keys())
STORAGE_ROOT = os.path.join(os.getcwd(), "user_data")


def get_user_storage(user_id: Optional[str]) -> str:
    if not user_id:
        return STORAGE_ROOT
    return os.path.join(STORAGE_ROOT, user_id)


def ensure_user_storage(user_id: Optional[str]) -> str:
    root = get_user_storage(user_id)
    os.makedirs(root, exist_ok=True)
    return root


def resolve_user_path(path: Optional[str], user_id: Optional[str]) -> Optional[str]:
    if not path or not user_id:
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(get_user_storage(user_id), path)


def apply_user_paths(config: Dict[str, Any], user_id: Optional[str]) -> Dict[str, Any]:
    if not user_id:
        return config

    ensure_user_storage(user_id)
    updated = config.copy()
    for key in ["input_file", "watermarked_file", "corrupted_file", "secret_file"]:
        if key not in updated:
            continue
        value = updated[key]
        if not value:
            continue
        if os.path.isabs(value):
            continue
        updated[key] = os.path.join(get_user_storage(user_id), value)
    return updated


def merge_request_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    merged = CONFIG.copy()
    for key, value in payload.items():
        if key in VALID_CONFIG_KEYS:
            merged[key] = value
    return merged


def validate_config(payload: Dict[str, Any]) -> None:
    unknown_keys = [k for k in payload.keys() if k not in VALID_CONFIG_KEYS]
    if unknown_keys:
        raise ValueError(f"Unknown config keys: {unknown_keys}")
    if "algorithm" in payload and payload["algorithm"] not in {"wavmark", "audioseal"}:
        raise ValueError("algorithm must be 'wavmark' or 'audioseal'")
    if "simulate_platform" in payload and payload["simulate_platform"] not in {"none", "youtube", "instagram", "whatsapp"}:
        raise ValueError("simulate_platform must be one of 'none', 'youtube', 'instagram', 'whatsapp'")


app = FastAPI(title="Audio Watermark API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/config")
def get_config() -> Dict[str, Any]:
    return {"config": CONFIG}


@app.post(
    "/config",
    summary="Update runtime watermarking settings",
    description="Update the active pipeline configuration. Only supplied values are changed.",
)
def set_config(payload: ConfigRequest = Body(default=ConfigRequest())) -> Dict[str, Any]:
    values = payload.dict(exclude_none=True)
    try:
        validate_config(values)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    CONFIG.update({k: v for k, v in values.items() if k in VALID_CONFIG_KEYS})
    return {"updated_config": CONFIG}


@app.post(
    "/encode",
    summary="Encode a watermark into audio",
    description="Embed a watermark into an input WAV file and optionally return the generated watermarked audio.",
)
async def encode(
    payload: EncodeRequest = Body(default=EncodeRequest()),
    input_file: Optional[UploadFile] = File(default=None),
) -> Any:
    config = merge_request_config(payload.dict(exclude_none=True))
    config = apply_user_paths(config, payload.user_id)

    if input_file:
        os.makedirs(os.path.dirname(config["input_file"]), exist_ok=True)
        with open(config["input_file"], "wb") as opened:
            opened.write(await input_file.read())

    success = encode_audio(
        config["input_file"],
        config["watermarked_file"],
        config["secret_file"],
        config["algorithm"],
    )
    if not success:
        raise HTTPException(status_code=500, detail="Encoding failed")

    append_user_secret(payload.user_id or "anonymous", config["secret_file"], "<secret-not-logged>")

    if payload.return_file:
        file_path = config["watermarked_file"]
        if not os.path.exists(file_path):
            raise HTTPException(status_code=500, detail=f"Encoded file not found: {file_path}")
        return FileResponse(file_path, media_type="audio/wav", filename=os.path.basename(file_path))

    return {"success": True, "config": config}


@app.post(
    "/corrupt",
    summary="Corrupt the watermarked audio",
    description="Apply attack transformations to the watermarked audio and optionally return the corrupted WAV file.",
)
async def corrupt(
    payload: CorruptRequest = Body(default=CorruptRequest()),
    watermarked_file: Optional[UploadFile] = File(default=None),
) -> Any:
    config = merge_request_config(payload.dict(exclude_none=True))
    config = apply_user_paths(config, payload.user_id)

    if watermarked_file:
        os.makedirs(os.path.dirname(config["watermarked_file"]), exist_ok=True)
        with open(config["watermarked_file"], "wb") as opened:
            opened.write(await watermarked_file.read())

    success = corrupt_audio(
        input_file=config["watermarked_file"],
        output_file=config["corrupted_file"],
        attack_mp3=config["attack_mp3"],
        mp3_bitrate=config["mp3_bitrate"],
        attack_cut=config["attack_cut"],
        cut_duration=config["cut_duration"],
        attack_speed=config["attack_speed"],
        speed_factor=config["speed_factor"],
        attack_noise=config["attack_noise"],
        noise_level=config["noise_level"],
        attack_lpf=config["attack_lpf"],
        lpf_cutoff=config["lpf_cutoff"],
        attack_volume=config["attack_volume"],
        volume_factor=config["volume_factor"],
        simulate_platform=config["simulate_platform"],
    )
    if not success:
        raise HTTPException(status_code=500, detail="Corruption failed")

    if payload.return_file:
        file_path = config["corrupted_file"]
        if not os.path.exists(file_path):
            raise HTTPException(status_code=500, detail=f"Corrupted file not found: {file_path}")
        return FileResponse(file_path, media_type="audio/wav", filename=os.path.basename(file_path))

    return {"success": True, "config": config}


@app.post(
    "/decode",
    summary="Decode a watermark from a corrupted file",
    description="Decode the watermark from the specified target file, defaulting to the current corrupted file.",
)
async def decode(
    payload: DecodeRequest = Body(default=DecodeRequest()),
    corrupted_file: Optional[UploadFile] = File(default=None),
) -> Dict[str, Any]:
    config = merge_request_config(payload.dict(exclude_none=True))
    config = apply_user_paths(config, payload.user_id)

    if corrupted_file:
        os.makedirs(os.path.dirname(config["corrupted_file"]), exist_ok=True)
        with open(config["corrupted_file"], "wb") as opened:
            opened.write(await corrupted_file.read())
        target_file = config["corrupted_file"]
    else:
        target_file = payload.target_file or config["corrupted_file"]
        target_file = resolve_user_path(target_file, payload.user_id)

    result = test_watermark(target_file, config["secret_file"], config["algorithm"])
    if result is None:
        raise HTTPException(status_code=500, detail="Decode failed")
    return {"success": True, "result": result}


@app.post(
    "/decode_original",
    summary="Decode the watermark from the original watermarked file",
    description="Decode the watermark from the most recently created watermarked audio file.",
)
async def decode_original(
    payload: DecodeOriginalRequest = Body(default=DecodeOriginalRequest()),
    watermarked_file: Optional[UploadFile] = File(default=None),
) -> Dict[str, Any]:
    config = merge_request_config(payload.dict(exclude_none=True))
    config = apply_user_paths(config, payload.user_id)

    if watermarked_file:
        os.makedirs(os.path.dirname(config["watermarked_file"]), exist_ok=True)
        with open(config["watermarked_file"], "wb") as opened:
            opened.write(await watermarked_file.read())

    result = test_watermark(config["watermarked_file"], config["secret_file"], config["algorithm"])
    if result is None:
        raise HTTPException(status_code=500, detail="Decode original failed")
    return {"success": True, "result": result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
