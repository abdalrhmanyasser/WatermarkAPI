import os
from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import Any, Dict

from encode import encode_audio
from corrupt import corrupt_audio
from decode import test_watermark

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

app = FastAPI(title="Audio Watermark API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/config")
def get_config() -> Dict[str, Any]:
    return {"config": CONFIG}


@app.post("/config")
def set_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        validate_config(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    CONFIG.update({k: v for k, v in payload.items() if k in VALID_CONFIG_KEYS})
    return {"updated_config": CONFIG}


@app.post("/encode")
def encode(payload: Dict[str, Any] = {}) -> Any:
    config = merge_request_config(payload)
    success = encode_audio(
        config["input_file"],
        config["watermarked_file"],
        config["secret_file"],
        config["algorithm"],
    )
    if not success:
        raise HTTPException(status_code=500, detail="Encoding failed")

    if payload.get("return_file", False):
        file_path = config["watermarked_file"]
        if not os.path.exists(file_path):
            raise HTTPException(status_code=500, detail=f"Encoded file not found: {file_path}")
        return FileResponse(file_path, media_type="audio/wav", filename=os.path.basename(file_path))

    return {"success": True, "config": config}


@app.post("/corrupt")
def corrupt(payload: Dict[str, Any] = {}) -> Any:
    config = merge_request_config(payload)
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

    if payload.get("return_file", False):
        file_path = config["corrupted_file"]
        if not os.path.exists(file_path):
            raise HTTPException(status_code=500, detail=f"Corrupted file not found: {file_path}")
        return FileResponse(file_path, media_type="audio/wav", filename=os.path.basename(file_path))

    return {"success": True, "config": config}


@app.post("/decode")
def decode(payload: Dict[str, Any] = {}) -> Dict[str, Any]:
    config = merge_request_config(payload)
    target_file = payload.get("target_file", config["corrupted_file"])
    result = test_watermark(target_file, config["secret_file"], config["algorithm"])
    if result is None:
        raise HTTPException(status_code=500, detail="Decode failed")
    return {"success": True, "result": result}


@app.post("/decode_original")
def decode_original(payload: Dict[str, Any] = {}) -> Dict[str, Any]:
    config = merge_request_config(payload)
    result = test_watermark(config["watermarked_file"], config["secret_file"], config["algorithm"])
    if result is None:
        raise HTTPException(status_code=500, detail="Decode original failed")
    return {"success": True, "result": result}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
