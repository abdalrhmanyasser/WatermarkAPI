from pydantic import BaseModel, Field
from typing import Literal, Optional


class EncodeRequest(BaseModel):
    return_file: bool = Field(default=False, description="Return the generated audio file instead of JSON")


class CorruptRequest(BaseModel):
    return_file: bool = Field(default=False, description="Return the generated audio file instead of JSON")


class DecodeRequest(BaseModel):
    pass


class DecodeOriginalRequest(BaseModel):
    pass


class ConfigRequest(BaseModel):
    input_file: Optional[str] = None
    watermarked_file: Optional[str] = None
    corrupted_file: Optional[str] = None
    secret_file: Optional[str] = None
    algorithm: Optional[Literal["wavmark", "audioseal"]] = None
    attack_mp3: Optional[bool] = None
    mp3_bitrate: Optional[float] = None
    attack_cut: Optional[bool] = None
    cut_duration: Optional[float] = None
    attack_speed: Optional[bool] = None
    speed_factor: Optional[float] = None
    attack_noise: Optional[bool] = None
    noise_level: Optional[float] = None
    attack_lpf: Optional[bool] = None
    lpf_cutoff: Optional[float] = None
    attack_volume: Optional[bool] = None
    volume_factor: Optional[float] = None
    simulate_platform: Optional[Literal["none", "youtube", "instagram", "whatsapp"]] = None
