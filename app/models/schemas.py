from pydantic import BaseModel, Field
from typing import Literal, Optional


class EncodeRequest(BaseModel):
    user_id: Optional[str] = Field(default=None, description="Optional user ID to scope files")
    input_file: Optional[str] = Field(default="input.wav", description="Source WAV file to encode")
    watermarked_file: Optional[str] = Field(default="watermarked.wav", description="Output file for the watermarked audio")
    secret_file: Optional[str] = Field(default="secret.npy", description="File used to store the watermark secret")
    algorithm: Optional[Literal["wavmark", "audioseal"]] = Field(default="wavmark", description="Algorithm to use for encoding")
    return_file: bool = Field(default=False, description="Return the generated audio file instead of JSON")


class CorruptRequest(BaseModel):
    user_id: Optional[str] = Field(default=None, description="Optional user ID to scope files")
    attack_mp3: bool = Field(default=False, description="Enable MP3 attack simulation")
    mp3_bitrate: float = Field(default=128.0, description="MP3 bitrate to use if attack_mp3 is enabled")
    attack_cut: bool = Field(default=False, description="Enable cutting the audio")
    cut_duration: float = Field(default=4.0, description="Duration in seconds to retain when cutting audio")
    attack_speed: bool = Field(default=False, description="Enable speed change attack")
    speed_factor: float = Field(default=1.15, description="Speed factor used when attack_speed is enabled")
    attack_noise: bool = Field(default=False, description="Enable noise injection")
    noise_level: float = Field(default=0.02, description="Noise level used when attack_noise is enabled")
    attack_lpf: bool = Field(default=False, description="Enable low-pass filtering attack")
    lpf_cutoff: float = Field(default=4000.0, description="Low-pass cutoff frequency in Hz")
    attack_volume: bool = Field(default=False, description="Enable volume adjustment attack")
    volume_factor: float = Field(default=0.5, description="Volume factor used when attack_volume is enabled")
    simulate_platform: Literal["none", "youtube", "instagram", "whatsapp"] = Field(default="none", description="Simulate platform audio processing artifacts")
    corrupted_file: Optional[str] = Field(default="corrupted.wav", description="Output file for the corrupted audio")
    return_file: bool = Field(default=False, description="Return the generated audio file instead of JSON")


class DecodeRequest(BaseModel):
    user_id: Optional[str] = Field(default=None, description="Optional user ID to scope files")
    target_file: Optional[str] = Field(default=None, description="Target file to decode; defaults to the corrupted file")
    algorithm: Optional[Literal["wavmark", "audioseal"]] = Field(default=None, description="Decoding algorithm to use")
    secret_file: Optional[str] = Field(default=None, description="Secret file used when decoding")


class DecodeOriginalRequest(BaseModel):
    user_id: Optional[str] = Field(default=None, description="Optional user ID to scope files")
    algorithm: Optional[Literal["wavmark", "audioseal"]] = Field(default=None, description="Decoding algorithm to use")
    secret_file: Optional[str] = Field(default=None, description="Secret file used when decoding the original watermarked file")


class ConfigRequest(BaseModel):
    user_id: Optional[str] = Field(default=None, description="Optional user ID to scope config changes")
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
