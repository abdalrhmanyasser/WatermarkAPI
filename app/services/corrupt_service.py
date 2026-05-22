import os
import subprocess

import torch
import torchaudio
import torchaudio.functional as F


def corrupt_audio(
    input_file="watermarked.wav",
    output_file="corrupted.wav",
    attack_mp3=False,
    mp3_bitrate=128.0,
    attack_cut=False,
    cut_duration=4.0,
    attack_speed=False,
    speed_factor=1.15,
    attack_noise=False,
    noise_level=0.01,
    attack_lpf=False,
    lpf_cutoff=4000.0,
    attack_volume=False,
    volume_factor=0.5,
    simulate_platform="none"
):
    print("\n--- CORRUPTING ---")
    if not os.path.exists(input_file):
        print(f"[!] Error: '{input_file}' not found. Run Encoder first.")
        return False

    print(f"Loading '{input_file}'...")
    wav, sr = torchaudio.load(input_file)

    if attack_volume:
        wav = wav * volume_factor

    if attack_noise:
        noise = torch.randn_like(wav) * noise_level
        wav = wav + noise

    if attack_lpf:
        wav = F.lowpass_biquad(wav, sample_rate=sr, cutoff_freq=lpf_cutoff)

    if attack_mp3:
        temp_mp3 = "temp_compressed.mp3"
        torchaudio.save(temp_mp3, wav, sr, format="mp3", compression=mp3_bitrate)
        wav, sr = torchaudio.load(temp_mp3)
        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)

    if attack_cut:
        start_sample = sr * 1
        end_sample = int(start_sample + (sr * cut_duration))
        if wav.shape[-1] > end_sample:
            wav = wav[:, start_sample:end_sample]

    if attack_speed and speed_factor != 1.0:
        fake_sr = int(sr * speed_factor)
        wav = F.resample(wav, orig_freq=fake_sr, new_freq=sr)

    if simulate_platform != "none":
        print(f" -> Attack: Simulating {simulate_platform.upper()} compression servers...")
        temp_in = "temp_platform_in.wav"
        torchaudio.save(temp_in, wav, sr)

        if simulate_platform == "youtube":
            temp_out = "temp_yt.m4a"
            cmd = [
                "ffmpeg", "-y",
                "-i", temp_in,
                "-af", "loudnorm=I=-14:TP=-1:LRA=11,acompressor=threshold=-20dB:ratio=4:attack=5:release=100",
                "-c:a", "aac", "-b:a", "128k", temp_out,
            ]
        elif simulate_platform == "instagram":
            temp_out = "temp_ig.m4a"
            cmd = [
                "ffmpeg", "-y",
                "-i", temp_in,
                "-af", "loudnorm=I=-14:TP=-1,acompressor=threshold=-18dB:ratio=3:attack=5:release=100",
                "-c:a", "aac", "-b:a", "64k", temp_out,
            ]
        elif simulate_platform == "whatsapp":
            temp_out = "temp_wa.ogg"
            cmd = [
                "ffmpeg", "-y",
                "-i", temp_in,
                "-af", "loudnorm=I=-16:TP=-1.5,acompressor=threshold=-20dB:ratio=2:attack=10:release=200",
                "-c:a", "libopus", "-b:a", "24k", "-ac", "1", temp_out,
            ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"[!] FFmpeg failed: {e.stderr.decode('utf-8')}")
            if os.path.exists(temp_in):
                os.remove(temp_in)
            return False

        wav, sr = torchaudio.load(temp_out)
        if os.path.exists(temp_in):
            os.remove(temp_in)
        if os.path.exists(temp_out):
            os.remove(temp_out)
    else:
        print(" -> Skipped: Social Media Simulation")

    print("Converting back to WAV...")
    torchaudio.save(output_file, wav, sr)
    print(f"[+] Success! Corrupted audio saved to '{output_file}'.")
    return True


if __name__ == "__main__":
    corrupt_audio()
