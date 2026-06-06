import os
import subprocess
import torch
import torchaudio
import torchaudio.functional as F

def corrupt_audio(
    input_file="watermarked.wav", 
    output_file="corrupted.wav", 
    attack_mp3=False, mp3_bitrate=128.0,
    attack_cut=False, cut_duration=4.0,
    attack_speed=False, speed_factor=1.15,
    attack_noise=False, noise_level=0.01,
    attack_lpf=False, lpf_cutoff=4000.0,
    attack_volume=False, volume_factor=0.5,
    attack_hpf=False, hpf_cutoff=500.0,
    attack_clipping=False, clip_threshold=0.5,
    attack_resample=False, downsample_sr=8000,
    attack_echo=False, echo_delay_ms=200, echo_decay=0.5,
    attack_pitch=False, pitch_steps=2,
    simulate_platform="none"
):
    print(f"[corrupt_audio] start: input={input_file} output={output_file} simulate_platform={simulate_platform}")
    if not os.path.exists(input_file):
        print(f"[corrupt_audio] error: input file not found: {input_file}")
        return False

    wav, sr = torchaudio.load(input_file)

    if attack_volume:
        print(f"[corrupt_audio] applying volume change: factor={volume_factor}")
        wav = wav * volume_factor

    if attack_noise:
        print(f"[corrupt_audio] adding noise: level={noise_level}")
        noise = torch.randn_like(wav) * noise_level
        wav = wav + noise

    if attack_lpf:
        print(f"[corrupt_audio] applying low-pass filter: cutoff={lpf_cutoff}")
        wav = F.lowpass_biquad(wav, sample_rate=sr, cutoff_freq=lpf_cutoff)

    if attack_hpf:
        print(f"[corrupt_audio] applying high-pass filter: cutoff={hpf_cutoff}")
        wav = F.highpass_biquad(wav, sample_rate=sr, cutoff_freq=hpf_cutoff)

    if attack_mp3:
        print(f"[corrupt_audio] applying MP3 compression: bitrate={mp3_bitrate}k")
        temp_mp3 = "temp_compressed.mp3"
        torchaudio.save(temp_mp3, wav, sr, format="mp3", compression=mp3_bitrate)
        wav, sr = torchaudio.load(temp_mp3)
        if os.path.exists(temp_mp3): os.remove(temp_mp3)

    if attack_cut:
        print(f"[corrupt_audio] cutting audio: duration={cut_duration}s")
        start_sample = sr * 1
        end_sample = int(start_sample + (sr * cut_duration))
        if wav.shape[-1] > end_sample:
            wav = wav[:, start_sample:end_sample]

    if attack_speed and speed_factor != 1.0:
        print(f"[corrupt_audio] changing speed: factor={speed_factor}")
        fake_sr = int(sr * speed_factor)
        wav = F.resample(wav, orig_freq=fake_sr, new_freq=sr)

    if attack_clipping:
        print(f"[corrupt_audio] applying clipping: threshold={clip_threshold}")
        wav = torch.clamp(wav, min=-clip_threshold, max=clip_threshold)

    if attack_resample:
        print(f"[corrupt_audio] resampling down/up: downsample_sr={downsample_sr}")
        wav = F.resample(wav, orig_freq=sr, new_freq=downsample_sr)
        wav = F.resample(wav, orig_freq=downsample_sr, new_freq=sr)

    if attack_echo:
        print(f"[corrupt_audio] adding echo: delay_ms={echo_delay_ms} decay={echo_decay}")
        delay_samples = int((echo_delay_ms / 1000.0) * sr)
        echo_wav = torch.zeros_like(wav)
        if wav.shape[-1] > delay_samples:
            echo_wav[:, delay_samples:] = wav[:, :-delay_samples]
        wav = wav + (echo_wav * echo_decay)
        wav = wav / torch.max(torch.abs(wav))

    if attack_pitch:
        print(f"[corrupt_audio] pitch shifting: steps={pitch_steps}")
        wav = F.pitch_shift(wav, sr, n_steps=pitch_steps)

    if simulate_platform != "none":
        print(f"[corrupt_audio] simulating platform: {simulate_platform}")
        temp_in = "temp_platform_in.wav"
        torchaudio.save(temp_in, wav, sr)
        cmd = None
        
        if simulate_platform == "youtube":
            temp_out = "temp_yt.m4a"
            cmd = ["ffmpeg", "-y", "-i", temp_in, "-af", "lowpass=f=16000,loudnorm=I=-14:TP=-1,acompressor=threshold=-20dB:ratio=4:attack=5:release=50", "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2", temp_out]
        elif simulate_platform == "instagram":
            temp_out = "temp_ig.m4a"
            cmd = ["ffmpeg", "-y", "-i", temp_in, "-af", "loudnorm=I=-14:TP=-1,acompressor=threshold=-18dB:ratio=3:attack=5:release=100", "-c:a", "aac", "-b:a", "64k", temp_out]
        elif simulate_platform == "tiktok":
            temp_out = "temp_tt.m4a"
            cmd = ["ffmpeg", "-y", "-i", temp_in, "-af", "loudnorm=I=-14:TP=-1,acompressor=threshold=-15dB:ratio=4:attack=5:release=50", "-c:a", "aac", "-b:a", "64k", temp_out]
        elif simulate_platform == "whatsapp":
            temp_out = "temp_wa.ogg"
            cmd = ["ffmpeg", "-y", "-i", temp_in, "-af", "loudnorm=I=-16:TP=-1.5,acompressor=threshold=-20dB:ratio=2:attack=10:release=200", "-c:a", "libopus", "-b:a", "24k", "-ac", "1", temp_out]

        if cmd:
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                wav, sr = torchaudio.load(temp_out)
                if os.path.exists(temp_out): os.remove(temp_out)
            except subprocess.CalledProcessError:
                print(f"[corrupt_audio] warning: platform simulation command failed for {simulate_platform}")
                pass
        if os.path.exists(temp_in): os.remove(temp_in)
        
    torchaudio.save(output_file, wav, sr)
    print(f"[corrupt_audio] saved corrupted file: {output_file}")
    return True