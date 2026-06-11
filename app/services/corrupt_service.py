import os
import subprocess
import uuid


def _ffprobe_info(path):
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=0", path
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, universal_newlines=True)
        info = {}
        for line in out.splitlines():
            if line.startswith("sample_rate="):
                info["sample_rate"] = int(line.split("=", 1)[1])
            elif line.startswith("channels="):
                info["channels"] = int(line.split("=", 1)[1])
            elif line.startswith("duration="):
                info["duration"] = float(line.split("=", 1)[1])
        return info
    except Exception:
        return {}


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
    attack_hpf=False,
    hpf_cutoff=500.0,
    attack_clipping=False,
    clip_threshold=0.5,
    attack_resample=False,
    downsample_sr=8000,
    attack_echo=False,
    echo_delay_ms=200,
    echo_decay=0.5,
    attack_pitch=False,
    pitch_steps=2,
    simulate_platform="none",
):
    """Apply audio corruptions using ffmpeg to minimize RAM use.

    Returns True on success, False on failure.
    """
    if not os.path.exists(input_file):
        return False

    info = _ffprobe_info(input_file)
    sr = info.get("sample_rate")
    duration = info.get("duration")

    working_file = input_file
    temps = []

    try:
        # MP3 compression pass (simulates lossy encode)
        if attack_mp3:
            temp_mp3 = os.path.join(os.getcwd(), f"tmp_{uuid.uuid4().hex}.mp3")
            cmd = ["ffmpeg", "-y", "-i", working_file, "-c:a", "libmp3lame", "-b:a", f"{int(mp3_bitrate)}k", temp_mp3]
            res_mp3 = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if res_mp3.returncode != 0:
                try:
                    err = res_mp3.stderr.decode('utf-8', errors='replace')
                except Exception:
                    err = str(res_mp3.stderr)
                print(f"[corrupt_audio] ffmpeg mp3 transcode failed: {err}")
                return False
            working_file = temp_mp3
            temps.append(temp_mp3)

        afilters = []
        filter_complex = None

        if attack_cut:
            # trim starting at 1s for cut_duration
            afilters.append(f"atrim=start=1:end={1+float(cut_duration)},asetpts=PTS-STARTPTS")

        if attack_volume:
            afilters.append(f"volume={float(volume_factor)}")

        if attack_speed and float(speed_factor) != 1.0:
            sf = float(speed_factor)
            atempo_parts = []
            # atempo supports 0.5-2.0 per filter
            while sf > 2.0:
                atempo_parts.append("atempo=2.0")
                sf /= 2.0
            while sf < 0.5:
                atempo_parts.append("atempo=0.5")
                sf /= 0.5
            atempo_parts.append(f"atempo={sf:.6f}")
            afilters.extend(atempo_parts)

        if attack_resample and downsample_sr:
            afilters.append(f"aresample={int(downsample_sr)}")
            if sr:
                afilters.append(f"aresample={sr}")

        if attack_lpf:
            afilters.append(f"lowpass=f={float(lpf_cutoff)}")

        if attack_hpf:
            afilters.append(f"highpass=f={float(hpf_cutoff)}")

        if attack_echo:
            delay_ms = int(echo_delay_ms)
            afilters.append(f"aecho=0.8:0.9:{delay_ms}:{float(echo_decay)}")

        if attack_pitch and sr:
            rate_mult = 2 ** (float(pitch_steps) / 12.0)
            target_sr = int(sr * rate_mult)
            afilters.append(f"asetrate={target_sr}")
            afilters.append(f"aresample={sr}")

        if attack_noise:
            dur = duration if duration and duration > 0 else None
            # If we also have afilters, chain them first and feed into the noise amix
            if afilters:
                af_chain = ",".join(afilters)
                if dur:
                    # apply afilters to input, then mix with noise
                    fc = f"[0:a]{af_chain}[a0];anoisesrc=d={dur}:amplitude={float(noise_level)}[n];[a0][n]amix=inputs=2:weights=1 1[out]"
                else:
                    fc = f"[0:a]{af_chain}[a0];anoisesrc=amplitude={float(noise_level)}[n];[a0][n]amix=inputs=2:weights=1 1[out]"
            else:
                if dur:
                    fc = f"[0:a]volume=1[a0];anoisesrc=d={dur}:amplitude={float(noise_level)}[n];[a0][n]amix=inputs=2:weights=1 1[out]"
                else:
                    fc = f"[0:a]volume=1[a0];anoisesrc=amplitude={float(noise_level)}[n];[a0][n]amix=inputs=2:weights=1 1[out]"
            filter_complex = fc

        platform_cmd = None
        if simulate_platform != "none":
            platform_cmd = simulate_platform

        # Build ffmpeg command
        if filter_complex:
            # filter_complex must produce labeled output [out]
            cmd = ["ffmpeg", "-y", "-i", working_file, "-filter_complex", filter_complex, "-map", "[out]", "-c:a", "pcm_s16le", output_file]
        else:
            cmd = ["ffmpeg", "-y", "-i", working_file]
            if afilters:
                cmd += ["-af", ",".join(afilters)]
            cmd += ["-c:a", "pcm_s16le", output_file]

        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if res.returncode != 0:
            try:
                err = res.stderr.decode('utf-8', errors='replace')
            except Exception:
                err = str(res.stderr)
            print(f"[corrupt_audio] ffmpeg failed during main pass (code={res.returncode})\n{err}")
            return False

        # Platform second pass: encode to a platform-like lossy format, then restore WAV
        if platform_cmd:
            tmp1 = os.path.join(os.getcwd(), f"tmp_{uuid.uuid4().hex}.wav")
            if os.path.exists(output_file):
                os.replace(output_file, tmp1)
            temps.append(tmp1)

            tmp2_ext = "m4a"
            codec = "aac"
            container = "m4a"
            if platform_cmd == "whatsapp":
                tmp2_ext = "opus"
                codec = "libopus"
                container = "opus"
            tmp2 = os.path.join(os.getcwd(), f"tmp_{uuid.uuid4().hex}.{tmp2_ext}")
            temps.append(tmp2)

            if platform_cmd == "youtube":
                encode_cmd = [
                    "ffmpeg", "-y", "-i", tmp1,
                    "-af", "lowpass=f=16000,loudnorm=I=-14:TP=-1,acompressor=threshold=-20dB:ratio=4:attack=5:release=50",
                    "-c:a", codec, "-b:a", "128k", tmp2
                ]
            elif platform_cmd == "instagram":
                encode_cmd = [
                    "ffmpeg", "-y", "-i", tmp1,
                    "-af", "loudnorm=I=-14:TP=-1,acompressor=threshold=-18dB:ratio=3:attack=5:release=100",
                    "-c:a", codec, "-b:a", "64k", tmp2
                ]
            elif platform_cmd == "tiktok":
                encode_cmd = [
                    "ffmpeg", "-y", "-i", tmp1,
                    "-af", "loudnorm=I=-14:TP=-1,acompressor=threshold=-15dB:ratio=4:attack=5:release=50",
                    "-c:a", codec, "-b:a", "64k", tmp2
                ]
            elif platform_cmd == "whatsapp":
                encode_cmd = [
                    "ffmpeg", "-y", "-i", tmp1,
                    "-af", "loudnorm=I=-16:TP=-1.5,acompressor=threshold=-20dB:ratio=2:attack=10:release=200",
                    "-c:a", codec, "-b:a", "24k", "-ac", "1", tmp2
                ]
            else:
                encode_cmd = None

            if encode_cmd:
                res2 = subprocess.run(encode_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                if res2.returncode != 0:
                    try:
                        err2 = res2.stderr.decode('utf-8', errors='replace')
                    except Exception:
                        err2 = str(res2.stderr)
                    print(f"[corrupt_audio] ffmpeg failed during platform encode pass (code={res2.returncode})\n{err2}")
                    return False

                decode_cmd = ["ffmpeg", "-y", "-i", tmp2, "-c:a", "pcm_s16le", output_file]
                res3 = subprocess.run(decode_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                if res3.returncode != 0:
                    try:
                        err3 = res3.stderr.decode('utf-8', errors='replace')
                    except Exception:
                        err3 = str(res3.stderr)
                    print(f"[corrupt_audio] ffmpeg failed while decoding platform output back to WAV (code={res3.returncode})\n{err3}")
                    return False

        return True
    finally:
        for t in temps:
            try:
                if os.path.exists(t):
                    os.remove(t)
            except Exception:
                pass
