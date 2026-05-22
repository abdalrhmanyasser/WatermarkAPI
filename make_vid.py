import os
from pathlib import Path
import subprocess

def create_mp4(audio_file="watermarked.wav", output_file="test_video.mp4"):
    if not Path(audio_file).exists():
        print(f"[!] Error: '{audio_file}' not found. Run encoder first.")
        return

    print(f"Measuring '{audio_file}' using torchcodec...")
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=1280x720:r=1",
        "-i",
        audio_file,
        "-c:v",
        "libx264",
        "-tune",
        "stillimage",
        "-c:a",
        "aac",
        "-b:a",
        "320k",
        "-t",
        str(10),
        output_file,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if Path(output_file).exists():
        print(f"[+] Success! Video saved to '{output_file}'")
    else:
        print('[!] Something went wrong. Make sure FFmpeg is in your PATH.')


if __name__ == "__main__":
    create_mp4()
