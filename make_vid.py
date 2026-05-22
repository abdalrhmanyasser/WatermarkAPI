import os
from torchcodec.decoders import AudioDecoder
import subprocess

def create_mp4(audio_file="watermarked.wav", output_file="test_video.mp4"):
    if not os.path.exists(audio_file):
        print(f"[!] Error: '{audio_file}' not found. Run your encoder first.")
        return

    # 1. Get the exact duration of the audio using torchcodec
    print(f"Measuring '{audio_file}' using torchcodec...")
    decoder = AudioDecoder(audio_file)
    
    # Extract the duration directly from the torchcodec metadata
    try:
        exact_duration = decoder.metadata.duration_seconds
    except AttributeError:
        # Fallback math if duration_seconds is missing in this version
        exact_duration = decoder.metadata.num_frames / decoder.metadata.sample_rate
        
    print(f" -> Duration: {exact_duration:.2f} seconds")

    print("Packaging into MP4...")
    
    # 2. Build the FFmpeg command using the explicit duration
    cmd = [
        "ffmpeg", 
        "-y",                                  
        "-f", "lavfi",                         
        "-i", "color=c=black:s=1280x720:r=1",  
        "-i", audio_file,                      
        "-c:v", "libx264",                     
        "-tune", "stillimage",                 
        "-c:a", "aac",                         
        "-b:a", "320k",                        
        "-t", str(exact_duration),             # Force cut exactly when audio ends
        output_file
    ]
    
    # Run the command silently
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if os.path.exists(output_file):
        print(f"[+] Success! Your video is exactly {exact_duration:.2f} seconds long: '{output_file}'")
    else:
        print("[!] Something went wrong. Make sure FFmpeg is in your PATH.")

if __name__ == "__main__":
    create_mp4()