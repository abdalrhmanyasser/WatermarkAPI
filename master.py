import os
from dotenv import load_dotenv
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

CONFIG = {
    "input_file": "input.wav",
    "watermarked_file": "watermarked.wav",
    "corrupted_file": "corrupted.wav",
    "secret_file": "secret.npy", # Changed to .npy
    
    # NEW: Algorithm Toggle
    "algorithm": "wavmark", # "audioseal" or "wavmark"
    
    "attack_mp3": False,  "mp3_bitrate": 128.0,
    "attack_cut": False,  "cut_duration": 4.0,
    "attack_speed": False, "speed_factor": 1.15,
    "attack_noise": False, "noise_level": 0.02,
    "attack_lpf": False,   "lpf_cutoff": 4000.0,
    "attack_volume": False,"volume_factor": 0.5,
    "simulate_platform": "none" 
}

def print_menu():
    print("\n" + "#"*65)
    print("#      AUDIO WATERMARK MASTER CONTROLLER")
    print("#"*65)
    print(f"  [Z] ENGINE: [ {CONFIG['algorithm'].upper()} ]")
    print("-" * 65)
    print(" [1] Encode: Embed Watermark")
    print(" [2] Corrupt: Run Attack Pipeline")
    print(" [3] Decode: Test Corrupted File")
    print(" [4] Decode: Test Original Watermarked File")
    print(" [5] Run Full Pipeline (1 -> 2 -> 3)")
    print("-" * 65)
    print(" SOCIAL MEDIA SIMULATION (Used in Step 2):")
    print(f"  [P] Platform: [ {CONFIG['simulate_platform'].upper()} ]")
    print("-" * 65)
    print(" STANDARD ATTACKS")
    print(f"  [Q] MP3 Compression: {'[ON]' if CONFIG['attack_mp3'] else '[OFF]'} \t| [A] Bitrate: {CONFIG['mp3_bitrate']} kbps")
    print(f"  [W] Cut Audio      : {'[ON]' if CONFIG['attack_cut'] else '[OFF]'} \t| [S] Length : {CONFIG['cut_duration']} sec")
    print(f"  [E] Change Speed   : {'[ON]' if CONFIG['attack_speed'] else '[OFF]'}  \t| [D] Speed Factor : {CONFIG['speed_factor']}x")
    print(f"  [R] Add White Noise: {'[ON]' if CONFIG['attack_noise'] else '[OFF]'}  \t| [F] Level  : {CONFIG['noise_level']}")
    print(f"  [T] Low Pass Filter: {'[ON]' if CONFIG['attack_lpf'] else '[OFF]'}  \t| [G] Cutoff : {CONFIG['lpf_cutoff']} Hz")
    print(f"  [Y] Volume Drop    : {'[ON]' if CONFIG['attack_volume'] else '[OFF]'}  \t| [H] Volume Factor : {CONFIG['volume_factor']}x")
    print("-" * 65)
    print(" [0] Exit")
    print("#"*65)

def get_float_input(prompt):
    try:
        return float(input(prompt))
    except ValueError:
        print("[!] Invalid input. Please enter a number.")
        return None

def main():
    while True:
        print_menu()
        choice = input("Select an option: ").strip().lower()

        if choice == 'z':
            CONFIG["algorithm"] = "audioseal" if CONFIG["algorithm"] == "wavmark" else "wavmark"
            
        elif choice == '1':
            encode_audio(CONFIG["input_file"], CONFIG["watermarked_file"], CONFIG["secret_file"], CONFIG["algorithm"])
            
        elif choice == '2':
            corrupt_audio(
                input_file=CONFIG["watermarked_file"], output_file=CONFIG["corrupted_file"],
                attack_mp3=CONFIG["attack_mp3"], mp3_bitrate=CONFIG["mp3_bitrate"],
                attack_cut=CONFIG["attack_cut"], cut_duration=CONFIG["cut_duration"],
                attack_speed=CONFIG["attack_speed"], speed_factor=CONFIG["speed_factor"],
                attack_noise=CONFIG["attack_noise"], noise_level=CONFIG["noise_level"],
                attack_lpf=CONFIG["attack_lpf"], lpf_cutoff=CONFIG["lpf_cutoff"],
                attack_volume=CONFIG["attack_volume"], volume_factor=CONFIG["volume_factor"],
                simulate_platform=CONFIG["simulate_platform"]
            )
            
        elif choice == '3':
            test_watermark(CONFIG["corrupted_file"], CONFIG["secret_file"], CONFIG["algorithm"])
            
        elif choice == '4':
            test_watermark(CONFIG["watermarked_file"], CONFIG["secret_file"], CONFIG["algorithm"])
            
        elif choice == '5':
            if encode_audio(CONFIG["input_file"], CONFIG["watermarked_file"], CONFIG["secret_file"], CONFIG["algorithm"]):
                if corrupt_audio(
                    input_file=CONFIG["watermarked_file"], output_file=CONFIG["corrupted_file"],
                    attack_mp3=CONFIG["attack_mp3"], mp3_bitrate=CONFIG["mp3_bitrate"],
                    attack_cut=CONFIG["attack_cut"], cut_duration=CONFIG["cut_duration"],
                    attack_speed=CONFIG["attack_speed"], speed_factor=CONFIG["speed_factor"],
                    attack_noise=CONFIG["attack_noise"], noise_level=CONFIG["noise_level"],
                    attack_lpf=CONFIG["attack_lpf"], lpf_cutoff=CONFIG["lpf_cutoff"],
                    attack_volume=CONFIG["attack_volume"], volume_factor=CONFIG["volume_factor"],
                    simulate_platform=CONFIG["simulate_platform"]
                ):
                    test_watermark(CONFIG["corrupted_file"], CONFIG["secret_file"], CONFIG["algorithm"])

        # Platform Toggle
        elif choice == 'p':
            platforms = ["none", "youtube", "instagram", "whatsapp"]
            current_idx = platforms.index(CONFIG["simulate_platform"])
            next_idx = (current_idx + 1) % len(platforms)
            CONFIG["simulate_platform"] = platforms[next_idx]

        # Toggles
        elif choice == 'q': CONFIG["attack_mp3"] = not CONFIG["attack_mp3"]
        elif choice == 'w': CONFIG["attack_cut"] = not CONFIG["attack_cut"]
        elif choice == 'e': CONFIG["attack_speed"] = not CONFIG["attack_speed"]
        elif choice == 'r': CONFIG["attack_noise"] = not CONFIG["attack_noise"]
        elif choice == 't': CONFIG["attack_lpf"] = not CONFIG["attack_lpf"]
        elif choice == 'y': CONFIG["attack_volume"] = not CONFIG["attack_volume"]
        
        # Values
        elif choice == 'a':
            val = get_float_input("Enter MP3 Bitrate (e.g., 64, 128, 320): ")
            if val is not None: CONFIG["mp3_bitrate"] = val
        elif choice == 's':
            val = get_float_input("Enter Cut Duration in seconds: ")
            if val is not None: CONFIG["cut_duration"] = val
        elif choice == 'd':
            val = get_float_input("Enter Speed Factor (e.g., 0.85, 1.15): ")
            if val is not None: CONFIG["speed_factor"] = val
        elif choice == 'f':
            val = get_float_input("Enter Noise Level: ")
            if val is not None: CONFIG["noise_level"] = val
        elif choice == 'g':
            val = get_float_input("Enter LPF Cutoff Frequency in Hz: ")
            if val is not None: CONFIG["lpf_cutoff"] = val
        elif choice == 'h':
            val = get_float_input("Enter Volume Factor: ")
            if val is not None: CONFIG["volume_factor"] = val

        elif choice == '0':
            print("Exiting...")
            break
        else:
            print("[!] Unknown command.")

if __name__ == "__main__":
    main()