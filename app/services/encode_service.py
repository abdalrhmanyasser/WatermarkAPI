import os

import torch
import torchaudio
from torchaudio.functional import resample
import numpy as np


def encode_audio(input_file="input.wav", output_file="watermarked.wav", secret_file="secret.npy", algorithm="wavmark"):
    print(f"\n--- ENCODING WITH {algorithm.upper()} ---")
    if not os.path.exists(input_file):
        print(f"[!] Error: Could not find '{input_file}'.")
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Engine: {device.type.upper()}")

    payload_np = np.random.choice([0, 1], size=16).astype(np.int32)
    np.save(secret_file, payload_np)
    print(f"Generated Secret Payload: {payload_np.tolist()}")

    print(f"Loading '{input_file}'...")
    wav_high_res, orig_sr = torchaudio.load(input_file)
    wav_16k = resample(wav_high_res, orig_freq=orig_sr, new_freq=16000)

    if wav_16k.shape[0] > 1:
        wav_16k_mono = wav_16k.mean(dim=0, keepdim=True)
    else:
        wav_16k_mono = wav_16k

    if algorithm == "audioseal":
        from audioseal import AudioSeal
        print("Loading AudioSeal model...")
        model = AudioSeal.load_generator("audioseal_wm_16bits").to(device)

        wav_16k_mono = wav_16k_mono.unsqueeze(0).to(device)
        secret_tensor = torch.from_numpy(payload_np).unsqueeze(0).to(device)

        print("Embedding Watermark...")
        with torch.no_grad():
            watermark_16k_tensor = model.get_watermark(wav_16k_mono, message=secret_tensor).cpu()
            noise_only_16k = watermark_16k_tensor.squeeze(0)

    elif algorithm == "wavmark":
        import wavmark
        print("Loading WavMark model...")
        model = wavmark.load_model().to(device)
        audio_np = wav_16k_mono.squeeze().numpy()

        print("Embedding Watermark...")
        watermarked_audio_np, _ = wavmark.encode_watermark(model, audio_np, payload_np, show_progress=True)

        watermark_16k_tensor = torch.from_numpy(watermarked_audio_np).unsqueeze(0)
        noise_only_16k = watermark_16k_tensor - wav_16k_mono

    else:
        print("[!] Unknown algorithm selected.")
        return False

    print("Upsampling and finalizing...")
    noise_high_res = resample(noise_only_16k, orig_freq=16000, new_freq=orig_sr)
    watermarked_high_res = wav_high_res + noise_high_res

    torchaudio.save(output_file, watermarked_high_res, orig_sr)
    print(f"[+] Success! Watermarked audio saved to '{output_file}'.")
    return True


if __name__ == "__main__":
    encode_audio()
