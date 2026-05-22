import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import torchaudio
from torchaudio.functional import resample
import numpy as np


def test_watermark(target_file="corrupted.wav", secret_file="secret.npy", algorithm="wavmark"):
    print(f"\n--- DECODING & TESTING WITH {algorithm.upper()} ---")
    if not os.path.exists(target_file) or not os.path.exists(secret_file):
        print("[!] Error: Missing target or secret file.")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Engine: {device.type.upper()}")
    
    original_secret = np.load(secret_file)

    print(f"Loading '{target_file}'...")
    wav, sr = torchaudio.load(target_file)
    wav_16k = resample(wav, orig_freq=sr, new_freq=16000)
    if wav_16k.shape[0] > 1:
        wav_16k_mono = wav_16k.mean(dim=0, keepdim=True)
    else:
        wav_16k_mono = wav_16k

    result = {
        "algorithm": algorithm,
        "original_secret": original_secret.tolist(),
    }

    if algorithm == "audioseal":
        from audioseal import AudioSeal
        print("Loading AudioSeal Detector...")
        detector = AudioSeal.load_detector("audioseal_detector_16bits").to(device)
        
        wav_16k_mono = wav_16k_mono.unsqueeze(0).to(device)
        
        print("Scanning audio...")
        with torch.no_grad():
            result_tensor, decoded_message = detector(wav_16k_mono)

        probability = result_tensor[:, 1, :].mean().item()
        decoded_binary = (decoded_message > 0.5).int().cpu().numpy()[0]
        accuracy = (decoded_binary == original_secret).mean() * 100
        is_watermarked = probability > 0.5
        confidence_str = f"{probability:.1%}"

        result.update({
            "decoded_secret": decoded_binary.tolist(),
            "accuracy": float(accuracy),
            "is_watermarked": bool(is_watermarked),
            "confidence": confidence_str,
        })

    elif algorithm == "wavmark":
        import wavmark
        print("Loading WavMark Detector...")
        model = wavmark.load_model().to(device)
        audio_np = wav_16k_mono.squeeze().numpy()

        print("Scanning audio...")
        decoded_binary, _ = wavmark.decode_watermark(model, audio_np, show_progress=True)
        accuracy = (original_secret == decoded_binary).mean() * 100
        ber = float((original_secret != decoded_binary).mean())
        is_watermarked = accuracy >= 85.0

        result.update({
            "decoded_secret": decoded_binary.tolist(),
            "accuracy": float(accuracy),
            "bit_error_rate": float(ber),
            "is_watermarked": bool(is_watermarked),
            "confidence": "N/A",
        })
    else:
        print("[!] Unknown algorithm selected.")
        return None

    print("\n===== RESULTS =====")
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Original ID: {original_secret.tolist()}")
    print(f"Recovered ID: {result['decoded_secret']}")
    print(f"Payload Accuracy: {result['accuracy']:.1f}%")
    print(f"Watermark Found: {result['is_watermarked']} (Confidence: {result['confidence']})")
    print("===================\n")

    return result


if __name__ == "__main__":
    result = test_watermark()
    print(result)
