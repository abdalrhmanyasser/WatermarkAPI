import os
import torch
import torchaudio
from torchaudio.functional import resample
import numpy as np
import sys
import contextlib

class HideSNRWarnings:
    def __init__(self, stdout):
        self.stdout = stdout
        self.skip_next_newline = False
        
    def write(self, text):
        if "snr too low" in text.lower() or "kip section" in text.lower():
            if not text.endswith('\n'):
                self.skip_next_newline = True
            return
        if self.skip_next_newline and text == '\n':
            self.skip_next_newline = False
            return
        self.skip_next_newline = False
        self.stdout.write(text)
        
    def flush(self):
        self.stdout.flush()

@contextlib.contextmanager
def suppress_wavmark_warnings():
    old_stdout = sys.stdout
    sys.stdout = HideSNRWarnings(old_stdout)
    try:
        yield
    finally:
        sys.stdout = old_stdout

os.environ["TORCH_COMPILE_DISABLE"] = "1"

def encode_audio(input_file="input.wav", output_file="watermarked.wav", secret_file="secret.npy", algorithm="wavmark"):
    print(f"[encode_audio] start: input={input_file} output={output_file} algorithm={algorithm}")
    if not os.path.exists(input_file):
        print(f"[encode_audio] error: input file not found: {input_file}")
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(secret_file):
        payload_np = np.load(secret_file)
        print(f"[encode_audio] loaded secret from {secret_file}")
    else:
        payload_np = np.random.choice([0, 1], size=16).astype(np.int32)
        np.save(secret_file, payload_np)
        print(f"[encode_audio] generated and saved new secret to {secret_file}")

    wav_high_res, orig_sr = torchaudio.load(input_file)
    wav_16k = resample(wav_high_res, orig_freq=orig_sr, new_freq=16000)
    print(f"[encode_audio] loaded audio sr={orig_sr} shape={tuple(wav_high_res.shape)} -> resampled to 16k shape={tuple(wav_16k.shape)}")
    
    if wav_16k.shape[0] > 1:
        wav_16k_mono = wav_16k.mean(dim=0, keepdim=True)
    else:
        wav_16k_mono = wav_16k

    if algorithm == "audioseal":
        print("[encode_audio] algorithm=audioseal")
        from audioseal import AudioSeal
        model = AudioSeal.load_generator("audioseal_wm_16bits").to(device)
        wav_16k_mono_tensor = wav_16k_mono.unsqueeze(0).to(device)
        secret_tensor = torch.from_numpy(payload_np).unsqueeze(0).to(device)
        with torch.no_grad():
            watermark_16k_tensor = model.get_watermark(wav_16k_mono_tensor, message=secret_tensor).cpu()
            noise_only_16k = watermark_16k_tensor.squeeze(0)

    elif algorithm == "wavmark":
        print("[encode_audio] algorithm=wavmark")
        import wavmark
        model = wavmark.load_model().to(device)
        audio_np = wav_16k_mono.squeeze().numpy()
        with suppress_wavmark_warnings():
            watermarked_audio_np, _ = wavmark.encode_watermark(model, audio_np, payload_np, show_progress=False)
        watermark_16k_tensor = torch.from_numpy(watermarked_audio_np).unsqueeze(0)
        noise_only_16k = watermark_16k_tensor - wav_16k_mono

    elif algorithm == "qim":
        print("[encode_audio] algorithm=qim")
        audio_len = wav_16k_mono.shape[1]
        chunk_size = audio_len // 16
        delta = 0.02
        audio_np = wav_16k_mono.squeeze().numpy()
        watermarked_audio_np = audio_np.copy()
        for i in range(16):
            bit = payload_np[i]
            start = i * chunk_size
            end = start + chunk_size
            chunk = audio_np[start:end]
            if bit == 0:
                quantized = np.round(chunk / delta) * delta
            else:
                quantized = np.round((chunk - delta/2) / delta) * delta + delta/2
            watermarked_audio_np[start:end] = quantized
        watermark_16k_tensor = torch.from_numpy(watermarked_audio_np).unsqueeze(0)
        noise_only_16k = watermark_16k_tensor - wav_16k_mono

    elif algorithm == "dwt_svd":
        print("[encode_audio] algorithm=dwt_svd")
        import pywt
        from scipy.linalg import svd
        audio_len = wav_16k_mono.shape[1]
        chunk_size = audio_len // 16
        delta = 5.0
        audio_np = wav_16k_mono.squeeze().numpy()
        watermarked_audio_np = audio_np.copy()
        for i in range(16):
            bit = payload_np[i]
            start = i * chunk_size
            end = start + chunk_size
            chunk = audio_np[start:end].copy()
            cA, cD = pywt.dwt(chunk, 'db1')
            pad_len = 0
            if len(cA) % 16 != 0:
                pad_len = 16 - (len(cA) % 16)
                cA = np.pad(cA, (0, pad_len))
            matrix = cA.reshape((16, len(cA) // 16))
            U, S, Vh = svd(matrix, full_matrices=False)
            if bit == 0:
                S[0] = np.round(S[0] / delta) * delta
            else:
                S[0] = np.round((S[0] - delta/2) / delta) * delta + delta/2
            modified_matrix = np.dot(U * np.diag(S), Vh)
            modified_cA = modified_matrix.flatten()
            if pad_len > 0:
                modified_cA = modified_cA[:-pad_len]
            watermarked_chunk = pywt.idwt(modified_cA, cD, 'db1')
            min_len = min(len(watermarked_chunk), len(chunk))
            watermarked_audio_np[start:start+min_len] = watermarked_chunk[:min_len]
        watermark_16k_tensor = torch.from_numpy(watermarked_audio_np).unsqueeze(0)
        noise_only_16k = watermark_16k_tensor - wav_16k_mono

    elif algorithm == "echo":
        print("[encode_audio] algorithm=echo")
        audio_len = wav_16k_mono.shape[1]
        chunk_size = audio_len // 16
        delay_0, delay_1, alpha = 240, 400, 0.25
        audio_np = wav_16k_mono.squeeze().numpy()
        watermarked_audio_np = np.zeros_like(audio_np)
        for i in range(16):
            bit = payload_np[i]
            start = i * chunk_size
            end = start + chunk_size
            chunk = audio_np[start:end]
            d = delay_1 if bit == 1 else delay_0
            echo_chunk = np.zeros_like(chunk)
            if len(chunk) > d:
                echo_chunk[d:] = chunk[:-d]
            watermarked_audio_np[start:end] = chunk + (alpha * echo_chunk)
        watermark_16k_tensor = torch.from_numpy(watermarked_audio_np).unsqueeze(0)
        noise_only_16k = watermark_16k_tensor - wav_16k_mono

    elif algorithm == "lsb":
        print("[encode_audio] algorithm=lsb")
        audio_int16 = (wav_16k_mono.numpy() * 32767).astype(np.int16)
        for i in range(16):
            bit = payload_np[i]
            audio_int16[0, i] = (audio_int16[0, i] & ~1) | bit
        watermarked_audio_np = audio_int16.astype(np.float32) / 32767.0
        watermark_16k_tensor = torch.from_numpy(watermarked_audio_np)
        noise_only_16k = watermark_16k_tensor - wav_16k_mono

    elif algorithm == "phase":
        print("[encode_audio] algorithm=phase")
        audio_np = wav_high_res.numpy()
        channel_data = audio_np[0]
        audio_len = len(channel_data)
        chunk_size = audio_len // 16
        for i in range(16):
            bit = payload_np[i]
            start = i * chunk_size
            end = start + chunk_size
            chunk = channel_data[start:end]
            if len(chunk) == 0: continue
            fft_data = np.fft.fft(chunk)
            magnitudes, phases = np.abs(fft_data), np.angle(fft_data)
            target_phase = np.pi/2 if bit == 1 else -np.pi/2
            for j in range(1, min(500, len(phases)//2)): 
                phases[j] = target_phase
                phases[-j] = -target_phase
            modified_fft = magnitudes * np.exp(1j * phases)
            channel_data[start:end] = np.real(np.fft.ifft(modified_fft))
        audio_np[0] = channel_data
        watermarked_high_res = torch.from_numpy(audio_np)
        torchaudio.save(output_file, watermarked_high_res, orig_sr)
        return True

    elif algorithm == "hybrid":
        print("[encode_audio] algorithm=hybrid")
        import scipy.io.wavfile as wavfile
        
        # --- LAYER 1: WAVMARK ---
        try:
            import wavmark
            model = wavmark.load_model().to(device)
            audio_np = wav_16k_mono.squeeze().numpy()
            
            with suppress_wavmark_warnings():
                watermarked_audio_np, _ = wavmark.encode_watermark(model, audio_np, payload_np, show_progress=False)
                
            watermark_16k_tensor = torch.from_numpy(watermarked_audio_np).unsqueeze(0)
            noise_only_16k = watermark_16k_tensor - wav_16k_mono
            
            noise_high_res = resample(noise_only_16k, orig_freq=16000, new_freq=orig_sr)
            target_len, current_len = wav_high_res.shape[1], noise_high_res.shape[1]
            if current_len > target_len:
                noise_high_res = noise_high_res[:, :target_len]
            elif current_len < target_len:
                import torch.nn.functional as F
                noise_high_res = F.pad(noise_high_res, (0, target_len - current_len))
                
            layer1_tensor = wav_high_res + noise_high_res
            layer1_audio_np = layer1_tensor.numpy()[0].copy()
        except Exception as e:
            print(f"[encode_audio] error: Hybrid Layer 1 (WavMark) failed: {e}")
            return False

        # --- LAYER 2: SYNC_ENV ---
        block_len, chunk_len, depth = orig_sr, orig_sr // 16, 0.05
        template = np.ones(block_len, dtype=np.float32)
        for i in range(16):
            bit = int(payload_np[i])
            start, end = i * chunk_len, (i + 1) * chunk_len
            t = np.linspace(0, 2 * np.pi, max(1, end - start))
            template[start:end] = 1.0 + (depth * np.sin(t)) if bit == 1 else 1.0 - (depth * np.sin(t))

        for start in range(0, len(layer1_audio_np), block_len):
            end = min(start + block_len, len(layer1_audio_np))
            layer1_audio_np[start:end] *= template[:end-start]

        audio_export = (np.clip(layer1_audio_np, -1.0, 1.0) * 32767).astype(np.int16)
        wavfile.write(output_file, orig_sr, audio_export)
        return True
    else:
        print(f"[encode_audio] error: unknown algorithm: {algorithm}")
        return False
    
    noise_high_res = resample(noise_only_16k, orig_freq=16000, new_freq=orig_sr)
    target_len, current_len = wav_high_res.shape[1], noise_high_res.shape[1]
    
    if current_len > target_len:
        noise_high_res = noise_high_res[:, :target_len]
    elif current_len < target_len:
        import torch.nn.functional as F
        noise_high_res = F.pad(noise_high_res, (0, target_len - current_len))

    watermarked_high_res = wav_high_res + noise_high_res
    torchaudio.save(output_file, watermarked_high_res, orig_sr)
    print(f"[encode_audio] saved watermarked file: {output_file}")
    return True