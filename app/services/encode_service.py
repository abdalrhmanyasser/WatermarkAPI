import os
import re
import sys
import contextlib
import torch
import torchaudio
from torchaudio.functional import resample
import numpy as np

# Import the global progress tracker
from app.services.progress_manager import progress_tracker

# ==========================================
# WARNING SUPPRESSION
# ==========================================
class WavMarkProgressWriter:
    def __init__(self, stdout, task_id=None, min_pct=40, max_pct=90, status_prefix=None):
        self.stdout = stdout
        self.task_id = task_id
        self.min_pct = min_pct
        self.max_pct = max_pct
        self.status_prefix = status_prefix or "Processing WavMark progress"
        self.skip_next_newline = False
        self.buffer = ""
        self.last_reported_pct = None

    def write(self, text):
        lowered = text.lower()
        if "snr too low" in lowered or "kip section" in lowered:
            if not text.endswith('\n'):
                self.skip_next_newline = True
            return
        if self.skip_next_newline and text == '\n':
            self.skip_next_newline = False
            return
        self.skip_next_newline = False

        if self.task_id is not None:
            self._update_task_progress(text)

        self.stdout.write(text)

    def flush(self):
        self.stdout.flush()

    def _update_task_progress(self, text):
        self.buffer += text
        for match in re.finditer(r"(\d{1,3})\s*%", self.buffer):
            percent = int(match.group(1))
            if percent == self.last_reported_pct:
                continue
            self.last_reported_pct = percent
            mapped = self.min_pct + round(percent * (self.max_pct - self.min_pct) / 100)
            mapped = max(self.min_pct, min(mapped, self.max_pct))
            progress_tracker.update_progress(
                self.task_id,
                mapped,
                f"{self.status_prefix} ({percent}%)"
            )
        if "\n" in self.buffer or "\r" in self.buffer:
            self.buffer = re.split(r"[\r\n]+", self.buffer)[-1]

@contextlib.contextmanager
def suppress_wavmark_warnings(task_id=None, min_pct=40, max_pct=90, status_prefix=None):
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    writer = WavMarkProgressWriter(old_stdout, task_id, min_pct, max_pct, status_prefix)
    sys.stdout = writer
    sys.stderr = writer
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

os.environ["TORCH_COMPILE_DISABLE"] = "1"

# ==========================================
# ALGORITHM HELPER FUNCTIONS
# ==========================================

def _encode_audioseal(wav_16k_mono, payload_np, device, task_id):
    if task_id: progress_tracker.update_progress(task_id, 40, "Generating AudioSeal noise track...")
    from audioseal import AudioSeal
    model = AudioSeal.load_generator("audioseal_wm_16bits").to(device)
    wav_16k_mono_tensor = wav_16k_mono.unsqueeze(0).to(device)
    secret_tensor = torch.from_numpy(payload_np).unsqueeze(0).to(device)
    
    with torch.no_grad():
        watermark_16k_tensor = model.get_watermark(wav_16k_mono_tensor, message=secret_tensor).cpu()
        noise_only_16k = watermark_16k_tensor.squeeze(0)
        
    return noise_only_16k, False

def _encode_wavmark(wav_16k_mono, payload_np, device, task_id):
    if task_id: progress_tracker.update_progress(task_id, 20, "Embedding payload via WavMark neural net...")
    import wavmark
    model = wavmark.load_model().to(device)
    audio_np = wav_16k_mono.squeeze().numpy()
    
    with suppress_wavmark_warnings(
        task_id,
        min_pct=20,
        max_pct=80,
        status_prefix="Embedding payload via WavMark neural net"
    ):
        watermarked_audio_np, _ = wavmark.encode_watermark(model, audio_np, payload_np, show_progress=True)
        
    watermark_16k_tensor = torch.from_numpy(watermarked_audio_np).unsqueeze(0)
    noise_only_16k = watermark_16k_tensor - wav_16k_mono
    
    return noise_only_16k, False

def _encode_qim(wav_16k_mono, payload_np, task_id):
    if task_id: progress_tracker.update_progress(task_id, 40, "Applying Quantization Index Modulation...")
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
    
    return noise_only_16k, False

def _encode_dwt_svd(wav_16k_mono, payload_np, task_id):
    if task_id: progress_tracker.update_progress(task_id, 40, "Executing Wavelet & SVD encoding...")
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
    
    return noise_only_16k, False

def _encode_echo(wav_16k_mono, payload_np, task_id):
    if task_id: progress_tracker.update_progress(task_id, 40, "Injecting Echo Hiding delays...")
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
    
    return noise_only_16k, False

def _encode_lsb(wav_16k_mono, payload_np, task_id):
    if task_id: progress_tracker.update_progress(task_id, 40, "Encoding fragile Least Significant Bits...")
    audio_int16 = (wav_16k_mono.numpy() * 32767).astype(np.int16)
    
    for i in range(16):
        bit = payload_np[i]
        audio_int16[0, i] = (audio_int16[0, i] & ~1) | bit
        
    watermarked_audio_np = audio_int16.astype(np.float32) / 32767.0
    watermark_16k_tensor = torch.from_numpy(watermarked_audio_np)
    noise_only_16k = watermark_16k_tensor - wav_16k_mono
    
    return noise_only_16k, False

def _encode_phase(wav_high_res, orig_sr, payload_np, output_file, task_id):
    if task_id: progress_tracker.update_progress(task_id, 40, "Shifting Fast Fourier Transform phases...")
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
    
    return None, True # True means it's already saved, bypass upsampling

def _encode_hybrid(wav_16k_mono, wav_high_res, orig_sr, payload_np, output_file, device, task_id):
    import scipy.io.wavfile as wavfile
    try:
        if task_id: progress_tracker.update_progress(task_id, 30, "Hybrid Layer 1: WavMark Generation...")
        import wavmark
        model = wavmark.load_model().to(device)
        audio_np = wav_16k_mono.squeeze().numpy()
        
        with suppress_wavmark_warnings(
            task_id,
            min_pct=30,
            max_pct=70,
            status_prefix="Generating hybrid WavMark layer"
        ):
            watermarked_audio_np, _ = wavmark.encode_watermark(model, audio_np, payload_np, show_progress=True)
            
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
    except Exception:
        return None, False

    if task_id: progress_tracker.update_progress(task_id, 65, "Hybrid Layer 2: Sync-Env Amplitude Stamping...")
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

    if task_id: progress_tracker.update_progress(task_id, 80, "Writing hybrid final file...")
    audio_export = (np.clip(layer1_audio_np, -1.0, 1.0) * 32767).astype(np.int16)
    wavfile.write(output_file, orig_sr, audio_export)
    
    return None, True # True means it's already saved, bypass upsampling

# ==========================================
# MAIN ENCODE ENTRY POINT
# ==========================================

def encode_audio(input_file="input.wav", output_file="watermarked.wav", secret_file="secret.npy", algorithm="wavmark", task_id=None):
    if task_id: progress_tracker.update_progress(task_id, 5, "Initializing encoding engine...")
    
    if not os.path.exists(input_file):
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(secret_file):
        payload_np = np.load(secret_file)
    else:
        payload_np = np.random.choice([0, 1], size=16).astype(np.int32)
        np.save(secret_file, payload_np)

    if task_id: progress_tracker.update_progress(task_id, 10, "Loading high-resolution audio...")
    wav_high_res, orig_sr = torchaudio.load(input_file)
    wav_16k = resample(wav_high_res, orig_freq=orig_sr, new_freq=16000)
    
    wav_16k_mono = wav_16k.mean(dim=0, keepdim=True) if wav_16k.shape[0] > 1 else wav_16k

    if task_id: progress_tracker.update_progress(task_id, 15, f"Booting {algorithm.upper()} core...")

    is_already_saved = False
    noise_only_16k = None

    if algorithm == "audioseal":
        noise_only_16k, is_already_saved = _encode_audioseal(wav_16k_mono, payload_np, device, task_id)
    elif algorithm == "wavmark":
        noise_only_16k, is_already_saved = _encode_wavmark(wav_16k_mono, payload_np, device, task_id)
    elif algorithm == "qim":
        noise_only_16k, is_already_saved = _encode_qim(wav_16k_mono, payload_np, task_id)
    elif algorithm == "dwt_svd":
        noise_only_16k, is_already_saved = _encode_dwt_svd(wav_16k_mono, payload_np, task_id)
    elif algorithm == "echo":
        noise_only_16k, is_already_saved = _encode_echo(wav_16k_mono, payload_np, task_id)
    elif algorithm == "lsb":
        noise_only_16k, is_already_saved = _encode_lsb(wav_16k_mono, payload_np, task_id)
    elif algorithm == "phase":
        noise_only_16k, is_already_saved = _encode_phase(wav_high_res, orig_sr, payload_np, output_file, task_id)
    elif algorithm == "hybrid":
        noise_only_16k, is_already_saved = _encode_hybrid(wav_16k_mono, wav_high_res, orig_sr, payload_np, output_file, device, task_id)
    else:
        return False
    
    # If the algorithm handled its own high-res save (Phase, Hybrid), we bypass the upsampling block.
    if is_already_saved:
        return True

    if task_id: progress_tracker.update_progress(task_id, 70, "Upsampling isolated noise track...")
    noise_high_res = resample(noise_only_16k, orig_freq=16000, new_freq=orig_sr)
    target_len = wav_high_res.shape[1]
    current_len = noise_high_res.shape[1]
    
    if current_len > target_len:
        noise_high_res = noise_high_res[:, :target_len]
    elif current_len < target_len:
        import torch.nn.functional as F
        noise_high_res = F.pad(noise_high_res, (0, target_len - current_len))

    if task_id: progress_tracker.update_progress(task_id, 85, "Mixing and exporting studio quality audio...")
    watermarked_high_res = wav_high_res + noise_high_res
    torchaudio.save(output_file, watermarked_high_res, orig_sr)
    
    return True