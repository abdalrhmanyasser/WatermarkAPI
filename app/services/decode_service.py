import os
import re
import sys
import contextlib
import traceback
# os.environ["TORCH_COMPILE_DISABLE"] = "1"
import torch
import torchaudio
from torchaudio.functional import resample
import numpy as np

# Import the global progress tracker
from app.services.progress_manager import progress_tracker


class WavMarkProgressWriter:
    def __init__(self, stdout, task_id=None, min_pct=40, max_pct=90, status_prefix=None):
        self.stdout = stdout
        self.task_id = task_id
        self.min_pct = min_pct
        self.max_pct = max_pct
        self.status_prefix = status_prefix or "Scanning audio track"
        self.skip_next_newline = False
        self.buffer = ""
        self.last_reported_pct = None

    def write(self, text):
        lowered = text.lower()
        if "snr too low" in lowered or "kip section" in lowered:
            if not text.endswith("\n"):
                self.skip_next_newline = True
            return
        if self.skip_next_newline and text == "\n":
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

# ==========================================
# ALGORITHM HELPER FUNCTIONS
# ==========================================

def _decode_audioseal(wav_16k_mono, original_secret, device, task_id):
    print("[test_watermark] algorithm=audioseal")
    if task_id: progress_tracker.update_progress(task_id, 40, "Running AudioSeal detector...")
    
    from audioseal import AudioSeal
    detector = AudioSeal.load_detector("audioseal_detector_16bits").to(device)
    wav_16k_mono_tensor = wav_16k_mono.unsqueeze(0).to(device)
    
    with torch.no_grad():
        res, decoded_message = detector(wav_16k_mono_tensor)
        
    probability = res[:, 1, :].mean().item()
    decoded_binary = (decoded_message > 0.5).int().cpu().numpy()[0]
    accuracy = (decoded_binary == original_secret).mean() * 100
    is_watermarked = probability > 0.5
    confidence_str = f"{probability:.1%}"
    
    return decoded_binary, accuracy, is_watermarked, confidence_str


def _decode_wavmark(wav_16k_mono, original_secret, device, task_id=None, is_hybrid_layer=False):
    layer_txt = " (hybrid layer)" if is_hybrid_layer else ""
    print(f"[test_watermark] algorithm=wavmark{layer_txt}")
    
    if task_id and not is_hybrid_layer: 
        progress_tracker.update_progress(task_id, 30, "Booting WavMark model...")
    elif task_id and is_hybrid_layer:
        progress_tracker.update_progress(task_id, 45, "Preparing WavMark hybrid verification layer...")
        
    import wavmark
    model = wavmark.load_model().to(device)
    audio_np = wav_16k_mono.squeeze().numpy()
    
    if task_id and not is_hybrid_layer: 
        progress_tracker.update_progress(task_id, 35, "Scanning audio track...")
        
    with suppress_wavmark_warnings(
        task_id,
        min_pct=35 if not is_hybrid_layer else 50,
        max_pct=85 if not is_hybrid_layer else 90,
        status_prefix="Scanning audio track" if not is_hybrid_layer else "Verifying WavMark hybrid layer"
    ):
        raw_decoded, _ = wavmark.decode_watermark(model, audio_np, show_progress=True)
    
    if isinstance(raw_decoded, str): raw_decoded = [int(b) for b in raw_decoded]
    elif raw_decoded is None: raw_decoded = [0] * 16
        
    decoded_binary = np.array(raw_decoded, dtype=np.int32)
    if len(decoded_binary) != len(original_secret):
        decoded_binary = np.resize(decoded_binary, len(original_secret))
        
    accuracy = (original_secret == decoded_binary).mean() * 100
    is_watermarked = accuracy >= 85.0
    ber = (original_secret != decoded_binary).mean()
    confidence_str = f"N/A (BER: {ber:.3f})"
    
    return decoded_binary, accuracy, is_watermarked, confidence_str


def _decode_qim(wav_16k_mono, original_secret, task_id):
    print("[test_watermark] algorithm=qim")
    if task_id: progress_tracker.update_progress(task_id, 40, "Running QIM distance checks...")
    
    audio_len = wav_16k_mono.shape[1]
    chunk_size = audio_len // 16
    delta = 0.02
    audio_np = wav_16k_mono.squeeze().numpy()
    decoded_bits = []
    
    for i in range(16):
        start = i * chunk_size
        chunk = audio_np[start:start + chunk_size]
        dist_0 = np.abs(chunk - np.round(chunk / delta) * delta).mean()
        dist_1 = np.abs(chunk - (np.round((chunk - delta/2) / delta) * delta + delta/2)).mean()
        decoded_bits.append(0 if dist_0 < dist_1 else 1)
        
    decoded_binary = np.array(decoded_bits, dtype=np.int32)
    accuracy = (original_secret == decoded_binary).mean() * 100
    is_watermarked, confidence_str = accuracy >= 80.0, "Mean distance spread"
    
    return decoded_binary, accuracy, is_watermarked, confidence_str


def _decode_dwt_svd(wav_16k_mono, original_secret, task_id):
    print("[test_watermark] algorithm=dwt_svd")
    if task_id: progress_tracker.update_progress(task_id, 40, "Running Wavelet Transforms & SVD...")
    
    import pywt
    from scipy.linalg import svd
    
    audio_len = wav_16k_mono.shape[1]
    chunk_size = audio_len // 16
    delta = 5.0
    audio_np = wav_16k_mono.squeeze().numpy()
    decoded_bits = []
    
    for i in range(16):
        start = i * chunk_size
        chunk = audio_np[start:start + chunk_size]
        if len(chunk) < 2:
            decoded_bits.append(0); continue
            
        cA, _ = pywt.dwt(chunk, 'db1')
        if len(cA) % 16 != 0: cA = np.pad(cA, (0, 16 - (len(cA) % 16)))
        
        matrix = cA.reshape((16, len(cA) // 16))
        _, S, _ = svd(matrix, full_matrices=False)
        
        dist_0 = np.abs(S[0] - np.round(S[0] / delta) * delta)
        dist_1 = np.abs(S[0] - (np.round((S[0] - delta/2) / delta) * delta + delta/2))
        decoded_bits.append(0 if dist_0 < dist_1 else 1)
        
    decoded_binary = np.array(decoded_bits, dtype=np.int32)
    accuracy = (original_secret == decoded_binary).mean() * 100
    is_watermarked, confidence_str = accuracy >= 85.0, "SVD Dominant Component Match"
    
    return decoded_binary, accuracy, is_watermarked, confidence_str


def _decode_echo(wav_16k_mono, original_secret, task_id):
    print("[test_watermark] algorithm=echo")
    if task_id: progress_tracker.update_progress(task_id, 50, "Extracting Cepstral peaks...")
    
    audio_len = wav_16k_mono.shape[1]
    chunk_size = audio_len // 16
    delay_0, delay_1, window = 240, 400, 3
    audio_np = wav_16k_mono.squeeze().numpy()
    decoded_bits = []
    
    for i in range(16):
        start = i * chunk_size
        chunk = audio_np[start:start + chunk_size]
        spectrum = np.where(np.abs(np.fft.fft(chunk)) == 0, 1e-10, np.abs(np.fft.fft(chunk)))
        cepstrum = np.real(np.fft.ifft(np.log(spectrum)))
        try:
            peak_0 = np.max(cepstrum[delay_0 - window : delay_0 + window + 1])
            peak_1 = np.max(cepstrum[delay_1 - window : delay_1 + window + 1])
            decoded_bits.append(1 if peak_1 > peak_0 else 0)
        except ValueError:
            decoded_bits.append(0)
            
    decoded_binary = np.array(decoded_bits, dtype=np.int32)
    accuracy = (original_secret == decoded_binary).mean() * 100
    is_watermarked, confidence_str = accuracy >= 80.0, "Cepstral Peak Detection"
    
    return decoded_binary, accuracy, is_watermarked, confidence_str


def _decode_lsb(wav_16k_mono, original_secret, task_id):
    print("[test_watermark] algorithm=lsb")
    if task_id: progress_tracker.update_progress(task_id, 50, "Extracting bit-level payload...")
    
    audio_int16 = (wav_16k_mono.numpy() * 32767).astype(np.int16)
    decoded_bits = [audio_int16[0, i] & 1 for i in range(16)]
    
    decoded_binary = np.array(decoded_bits, dtype=np.int32)
    accuracy = (original_secret == decoded_binary).mean() * 100
    is_watermarked, confidence_str = accuracy == 100.0, "Fragile (Bitwise Extraction)"
    
    return decoded_binary, accuracy, is_watermarked, confidence_str


def _decode_phase(target_file, original_secret, task_id):
    print("[test_watermark] algorithm=phase")
    if task_id: progress_tracker.update_progress(task_id, 50, "Calculating phase angles...")
    
    wav_high_res, _ = torchaudio.load(target_file)
    channel_data = wav_high_res.numpy()[0]
    audio_len = len(channel_data)
    chunk_size = audio_len // 16
    decoded_bits = []
    
    for i in range(16):
        start = i * chunk_size
        chunk = channel_data[start:start + chunk_size]
        if len(chunk) == 0: 
            decoded_bits.append(0); continue
            
        phases = np.angle(np.fft.fft(chunk))
        phase_sum, count = sum(phases[1:min(500, len(phases)//2)]), min(499, len(phases)//2 - 1)
        avg_phase = phase_sum / count if count > 0 else 0
        decoded_bits.append(1 if avg_phase > 0 else 0)
        
    decoded_binary = np.array(decoded_bits, dtype=np.int32)
    accuracy = (original_secret == decoded_binary).mean() * 100
    is_watermarked, confidence_str = accuracy >= 80.0, "Phase Bin Extraction"
    
    return decoded_binary, accuracy, is_watermarked, confidence_str


# Helper function to remove nesting from Hybrid
def _analyze_stretch_offset(audio_np, test_block_len, test_chunk_len, reference_sin):
    max_confidence = -1
    best_bits = [0] * 16
    for offset_step in range(16):
        offset = int((offset_step / 16.0) * test_chunk_len)
        bit_correlations = np.zeros(16)
        
        for block_start in range(offset, len(audio_np) - test_block_len, test_block_len):
            for i in range(16):
                chunk_start = block_start + i * test_chunk_len
                chunk = audio_np[chunk_start:chunk_start+test_chunk_len]
                if len(chunk) == test_chunk_len:
                    bit_correlations[i] += np.sum(np.abs(chunk) * reference_sin)
                    
        confidence = np.sum(np.abs(bit_correlations))
        if confidence > max_confidence:
            max_confidence = confidence
            best_bits = [1 if c > 0 else 0 for c in bit_correlations]
            
    return max_confidence, best_bits


def _decode_hybrid(target_file, wav_16k_mono, original_secret, device, task_id=None):
    print("[test_watermark] algorithm=hybrid")
    
    # 1. TEST SYNC_ENV
    sync_accuracy = 0.0
    best_bits = [0]*16
    try:
        import scipy.io.wavfile as wavfile
        sr_file, audio_int16 = wavfile.read(target_file)
        channel_data = audio_int16[:, 0] if len(audio_int16.shape) > 1 else audio_int16
        audio_np = channel_data.astype(np.float32) / 32767.0
        
        stretches = np.arange(0.85, 1.25, 0.05)
        total_steps = len(stretches)
        best_overall_confidence = -1
        
        for step_idx, stretch in enumerate(stretches):
            if task_id: 
                current_pct = int(25 + ((step_idx / total_steps) * 20))
                progress_tracker.update_progress(task_id, current_pct, f"Analyzing amplitude envelope (Stretch {stretch:.2f})...")
                
            test_block_len = int(sr_file / stretch)
            test_chunk_len = test_block_len // 16
            if test_chunk_len == 0: continue
            reference_sin = np.sin(np.linspace(0, 2 * np.pi, test_chunk_len))
            
            conf, bits = _analyze_stretch_offset(audio_np, test_block_len, test_chunk_len, reference_sin)
            
            if conf > best_overall_confidence:
                best_overall_confidence = conf
                best_bits = bits
                
        sync_accuracy = (original_secret == np.array(best_bits, dtype=np.int32)).mean() * 100
    except Exception as e: 
        print(f"[test_watermark] warning: Sync_Env failed: {e}")

    # 2. TEST WAVMARK
    if task_id: progress_tracker.update_progress(task_id, 45, "Running WavMark verification layer...")
    wavmark_accuracy = 0.0 
    wm_decoded_binary = np.array([0]*16, dtype=np.int32)
    try:
        # Re-use the separated WavMark logic
        wm_decoded_binary, wavmark_accuracy, _, _ = _decode_wavmark(wav_16k_mono, original_secret, device, task_id, is_hybrid_layer=True)
    except Exception as e: 
        print(f"[test_watermark] warning: WavMark failed: {e}")

    # DECISION
    accuracy = max(sync_accuracy, wavmark_accuracy)
    is_watermarked = accuracy >= 80.0
    
    if sync_accuracy >= 80.0 and wavmark_accuracy >= 80.0:
        confidence_str = f"BOTH DETECTED (Sync: {sync_accuracy:.1f}%, WavMark: {wavmark_accuracy:.1f}%)"
    elif sync_accuracy >= 80.0: 
        confidence_str = f"SYNC_ENV SURVIVED ({sync_accuracy:.1f}%)"
    elif wavmark_accuracy >= 80.0: 
        confidence_str = f"WAVMARK SURVIVED ({wavmark_accuracy:.1f}%)"
    else: 
        confidence_str = f"HYBRID FAILED (Best was {accuracy:.1f}%)"
    
    decoded_binary = np.array(best_bits if sync_accuracy > wavmark_accuracy else wm_decoded_binary, dtype=np.int32)
    return decoded_binary, accuracy, is_watermarked, confidence_str


# ==========================================
# MAIN DECODE ENTRY POINT
# ==========================================

def test_watermark(target_file="corrupted.wav", secret_file="secret.npy", algorithm="wavmark", task_id=None):
    try:
        print(f"[test_watermark] start: target={target_file} secret={secret_file} algorithm={algorithm}")
        if task_id: progress_tracker.update_progress(task_id, 5, "Loading audio files...")

        if not os.path.exists(target_file) or not os.path.exists(secret_file):
            msg = "missing target or secret file"
            print(f"[test_watermark] error: {msg}")
            if task_id: progress_tracker.update_progress(task_id, -1, msg)
            return None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        original_secret = np.load(secret_file)
        wav, sr = torchaudio.load(target_file)

        if task_id: progress_tracker.update_progress(task_id, 10, "Resampling to 16kHz...")
        wav_16k = resample(wav, orig_freq=sr, new_freq=16000)

        wav_16k_mono = wav_16k.mean(dim=0, keepdim=True) if wav_16k.shape[0] > 1 else wav_16k

        print(f"[test_watermark] loaded audio sr={sr} shape={tuple(wav.shape)} -> resampled to 16k shape={tuple(wav_16k.shape)}")

        # Route to specific algorithm handlers to avoid deep nesting
        if algorithm == "audioseal":
            decoded_binary, accuracy, is_watermarked, confidence_str = _decode_audioseal(wav_16k_mono, original_secret, device, task_id)
        elif algorithm == "wavmark":
            decoded_binary, accuracy, is_watermarked, confidence_str = _decode_wavmark(wav_16k_mono, original_secret, device, task_id)
        elif algorithm == "qim":
            decoded_binary, accuracy, is_watermarked, confidence_str = _decode_qim(wav_16k_mono, original_secret, task_id)
        elif algorithm == "dwt_svd":
            decoded_binary, accuracy, is_watermarked, confidence_str = _decode_dwt_svd(wav_16k_mono, original_secret, task_id)
        elif algorithm == "echo":
            decoded_binary, accuracy, is_watermarked, confidence_str = _decode_echo(wav_16k_mono, original_secret, task_id)
        elif algorithm == "lsb":
            decoded_binary, accuracy, is_watermarked, confidence_str = _decode_lsb(wav_16k_mono, original_secret, task_id)
        elif algorithm == "phase":
            decoded_binary, accuracy, is_watermarked, confidence_str = _decode_phase(target_file, original_secret, task_id)
        elif algorithm == "hybrid":
            decoded_binary, accuracy, is_watermarked, confidence_str = _decode_hybrid(target_file, wav_16k_mono, original_secret, device, task_id)
        else:
            msg = f"unsupported algorithm: {algorithm}"
            print(f"[test_watermark] error: {msg}")
            if task_id: progress_tracker.update_progress(task_id, -1, msg)
            return None

        if task_id: progress_tracker.update_progress(task_id, 95, "Formatting payload data...")

        result = {
            "algorithm": algorithm,
            "original_secret": original_secret.tolist(),
            "decoded_secret": decoded_binary.tolist(),
            "accuracy": float(accuracy),
            "is_watermarked": bool(is_watermarked),
            "confidence": confidence_str,
        }

        print(f"[test_watermark] result: accuracy={accuracy:.2f}% is_watermarked={is_watermarked} confidence={confidence_str}")
        return result

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[test_watermark] unexpected error: {e}\n{tb}")
        if task_id:
            progress_tracker.update_progress(task_id, -1, f"Crash in test_watermark: {e}")
        return None