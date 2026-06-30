import os
import re
import sys
import contextlib
import torch
import torchaudio
from torchaudio.functional import resample
import numpy as np

# Import the progress tracker
from app.services.progress_manager import progress_tracker

class WavMarkProgressWriter:
    def __init__(self, stdout, task_id=None, base_progress=0, max_progress=100):
        self.stdout = stdout
        self.task_id = task_id
        self.base_progress = base_progress
        self.max_progress = max_progress
        self.buffer = ""

    def write(self, text):
        if self.task_id and "%" in text:
            match = re.search(r'%(\d+)', text)
            if match:
                pct = int(match.group(1))
                scaled_pct = self.base_progress + (pct / 100.0) * (self.max_progress - self.base_progress)
                progress_tracker.update_progress(
                    self.task_id, 
                    int(scaled_pct), 
                    f"Running Deep Learning Model ({pct}%)..."
                )

    def flush(self):
        self.stdout.flush()

@contextlib.contextmanager
def suppress_wavmark_warnings(task_id=None, base_progress=0, max_progress=100):
    o_stdout, o_stderr = sys.stdout, sys.stderr
    w = WavMarkProgressWriter(o_stdout, task_id, base_progress, max_progress)
    sys.stdout, sys.stderr = w, w
    try: 
        yield
    finally: 
        sys.stdout, sys.stderr = o_stdout, o_stderr

def _decode_audioseal(wav, device):
    from audioseal import AudioSeal
    det = AudioSeal.load_detector("audioseal_detector_16bits").to(device)
    _, msg = det(wav.unsqueeze(0).to(device))
    return (msg > 0.5).int().cpu().numpy()[0]

def _decode_wavmark(wav, device, task_id=None):
    import wavmark
    model = wavmark.load_model().to(device)
    
    with suppress_wavmark_warnings(task_id, base_progress=60, max_progress=85): 
        dec, _ = wavmark.decode_watermark(model, wav.squeeze().numpy())
        
    # Bugfix: safely handle None and NumPy array evaluations
    if dec is None:
        dec = [0] * 16
    elif isinstance(dec, str): 
        dec = [int(b) for b in dec]
    elif isinstance(dec, (list, tuple, np.ndarray)) and len(dec) == 0:
        dec = [0] * 16
        
    return np.array(dec[:16], dtype=np.int32)

def _decode_qim(wav):
    a = wav.squeeze().numpy()
    c_len = a.shape[0] // 16
    bits = []
    for i in range(16):
        c = a[i*c_len:(i+1)*c_len]
        d0 = np.abs(c - np.round(c/0.02)*0.02).mean()
        d1 = np.abs(c - (np.round((c-0.01)/0.02)*0.02 + 0.01)).mean()
        bits.append(0 if d0 < d1 else 1)
    return np.array(bits, dtype=np.int32)

def _decode_dwt_svd(wav):
    import pywt
    from scipy.linalg import svd
    a = wav.squeeze().numpy()
    c_len = a.shape[0] // 16
    bits = []
    for i in range(16):
        c = a[i*c_len:(i+1)*c_len]
        if len(c)<2: 
            bits.append(0); continue
        cA, _ = pywt.dwt(c, 'db1')
        if len(cA)%16!=0: cA = np.pad(cA, (0, 16-(len(cA)%16)))
        _, S, _ = svd(cA.reshape((16, len(cA)//16)), full_matrices=False)
        d0 = np.abs(S[0] - np.round(S[0]/5.0)*5.0)
        d1 = np.abs(S[0] - (np.round((S[0]-2.5)/5.0)*5.0 + 2.5))
        bits.append(0 if d0 < d1 else 1)
    return np.array(bits, dtype=np.int32)

def _decode_echo(wav):
    a = wav.squeeze().numpy()
    c_len = a.shape[0] // 16
    bits = []
    for i in range(16):
        c = a[i*c_len:(i+1)*c_len]
        cep = np.real(np.fft.ifft(np.log(np.where(np.abs(np.fft.fft(c))==0, 1e-10, np.abs(np.fft.fft(c))))))
        try: bits.append(1 if np.max(cep[397:404]) > np.max(cep[237:244]) else 0)
        except: bits.append(0)
    return np.array(bits, dtype=np.int32)

def _decode_lsb(wav):
    a = (wav.numpy() * 32767).astype(np.int16)
    return np.array([a[0, i] & 1 for i in range(16)], dtype=np.int32)

def _decode_phase(target_file):
    wav, _ = torchaudio.load(target_file)
    a = wav.numpy()[0]
    c_len = len(a) // 16
    bits = []
    for i in range(16):
        c = a[i*c_len:(i+1)*c_len]
        if not len(c): 
            bits.append(0); continue
        p = np.angle(np.fft.fft(c))
        bits.append(1 if sum(p[1:min(500, len(p)//2)]) > 0 else 0)
    return np.array(bits, dtype=np.int32)

def _decode_hybrid(wav, device, task_id=None):
    if task_id: progress_tracker.update_progress(task_id, 65, "Decoding primary WavMark layer...")
    
    # 1. Try Deep Learning Layer First
    wm_bits = _decode_wavmark(wav, device, task_id)
    
    if task_id: progress_tracker.update_progress(task_id, 80, "Decoding secondary Sync Envelope layer...")
    
    # 2. Extract the Amplitude/Sync Envelope Layer
    a = wav.squeeze().numpy()
    sr = 16000 # Wav is already resampled to 16k in test_watermark
    c_len = sr // 16
    
    # We analyze up to the first 3 seconds of audio to cast "votes" on the envelope bits
    sec_to_check = min(3, len(a) // sr)
    if sec_to_check < 1:
        return wm_bits # Audio is too short, rely entirely on WavMark
        
    bit_votes = np.zeros(16)
    
    for sec in range(sec_to_check):
        sec_audio = a[sec*sr : (sec+1)*sr]
        for i in range(16):
            chunk = sec_audio[i*c_len : (i+1)*c_len]
            if not len(chunk): continue
            
            # Recreate the reference sine wave used during encoding
            t = np.linspace(0, 2*np.pi, len(chunk))
            sine_ref = np.sin(t)
            
            # Check the correlation between the chunk's magnitude and the sine wave
            correlation = np.dot(np.abs(chunk), sine_ref)
            
            if correlation > 0:
                bit_votes[i] += 1
            else:
                bit_votes[i] -= 1
                
    env_bits = np.array([1 if v > 0 else 0 for v in bit_votes], dtype=np.int32)
    
    # 3. Hybrid Decision Logic
    # If the deep learning payload was destroyed (returns all 0s), fallback to the Envelope
    if np.sum(wm_bits) == 0:
        if task_id: progress_tracker.update_progress(task_id, 85, "WavMark corrupted, falling back to Sync Env...")
        return env_bits
        
    # Otherwise, return the primary WavMark payload
    return wm_bits

def test_watermark(target_file, algorithm, task_id=None):
    if task_id: progress_tracker.update_progress(task_id, 10, "Initializing decoder...")
    if not os.path.exists(target_file): return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if task_id: progress_tracker.update_progress(task_id, 30, "Loading audio file...")
    wav, sr = torchaudio.load(target_file)
    wav = resample(wav, sr, 16000)
    wav_mono = wav.mean(dim=0, keepdim=True) if wav.shape[0] > 1 else wav

    if task_id: progress_tracker.update_progress(task_id, 60, f"Extracting {algorithm} payload...")
    if algorithm == "audioseal": bits = _decode_audioseal(wav_mono, device)
    elif algorithm == "wavmark": bits = _decode_wavmark(wav_mono, device, task_id)
    elif algorithm == "qim": bits = _decode_qim(wav_mono)
    elif algorithm == "dwt_svd": bits = _decode_dwt_svd(wav_mono)
    elif algorithm == "echo": bits = _decode_echo(wav_mono)
    elif algorithm == "lsb": bits = _decode_lsb(wav_mono)
    elif algorithm == "phase": bits = _decode_phase(target_file)
    elif algorithm == "hybrid": bits = _decode_hybrid(wav_mono, device, task_id)
    else: return None

    if task_id: progress_tracker.update_progress(task_id, 90, "Parsing user ID...")
    user_id = int("".join(str(b) for b in bits), 2)
    return {"owner_id": user_id, "algorithm": algorithm, "bits": bits.tolist()}