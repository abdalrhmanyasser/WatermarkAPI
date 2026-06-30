import os
import contextlib
import sys
import torch
import torchaudio
from torchaudio.functional import resample
import numpy as np

# Import the progress tracker
from app.services.progress_manager import progress_tracker
import re

class WavMarkProgressWriter:
    def __init__(self, stdout, task_id=None, base_progress=0, max_progress=100):
        self.stdout = stdout
        self.task_id = task_id
        self.base_progress = base_progress
        self.max_progress = max_progress

    def write(self, text):
        # Intercept tqdm-style percentage outputs from the WavMark deep learning model
        if self.task_id and "%" in text:
            match = re.search(r'(\d+)%', text)
            if match:
                pct = int(match.group(1))
                # Scale the model's 0-100% to our API's allocated progress window
                scaled_pct = self.base_progress + (pct / 100.0) * (self.max_progress - self.base_progress)
                progress_tracker.update_progress(
                    self.task_id, 
                    int(scaled_pct), 
                    f"Running Deep Learning Model ({pct}%)..."
                )
        
        # We don't actually write it to stdout to keep the server console clean
        # self.stdout.write(text) 

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

os.environ["TORCH_COMPILE_DISABLE"] = "1"

def _encode_audioseal(wav, payload, device):
    from audioseal import AudioSeal
    model = AudioSeal.load_generator("audioseal_wm_16bits").to(device)
    return model.get_watermark(wav.unsqueeze(0).to(device), message=torch.from_numpy(payload).unsqueeze(0).to(device)).cpu().squeeze(0), False
def _encode_wavmark(wav, payload, device, task_id=None):
    import wavmark
    model = wavmark.load_model().to(device)
    
    # Map WavMark's internal 0-100% to the API's 50% -> 80% phase
    with suppress_wavmark_warnings(task_id, base_progress=50, max_progress=80):
        wm, _ = wavmark.encode_watermark(model, wav.squeeze().numpy(), payload)
        
    return torch.from_numpy(wm).unsqueeze(0) - wav, False

def _encode_hybrid(wav, wav_hi, orig_sr, payload, out, device, task_id=None):
    import scipy.io.wavfile as wavfile
    import wavmark
    model = wavmark.load_model().to(device)
    
    # Map WavMark's pass to 50% -> 70%
    with suppress_wavmark_warnings(task_id, base_progress=50, max_progress=70): 
        wm, _ = wavmark.encode_watermark(model, wav.squeeze().numpy(), payload)
        
    if task_id: progress_tracker.update_progress(task_id, 75, "Applying Hybrid Phase/Echo Layers...")
    n16 = torch.from_numpy(wm).unsqueeze(0) - wav
    n_hi = resample(n16, 16000, orig_sr)
    tl, cl = wav_hi.shape[1], n_hi.shape[1]
    
    if cl > tl: n_hi = n_hi[:, :tl]
    elif cl < tl: 
        import torch.nn.functional as F
        n_hi = F.pad(n_hi, (0, tl - cl))
    l1 = (wav_hi + n_hi).numpy()[0].copy()

    tmp = np.ones(orig_sr, dtype=np.float32)
    for i in range(16):
        s, e = i*(orig_sr//16), (i+1)*(orig_sr//16)
        t = np.linspace(0, 2*np.pi, max(1, e-s))
        tmp[s:e] = 1.0 + 0.05*np.sin(t) if payload[i]==1 else 1.0 - 0.05*np.sin(t)
        
    for s in range(0, len(l1), orig_sr):
        e = min(s+orig_sr, len(l1))
        l1[s:e] *= tmp[:e-s]
        
    if task_id: progress_tracker.update_progress(task_id, 80, "Writing hybrid output...")
    wavfile.write(out, orig_sr, (np.clip(l1, -1.0, 1.0)*32767).astype(np.int16))
    return None, True
def _encode_qim(wav, payload):
    a = wav.squeeze().numpy()
    wm = a.copy()
    c_len = a.shape[0] // 16
    for i in range(16):
        s, e = i*c_len, (i+1)*c_len
        c = a[s:e]
        wm[s:e] = np.round(c/0.02)*0.02 if payload[i]==0 else np.round((c-0.01)/0.02)*0.02 + 0.01
    return torch.from_numpy(wm).unsqueeze(0) - wav, False

def _encode_dwt_svd(wav, payload):
    import pywt
    from scipy.linalg import svd
    a = wav.squeeze().numpy()
    wm = a.copy()
    c_len = a.shape[0] // 16
    for i in range(16):
        s, e = i*c_len, (i+1)*c_len
        c = a[s:e].copy()
        cA, cD = pywt.dwt(c, 'db1')
        pad = 16 - (len(cA)%16) if len(cA)%16!=0 else 0
        if pad: cA = np.pad(cA, (0, pad))
        U, S, Vh = svd(cA.reshape((16, len(cA)//16)), full_matrices=False)
        S[0] = np.round(S[0]/5.0)*5.0 if payload[i]==0 else np.round((S[0]-2.5)/5.0)*5.0 + 2.5
        mod = np.dot(U*np.diag(S), Vh).flatten()
        if pad: mod = mod[:-pad]
        wc = pywt.idwt(mod, cD, 'db1')
        wm[s:s+min(len(wc), len(c))] = wc[:min(len(wc), len(c))]
    return torch.from_numpy(wm).unsqueeze(0) - wav, False

def _encode_echo(wav, payload):
    a = wav.squeeze().numpy()
    wm = np.zeros_like(a)
    c_len = a.shape[0] // 16
    for i in range(16):
        s, e = i*c_len, (i+1)*c_len
        c = a[s:e]
        d = 400 if payload[i]==1 else 240
        ec = np.zeros_like(c)
        if len(c)>d: ec[d:] = c[:-d]
        wm[s:e] = c + 0.25*ec
    return torch.from_numpy(wm).unsqueeze(0) - wav, False

def _encode_lsb(wav, payload):
    audio = (wav.numpy() * 32767).astype(np.int16)
    for i in range(16): audio[0, i] = (audio[0, i] & ~1) | payload[i]
    return torch.from_numpy(audio.astype(np.float32)/32767.0) - wav, False

def _encode_phase(wav_hi, orig_sr, payload, out):
    audio = wav_hi.numpy()
    ch = audio[0]
    c_len = len(ch) // 16
    for i in range(16):
        s, e = i*c_len, (i+1)*c_len
        c = ch[s:e]
        if not len(c): continue
        fft = np.fft.fft(c)
        m, p = np.abs(fft), np.angle(fft)
        tp = np.pi/2 if payload[i]==1 else -np.pi/2
        for j in range(1, min(500, len(p)//2)):
            p[j], p[-j] = tp, -tp
        ch[s:e] = np.real(np.fft.ifft(m * np.exp(1j * p)))
    audio[0] = ch
    torchaudio.save(out, torch.from_numpy(audio), orig_sr)
    return None, True

def encode_audio(in_path, out_path, algo, user_id, task_id=None):
    if task_id: progress_tracker.update_progress(task_id, 10, "Initializing encoder...")
    if not os.path.exists(in_path): return False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    payload = np.array([int(x) for x in format(user_id, '016b')], dtype=np.int32)
    
    if task_id: progress_tracker.update_progress(task_id, 30, "Loading audio file...")
    wav_hi, orig_sr = torchaudio.load(in_path)
    wav = resample(wav_hi, orig_sr, 16000)
    wav_mono = wav.mean(dim=0, keepdim=True) if wav.shape[0] > 1 else wav
    if task_id: progress_tracker.update_progress(task_id, 50, f"Applying {algo} watermark...")
    
    if algo == "audioseal": n, b = _encode_audioseal(wav_mono, payload, device)
    elif algo == "wavmark": n, b = _encode_wavmark(wav_mono, payload, device, task_id) # PASSED
    elif algo == "qim": n, b = _encode_qim(wav_mono, payload)
    elif algo == "dwt_svd": n, b = _encode_dwt_svd(wav_mono, payload)
    elif algo == "echo": n, b = _encode_echo(wav_mono, payload)
    elif algo == "lsb": n, b = _encode_lsb(wav_mono, payload)
    elif algo == "phase": n, b = _encode_phase(wav_hi, orig_sr, payload, out_path)
    elif algo == "hybrid": n, b = _encode_hybrid(wav_mono, wav_hi, orig_sr, payload, out_path, device, task_id) # PASSED
    else: return False

    if task_id: progress_tracker.update_progress(task_id, 85, "Finalizing audio buffer...")
    if b: return True

    n_hi = resample(n, 16000, orig_sr)
    tl, cl = wav_hi.shape[1], n_hi.shape[1]
    if cl > tl: n_hi = n_hi[:, :tl]
    elif cl < tl: 
        import torch.nn.functional as F
        n_hi = F.pad(n_hi, (0, tl - cl))
        
    if task_id: progress_tracker.update_progress(task_id, 95, "Saving output file...")
    torchaudio.save(out_path, wav_hi + n_hi, orig_sr)
    return True