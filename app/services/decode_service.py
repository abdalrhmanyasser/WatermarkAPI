import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
import torch
import torchaudio
from torchaudio.functional import resample
import numpy as np

def test_watermark(target_file="corrupted.wav", secret_file="secret.npy", algorithm="wavmark"):
    print(f"[test_watermark] start: target={target_file} secret={secret_file} algorithm={algorithm}")
    if not os.path.exists(target_file) or not os.path.exists(secret_file):
        print(f"[test_watermark] error: missing target or secret file")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_secret = np.load(secret_file)
    wav, sr = torchaudio.load(target_file)
    wav_16k = resample(wav, orig_freq=sr, new_freq=16000)
    
    wav_16k_mono = wav_16k.mean(dim=0, keepdim=True) if wav_16k.shape[0] > 1 else wav_16k

    result = {"algorithm": algorithm, "original_secret": original_secret.tolist()}
    print(f"[test_watermark] loaded audio sr={sr} shape={tuple(wav.shape)} -> resampled to 16k shape={tuple(wav_16k.shape)}")
    
    if algorithm == "audioseal":
        print("[test_watermark] algorithm=audioseal")
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

    elif algorithm == "wavmark":
        print("[test_watermark] algorithm=wavmark")
        import wavmark
        model = wavmark.load_model().to(device)
        audio_np = wav_16k_mono.squeeze().numpy()
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

    elif algorithm == "qim":
        print("[test_watermark] algorithm=qim")
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

    elif algorithm == "dwt_svd":
        print("[test_watermark] algorithm=dwt_svd")
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

    elif algorithm == "echo":
        print("[test_watermark] algorithm=echo")
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

    elif algorithm == "lsb":
        print("[test_watermark] algorithm=lsb")
        audio_int16 = (wav_16k_mono.numpy() * 32767).astype(np.int16)
        decoded_bits = [audio_int16[0, i] & 1 for i in range(16)]
        decoded_binary = np.array(decoded_bits, dtype=np.int32)
        accuracy = (original_secret == decoded_binary).mean() * 100
        is_watermarked, confidence_str = accuracy == 100.0, "Fragile (Bitwise Extraction)"

    elif algorithm == "phase":
        print("[test_watermark] algorithm=phase")
        wav_high_res, orig_sr = torchaudio.load(target_file)
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

    elif algorithm == "hybrid":
        print("[test_watermark] algorithm=hybrid")
        # 1. TEST SYNC_ENV
        sync_accuracy = 0.0
        best_bits = [0]*16
        try:
            import scipy.io.wavfile as wavfile
            sr_file, audio_int16 = wavfile.read(target_file)
            channel_data = audio_int16[:, 0] if len(audio_int16.shape) > 1 else audio_int16
            audio_np = channel_data.astype(np.float32) / 32767.0
            max_confidence = -1
            for stretch in np.arange(0.85, 1.25, 0.05):
                test_block_len = int(sr_file / stretch)
                test_chunk_len = test_block_len // 16
                if test_chunk_len == 0: continue
                reference_sin = np.sin(np.linspace(0, 2 * np.pi, test_chunk_len))
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
                        max_confidence, best_bits = confidence, [1 if c > 0 else 0 for c in bit_correlations]
            sync_accuracy = (original_secret == np.array(best_bits, dtype=np.int32)).mean() * 100
        except Exception as e: 
            print(f"[test_watermark] warning: Sync_Env failed: {e}")

        # 2. TEST WAVMARK
        wavmark_accuracy = 0.0 
        wm_decoded_binary = np.array([0]*16, dtype=np.int32)
        try:
            import wavmark
            model = wavmark.load_model().to(device)
            audio_np = wav_16k_mono.squeeze().numpy()
            raw_decoded, _ = wavmark.decode_watermark(model, audio_np, show_progress=True)
            
            if isinstance(raw_decoded, str): raw_decoded = [int(b) for b in raw_decoded]
            elif raw_decoded is None: raw_decoded = [0] * 16
                
            wm_decoded_binary = np.array(raw_decoded, dtype=np.int32)
            if len(wm_decoded_binary) != len(original_secret):
                wm_decoded_binary = np.resize(wm_decoded_binary, len(original_secret))
                
            wavmark_accuracy = (original_secret == wm_decoded_binary).mean() * 100
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
        
        # Output whichever layer performed better
        decoded_binary = np.array(best_bits if sync_accuracy > wavmark_accuracy else wm_decoded_binary, dtype=np.int32)
    else:
        print(f"[test_watermark] error: unsupported algorithm: {algorithm}")
        return None

    result.update({
        "decoded_secret": decoded_binary.tolist(),
        "accuracy": float(accuracy),
        "is_watermarked": bool(is_watermarked),
        "confidence": confidence_str,
    })
    print(f"[test_watermark] result: accuracy={accuracy:.2f}% is_watermarked={is_watermarked} confidence={confidence_str}")
    return result