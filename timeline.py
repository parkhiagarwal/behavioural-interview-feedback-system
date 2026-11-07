import librosa
import numpy as np
 
HOP = 512
 
 
def audio_timelines(y, sr=16000):
    """Returns {"volume": [dB per second], "pitch": [Hz per second]} (None where there is no speech)."""
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0]
    speaking = rms > max(np.percentile(rms, 90) * 0.15, 1e-4)
    db = librosa.amplitude_to_db(rms, ref=1.0)
    f0, _, _ = librosa.pyin(y, fmin=70, fmax=400, sr=sr, hop_length=HOP)
 
    n = min(len(rms), len(f0))
    idx = np.where(speaking[:n])[0]
    if len(idx) == 0:
        return {"volume": [], "pitch": []}
    first, last = int(idx[0]), int(idx[-1]) + 1
    fps = sr / HOP
 
    volume, pitch = [], []
    for s in range(int((last - first) / fps) + 1):
        a, b = first + int(s * fps), min(first + int((s + 1) * fps), last)
        if b <= a:
            break
        sp = speaking[a:b]
        volume.append(float(np.mean(db[a:b][sp])) if sp.any() else None)
        seg = f0[a:b]
        seg = seg[~np.isnan(seg)]
        pitch.append(float(np.median(seg)) if len(seg) >= 3 else None)
    return {"volume": volume, "pitch": pitch}
 