import sys
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
 
SR = 16000      # sample rate (samples per second)
HOP = 512       # one "frame" = 512 samples = 32 ms
FRAME = 1024
MIN_PAUSE = 0.4  # a silence shorter than this is just breathing between words
 
 
def record(seconds=20):
    print(f"Recording for {seconds} seconds... start speaking now!")
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    y = audio[:, 0]
    sf.write("last_recording.wav", y, SR)
    print("Saved as last_recording.wav")
    return y
 
 
def find_runs(mask):
    """Return (start, end) index pairs for every stretch of consecutive True values."""
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs
 
 
def analyze(y):
    frame_s = HOP / SR
 
    # 1) Loudness of each frame (RMS energy)
    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0]
 
    # 2) Decide which frames are "speaking": above a threshold relative to loud parts
    threshold = max(np.percentile(rms, 90) * 0.15, 1e-4)
    speaking = rms > threshold
    speech_runs = find_runs(speaking)
    if not speech_runs:
        return None
 
    # 3) Ignore silence before the first word and after the last word
    first, last = speech_runs[0][0], speech_runs[-1][1]
    trimmed = speaking[first:last]
    total_time = (last - first) * frame_s
    speaking_time = trimmed.sum() * frame_s
 
    # 4) Pauses = silent stretches inside the speech that are long enough
    silent_runs = find_runs(~trimmed)
    pauses = [(e - s) * frame_s for s, e in silent_runs if (e - s) * frame_s >= MIN_PAUSE]
    pause_ratio = 1 - speaking_time / total_time
 
    # 5) Speaking rate (rough): count sound onsets (about one per syllable)
    onsets = librosa.onset.onset_detect(y=y, sr=SR, hop_length=HOP, units="frames")
    onsets = [o for o in onsets if first <= o < last]
    rate = len(onsets) / speaking_time if speaking_time > 0 else 0
 
    # 6) Volume level and steadiness (in decibels, only while speaking)
    db = librosa.amplitude_to_db(rms[first:last][trimmed], ref=1.0)
    volume_mean, volume_std = float(np.mean(db)), float(np.std(db))
 
    # 7) Pitch variation in semitones (flat voice = low value)
    f0, voiced, _ = librosa.pyin(y, fmin=70, fmax=400, sr=SR, hop_length=HOP)
    f0 = f0[voiced & ~np.isnan(f0)]
    pitch_std = float(np.std(12 * np.log2(f0 / np.median(f0)))) if len(f0) > 10 else 0.0
 
    return {
        "total_time_s": total_time,
        "speaking_time_s": speaking_time,
        "pause_count": len(pauses),
        "longest_pause_s": max(pauses) if pauses else 0.0,
        "pause_ratio": pause_ratio,
        "syllables_per_sec": rate,
        "volume_mean_db": volume_mean,
        "volume_std_db": volume_std,
        "pitch_variation_semitones": pitch_std,
    }
 
 
def show_report(f):
    print("\n===== AUDIO BEHAVIOUR REPORT =====")
    print(f"Speech length        : {f['total_time_s']:.1f} s")
    print(f"Pauses (>0.4 s)      : {f['pause_count']} (longest {f['longest_pause_s']:.1f} s)")
    print(f"Pause ratio          : {f['pause_ratio']*100:.0f}% of the time silent")
    print(f"Speaking rate        : {f['syllables_per_sec']:.1f} syllables/sec")
    print(f"Volume steadiness    : {f['volume_std_db']:.1f} dB variation")
    print(f"Pitch variation      : {f['pitch_variation_semitones']:.1f} semitones")
 
    print("\nQuick feedback:")
    r = f["syllables_per_sec"]
    print("- Pace: too slow" if r < 7 else "- Pace: too fast" if r > 12 else "- Pace: good")
    print("- Many long pauses, try to keep a flow" if f["pause_ratio"] > 0.35 else "- Pause level: good")
    print("- Voice sounds flat, add more expression" if f["pitch_variation_semitones"] < 2 else "- Voice expression: good")
 
 
if __name__ == "__main__":
    if len(sys.argv) > 1:
        y, _ = librosa.load(sys.argv[1], sr=SR)
    else:
        y = record(20)
    features = analyze(y)
    if features is None:
        print("No speech detected. Check your microphone.")
    else:
        show_report(features)