import json
import time
 
import sounddevice as sd
import soundfile as sf
 
from step1_audio_analysis import SR, analyze, show_report
from step2_transcript_analysis import analyze_words, show_report2, transcribe
from step3_video_analysis import (BODY_MOVE_HIGH, HEAD_MOVE_HIGH, SLOUCH_LIMIT,
                                  analyze_video, show_video_report)
 
DURATION = 30  # seconds
 
 
# ---------- scoring helpers ----------
def band(v, lo, hi, tol):
    """100 points inside [lo, hi]; points fall linearly to 0 at a distance of tol."""
    if lo <= v <= hi:
        return 100.0
    d = lo - v if v < lo else v - hi
    return max(0.0, 100.0 * (1 - d / tol))
 
 
def falling(v, zero_at):
    """100 points when v = 0, falling linearly to 0 points when v reaches zero_at."""
    return max(0.0, min(100.0, 100.0 * (1 - v / zero_at)))
 
 
# ---------- recording ----------
def record_both(seconds):
    print(f"Recording {seconds}s of audio and video. Look at the camera and start your answer!")
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32")
    t0 = time.time()
    video = analyze_video(0, max_seconds=seconds, show=True)
    elapsed = time.time() - t0
    sd.stop()
    y = audio[: int(min(elapsed, seconds) * SR), 0].copy()  # cut if you pressed q early
    sf.write("last_recording.wav", y, SR)
    return y, video
 
 
# ---------- scoring ----------
def build_components(a, t, v):
    """Every component is a score from 0 to 100 plus advice shown when it is low."""
    n = max(t["word_count"], 1)
    filler_rate = t["filler_total"] / n * 100
    hedge_rate = t["hedge_total"] / n * 100
    unclear_ratio = len(t["unclear_words"]) / n
 
    comp = {
        "pace": (band(t["wpm"], 120, 160, 60),
                 "Keep your pace between about 120 and 160 words per minute."),
        "pauses": (0.5 * band(a["pause_ratio"], 0.05, 0.25, 0.30) + 0.5 * falling(t["long_pauses"], 8),
                   "Avoid long silences. Plan your next sentence while finishing the current one."),
        "fillers": (falling(filler_rate, 8),
                    "Replace 'um', 'uh' and 'like' with a short silent pause."),
        "hedging": (falling(hedge_rate, 6),
                    "Sound more direct: say 'I led the project' instead of 'I think I kind of led it'."),
        "voice_expression": (band(a["pitch_variation_semitones"], 2.5, 6, 3),
                             "Vary your pitch and stress key words so your voice is not flat."),
        "word_clarity": (band(t["mean_word_confidence"], 0.9, 1.0, 0.4),
                         "Speak more clearly and do not rush your words."),
        "unclear_words": (falling(unclear_ratio, 0.25),
                          "Pronounce the ends of words fully and open your mouth more."),
    }
    if v:
        comp["eye_contact"] = (min(100.0, v["eye_contact_ratio"] / 0.8 * 100),
                               "Look at the camera lens as if it were the interviewer's eyes.")
        comp["head_steady"] = (falling(v["head_movement"], HEAD_MOVE_HIGH * 2),
                               "Keep your head steadier while you talk.")
        comp["body_calm"] = (falling(v["body_movement"], BODY_MOVE_HIGH * 2),
                             "Reduce fidgeting and keep your hand movements purposeful.")
        comp["posture"] = (falling(max(0.0, v["slouch_drift"]), SLOUCH_LIMIT * 2),
                           "Sit upright and keep your shoulders up through the whole answer.")
    if v and "looking_down_ratio" in v:  # only available with MediaPipe eye tracking
        comp["looking_down"] = (falling(v["looking_down_ratio"], 0.4),
                                "Avoid looking down. Keep your head and eyes up, even while thinking.")
        comp["blink_calm"] = (band(v["blink_rate"], 8, 28, 20),
                              "Stay relaxed: very fast blinking can look nervous.")
        comp["warmth"] = (band(v["smile_ratio"], 0.08, 0.7, 0.3),
                          "Smile naturally a few times to sound warm and friendly.")
    return comp
 
 
# How much each component counts toward each final score (weights in each row sum to 1)
DIMENSIONS = {
    "Confidence": {"eye_contact": .28, "looking_down": .07, "head_steady": .08, "body_calm": .04,
                   "voice_expression": .13, "hedging": .17, "fillers": .09, "posture": .07,
                   "blink_calm": .04, "warmth": .03},
    "Clarity": {"pace": .35, "pauses": .30, "fillers": .20, "word_clarity": .15},
    "Pronunciation": {"word_clarity": .70, "unclear_words": .30},
}
 
 
def dimension_score(weights, comp):
    avail = {k: w for k, w in weights.items() if k in comp}  # skips video parts if no camera
    total = sum(avail.values())
    return sum(comp[k][0] * w for k, w in avail.items()) / total
 
 
def label(score):
    return ("Excellent" if score >= 80 else "Good" if score >= 65
            else "Needs work" if score >= 50 else "Weak")
 
 
def final_report(a, t, v):
    comp = build_components(a, t, v)
    scores = {name: dimension_score(w, comp) for name, w in DIMENSIONS.items()}
    overall = sum(scores.values()) / len(scores)
 
    print("\n" + "=" * 44)
    print("        FINAL BEHAVIOUR FEEDBACK")
    print("=" * 44)
    for name, s in scores.items():
        print(f"{name:<15}: {s:5.1f} / 100   {label(s)}")
    print(f"{'Overall':<15}: {overall:5.1f} / 100   {label(overall)}")
 
    weakest = sorted(comp.items(), key=lambda kv: kv[1][0])[:3]
    weakest = [(k, s, adv) for k, (s, adv) in weakest if s < 75]
    print("\nTop things to improve:")
    if not weakest:
        print("- Nothing major. Great delivery!")
    for k, s, adv in weakest:
        print(f"- {k.replace('_', ' ').title()} ({s:.0f}/100): {adv}")
 
    with open("report.json", "w", encoding="utf-8") as f:
        json.dump({
            "scores": {k: round(x, 1) for k, x in scores.items()},
            "overall": round(overall, 1),
            "components": {k: round(s, 1) for k, (s, _) in comp.items()},
            "audio": a, "transcript": t, "video": v,
        }, f, indent=2)
    print("\nFull details saved to report.json")
 
 
if __name__ == "__main__":
    y, video = record_both(DURATION)
    if len(y) < SR * 3:
        raise SystemExit("Recording too short. Please speak for at least 3 seconds.")
 
    print("Analyzing audio...")
    audio = analyze(y)
    if audio is None:
        raise SystemExit("No speech detected. Check your microphone.")
 
    words = transcribe(y)
    if not words:
        raise SystemExit("No words recognized. Speak a bit louder and clearer.")
    text = analyze_words(words)
 
    show_report(audio)
    show_report2(text)
    if video:
        show_video_report(video)
    else:
        print("\nNo video result (camera problem or face not found). Scoring audio only.")
 
    final_report(audio, text, video)
 