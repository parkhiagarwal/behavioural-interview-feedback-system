FEATURES = [
    # audio
    "pause_ratio", "pause_count", "longest_pause_s", "syllables_per_sec",
    "volume_mean_db", "volume_std_db", "pitch_variation",
    # transcript
    "wpm", "long_word_pauses", "word_confidence", "filler_rate", "hedge_rate",
    "repeat_rate", "unclear_ratio",
    # video
    "eye_contact", "face_visible", "head_movement", "lean_variation",
    "slouch_drift", "body_movement",
    # new eye-tracking features
    "looking_down_ratio", "blink_rate", "smile_ratio", "head_yaw_std",
]
 
LABELS = ["confidence", "clarity", "pronunciation"]
 
 
def extract_features(a, t, v):
    """a = step 1 result, t = step 2 result, v = step 3 result."""
    n = max(t["word_count"], 1)
    row = {
        "pause_ratio": a["pause_ratio"],
        "pause_count": a["pause_count"],
        "longest_pause_s": a["longest_pause_s"],
        "syllables_per_sec": a["syllables_per_sec"],
        "volume_mean_db": a["volume_mean_db"],
        "volume_std_db": a["volume_std_db"],
        "pitch_variation": a["pitch_variation_semitones"],
        "wpm": t["wpm"],
        "long_word_pauses": t["long_pauses"],
        "word_confidence": t["mean_word_confidence"],
        "filler_rate": t["filler_total"] / n * 100,
        "hedge_rate": t["hedge_total"] / n * 100,
        "repeat_rate": t["repeats"] / n * 100,
        "unclear_ratio": len(t["unclear_words"]) / n,
        "eye_contact": v["eye_contact_ratio"],
        "face_visible": v["face_visible_ratio"],
        "head_movement": v["head_movement"],
        "lean_variation": v["lean_variation"],
        "slouch_drift": v["slouch_drift"],
        "body_movement": v["body_movement"],
        "looking_down_ratio": v.get("looking_down_ratio", 0.0),
        "blink_rate": v.get("blink_rate", 17.0),
        "smile_ratio": v.get("smile_ratio", 0.0),
        "head_yaw_std": v.get("head_yaw_std", 0.0),
    }
    return {k: float(row[k]) for k in FEATURES}
 