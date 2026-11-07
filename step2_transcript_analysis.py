import re
import sys
 
import librosa
from faster_whisper import WhisperModel
 
from step1_audio_analysis import SR, analyze, record, show_report
 
FILLERS = ["um", "umm", "uh", "uhh", "hmm", "like", "you know", "basically", "actually"]
HEDGES = ["maybe", "i think", "i guess", "kind of", "sort of", "probably",
          "perhaps", "i feel like", "not sure"]
 
# This prompt nudges the model to KEEP "um/uh" instead of cleaning them away.
FILLER_PROMPT = "Umm, let me think, like, hmm... Okay, so, uh, here is what I think."
 
 
def transcribe(y):
    """Speech to text with a small model. Returns a list of words with time + confidence."""
    print("Loading speech model (first time downloads it)...")
    model = WhisperModel("base.en", device="cpu", compute_type="int8")
    print("Transcribing...")
    segments, _ = model.transcribe(
        y, language="en", word_timestamps=True,
        initial_prompt=FILLER_PROMPT, beam_size=5,
    )
    return [w for seg in segments for w in seg.words]
 
 
def count_phrase(text, phrase):
    return len(re.findall(r"\b" + re.escape(phrase) + r"\b", text))
 
 
def analyze_words(words):
    tokens = [w.word.strip().lower().strip(".,!?;:\"'") for w in words]
    text = " ".join(tokens)
 
    duration = words[-1].end - words[0].start
    wpm = len(words) / duration * 60
 
    # Pauses between words (more accurate than the energy-based ones in step 1)
    gaps = [words[i + 1].start - words[i].end for i in range(len(words) - 1)]
    long_pauses = [g for g in gaps if g >= 0.5]
 
    # Pronunciation proxy: how sure the recognizer was about each word
    probs = [w.probability for w in words]
    mean_prob = sum(probs) / len(probs)
    unclear = [(t, round(p, 2)) for t, p in zip(tokens, probs) if p < 0.5]
 
    fillers = {f: count_phrase(text, f) for f in FILLERS if count_phrase(text, f)}
    hedges = {h: count_phrase(text, h) for h in HEDGES if count_phrase(text, h)}
    repeats = sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i - 1])
 
    return {
        "transcript": " ".join(w.word.strip() for w in words),
        "word_count": len(words),
        "wpm": wpm,
        "long_pauses": len(long_pauses),
        "mean_word_confidence": mean_prob,
        "unclear_words": unclear,
        "fillers": fillers,
        "filler_total": sum(fillers.values()),
        "hedges": hedges,
        "hedge_total": sum(hedges.values()),
        "repeats": repeats,
    }
 
 
def show_report2(f):
    n = f["word_count"]
    print("\n===== TRANSCRIPT BEHAVIOUR REPORT =====")
    print(f"Transcript: {f['transcript']}\n")
    print(f"Words                : {n}")
    print(f"Speaking speed       : {f['wpm']:.0f} words/min")
    print(f"Long pauses (>0.5 s) : {f['long_pauses']}")
    print(f"Filler words         : {f['filler_total']} {f['fillers']}")
    print(f"Hedging phrases      : {f['hedge_total']} {f['hedges']}")
    print(f"Repeated words       : {f['repeats']}")
    print(f"Word clarity score   : {f['mean_word_confidence']*100:.0f}/100")
    print(f"Unclear words        : {f['unclear_words']}")
 
    print("\nQuick feedback:")
    if f["wpm"] < 110:
        print("- Pace: a bit slow, try to speak with more energy")
    elif f["wpm"] > 170:
        print("- Pace: too fast, slow down so the listener can follow")
    else:
        print("- Pace: good")
    if n and f["filler_total"] / n * 100 > 5:
        print("- Too many filler words, pause silently instead of saying um/like")
    else:
        print("- Filler words: under control")
    if n and f["hedge_total"] / n * 100 > 3:
        print("- Lots of uncertain phrases (maybe / I think), sound more direct")
    if f["mean_word_confidence"] < 0.75:
        print("- Some words were unclear, speak more clearly and open your mouth more")
    else:
        print("- Pronunciation clarity: good")
 
 
if __name__ == "__main__":
    if len(sys.argv) > 1:
        y, _ = librosa.load(sys.argv[1], sr=SR)
    else:
        y = record(30)
 
    audio_features = analyze(y)
    if audio_features:
        show_report(audio_features)
 
    words = transcribe(y)
    if not words:
        print("No words recognized. Check your microphone.")
    else:
        show_report2(analyze_words(words))