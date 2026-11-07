import difflib
import json
import random
import re
import time
 
import numpy as np
import sounddevice as sd
 
from step1_audio_analysis import SR
from step2_transcript_analysis import transcribe
 
PASSAGES = [
    "I am excited to apply for this position because it matches my skills and my interests. "
    "In my previous project, I worked with a small team to solve a difficult problem under a "
    "tight deadline. I learned that clear communication and steady effort make the biggest "
    "difference to the final result.",
    "Throughout my studies I have enjoyed turning complicated ideas into simple working systems. "
    "When something fails, I stay calm, look carefully at the evidence, and try one change at a "
    "time. I believe this patient approach helps me grow quickly and support the people around me.",
    "The weather was unusually pleasant, so the whole team decided to discuss the quarterly "
    "schedule outside. Everyone shared their thoughts, and we agreed to finish the important "
    "tasks before Thursday and review our progress together.",
]
 
 
def tokens(text):
    return re.findall(r"[a-z']+", text.lower())
 
 
def record_until_enter():
    frames = []
 
    def callback(indata, n, t, s):
        frames.append(indata.copy())
 
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", callback=callback):
        input("Recording... read the passage aloud, then press Enter to stop.\n")
    return np.concatenate(frames)[:, 0] if frames else np.zeros(0, dtype="float32")
 
 
def compare(reference, heard):
    """reference: text. heard: list of (word, confidence). Returns the comparison results."""
    ref = tokens(reference)
    hyp = [re.sub(r"[^a-z']", "", w.lower()) for w, _ in heard]
    prob = [p for _, p in heard]
 
    sm = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    correct, missed, wrong, shaky, matched_conf = 0, [], [], [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            correct += i2 - i1
            for k in range(i2 - i1):
                matched_conf.append(prob[j1 + k])
                if prob[j1 + k] < 0.6:
                    shaky.append(ref[i1 + k])
        elif tag == "delete":
            missed += ref[i1:i2]
        elif tag == "replace":
            pairs = min(i2 - i1, j2 - j1)
            wrong += [(ref[i1 + k], hyp[j1 + k]) for k in range(pairs)]
            missed += ref[i1 + pairs:i2]
 
    accuracy = correct / max(1, len(ref)) * 100
    conf = float(np.mean(matched_conf)) if matched_conf else 0.0
    conf_score = max(0.0, min(100.0, (conf - 0.5) / 0.5 * 100))
    return {"accuracy": accuracy, "pronunciation_score": 0.6 * accuracy + 0.4 * conf_score,
            "missed": missed, "wrong": wrong, "shaky": shaky, "word_count": len(ref)}
 
 
if __name__ == "__main__":
    passage = random.choice(PASSAGES)
    print("\nREAD THIS PASSAGE ALOUD, clearly and at a natural pace:\n")
    print(passage + "\n")
    input("Press Enter when you are ready to start...")
    y = record_until_enter()
    if len(y) < SR * 3:
        raise SystemExit("Recording too short.")
 
    print("Analyzing...")
    words = transcribe(y)
    if not words:
        raise SystemExit("No words recognized. Speak louder and clearer.")
    r = compare(passage, [(w.word, w.probability) for w in words])
 
    print("\n===== READING / PRONUNCIATION RESULT =====")
    print(f"Pronunciation score : {r['pronunciation_score']:.0f}/100")
    print(f"Words read correctly: {r['accuracy']:.0f}% of {r['word_count']}")
    if r["wrong"]:
        print("Said differently    :", ", ".join(f"{a} -> {b}" for a, b in r["wrong"]))
    if r["missed"]:
        print("Skipped / unclear   :", ", ".join(r["missed"]))
    if r["shaky"]:
        print("Said but unclear    :", ", ".join(sorted(set(r["shaky"]))))
    if not (r["wrong"] or r["missed"] or r["shaky"]):
        print("Excellent! Every word was clear.")
 
    with open("reading_results.json", "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M"), **r}) + "\n")
 