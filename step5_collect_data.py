import csv
import os
import time
 
import soundfile as sf
 
from features import FEATURES, LABELS, extract_features
from step1_audio_analysis import SR, analyze
from step2_transcript_analysis import analyze_words, transcribe
from step4_full_system import DURATION, record_both
 
DATASET = "dataset.csv"
COLUMNS = ["speaker", "audio_file"] + FEATURES + LABELS
 
QUESTIONS = [
    "Tell me about yourself.",
    "Describe a time you faced a challenge and how you handled it.",
    "Tell me about a time you worked in a team.",
    "What is your biggest strength?",
    "Describe a mistake you made and what you learned.",
    "Why do you want this job?",
]
 
 
def check_header():
    if os.path.exists(DATASET):
        with open(DATASET, encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        if header != COLUMNS:
            raise SystemExit("dataset.csv was made with older features. Rename it (for example "
                             "dataset_old.csv) and collect again, otherwise the columns will not match.")
 
 
def ask_rating(name):
    while True:
        s = input(f"  {name} (1-5): ").strip()
        try:
            r = float(s)
            if 1 <= r <= 5:
                return r
        except ValueError:
            pass
        print("  Please type a number from 1 to 5 (decimals like 3.5 are fine).")
 
 
def save_row(row):
    new_file = not os.path.exists(DATASET)
    with open(DATASET, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
 
 
def count_rows():
    if not os.path.exists(DATASET):
        return 0
    with open(DATASET, encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)
 
 
if __name__ == "__main__":
    check_header()
    os.makedirs("recordings", exist_ok=True)
    print(f"Samples collected so far: {count_rows()}  (aim for 40 or more)\n")
 
    i = count_rows()
    while True:
        speaker = input("Speaker name or ID (press Enter to finish): ").strip()
        if not speaker:
            break
 
        print(f"\nQuestion: {QUESTIONS[i % len(QUESTIONS)]}")
        input("Press Enter when ready to start recording...")
        y, video = record_both(DURATION)
 
        if video is None or len(y) < SR * 5:
            print("Bad recording (face not found or too short). Skipped, try again.\n")
            continue
 
        print("Analyzing... (the scores stay hidden so the rater is not biased)")
        audio = analyze(y)
        words = transcribe(y)
        if audio is None or not words:
            print("No speech detected. Skipped, try again.\n")
            continue
        feats = extract_features(audio, analyze_words(words), video)
 
        path = f"recordings/{speaker}_{int(time.time())}.wav"
        sf.write(path, y, SR)
 
        print("\nRATER: judge ONLY how it was delivered, not whether the answer was good.")
        print("1 = very poor, 3 = average, 5 = excellent")
        labels = {name: ask_rating(name.capitalize()) for name in LABELS}
 
        save_row({"speaker": speaker, "audio_file": path, **feats, **labels})
        i += 1
        print(f"Saved! Total samples: {count_rows()}\n")
 
    print("Done. Next, run step6_train_model.py")
 