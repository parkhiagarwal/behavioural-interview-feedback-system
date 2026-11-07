import html
import json
import os
import random
import re
import time
import webbrowser
from pathlib import Path
 
from features import extract_features
from step1_audio_analysis import SR, analyze
from step2_transcript_analysis import FILLERS, HEDGES, analyze_words, transcribe
from step3_video_analysis import calibrate
from step4_full_system import (DIMENSIONS, DURATION, build_components,
                               dimension_score, label, record_both)
from step5_collect_data import QUESTIONS
from timeline import audio_timelines
 
try:
    from step6_train_model import MODEL_FILE, predict
    HAS_MODEL = os.path.exists(MODEL_FILE)
except ImportError:
    HAS_MODEL = False
 
# How much the trained model counts in the final score (0 = formulas only, 1 = model only).
# With very little training data, keep this at 0.5 or lower.
LEARNED_WEIGHT = 0.5
HISTORY_FILE = "history.json"
 
DRILLS = {
    "pace": "Read a news article aloud for 1 minute with a timer. Aim for about 140 words per minute. Repeat daily.",
    "pauses": "Answer 'Tell me about your day' without stopping. When you lose your thought, finish the sentence calmly and move on.",
    "fillers": "Answer any question for 30 seconds. Each time you want to say um or like, close your mouth and take a silent breath instead. Do it 3 times.",
    "hedging": "Rewrite 5 of your answers replacing 'I think I maybe' with 'I did' or 'I decided'. Then say them aloud.",
    "voice_expression": "Read a short story aloud and exaggerate the emotion and stress on key words. Then bring it down to a natural level.",
    "word_clarity": "Read a passage slowly and finish every word. Then use reading mode (step_reading_mode.py) to check which words were unclear.",
    "unclear_words": "Do tongue twisters slowly, then faster, opening your mouth wide on every word. Use reading mode to track improvement.",
    "eye_contact": "Stick a small arrow beside the camera lens. Answer 3 questions looking only at it, and glance away only when you finish a thought.",
    "looking_down": "Raise the screen or laptop so the camera is at eye level, and keep notes out of sight while practising.",
    "head_steady": "Place a sticky note on the wall behind the screen and keep your nose pointing at the camera for a whole answer.",
    "body_calm": "Rest your hands on the desk and use at most two deliberate gestures per answer.",
    "posture": "Sit with your back against the chair and shoulders level. Reset your posture after each sentence for a week.",
    "blink_calm": "Take one slow breath before you start and relax your face. Fast blinking usually slows when you breathe slowly.",
    "warmth": "Start the answer with a small natural smile and smile again when you mention something you are proud of.",
}
 
 
def to_100(rating):
    """Convert a 1-5 rating to a 0-100 score."""
    return (rating - 1) / 4 * 100
 
 
def evaluate(y, video):
    audio = analyze(y)
    words = transcribe(y)
    if audio is None or not words:
        return None
    text = analyze_words(words)
 
    comp = build_components(audio, text, video)
    formula = {name: dimension_score(w, comp) for name, w in DIMENSIONS.items()}
 
    learned = None
    if HAS_MODEL:
        try:
            pred = predict(extract_features(audio, text, video))
            learned = {name: to_100(pred[name.lower()]) for name in formula}
        except Exception as e:  # for example, model trained with older features
            print(f"(Trained model skipped: {e})")
 
    final = {}
    for name in formula:
        final[name] = ((1 - LEARNED_WEIGHT) * formula[name] + LEARNED_WEIGHT * learned[name]
                       if learned else formula[name])
 
    tips = sorted(comp.items(), key=lambda kv: kv[1][0])[:3]
    tips = [(k, s, adv) for k, (s, adv) in tips if s < 75]
    return {"audio": audio, "text": text, "video": video, "formula": formula, "learned": learned,
            "final": final, "overall": sum(final.values()) / len(final), "tips": tips,
            "tl": audio_timelines(y)}
 
 
# ---------------- history ----------------
def load_history():
    p = Path(HISTORY_FILE)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}
 
 
def save_attempt(name, question, res):
    history = load_history()
    history.setdefault(name, []).append({
        "time": time.strftime("%Y-%m-%d %H:%M"), "question": question,
        "scores": {k: round(v, 1) for k, v in res["final"].items()},
        "overall": round(res["overall"], 1)})
    Path(HISTORY_FILE).write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history[name]
 
 
# ---------------- report pieces ----------------
def color(score):
    return "#2e9e5b" if score >= 80 else "#6aa84f" if score >= 65 else "#e69f00" if score >= 50 else "#d9534f"
 
 
def delta_html(now, before):
    d = now - before
    if abs(d) < 0.5:
        return '<span style="color:#777">no change</span>'
    arrow, c = ("&#9650;", "#2e9e5b") if d > 0 else ("&#9660;", "#d9534f")
    return f'<span style="color:{c}">{arrow} {d:+.0f}</span>'
 
 
def svg_chart(values, lo, hi, kind="line", fill="#4f7cff", unit=""):
    """Small dependency-free SVG chart. None values leave a gap."""
    if not values or all(v is None for v in values):
        return '<p class="note">Not enough data for this chart.</p>'
    W, H, P = 680, 130, 26
    n = len(values)
    if hi - lo < 1e-9:
        hi = lo + 1
 
    def pos(i, v):
        x = P + (W - 2 * P) * ((i / max(n - 1, 1)) if kind == "line" else (i + 0.5) / n)
        y = H - P - (H - 2 * P) * max(0.0, min(1.0, (v - lo) / (hi - lo)))
        return x, y
 
    parts = [f'<line x1="{P}" y1="{H - P}" x2="{W - P}" y2="{H - P}" stroke="#ccc"/>']
    if kind == "bar":
        bw = (W - 2 * P) / n * 0.7
        for i, v in enumerate(values):
            if v is not None:
                x, y = pos(i, v)
                parts.append(f'<rect x="{x - bw / 2:.1f}" y="{y:.1f}" width="{bw:.1f}" '
                             f'height="{H - P - y:.1f}" fill="{fill}" rx="2"/>')
    else:
        run = []
        for i, v in enumerate(list(values) + [None]):
            if v is not None:
                run.append(pos(i, v))
                continue
            if len(run) == 1:
                parts.append(f'<circle cx="{run[0][0]:.1f}" cy="{run[0][1]:.1f}" r="3" fill="{fill}"/>')
            elif len(run) > 1:
                pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in run)
                parts.append(f'<polyline points="{pts}" fill="none" stroke="{fill}" stroke-width="2.5"/>')
            run = []
    parts.append(f'<text x="2" y="{P + 4}" font-size="11" fill="#888">{hi:.0f}{unit}</text>')
    parts.append(f'<text x="2" y="{H - P}" font-size="11" fill="#888">{lo:.0f}{unit}</text>')
    parts.append(f'<text x="{P}" y="{H - 6}" font-size="11" fill="#888">0s</text>')
    parts.append(f'<text x="{W - P - 24}" y="{H - 6}" font-size="11" fill="#888">{n}s</text>')
    return f'<svg viewBox="0 0 {W} {H}" width="100%">{"".join(parts)}</svg>'
 
 
def highlight(text, unclear):
    """Transcript with filler words, hedging phrases and unclear words coloured."""
    toks = re.findall(r"\S+", text)
    clean = [re.sub(r"[^a-z']", "", t.lower()) for t in toks]
    cls = [""] * len(toks)
    phrases = [(p.split(), "hedge") for p in HEDGES] + [(f.split(), "filler") for f in FILLERS]
    for words, c in sorted(phrases, key=lambda x: -len(x[0])):
        n = len(words)
        for i in range(len(toks) - n + 1):
            if clean[i:i + n] == words and not any(cls[i:i + n]):
                cls[i:i + n] = [c] * n
    for i, w in enumerate(clean):
        if not cls[i] and w in unclear:
            cls[i] = "unclear"
    return " ".join(f'<span class="{c}">{html.escape(t)}</span>' if c else html.escape(t)
                    for t, c in zip(toks, cls))
 
 
CSS = """
body{font-family:Segoe UI,Arial,sans-serif;max-width:780px;margin:30px auto;padding:0 16px;color:#222;background:#f6f7f9}
.card{background:#fff;border-radius:12px;padding:20px 24px;margin-bottom:16px;box-shadow:0 1px 4px rgba(0,0,0,.08)}
h1{margin:0 0 4px} h2{margin-top:0} .q{color:#555}
.row{margin-bottom:16px} .top{display:flex;justify-content:space-between;margin-bottom:6px}
.track{background:#e6e8eb;border-radius:8px;height:14px;overflow:hidden} .fill{height:100%}
small,.note{color:#777;font-size:13px} table{width:100%;border-collapse:collapse} td{padding:7px 4px;border-bottom:1px solid #eee}
td:last-child{text-align:right;font-weight:600} li{margin-bottom:10px} .drill{color:#444;font-size:14px}
span.filler{background:#ffe0b2;border-radius:4px;padding:0 3px}
span.hedge{background:#d1c4ff;border-radius:4px;padding:0 3px} span.unclear{background:#ffc9c9;border-radius:4px;padding:0 3px}
.legend span{margin-right:12px;font-size:13px} button{padding:10px 18px;border:0;border-radius:8px;background:#4f7cff;color:#fff;font-size:15px;cursor:pointer}
@media print{.noprint{display:none} body{background:#fff}}
"""
 
 
def render_report(res, question, name, attempts):
    a, t, v = res["audio"], res["text"], res["video"]
    prev = attempts[-2] if len(attempts) >= 2 else None
 
    overall_delta = f" &middot; {delta_html(res['overall'], prev['overall'])} vs last attempt" if prev else ""
    bars = ""
    for dim, s in res["final"].items():
        detail = f"Formula score {res['formula'][dim]:.0f}"
        if res["learned"]:
            detail += f" &middot; Learned model {res['learned'][dim]:.0f}"
        change = f" &nbsp;{delta_html(s, prev['scores'][dim])}" if prev and dim in prev["scores"] else ""
        bars += (f'<div class="row"><div class="top"><b>{dim}</b><span>{s:.0f}/100 &middot; {label(s)}{change}</span></div>'
                 f'<div class="track"><div class="fill" style="width:{s:.0f}%;background:{color(s)}"></div></div>'
                 f'<small>{detail}</small></div>')
 
    if res["tips"]:
        tips = "".join(
            f"<li><b>{k.replace('_', ' ').title()}</b> ({s:.0f}/100): {html.escape(adv)}"
            f"<div class='drill'>&#127939; Practice: {html.escape(DRILLS.get(k, ''))}</div></li>"
            for k, s, adv in res["tips"])
    else:
        tips = "<li>Nothing major to fix. Great delivery!</li>"
 
    vol, pit = res["tl"]["volume"], res["tl"]["pitch"]
    vv, pp = [x for x in vol if x is not None], [x for x in pit if x is not None]
    charts = ""
    if vv:
        charts += f"<h3>Volume (dB, higher = louder)</h3>{svg_chart(vol, min(vv) - 3, max(vv) + 3, 'line', '#4f7cff', ' dB')}"
    if pp:
        charts += f"<h3>Pitch (Hz): flat line = monotone voice</h3>{svg_chart(pit, min(pp) - 10, max(pp) + 10, 'line', '#9b59b6', ' Hz')}"
    if v.get("eye_timeline"):
        charts += f"<h3>Eye contact in each second (tall bar = looking at the camera)</h3>{svg_chart(v['eye_timeline'], 0, 1, 'bar', '#2e9e5b')}"
 
    rows = [("Speaking speed", f"{t['wpm']:.0f} words/min"), ("Filler words", f"{t['filler_total']}"),
            ("Hedging phrases", f"{t['hedge_total']}"), ("Long pauses", f"{t['long_pauses']}"),
            ("Word clarity", f"{t['mean_word_confidence'] * 100:.0f}/100"),
            ("Pitch variation", f"{a['pitch_variation_semitones']:.1f} semitones"),
            ("Eye contact", f"{v['eye_contact_ratio'] * 100:.0f}% of the time")]
    if "looking_down_ratio" in v:
        rows += [("Looking down", f"{v['looking_down_ratio'] * 100:.0f}% of the time"),
                 ("Blink rate", f"{v['blink_rate']:.0f} per minute"),
                 ("Smiling", f"{v['smile_ratio'] * 100:.0f}% of the time")]
    rows += [("Head movement", f"{v['head_movement']:.3f}"), ("Body movement", f"{v['body_movement']:.3f}")]
    table = "".join(f"<tr><td>{k}</td><td>{val}</td></tr>" for k, val in rows)
 
    unclear = {w for w, _ in t["unclear_words"]}
    mode = "MediaPipe eye tracking" if v.get("mode") == "mediapipe" else "basic eye detection"
    calibrated = ", personally calibrated" if v.get("calibrated") else ""
 
    progress = ""
    if len(attempts) >= 2:
        progress = ('<div class="card"><h2>Your progress</h2>'
                    + svg_chart([x["overall"] for x in attempts], 0, 100, "bar", "#4f7cff")
                    + '<p class="note">Overall score for each attempt, oldest to newest.</p></div>')
 
    page = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Interview Feedback</title>
<style>{CSS}</style></head><body>
<div class="card"><h1>Interview Feedback Report</h1>
<div class="q">{html.escape(name)} &middot; {time.strftime('%d %b %Y %H:%M')}</div>
<div class="q">Question: {html.escape(question)}</div>
<div style="font-size:48px;font-weight:700;color:{color(res['overall'])}">{res['overall']:.0f}<span style="font-size:20px;color:#777">/100 overall &middot; {label(res['overall'])}{overall_delta}</span></div></div>
<div class="card"><h2>Scores</h2>{bars}</div>
<div class="card"><h2>What to improve</h2><ul>{tips}</ul></div>
<div class="card"><h2>Your delivery over time</h2>{charts or '<p class="note">Not enough data for charts.</p>'}</div>
<div class="card"><h2>Measurements</h2><table>{table}</table><p class="note">Video analysis: {mode}{calibrated}.</p></div>
<div class="card"><h2>Transcript</h2><p>{highlight(t['transcript'], unclear)}</p>
<div class="legend"><span class="filler">filler word</span><span class="hedge">uncertain phrase</span><span class="unclear">unclear pronunciation</span></div>
<p class="note">This system judges delivery (pace, pauses, fillers, clarity, eye contact, steadiness), never whether the answer is correct.</p></div>
{progress}
<p class="noprint"><button onclick="window.print()">Save as PDF / Print</button></p>
</body></html>"""
 
    Path("reports").mkdir(exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_-]", "", name) or "guest"
    path = Path("reports") / f"report_{safe}_{int(time.time())}.html"
    path.write_text(page, encoding="utf-8")
    return path
 
 
if __name__ == "__main__":
    print("=" * 50)
    print(" BEHAVIOURAL INTERVIEW FEEDBACK SYSTEM")
    print("=" * 50)
    print("Trained model found: scores blend formulas + model." if HAS_MODEL
          else "No model.pkl found: using formula scores only.")
 
    name = input("\nYour name (used to track your progress): ").strip() or "guest"
    input("Calibration: sit as you will during the interview, then press Enter and look at the camera for 4 seconds...")
    calibrate()
 
    while True:
        question = random.choice(QUESTIONS)
        print(f"\nQUESTION: {question}")
        input("Press Enter to start recording (a camera window will open)...")
 
        y, video = record_both(DURATION)
        if video is None or len(y) < SR * 5:
            print("Face not found or recording too short. Please try again.")
        else:
            print("Analyzing, please wait...")
            result = evaluate(y, video)
            if result is None:
                print("No speech detected. Please try again.")
            else:
                attempts = save_attempt(name, question, result)
                report = render_report(result, question, name, attempts)
                print(f"Report saved: {report}")
                webbrowser.open(report.resolve().as_uri())
 
        if input("\nAnother question? (y/n): ").strip().lower() != "y":
            break
    print("Goodbye!")
 