import os
import sys
import time
from collections import Counter
 
import cv2
import numpy as np
 
MP_ERROR = ""
try:
    import mediapipe as mp
    HAS_MP = hasattr(mp, "solutions")
    if not HAS_MP:
        MP_ERROR = f"mediapipe {getattr(mp, '__version__', '?')} has no 'solutions' (old face mesh removed)"
except Exception as e:  # not installed, or broken / incompatible install
    mp = None
    HAS_MP = False
    MP_ERROR = f"{type(e).__name__}: {e}"
 
 
def _load_cascade(name):
    """Find an OpenCV Haar cascade file. Returns None if this OpenCV build does not ship it."""
    places = []
    if hasattr(cv2, "data"):
        places.append(cv2.data.haarcascades + name)
    places.append(os.path.join(os.path.dirname(cv2.__file__), "data", name))
    for path in places:
        if os.path.exists(path):
            clf = cv2.CascadeClassifier(path)
            if not clf.empty():
                return clf
    return None
 
# ---- Limits you can CALIBRATE (other files import the first four names) ----
EYE_CONTACT_GOOD = 0.65   # share of the time looking at the camera
HEAD_MOVE_HIGH = 0.02     # average head step per frame, relative to face width
BODY_MOVE_HIGH = 0.08     # fraction of pixels changing below the face
SLOUCH_LIMIT = 0.25       # face sinking, relative to face width
 
GAZE_H_TOL = 0.15         # how far the iris may move sideways inside the eye
GAZE_V_TOL = 0.22         # how far the iris may move downward inside the eye
YAW_TOL = 0.14            # head turn left/right
PITCH_TOL = 0.10          # head nod up/down
DOWN_PITCH = 0.10         # head tilted down by this much counts as "looking down"
BLINK_FRACTION = 0.60     # eye opening below 60% of normal = blink
SMILE_FACTOR = 1.10       # mouth 10% wider than normal = smile
 
BASELINE = None           # filled in by calibrate()
TOL = None                # personal limits, filled in by calibrate()
DEBUG = False             # run:  python step3_video_analysis.py debug
DEFAULT_TOL = {"gaze_h": GAZE_H_TOL, "gaze_v": GAZE_V_TOL, "yaw": YAW_TOL,
               "pitch": PITCH_TOL, "down": DOWN_PITCH}
 
# MediaPipe Face Mesh landmark numbers
L_IRIS, R_IRIS = 468, 473
L_C1, L_C2, R_C1, R_C2 = 33, 133, 362, 263      # eye corners
L_UP, L_LOW, R_UP, R_LOW = 159, 145, 386, 374   # eyelids
NOSE, CHIN, FACE_L, FACE_R = 1, 152, 234, 454
MOUTH_L, MOUTH_R = 61, 291
 
STATES = {
    "contact": ("Eye contact", (0, 200, 0)),
    "blink": ("Eye contact", (0, 200, 0)),
    "away": ("Looking away", (0, 0, 255)),
    "down": ("Looking down", (0, 140, 255)),
}
 
 
# --------------------------------------------------------------------------
# MediaPipe helpers
# --------------------------------------------------------------------------
def _resize(frame):
    h0, w0 = frame.shape[:2]
    return cv2.resize(frame, (640, int(h0 * 640 / w0)))
 
 
def _new_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
 
 
def _read_points(mesh, frame):
    H, W = frame.shape[:2]
    res = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    if len(lm) < 478:
        return None
    return np.array([[p.x * W, p.y * H] for p in lm])
 
 
def _dist(a, b):
    return float(np.linalg.norm(a - b))
 
 
def _eye_ratio(iris, c1, c2):
    """Position of the iris between the eye corners, left-to-right in the image (0.5 = centered)."""
    a, b = (c1, c2) if c1[0] <= c2[0] else (c2, c1)
    ab = b - a
    return float(np.dot(iris - a, ab) / (np.dot(ab, ab) + 1e-9))
 
 
def _metrics(p):
    face_w = _dist(p[FACE_L], p[FACE_R]) + 1e-9
    eye_mid_y = (p[L_UP][1] + p[R_UP][1]) / 2
    return {
        "gaze_h": (_eye_ratio(p[L_IRIS], p[L_C1], p[L_C2]) + _eye_ratio(p[R_IRIS], p[R_C1], p[R_C2])) / 2,
        "gaze_v": ((p[L_IRIS][1] - p[L_UP][1]) / (p[L_LOW][1] - p[L_UP][1] + 1e-9)
                   + (p[R_IRIS][1] - p[R_UP][1]) / (p[R_LOW][1] - p[R_UP][1] + 1e-9)) / 2,
        "ear": (_dist(p[L_UP], p[L_LOW]) / (_dist(p[L_C1], p[L_C2]) + 1e-9)
                + _dist(p[R_UP], p[R_LOW]) / (_dist(p[R_C1], p[R_C2]) + 1e-9)) / 2,
        "yaw": float((p[NOSE][0] - (p[FACE_L][0] + p[FACE_R][0]) / 2) / face_w),
        "pitch": float((p[NOSE][1] - eye_mid_y) / (p[CHIN][1] - eye_mid_y + 1e-9)),
        "mouth": _dist(p[MOUTH_L], p[MOUTH_R]) / face_w,
        "face_w": face_w,
        "nose_x": float(p[NOSE][0] / face_w),
        "nose_y": float(p[NOSE][1] / face_w),
    }
 
 
def _baseline_from(ms, session=False):
    """YOUR normal values. For a whole session, open-eye and neutral-mouth values are
    estimated from percentiles so blinks and smiles do not distort them."""
    g = lambda k: [m[k] for m in ms]
    return {
        "gaze_h": float(np.median(g("gaze_h"))), "gaze_v": float(np.median(g("gaze_v"))),
        "yaw": float(np.median(g("yaw"))), "pitch": float(np.median(g("pitch"))),
        "ear": float(np.percentile(g("ear"), 75)),
        "mouth": float(np.percentile(g("mouth"), 20 if session else 50)),
    }
 
 
def _tol():
    return TOL or DEFAULT_TOL
 
 
def _tol_from(ms):
    """Personal limits: never tighter than the defaults, wider if YOUR tracking is jittery."""
    sd = lambda k: float(np.std([m[k] for m in ms]))
    return {"gaze_h": max(GAZE_H_TOL, 4 * sd("gaze_h")), "gaze_v": max(GAZE_V_TOL, 4 * sd("gaze_v")),
            "yaw": max(YAW_TOL, 4 * sd("yaw")), "pitch": max(PITCH_TOL, 4 * sd("pitch")),
            "down": max(DOWN_PITCH, 4 * sd("pitch"))}
 
 
def _classify(m, base):
    t = _tol()
    if m["ear"] < BLINK_FRACTION * base["ear"]:
        return "blink"
    if m["pitch"] - base["pitch"] > t["down"] or m["gaze_v"] - base["gaze_v"] > t["gaze_v"]:
        return "down"
    if (abs(m["gaze_h"] - base["gaze_h"]) > t["gaze_h"] or abs(m["yaw"] - base["yaw"]) > t["yaw"]
            or abs(m["pitch"] - base["pitch"]) > t["pitch"]):
        return "away"
    return "contact"
 
 
def _reasons(m, base):
    """Why a frame counts as 'looking away' (used for the debug summary)."""
    t, r = _tol(), []
    if abs(m["gaze_h"] - base["gaze_h"]) > t["gaze_h"]:
        r.append("eyes sideways")
    if abs(m["yaw"] - base["yaw"]) > t["yaw"]:
        r.append("head turned")
    if abs(m["pitch"] - base["pitch"]) > t["pitch"]:
        r.append("head up/down")
    return r
 
 
def _smooth(states, k=5):
    """Remove flicker: a blink keeps the previous state, then each frame takes the majority of its neighbours."""
    filled, last = [], "contact"
    for st in states:
        if st != "blink":
            last = st
        filled.append(last)
    half, out = k // 2, []
    for i in range(len(filled)):
        out.append(Counter(filled[max(0, i - half):i + half + 1]).most_common(1)[0][0])
    return out
 
 
def calibrate(seconds=4, source=0):
    """Look at the camera and sit naturally. Stores your personal baseline."""
    global BASELINE, TOL
    if not HAS_MP:
        print(f"MediaPipe unavailable ({MP_ERROR}). Skipping calibration (basic mode).")
        return None
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Could not open the camera for calibration.")
        return None
    mesh, ms, start = _new_mesh(), [], time.time()
    while time.time() - start < seconds:
        ok, frame = cap.read()
        if not ok:
            break
        frame = _resize(frame)
        pts = _read_points(mesh, frame)
        if pts is not None:
            ms.append(_metrics(pts))
        left = max(0, seconds - (time.time() - start))
        cv2.putText(frame, f"CALIBRATION: look at the camera, sit naturally ({left:.0f}s)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    mesh.close()
    if len(ms) < 15:
        print("Calibration failed (face not found). Continuing without it.")
        return None
    BASELINE = _baseline_from(ms)
    TOL = _tol_from(ms)
    print("Calibration done.")
    return BASELINE
 
 
# --------------------------------------------------------------------------
# Main analysis with MediaPipe
# --------------------------------------------------------------------------
def _analyze_mesh(source, max_seconds, show):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Could not open the camera / video file.")
        return None
    mesh = _new_mesh()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames, bin_total, body = [], {}, []
    total, prev_blur, now = 0, None, 0.0
    live_contact = live_total = 0
    start = time.time()
 
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = (time.time() - start) if source == 0 else total / fps
        if source == 0 and now > max_seconds:
            break
        frame = _resize(frame)
        H, W = frame.shape[:2]
        blur = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        total += 1
        sec = int(now)
        bin_total[sec] = bin_total.get(sec, 0) + 1
 
        pts = _read_points(mesh, frame)
        label, color = "No face: looking away", (0, 0, 255)
        if pts is None:
            live_total += 1
        else:
            m = _metrics(pts)
            m["t"], m["sec"] = now, sec
            frames.append(m)
            base = BASELINE if BASELINE else (_baseline_from(frames[-120:], True) if len(frames) >= 15 else None)
            state = _classify(m, base) if base else "contact"
            label, color = STATES[state]
            if DEBUG and base:
                t = _tol()
                dbg = (f"eyes {m['gaze_h'] - base['gaze_h']:+.2f}/{t['gaze_h']:.2f}  "
                       f"yaw {m['yaw'] - base['yaw']:+.2f}/{t['yaw']:.2f}  "
                       f"pitch {m['pitch'] - base['pitch']:+.2f}/{t['pitch']:.2f}  "
                       f"down {m['gaze_v'] - base['gaze_v']:+.2f}/{t['gaze_v']:.2f}")
                cv2.putText(frame, dbg, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            if state != "blink":
                live_total += 1
                live_contact += state == "contact"
 
            if prev_blur is not None:  # movement of the area below the face
                fw = m["face_w"]
                y0 = min(int(pts[:, 1].max()), H - 2)
                x0, x1 = max(0, int(pts[:, 0].min() - fw)), min(W, int(pts[:, 0].max() + fw))
                diff = cv2.absdiff(blur[y0:, x0:x1], prev_blur[y0:, x0:x1])
                if diff.size > 0:
                    body.append(float((diff > 25).mean()))
            for i in (L_IRIS, R_IRIS):
                cv2.circle(frame, tuple(int(v) for v in pts[i]), 2, (0, 255, 255), -1)
        prev_blur = blur
 
        if show:
            pct = live_contact / live_total * 100 if live_total else 0
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Eye contact so far: {pct:.0f}%   time: {now:.0f}s",
                        (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Interview camera (press q to stop)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
 
    cap.release()
    cv2.destroyAllWindows()
    mesh.close()
    if total == 0 or len(frames) < 10:
        return None
 
    base = BASELINE if BASELINE else _baseline_from(frames, session=True)
    raw = [_classify(m, base) for m in frames]
    blinking = [s == "blink" for s in raw]
    states = _smooth(raw)
    away_reasons = Counter()
    for m, st in zip(frames, raw):
        if st == "away":
            away_reasons.update(_reasons(m, base))
    state_share = {k: round(v / len(states), 2) for k, v in Counter(states).items()}
    duration = max(1.0, frames[-1]["t"] - frames[0]["t"])
    blink_events = sum(1 for i, b in enumerate(blinking) if b and (i == 0 or not blinking[i - 1]))
 
    nose = np.array([[m["nose_x"], m["nose_y"]] for m in frames])
    steps = np.linalg.norm(np.diff(nose, axis=0), axis=1)
    fw = np.array([m["face_w"] for m in frames])
    third = max(1, len(frames) // 3)
 
    timeline = []
    for s in range(int(now) + 1):
        c = sum(1 for m, st in zip(frames, states) if m["sec"] == s and st == "contact")
        timeline.append(c / max(1, bin_total.get(s, 0)))
 
    return {
        "mode": "mediapipe",
        "calibrated": BASELINE is not None,
        "duration_s": duration,
        "face_visible_ratio": len(frames) / total,
        "eye_contact_ratio": sum(1 for s in states if s == "contact") / max(1, total),
        "looking_down_ratio": sum(1 for s in states if s == "down") / max(1, total),
        "state_share": state_share,
        "away_reasons": dict(away_reasons),
        "blink_rate": blink_events / duration * 60,
        "smile_ratio": sum(1 for m in frames if m["mouth"] > SMILE_FACTOR * base["mouth"]) / len(frames),
        "head_movement": float(np.mean(steps)) if len(steps) else 0.0,
        "head_yaw_std": float(np.std([m["yaw"] for m in frames])),
        "lean_variation": float(np.std(fw) / np.mean(fw)),
        "slouch_drift": float(nose[-third:, 1].mean() - nose[:third, 1].mean()),
        "body_movement": float(np.mean(body)) if body else 0.0,
        "eye_timeline": [round(float(x), 2) for x in timeline],
    }
 
 
# --------------------------------------------------------------------------
# Fallback: older OpenCV-only analysis (used if MediaPipe is missing)
# --------------------------------------------------------------------------
def _analyze_basic(source, max_seconds, show):
    FACE = _load_cascade("haarcascade_frontalface_default.xml")
    EYE = _load_cascade("haarcascade_eye.xml")
    if FACE is None or EYE is None:
        print("This OpenCV version has no face detector files (OpenCV 5 removed them).")
        print('Fix: pip uninstall -y opencv-python opencv-contrib-python, then: pip install "opencv-contrib-python<5"')
        return None
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Could not open the camera / video file.")
        return None
 
    total = face_frames = eye_frames = 0
    centers, sizes, body_motion = [], [], []
    prev_blur = None
    start = time.time()
 
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if source == 0 and time.time() - start > max_seconds:
            break
        frame = _resize(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        H, W = gray.shape
        total += 1
 
        faces = FACE.detectMultiScale(gray_eq, 1.1, 5, minSize=(80, 80))
        face = max(faces, key=lambda f: f[2] * f[3]) if len(faces) else None
        label, color = "Looking away / no face", (0, 0, 255)
        if face is not None:
            x, y, w, h = face
            face_frames += 1
            centers.append((x + w / 2, y + h / 2))
            sizes.append(w)
            eyes = EYE.detectMultiScale(gray_eq[y:y + h // 2, x:x + w], 1.1, 5,
                                        minSize=(max(10, w // 10), max(10, w // 10)))
            if len(eyes) >= 2:
                eye_frames += 1
                label, color = "Eye contact", (0, 200, 0)
            else:
                label, color = "Face found, eyes unclear", (0, 200, 255)
            if prev_blur is not None:
                y0, x0, x1 = min(y + h, H - 2), max(0, int(x - w)), min(W, int(x + 2 * w))
                diff = cv2.absdiff(blur[y0:, x0:x1], prev_blur[y0:, x0:x1])
                if diff.size > 0:
                    body_motion.append(float((diff > 25).mean()))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        prev_blur = blur
 
        if show:
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("Interview camera (press q to stop)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
 
    cap.release()
    cv2.destroyAllWindows()
    if total == 0 or len(centers) < 10:
        return None
 
    c, s = np.array(centers), np.array(sizes)
    face_width = float(np.median(s))
    steps = np.linalg.norm(np.diff(c, axis=0), axis=1) / face_width
    third = max(1, len(c) // 3)
    return {
        "mode": "basic",
        "calibrated": False,
        "duration_s": time.time() - start if source == 0 else total / 30,
        "face_visible_ratio": face_frames / total,
        "eye_contact_ratio": eye_frames / total,
        # the old detector is jittery, so rescale to the same limits as the new mode
        "head_movement": float(np.mean(steps)) * (HEAD_MOVE_HIGH / 0.06),
        "lean_variation": float(np.std(s) / np.mean(s)),
        "slouch_drift": float((c[-third:, 1].mean() - c[:third, 1].mean()) / face_width),
        "body_movement": float(np.mean(body_motion)) if body_motion else 0.0,
    }
 
 
def analyze_video(source=0, max_seconds=20, show=True):
    if HAS_MP:
        try:
            return _analyze_mesh(source, max_seconds, show)
        except Exception as e:
            print(f"MediaPipe mode failed ({e}). Using basic mode.")
    else:
        print(f"(MediaPipe unavailable: {MP_ERROR})")
    return _analyze_basic(source, max_seconds, show)
 
 
def show_video_report(f):
    print("\n===== VIDEO BEHAVIOUR REPORT =====")
    print(f"Mode                  : {'MediaPipe eye tracking' if f.get('mode') == 'mediapipe' else 'basic (rough)'}"
          f"{' + calibrated' if f.get('calibrated') else ''}")
    print(f"Face visible          : {f['face_visible_ratio']*100:.0f}% of the time")
    print(f"Eye contact           : {f['eye_contact_ratio']*100:.0f}% of the time")
    if "looking_down_ratio" in f:
        print(f"Looking down          : {f['looking_down_ratio']*100:.0f}% of the time")
        print(f"Blink rate            : {f['blink_rate']:.0f} per minute")
        print(f"Smiling               : {f['smile_ratio']*100:.0f}% of the time")
    if "state_share" in f:
        print("Eyes/head state       : " + ", ".join(f"{k} {v*100:.0f}%" for k, v in f["state_share"].items()))
        if f.get("away_reasons"):
            print("Reasons for 'away'    : " + ", ".join(f"{k} ({v} frames)" for k, v in f["away_reasons"].items()))
    print(f"Head movement         : {f['head_movement']:.3f}")
    print(f"Leaning back/forward  : {f['lean_variation']:.3f}")
    print(f"Sinking (slouch) drift: {f['slouch_drift']:.2f}")
    print(f"Body/hand movement    : {f['body_movement']:.3f}")
 
    print("\nQuick feedback:")
    print("- Eye contact: good" if f["eye_contact_ratio"] >= EYE_CONTACT_GOOD
          else "- Look at the camera lens more often, like you would look at the interviewer")
    if f.get("looking_down_ratio", 0) > 0.25:
        print("- You looked down a lot. Keep your head and eyes up, even while thinking")
    if f["face_visible_ratio"] < 0.8:
        print("- You moved out of the frame or turned away too often")
    print("- Lots of head movement, try to keep your head steadier" if f["head_movement"] > HEAD_MOVE_HIGH
          else "- Head steadiness: good")
    if f["slouch_drift"] > SLOUCH_LIMIT:
        print("- You slowly sank down during the answer, sit upright")
    print("- Lots of body or hand movement, it may look like fidgeting" if f["body_movement"] > BODY_MOVE_HIGH
          else "- Body movement: calm")
 
 
if __name__ == "__main__":
    if "debug" in sys.argv:
        DEBUG = True
        sys.argv.remove("debug")
    src = sys.argv[1] if len(sys.argv) > 1 else 0
    if src == 0:
        calibrate()
    result = analyze_video(src, max_seconds=20)
    if result is None:
        print("Not enough face detected. Check the camera, light, and sit facing it.")
    else:
        show_video_report(result)
 