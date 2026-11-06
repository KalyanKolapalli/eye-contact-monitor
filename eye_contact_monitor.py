"""
Eye-Contact Monitor  (eyes + head + presence + hand gestures)

Run:   python eye_contact_monitor.py
Keys:  SPACE = confirm a calibration step
       R     = redo calibration
       Q     = quit

Gestures:
   Thumbs up (one hand)                         -> thumbs-up symbol
   Heart: both hands (index tips + thumb tips
          touching) OR finger-heart (thumb and
          index tip pinched, other fingers closed) -> heart symbol
"""

import json
import os
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

# ---------------- Settings you can tune ----------------
SENSITIVITY = 0.35        # eyes: 0.2 strict ... 0.5 relaxed
HEAD_YAW_LIMIT = 0.07     # head turn left/right allowance
HEAD_PITCH_LIMIT = 0.08   # head tilt up/down allowance
SMOOTHING = 0.6
MOUTH_WINDOW = 15
SPEAK_VARIATION = 0.012
ABSENT_SECONDS = 1.0      # no face for this long -> "not in front of camera"
POSITION_SHIFT = 0.18     # face moved this fraction of screen width from your seat -> error
SIZE_CHANGE = 0.40        # face size changed 40% (leaned far back/forward) -> error
GESTURE_FRAMES = 4        # gesture must be held this many frames
ANIM_SECONDS = 1.6        # how long the symbol animation lasts
CALIB_FILE = "eye_calibration.json"
# --------------------------------------------------------

L_IRIS, R_IRIS = 468, 473
EYE_A = (33, 133, 159, 145)
EYE_B = (263, 362, 386, 374)
LIP_TOP, LIP_BOT, MOUTH_L, MOUTH_R = 13, 14, 78, 308
NOSE, FACE_L, FACE_R, FOREHEAD, CHIN = 1, 234, 454, 10, 152

GREEN, RED, ORANGE, WHITE, YELLOW, BLUE = (0, 200, 0), (0, 0, 255), (0, 165, 255), (255, 255, 255), (0, 255, 255), (255, 120, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX

CALIB_STEPS = [
    ("center", "Sit normally, look STRAIGHT at the CAMERA, press SPACE"),
    ("left",   "Head still. Move only your EYES to the LEFT edge of the screen, press SPACE"),
    ("right",  "Head still. Move only your EYES to the RIGHT edge of the screen, press SPACE"),
    ("up",     "Head still. Move only your EYES to the TOP of the screen, press SPACE"),
    ("down",   "Head still. Move only your EYES to the BOTTOM of the screen, press SPACE"),
]


# =============== Symbols (emoji images) ===============
def _emoji_from_font(ch):
    """Render a colour emoji using the Mac emoji font. Returns BGRA image or None."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype("/System/Library/Fonts/Apple Color Emoji.ttc", 160)
        img = Image.new("RGBA", (260, 260), (0, 0, 0, 0))
        ImageDraw.Draw(img).text((10, 10), ch, font=font, embedded_color=True)
        arr = np.array(img)
        ys, xs = np.where(arr[:, :, 3] > 0)
        if len(xs) == 0:
            return None
        arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    except Exception:
        return None


def _drawn_heart(size=200):
    img = np.zeros((size, size, 4), np.uint8)
    t = np.linspace(0, 2 * np.pi, 200)
    x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    pts = np.stack([x, y], 1) * (size / 36) + size / 2
    cv2.fillPoly(img, [pts.astype(np.int32)], (60, 40, 230, 255), cv2.LINE_AA)
    return img


def _drawn_thumbs(size=200):
    img = np.zeros((size, size, 4), np.uint8)
    skin, edge = (80, 200, 255, 255), (30, 120, 200, 255)
    cv2.rectangle(img, (50, 90), (160, 180), skin, -1)               # fist
    cv2.rectangle(img, (70, 20), (110, 100), skin, -1)               # thumb
    cv2.circle(img, (90, 22), 20, skin, -1)
    for yy in (112, 134, 156):
        cv2.line(img, (110, yy), (160, yy), edge, 3)                 # finger lines
    cv2.rectangle(img, (20, 95), (48, 180), (200, 120, 40, 255), -1)  # cuff
    return img


def load_symbols():
    heart = _emoji_from_font("❤️")
    thumbs = _emoji_from_font("\U0001F44D")
    return {"HEART": heart if heart is not None else _drawn_heart(),
            "THUMBS_UP": thumbs if thumbs is not None else _drawn_thumbs()}


def overlay(frame, img, cx, cy, scale, alpha):
    nh, nw = max(int(img.shape[0] * scale), 1), max(int(img.shape[1] * scale), 1)
    sym = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x1, y1 = cx - nw // 2, cy - nh // 2
    fx1, fy1 = max(x1, 0), max(y1, 0)
    fx2, fy2 = min(x1 + nw, frame.shape[1]), min(y1 + nh, frame.shape[0])
    if fx1 >= fx2 or fy1 >= fy2:
        return
    s = sym[fy1 - y1:fy2 - y1, fx1 - x1:fx2 - x1]
    a = (s[:, :, 3:4].astype(np.float32) / 255.0) * alpha
    roi = frame[fy1:fy2, fx1:fx2].astype(np.float32)
    frame[fy1:fy2, fx1:fx2] = (roi * (1 - a) + s[:, :, :3] * a).astype(np.uint8)


def draw_animation(frame, img, start):
    """Pop-up: starts small, grows bigger, then fades out."""
    t = (time.time() - start) / ANIM_SECONDS
    if t >= 1:
        return False
    ease = 1 - (1 - t) ** 3
    scale = 0.3 + 1.4 * ease
    alpha = 1.0 if t < 0.7 else (1 - t) / 0.3
    h, w = frame.shape[:2]
    overlay(frame, img, w // 2, int(h * 0.45), scale, alpha)
    return True


# =============== Hands & gestures ===============
def hpts(hand, w, h):
    return np.array([[p.x * w, p.y * h] for p in hand.landmark])


def folded(p, tip, pip):
    return np.linalg.norm(p[tip] - p[0]) < np.linalg.norm(p[pip] - p[0])


def is_thumbs_up(p):
    size = np.linalg.norm(p[9] - p[0])
    others_folded = all(folded(p, t, q) for t, q in ((8, 6), (12, 10), (16, 14), (20, 18)))
    thumb_up = p[4][1] < p[3][1] < p[2][1] and p[4][1] < p[:, 1].min() + 0.05 * size
    return others_folded and thumb_up and (p[2][1] - p[4][1]) > 0.5 * size


def is_finger_heart(p):
    size = np.linalg.norm(p[9] - p[0])
    pinched = np.linalg.norm(p[4] - p[8]) < 0.30 * size
    rest_folded = all(folded(p, t, q) for t, q in ((12, 10), (16, 14), (20, 18)))
    return pinched and rest_folded


def is_two_hand_heart(a, b):
    size = (np.linalg.norm(a[9] - a[0]) + np.linalg.norm(b[9] - b[0])) / 2
    index_touch = np.linalg.norm(a[8] - b[8]) < 0.45 * size
    thumb_touch = np.linalg.norm(a[4] - b[4]) < 0.45 * size
    index_above = (a[8][1] + b[8][1]) / 2 < (a[4][1] + b[4][1]) / 2
    return index_touch and thumb_touch and index_above


def detect_gesture(hands_pts):
    if len(hands_pts) == 2 and is_two_hand_heart(*hands_pts):
        return "HEART"
    for p in hands_pts:
        if is_thumbs_up(p):
            return "THUMBS_UP"
        if is_finger_heart(p):
            return "HEART"
    return None


# =============== Face / eyes / head ===============
def P(lm, i, w, h):
    return np.array([lm[i].x * w, lm[i].y * h])


def eye_gaze(lm, w, h, iris, eye):
    outer, inner, top, bot = (P(lm, i, w, h) for i in eye)
    c = P(lm, iris, w, h)
    a, b = (outer, inner) if outer[0] < inner[0] else (inner, outer)
    axis = b - a
    length = np.linalg.norm(axis) + 1e-6
    u = axis / length
    gh = np.dot(c - a, u) / length - 0.5
    n = np.array([-u[1], u[0]])
    gv = np.dot(c - (top + bot) / 2, n) / length
    return gh, gv


def measure(lm, w, h):
    ah, av = eye_gaze(lm, w, h, L_IRIS, EYE_A)
    bh, bv = eye_gaze(lm, w, h, R_IRIS, EYE_B)
    nose = P(lm, NOSE, w, h)
    fl, fr = P(lm, FACE_L, w, h), P(lm, FACE_R, w, h)
    ft, ch = P(lm, FOREHEAD, w, h), P(lm, CHIN, w, h)
    lo, hi = min(fl[0], fr[0]), max(fl[0], fr[0])
    yaw = (nose[0] - lo) / (hi - lo + 1e-6) - 0.5
    pitch = (nose[1] - ft[1]) / (ch[1] - ft[1] + 1e-6) - 0.5
    return np.array([(ah + bh) / 2, (av + bv) / 2, yaw, pitch])


def face_place(lm):
    """Face centre (as fraction of screen) and face width (fraction of screen)."""
    xs = [p.x for p in lm]; ys = [p.y for p in lm]
    return [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs)]


def build_thresholds(cal):
    c = np.array(cal["center"])
    t = {"center": c.tolist()}
    defaults = {"left": 0.08, "right": 0.08, "up": 0.06, "down": 0.06}
    for d, idx in (("left", 0), ("right", 0), ("up", 1), ("down", 1)):
        dist = abs(cal[d][idx] - c[idx])
        if dist < 0.015:
            dist = defaults[d] / SENSITIVITY
        t[d] = dist * SENSITIVITY
    return t


def eye_direction(m, t):
    c = t["center"]
    dx, dy = m[0] - c[0], m[1] - c[1]
    parts = []
    if dy < -t["up"]: parts.append("UP")
    if dy > t["down"]: parts.append("DOWN")
    if dx < -t["left"]: parts.append("LEFT")
    if dx > t["right"]: parts.append("RIGHT")
    return "-".join(parts) or "CENTER", dx, dy


def head_direction(m, base):
    yaw, pitch = m[2] - base[2], m[3] - base[3]
    parts = []
    if pitch < -HEAD_PITCH_LIMIT: parts.append("UP")
    if pitch > HEAD_PITCH_LIMIT: parts.append("DOWN")
    if yaw < -HEAD_YAW_LIMIT: parts.append("LEFT")
    if yaw > HEAD_YAW_LIMIT: parts.append("RIGHT")
    return "-".join(parts) or "CENTER", yaw, pitch


def mouth_open(lm, w, h):
    gap = np.linalg.norm(P(lm, LIP_TOP, w, h) - P(lm, LIP_BOT, w, h))
    width = np.linalg.norm(P(lm, MOUTH_L, w, h) - P(lm, MOUTH_R, w, h))
    return gap / width if width > 0 else 0.0


def full_face_box(lm, w, h):
    xs = np.array([p.x * w for p in lm]); ys = np.array([p.y * h for p in lm])
    fw, fh = xs.max() - xs.min(), ys.max() - ys.min()
    return (max(int(xs.min() - 0.15 * fw), 0), max(int(ys.min() - 0.30 * fh), 0),
            min(int(xs.max() + 0.15 * fw), w - 1), min(int(ys.max() + 0.10 * fh), h - 1))


# =============== Drawing helpers ===============
def tag(frame, text, x, y, color, scale=0.6):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 2)
    y = max(y, th + 10)
    cv2.rectangle(frame, (x, y - th - 10), (x + tw + 12, y), color, -1)
    cv2.putText(frame, text, (x + 6, y - 5), FONT, scale, WHITE, 2)
    return th + 12


def banner(frame, text, color=BLUE):
    w = frame.shape[1]
    cv2.rectangle(frame, (0, 40), (w, 85), color, -1)
    cv2.putText(frame, text, (10, 70), FONT, 0.6, WHITE, 2)


def big_error(frame, line1, line2=""):
    h, w = frame.shape[:2]
    red = frame.copy()
    red[:] = (0, 0, 180)
    cv2.addWeighted(red, 0.35, frame, 0.65, 0, frame)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), RED, 14)
    for i, text in enumerate((line1, line2)):
        if text:
            (tw, _), _ = cv2.getTextSize(text, FONT, 0.9, 2)
            cv2.putText(frame, text, ((w - tw) // 2, h // 2 + i * 40), FONT, 0.9, WHITE, 2)


def load_calibration():
    if os.path.exists(CALIB_FILE):
        try:
            with open(CALIB_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return None


# =============== Main loop ===============
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open the webcam. Check camera permission.")
        return
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                                min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.6,
                                     min_tracking_confidence=0.5)
    draw_utils, hand_conn = mp.solutions.drawing_utils, mp.solutions.hands.HAND_CONNECTIONS
    symbols = load_symbols()

    cal = load_calibration()
    if cal and "home" not in cal:
        cal = None                       # old calibration file -> redo
    step = None if cal else 0
    thresholds = build_thresholds(cal) if cal else None
    collected, recent, places = {}, deque(maxlen=15), deque(maxlen=15)
    mouth_hist = deque(maxlen=MOUTH_WINDOW)
    smooth, last_face_time = None, time.time()
    gesture_count, last_gesture, anim = 0, None, None   # anim = (name, start_time)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_res = face_mesh.process(rgb)
        hand_res = hands.process(rgb)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("r"), ord("R")):
            step, collected = 0, {}

        # ---------- Hands ----------
        hand_list = hand_res.multi_hand_landmarks or []
        hands_pts = [hpts(hl, w, h) for hl in hand_list]
        for hl in hand_list:
            draw_utils.draw_landmarks(frame, hl, hand_conn)
        gesture = detect_gesture(hands_pts) if hands_pts else None
        gesture_count = gesture_count + 1 if gesture and gesture == last_gesture else (1 if gesture else 0)
        last_gesture = gesture
        if gesture and gesture_count == GESTURE_FRAMES and (anim is None or anim[0] != gesture):
            anim = (gesture, time.time())
        hands_error = bool(hands_pts) and gesture is None

        face = face_res.multi_face_landmarks[0].landmark if face_res.multi_face_landmarks else None

        # ---------- Calibration mode ----------
        if step is not None:
            banner(frame, f"CALIBRATION {step + 1}/{len(CALIB_STEPS)}: {CALIB_STEPS[step][1]}")
            if face is None:
                cv2.putText(frame, "No face detected - sit in front of the camera", (10, 120), FONT, 0.7, ORANGE, 2)
            else:
                recent.append(measure(face, w, h))
                places.append(face_place(face))
                for i in (L_IRIS, R_IRIS):
                    cv2.circle(frame, tuple(P(face, i, w, h).astype(int)), 3, YELLOW, -1)
                if key == ord(" ") and len(recent) >= 10:
                    name = CALIB_STEPS[step][0]
                    collected[name] = np.mean(recent, axis=0).tolist()
                    if name == "center":
                        collected["home"] = np.mean(places, axis=0).tolist()
                    recent.clear(); places.clear()
                    step += 1
                    if step == len(CALIB_STEPS):
                        cal, step = collected, None
                        thresholds = build_thresholds(cal)
                        with open(CALIB_FILE, "w") as f:
                            json.dump(cal, f)
            cv2.putText(frame, "R = restart calibration   Q = quit", (10, 25), FONT, 0.55, WHITE, 1)
            cv2.imshow("Eye-Contact Monitor", frame)
            continue

        # ---------- Not in front of camera ----------
        if face is None:
            if time.time() - last_face_time > ABSENT_SECONDS:
                if hands_pts and gesture is None:
                    big_error(frame, "ERROR: Your face is hidden!", "Move your hand away from the camera")
                else:
                    big_error(frame, "ERROR: You are not in front of the camera!", "Please come back to your seat")
            if anim and not draw_animation(frame, symbols[anim[0]], anim[1]):
                anim = None
            cv2.putText(frame, "R = recalibrate   Q = quit", (10, 25), FONT, 0.55, WHITE, 1)
            cv2.imshow("Eye-Contact Monitor", frame)
            continue
        last_face_time = time.time()

        # ---------- Face present ----------
        m = measure(face, w, h)
        smooth = m if smooth is None else SMOOTHING * smooth + (1 - SMOOTHING) * m
        eye_dir, dx, dy = eye_direction(smooth, thresholds)
        head_dir, yaw, pitch = head_direction(smooth, cal["center"])
        eyes_ok, head_ok = eye_dir == "CENTER", head_dir == "CENTER"

        cx, cy, fw = face_place(face)
        hx, hy, hw = cal["home"]
        moved = np.hypot(cx - hx, cy - hy) > POSITION_SHIFT or abs(fw - hw) / hw > SIZE_CHANGE

        mouth_hist.append(mouth_open(face, w, h))
        speaking = len(mouth_hist) == MOUTH_WINDOW and np.std(mouth_hist) > SPEAK_VARIATION

        if moved:
            main_text = "ERROR: Not in your seat position"
        elif hands_error:
            main_text = "ERROR: Hands in front of camera"
        elif eyes_ok and head_ok:
            main_text = "LOOKING AT CAMERA"
        else:
            main_text = "NOT LOOKING AT CAMERA"
        good = main_text == "LOOKING AT CAMERA"
        if speaking:
            main_text += " (speaking)"
        if gesture:
            main_text += f"  [{gesture.replace('_', ' ')}]"

        color = GREEN if good else RED
        x1, y1, x2, y2 = full_face_box(face, w, h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        y = y1
        y -= tag(frame, "Head: STRAIGHT" if head_ok else f"Head: TURNED {head_dir}", x1, y, GREEN if head_ok else RED, 0.55)
        y -= tag(frame, "Eyes: ON CAMERA" if eyes_ok else f"Eyes: LOOKING {eye_dir}", x1, y, GREEN if eyes_ok else RED, 0.55)
        tag(frame, main_text, x1, y, color, 0.65)

        for i in (L_IRIS, R_IRIS):
            c = P(face, i, w, h).astype(int)
            cv2.circle(frame, tuple(c), 3, YELLOW, -1)
            cv2.arrowedLine(frame, tuple(c), (int(c[0] + dx * 400), int(c[1] + dy * 400)), YELLOW, 2, tipLength=0.3)

        if moved:
            hx_px, hy_px = int(hx * w), int(hy * h)
            cv2.drawMarker(frame, (hx_px, hy_px), ORANGE, cv2.MARKER_CROSS, 40, 3)
            cv2.putText(frame, "move back here", (hx_px - 70, hy_px + 40), FONT, 0.6, ORANGE, 2)
        if speaking and not good:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), RED, 12)
            cv2.putText(frame, "ERROR: Speaking without eye contact!", (10, h - 45), FONT, 0.8, RED, 2)

        if anim and not draw_animation(frame, symbols[anim[0]], anim[1]):
            anim = None

        cv2.putText(frame, f"eyes dx={dx:+.3f} dy={dy:+.3f}  head yaw={yaw:+.3f} pitch={pitch:+.3f}",
                    (10, h - 15), FONT, 0.5, WHITE, 1)
        cv2.putText(frame, "R = recalibrate   Q = quit", (10, 25), FONT, 0.55, WHITE, 1)
        cv2.imshow("Eye-Contact Monitor", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()