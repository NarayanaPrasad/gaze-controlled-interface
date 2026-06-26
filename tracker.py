import argparse
import cv2
import mediapipe as mp
import numpy as np
import ctypes
import math
import time
import platform
import asyncio
import json
import threading
import websockets
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

# ─────────────────────────────────────────────────────────────────────────────
#  1. SCREEN RESOLUTION (cross-platform)
# ─────────────────────────────────────────────────────────────────────────────
def get_screen_resolution():
    system = platform.system()
    if system == "Windows":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except AttributeError:
            pass
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    elif system == "Darwin":
        import subprocess
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True
        )
        for line in result.stdout.split("\n"):
            if "Resolution" in line:
                parts = line.strip().split()
                try:
                    return int(parts[1]), int(parts[3])
                except (IndexError, ValueError):
                    pass
        return 1440, 900
    else:
        try:
            import subprocess
            result = subprocess.run(["xrandr"], capture_output=True, text=True)
            for line in result.stdout.split("\n"):
                if "*" in line:
                    parts = line.strip().split()
                    w, h = parts[0].split("x")
                    return int(w), int(h)
        except Exception:
            pass
        return 1920, 1080

SCREEN_W, SCREEN_H = get_screen_resolution()

# ─────────────────────────────────────────────────────────────────────────────
#  2. WEBCAM (cross-platform)
# ─────────────────────────────────────────────────────────────────────────────
if platform.system() == "Windows":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ─────────────────────────────────────────────────────────────────────────────
#  3. MEDIAPIPE
# ─────────────────────────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.6, min_tracking_confidence=0.85
)

LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
L_INNER, L_OUTER = 133, 33
R_INNER, R_OUTER = 362, 263
L_TOP, L_BOT     = 159, 145
R_TOP, R_BOT     = 386, 374

# ─────────────────────────────────────────────────────────────────────────────
#  4. ML MODELS
# ─────────────────────────────────────────────────────────────────────────────
model_x = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2), Ridge(alpha=10.0))
model_y = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2), Ridge(alpha=10.0))
ml_trained = False

calib_features = []
calib_targets  = []

# ─────────────────────────────────────────────────────────────────────────────
#  5. TRACKING STATE
# ─────────────────────────────────────────────────────────────────────────────
cursor_x, cursor_y    = SCREEN_W // 2, SCREEN_H // 2
history_buffer        = []
BUFFER_SIZE           = 8
DEADZONE_RADIUS       = 35
SMOOTHING             = 0.12

wink_start_time       = None
click_executed        = False
is_frozen             = False
click_animation_timer = 0

WINK_TIME_THRESHOLD   = 0.5
EAR_FREEZE_THRESH     = 0.22
EAR_CLOSE_THRESH      = 0.15
EAR_OPEN_THRESH       = 0.20

# ─────────────────────────────────────────────────────────────────────────────
#  6. SHARED STATE
# ─────────────────────────────────────────────────────────────────────────────
shared_state = {
    "type":      "gaze",
    "x":         SCREEN_W // 2,
    "y":         SCREEN_H // 2,
    "is_frozen": False,
    "clicked":   False,
}
state_lock        = threading.Lock()
connected_clients = set()

latest_feature = None
feature_lock   = threading.Lock()

# ─────────────────────────────────────────────────────────────────────────────
#  7. HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def pt(marks, id, img_w, img_h):
    return np.array([marks[id].x * img_w, marks[id].y * img_h])

def extract_gaze_features(marks, img_w, img_h):
    l_iris = np.mean([pt(marks, i, img_w, img_h) for i in LEFT_IRIS],  axis=0)
    r_iris = np.mean([pt(marks, i, img_w, img_h) for i in RIGHT_IRIS], axis=0)
    l_in,  l_out = pt(marks, L_INNER, img_w, img_h), pt(marks, L_OUTER, img_w, img_h)
    r_in,  r_out = pt(marks, R_INNER, img_w, img_h), pt(marks, R_OUTER, img_w, img_h)
    l_width = np.linalg.norm(l_out - l_in)
    r_width = np.linalg.norm(r_out - r_in)
    if l_width < 1 or r_width < 1:
        return None
    return [
        (l_iris[0] - l_in[0]) / l_width,
        (l_iris[1] - l_in[1]) / l_width,
        (r_iris[0] - r_in[0]) / r_width,
        (r_iris[1] - r_in[1]) / r_width,
    ]

def get_ear(marks, img_w, img_h, top_id, bot_id, in_id, out_id):
    top,   bot   = pt(marks, top_id, img_w, img_h), pt(marks, bot_id, img_w, img_h)
    inner, outer = pt(marks, in_id,  img_w, img_h), pt(marks, out_id, img_w, img_h)
    v_dist = np.linalg.norm(top - bot)
    h_dist = np.linalg.norm(inner - outer)
    return v_dist / h_dist if h_dist != 0 else 0.0

# ─────────────────────────────────────────────────────────────────────────────
#  8. WEBSOCKET SERVER
# ─────────────────────────────────────────────────────────────────────────────
async def ws_handler(websocket):
    global ml_trained
    connected_clients.add(websocket)
    print(f"[WS] Browser connected  ({len(connected_clients)} client(s))")
    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type")

            if msg_type == "calib_collect":
                dot_x = msg.get("dot_x")
                dot_y = msg.get("dot_y")
                with feature_lock:
                    fv = latest_feature
                if fv is not None and dot_x is not None:
                    calib_features.append(fv)
                    calib_targets.append([dot_x, dot_y])
                    n = len(calib_features)
                    print(f"[CALIB] Sample {n} collected  dot=({dot_x},{dot_y})")
                    await websocket.send(json.dumps({"type": "calib_ack", "collected": n}))
                else:
                    print("[CALIB] WARNING — no face detected when spacebar pressed")
                    await websocket.send(json.dumps({
                        "type": "calib_ack",
                        "collected": len(calib_features),
                        "warning": "No face detected — try again"
                    }))

            elif msg_type == "calib_done":
                if len(calib_features) >= 9:
                    X   = np.array(calib_features)
                    Y_x = np.array([t[0] for t in calib_targets])
                    Y_y = np.array([t[1] for t in calib_targets])
                    model_x.fit(X, Y_x)
                    model_y.fit(X, Y_y)
                    ml_trained = True
                    print("[CALIB] Model trained! Gaze tracking is now active.")
                    await websocket.send(json.dumps({"type": "calib_complete"}))
                else:
                    print(f"[CALIB] Not enough samples ({len(calib_features)})")

    finally:
        connected_clients.discard(websocket)
        print(f"[WS] Browser disconnected ({len(connected_clients)} remaining)")

async def broadcast_loop():
    while True:
        if connected_clients:
            with state_lock:
                payload = json.dumps(shared_state)
                if shared_state["clicked"]:
                    shared_state["clicked"] = False
            websockets.broadcast(connected_clients, payload)
        await asyncio.sleep(1 / 30)

async def ws_main(host, port):
    async with websockets.serve(ws_handler, host, port):
        print(f"[WS] WebSocket server ready on ws://{host}:{port}")
        await broadcast_loop()

def start_ws_thread(host, port):
    asyncio.run(ws_main(host, port))

# ─────────────────────────────────────────────────────────────────────────────
#  9. MAIN EXECUTION BLOCK
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Command Line Arguments for Dynamic Port/Host config
    parser = argparse.ArgumentParser(description="Start the Gaze Memory ML Engine")
    parser.add_argument("--host", type=str, default="localhost", help="Host IP address (default: localhost)")
    parser.add_argument("--port", type=int, default=8765, help="Port to run the WebSocket server on (default: 8765)")
    args = parser.parse_args()

    # Start the WebSocket server in a background thread using the provided arguments
    ws_thread = threading.Thread(target=start_ws_thread, args=(args.host, args.port), daemon=True)
    ws_thread.start()

    # Small debug window setup
    DEBUG_W, DEBUG_H = 560, 110
    WIN = "Gaze Tracker  |  press Q to quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, DEBUG_W, DEBUG_H)

    print(f"Screen: {SCREEN_W}x{SCREEN_H}")
    print("Open the HTML frontend in Chrome and follow the calibration steps there.")

    # ─────────────────────────────────────────────────────────────────────────
    #  10. MAIN CV LOOP
    # ─────────────────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame     = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = face_mesh.process(rgb_frame)

        # Small debug canvas
        debug = np.zeros((DEBUG_H, DEBUG_W, 3), dtype=np.uint8)

        face_detected = False
        l_ear_disp    = 0.0
        r_ear_disp    = 0.0

        if results.multi_face_landmarks:
            marks            = results.multi_face_landmarks[0].landmark
            img_h, img_w, _  = frame.shape
            feature_vector   = extract_gaze_features(marks, img_w, img_h)
            face_detected    = True

            if feature_vector is not None:
                # Keep latest feature fresh for calibration sampling
                with feature_lock:
                    latest_feature = feature_vector

                # ── TRACKING PHASE ───────────────────────────────────────────
                if ml_trained:
                    l_ear_disp = get_ear(marks, img_w, img_h, L_TOP, L_BOT, L_INNER, L_OUTER)
                    r_ear_disp = get_ear(marks, img_w, img_h, R_TOP, R_BOT, R_INNER, R_OUTER)

                    # Snap-Freeze logic
                    if r_ear_disp < EAR_FREEZE_THRESH and l_ear_disp > EAR_OPEN_THRESH:
                        is_frozen = True
                        if r_ear_disp < EAR_CLOSE_THRESH:
                            if wink_start_time is None:
                                wink_start_time = time.time()
                            elif (time.time() - wink_start_time) >= WINK_TIME_THRESHOLD:
                                if not click_executed:
                                    print(f"[CLICK] Wink at ({cursor_x},{cursor_y})")
                                    click_executed        = True
                                    click_animation_timer = 15
                                    with state_lock:
                                        shared_state["clicked"] = True
                    else:
                        is_frozen       = False
                        wink_start_time = None
                        click_executed  = False

                    # Predict & smooth
                    pred_x = float(model_x.predict([feature_vector])[0])
                    pred_y = float(model_y.predict([feature_vector])[0])
                    history_buffer.append((pred_x, pred_y))
                    if len(history_buffer) > BUFFER_SIZE:
                        history_buffer.pop(0)

                    avg_x = sum(p[0] for p in history_buffer) / len(history_buffer)
                    avg_y = sum(p[1] for p in history_buffer) / len(history_buffer)

                    if not is_frozen:
                        dist = math.hypot(avg_x - cursor_x, avg_y - cursor_y)
                        if dist > DEADZONE_RADIUS:
                            cursor_x = int(cursor_x + (avg_x - cursor_x) * SMOOTHING)
                            cursor_y = int(cursor_y + (avg_y - cursor_y) * SMOOTHING)

                    with state_lock:
                        shared_state["x"]         = cursor_x
                        shared_state["y"]         = cursor_y
                        shared_state["is_frozen"] = bool(is_frozen)

                    if click_animation_timer > 0:
                        click_animation_timer -= 1

        # ── Debug text ───────────────────────────────────────────────────────
        face_col   = (0, 220, 100)  if face_detected  else (60, 60, 255)
        status_col = (0, 229, 200)  if ml_trained      else (200, 200, 0)
        clients    = len(connected_clients)

        cv2.putText(debug, "Face: " + ("DETECTED"    if face_detected else "NOT FOUND"),
                    (12, 26),  cv2.FONT_HERSHEY_SIMPLEX, 0.62, face_col,   1)
        cv2.putText(debug, "Model: " + ("TRACKING"   if ml_trained    else "Awaiting calibration in browser"),
                    (12, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_col, 1)
        cv2.putText(debug, f"L-EAR:{l_ear_disp:.2f}  R-EAR:{r_ear_disp:.2f}  FROZEN:{is_frozen}",
                    (12, 74),  cv2.FONT_HERSHEY_SIMPLEX, 0.54, (150,150,150), 1)
        cv2.putText(debug, f"Browser clients: {clients}",
                    (12, 96),  cv2.FONT_HERSHEY_SIMPLEX, 0.54, (150,150,150), 1)

        cv2.imshow(WIN, debug)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()