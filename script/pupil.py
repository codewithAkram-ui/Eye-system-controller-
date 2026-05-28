import cv2
import ctypes
import numpy as np
import mediapipe as mp
import pyautogui
import time
import os
from scipy.spatial import distance as dist

try:
    import win32api, win32con
    _HAS_WIN32 = True
except ImportError:
    _HAS_WIN32 = False


CAM_W, CAM_H = 640, 480

EAR_THRESHOLD    = 0.21
CONSEC_FRAMES    = 3
DOUBLE_BLINK_GAP = 0.60

CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "calibration_nose.npz")

IOD_MIN_PX = 85;  IOD_MAX_PX = 145
IOD_OK_MIN = 90;  IOD_OK_MAX = 135
STEP1_HOLD = 2.0

CALIB_POINTS_NORM = [
    (0.10, 0.10), (0.50, 0.10), (0.90, 0.10),
    (0.10, 0.50), (0.50, 0.50), (0.90, 0.50),
    (0.10, 0.90), (0.50, 0.90), (0.90, 0.90),
]
CALIB_FRAMES_PER_POINT = 60
MEASURE_FRAMES         = 60

# Nose movement sensitivity — larger = more movement per head tilt
NOSE_SENSITIVITY = 8.0

NOSE_SMOOTH  = 0.18
KALMAN_Q     = 1e-3
KALMAN_R     = 4.0

MP_LEFT_EYE  = [362, 385, 387, 263, 373, 380]
MP_RIGHT_EYE = [33,  160, 158, 133, 153, 144]
L_OUT = 263;  L_IN = 362
R_OUT = 33;   R_IN = 133
NOSE  = 4          # nose tip landmark



cap       = None
face_mesh = None

SCREEN_W, SCREEN_H = pyautogui.size()
pyautogui.FAILSAFE  = False

dot_x = dot_y = 0.0
_last_ear       = 0.3
_overlay_canvas = None

_blink_state      = "IDLE"
_closed_count     = 0
_first_blink_time = 0.0

_nose_rest_x = 0.5
_nose_rest_y = 0.5
_iod_rest    = 110.0   # inter-ocular distance at calibration distance

_calib_cx = _calib_cy = None
_is_calibrated = False

_smooth_ox = _smooth_oy = 0.0
_nose_init = False


# =============================================================================
# KALMAN FILTER
# =============================================================================

class Kalman2D:
    def __init__(self, q=KALMAN_Q, r=KALMAN_R):
        dt = 1/30.0
        self.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], np.float64)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], np.float64)
        self.Q = np.eye(4) * q
        self.R = np.eye(2) * r
        self.P = np.eye(4) * 500.0
        self.x = np.zeros((4,1))
        self._ok = False

    def update(self, mx, my):
        z = np.array([[mx],[my]], np.float64)
        if not self._ok:
            self.x[0,0]=mx; self.x[1,0]=my; self._ok=True; return mx,my
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ (z - self.H @ self.x)
        self.P  = (np.eye(4) - K @ self.H) @ self.P
        return float(self.x[0,0]), float(self.x[1,0])

    def reset(self, x, y):
        self.x[:]=0; self.x[0,0]=x; self.x[1,0]=y
        self.P=np.eye(4)*500.0; self._ok=True

_kalman = Kalman2D()


def _set_cursor(x, y):
    ctypes.windll.user32.SetCursorPos(int(np.clip(x, 0, SCREEN_W-1)),
                                       int(np.clip(y, 0, SCREEN_H-1)))

def _scroll_down():
    if _HAS_WIN32:
        win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, -480, 0)
    else:
        pyautogui.scroll(-5)


# =============================================================================
# FRAME GRAB
# =============================================================================

def _grab_full():
    ret, raw = cap.read()
    if not ret:
        return None, None, None, None, None
    h, w = raw.shape[:2]
    rgb  = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res = face_mesh.process(rgb)
    rgb.flags.writeable = True
    display = cv2.flip(raw, 1)
    lm = res.multi_face_landmarks[0].landmark if res.multi_face_landmarks else None
    return raw, display, lm, h, w


# =============================================================================
# HELPERS
# =============================================================================

def _dp(lm, idx, w, h):
    return int((1.0 - lm[idx].x) * w), int(lm[idx].y * h)

def _iod(lm, w, h):
    lx,ly = lm[L_OUT].x*w, lm[L_OUT].y*h
    rx,ry = lm[R_OUT].x*w, lm[R_OUT].y*h
    return dist.euclidean((lx,ly),(rx,ry))

def _calc_ear(lm, indices, iw, ih):
    pts = np.array([(lm[i].x*iw, lm[i].y*ih) for i in indices], np.float32)
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C)


# =============================================================================
# NOSE OFFSET  (replaces iris offset)
# =============================================================================

def _nose_offset(lm, w, h):
    """
    Returns (ox, oy) — normalised nose displacement from rest position.
    ox > 0 → looking right   oy > 0 → looking down

    We divide by _iod_rest so the offset is scale-invariant with head distance.
    The sign of ox is flipped because the camera image is mirrored for display
    but the raw landmark x is NOT flipped.
    """
    nx = lm[NOSE].x   # 0..1, raw (not mirrored)
    ny = lm[NOSE].y

    # Displacement from rest (in normalised 0..1 units)
    dx = nx - _nose_rest_x   # positive = nose moved right in raw frame = left on display
    dy = ny - _nose_rest_y   # positive = nose moved down

    # Scale by sensitivity and IOD so the range sits nicely in calibration space
    scale = max(_iod_rest, 1.0)
    ox = -dx * scale * NOSE_SENSITIVITY   # flip x for mirrored view
    oy =  dy * scale * NOSE_SENSITIVITY

    return ox, oy


# =============================================================================
# CALIBRATION MATH
# =============================================================================

def _fvec(ox, oy):
    return np.array([1.0, ox, oy, ox*ox, ox*oy, oy*oy], np.float64)

def _fit_calib(offsets, pts):
    A  = np.array([_fvec(ox,oy) for ox,oy in offsets])
    sx = np.array([p[0] for p in pts], np.float64)
    sy = np.array([p[1] for p in pts], np.float64)
    lam = 1e-3
    ATA = A.T @ A + lam * np.eye(6)
    cx  = np.linalg.solve(ATA, A.T @ sx)
    cy  = np.linalg.solve(ATA, A.T @ sy)
    return cx, cy

def _apply_calib(ox, oy):
    fv = _fvec(ox, oy)
    sx = int(np.clip(np.dot(_calib_cx, fv), 0, SCREEN_W-1))
    sy = int(np.clip(np.dot(_calib_cy, fv), 0, SCREEN_H-1))
    return sx, sy

def _save_calib(cx, cy, nx, ny, iod):
    np.savez(CALIB_FILE, cx=cx, cy=cy,
             nose_rest=np.array([nx, ny]),
             iod_rest=np.array([iod]))
    print(f"[pupil] Saved -> {CALIB_FILE}")

def _load_calib():
    global _calib_cx, _calib_cy, _is_calibrated
    global _nose_rest_x, _nose_rest_y, _iod_rest
    if not os.path.exists(CALIB_FILE):
        return False
    d = np.load(CALIB_FILE)
    _calib_cx    = d['cx']
    _calib_cy    = d['cy']
    _nose_rest_x = float(d['nose_rest'][0])
    _nose_rest_y = float(d['nose_rest'][1])
    _iod_rest    = float(d['iod_rest'][0])
    _is_calibrated = True
    print(f"[pupil] Loaded | nose_rest=({_nose_rest_x:.3f},{_nose_rest_y:.3f})  iod={_iod_rest:.1f}px")
    return True


# =============================================================================
# SETUP STEPS
# =============================================================================

def _step1_face_distance():
    win = "STEP 1/3 — Position Face (fill the oval)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, CAM_W, CAM_H)
    ok_since = None

    while True:
        raw, display, lm, h, w = _grab_full()
        if display is None:
            continue
        ov = display.copy()
        cxo, cyo = w//2, h//2
        cv2.ellipse(ov,(cxo,cyo),(100,130),0,0,360,(80,80,80),2)

        iod_val=None; txt="No face — look at the camera"; good=False

        if lm:
            iod_val = _iod(lm, w, h)
            for idx in MP_LEFT_EYE + MP_RIGHT_EYE:
                cv2.circle(ov, _dp(lm,idx,w,h), 2, (0,220,0), -1)
            # Draw nose tip
            cv2.circle(ov, _dp(lm, NOSE, w, h), 6, (0,165,255), -1)

            if iod_val < IOD_MIN_PX:
                txt = f"Move CLOSER  (IOD={iod_val:.0f}px)"
                cv2.ellipse(ov,(cxo,cyo),(100,130),0,0,360,(0,80,255),3)
                ok_since = None
            elif iod_val > IOD_MAX_PX:
                txt = f"Move BACK  (IOD={iod_val:.0f}px)"
                cv2.ellipse(ov,(cxo,cyo),(100,130),0,0,360,(0,80,255),3)
                ok_since = None
            else:
                txt  = f"PERFECT — hold still  (IOD={iod_val:.0f}px)"
                good = True
                cv2.ellipse(ov,(cxo,cyo),(100,130),0,0,360,(0,255,0),3)
                if ok_since is None:
                    ok_since = time.time()
                else:
                    held = time.time() - ok_since
                    ang  = int(360 * min(held/STEP1_HOLD, 1.0))
                    cv2.ellipse(ov,(cxo,cyo),(112,142),-90,0,ang,(0,255,100),4)
                    if held >= STEP1_HOLD:
                        cv2.destroyWindow(win)
                        return True

        bx,by,bl = 20,h-40,w-40
        cv2.rectangle(ov,(bx,by),(bx+bl,by+18),(50,50,50),-1)
        if iod_val:
            f = int(np.clip((iod_val-IOD_MIN_PX)/(IOD_MAX_PX-IOD_MIN_PX),0,1)*bl)
            color = (0,200,0) if good else (0,80,255)
            cv2.rectangle(ov,(bx,by),(bx+f,by+18),color,-1)
        lo = int((IOD_OK_MIN-IOD_MIN_PX)/(IOD_MAX_PX-IOD_MIN_PX)*bl)+bx
        hi = int((IOD_OK_MAX-IOD_MIN_PX)/(IOD_MAX_PX-IOD_MIN_PX)*bl)+bx
        cv2.rectangle(ov,(lo,by-2),(hi,by+20),(0,255,0),2)

        cv2.putText(ov,"STEP 1/3 — Face distance",
                    (10,24),cv2.FONT_HERSHEY_SIMPLEX,0.65,(200,200,200),2)
        cv2.putText(ov,txt,(10,52),cv2.FONT_HERSHEY_SIMPLEX,0.55,
                    (0,255,0) if good else (0,120,255),2)
        cv2.putText(ov,"Orange dot = nose tip | keep face in oval",
                    (10,h-55),cv2.FONT_HERSHEY_SIMPLEX,0.44,(150,150,150),1)
        cv2.imshow(win, ov)
        if cv2.waitKey(1)&0xFF == ord('q'):
            cv2.destroyWindow(win); return False
    return True


def _step2_measure_nose():
    """Capture rest-pose nose position and IOD. User looks straight ahead."""
    global _nose_rest_x, _nose_rest_y, _iod_rest

    win = "STEP 2/3 — Nose Rest-Pose (look STRAIGHT AHEAD, hold still)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, CAM_W, CAM_H)

    nose_xs, nose_ys, iods = [], [], []

    while len(nose_xs) < MEASURE_FRAMES:
        raw, display, lm, h, w = _grab_full()
        if display is None:
            continue
        pct = int(len(nose_xs)/MEASURE_FRAMES*100)

        # Cross-hair guide at centre
        cv2.line(display, (w//2-30, h//2), (w//2+30, h//2), (200,200,200), 1)
        cv2.line(display, (w//2, h//2-30), (w//2, h//2+30), (200,200,200), 1)
        cv2.circle(display, (w//2, h//2), 6, (0,0,255), -1)

        if lm:
            nose_xs.append(lm[NOSE].x)
            nose_ys.append(lm[NOSE].y)
            iods.append(_iod(lm, w, h))

            for idx in MP_LEFT_EYE + MP_RIGHT_EYE:
                cv2.circle(display, _dp(lm,idx,w,h), 2, (0,220,0), -1)
            cv2.circle(display, _dp(lm, NOSE, w, h), 8, (0,165,255), -1)

        cv2.rectangle(display,(10,h-35),(w-10,h-15),(50,50,50),-1)
        cv2.rectangle(display,(10,h-35),
                      (10+int((w-20)*len(nose_xs)/MEASURE_FRAMES),h-15),(0,200,100),-1)
        cv2.putText(display,"STEP 2/3 — Recording nose rest-pose",
                    (10,24),cv2.FONT_HERSHEY_SIMPLEX,0.60,(200,200,200),2)
        cv2.putText(display,f"Look at RED dot, HOLD STILL  ({pct}%)",
                    (10,50),cv2.FONT_HERSHEY_SIMPLEX,0.52,(0,200,255),2)
        cv2.putText(display,"Orange dot = your nose tip",
                    (10,74),cv2.FONT_HERSHEY_SIMPLEX,0.42,(100,165,80),1)

        cv2.imshow(win, display)
        if cv2.waitKey(1)&0xFF == ord('q'):
            cv2.destroyWindow(win); return False

    _nose_rest_x = float(np.median(nose_xs))
    _nose_rest_y = float(np.median(nose_ys))
    _iod_rest    = float(np.median(iods))
    cv2.destroyWindow(win)
    print(f"[pupil] nose_rest=({_nose_rest_x:.3f},{_nose_rest_y:.3f})  iod={_iod_rest:.1f}px")
    return True


def _step3_calibrate():
    """9-point calibration using nose offsets."""
    global _calib_cx, _calib_cy, _is_calibrated, _nose_init

    win = "STEP 3/3 — Gaze Calibration (move nose toward each dot)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    canvas     = np.zeros((SCREEN_H, SCREEN_W, 3), np.uint8)
    offsets    = []
    screen_pts = []
    total      = len(CALIB_POINTS_NORM)

    for i, (nx, ny) in enumerate(CALIB_POINTS_NORM):
        tx, ty    = int(nx*SCREEN_W), int(ny*SCREEN_H)
        collected = None
        tick      = 0

        while True:
            raw, display, lm, h, w = _grab_full()
            if display is None:
                continue

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyWindow(win); return False
            if key == ord(' ') and collected is None:
                collected = []
                print(f"[pupil] Collecting point {i+1}/{total}...")

            canvas[:] = 0
            tick = (tick + 1) % 30

            cv2.putText(canvas,
                        f"STEP 3/3  ({i+1}/{total})  —  Aim NOSE at RED dot, press SPACE",
                        (SCREEN_W//2-400, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.80, (170,170,170), 2)
            cv2.putText(canvas,
                        "Tilt your head so nose points toward the dot",
                        (SCREEN_W//2-300, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (120,120,120), 1)

            for j, (px, py) in enumerate(CALIB_POINTS_NORM):
                c = (35,35,35) if j != i else (0,0,200)
                r = 6 if j != i else (14 + int(5*abs(np.sin(tick*0.21))))
                cv2.circle(canvas,(int(px*SCREEN_W), int(py*SCREEN_H)), r, c, -1)
            cv2.circle(canvas,(tx,ty), 22, (0,0,255), 2)
            cv2.line(canvas,(tx-30,ty),(tx+30,ty),(70,70,70),1)
            cv2.line(canvas,(tx,ty-30),(tx,ty+30),(70,70,70),1)

            cur_off = None
            face_ok = False

            if lm:
                face_ok = True
                ox, oy  = _nose_offset(lm, w, h)
                cur_off = (ox, oy)

            # Status text
            status_txt = "FACE LOCKED — press SPACE" if face_ok else "No face detected..."
            status_col = (0,255,100) if face_ok else (0,80,200)
            cv2.putText(canvas, status_txt,
                        (SCREEN_W//2-200, SCREEN_H-40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_col, 2)

            # Collecting
            if collected is not None:
                if cur_off is not None:
                    collected.append(cur_off)

                pct = int(len(collected) / CALIB_FRAMES_PER_POINT * 100)
                bw  = int(300 * len(collected) / CALIB_FRAMES_PER_POINT)
                bx_ = tx - 150; by_ = ty + 32
                cv2.rectangle(canvas,(bx_,by_),(bx_+300,by_+14),(40,40,40),-1)
                cv2.rectangle(canvas,(bx_,by_),(bx_+bw, by_+14),(0,210,80),-1)
                cv2.putText(canvas, f"Recording... {pct}%",
                            (bx_, by_+32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,210,80), 2)

                if len(collected) >= CALIB_FRAMES_PER_POINT:
                    valid = [s for s in collected if s is not None]
                    if len(valid) < 5:
                        canvas[:] = 0
                        cv2.putText(canvas,"Not enough data — aim nose at dot and press SPACE again",
                                    (SCREEN_W//2-440, SCREEN_H//2),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,80,255),2)
                        cv2.imshow(win, canvas); cv2.waitKey(1200)
                        collected = None
                    else:
                        ax = np.array([s[0] for s in valid])
                        ay = np.array([s[1] for s in valid])
                        def iqr(a):
                            q1,q3=np.percentile(a,25),np.percentile(a,75); iq=q3-q1
                            return (a>=q1-1.5*iq)&(a<=q3+1.5*iq)
                        mask = iqr(ax) & iqr(ay)
                        if mask.sum() < 5:
                            mask = np.ones(len(ax), bool)
                        offsets.append((float(np.mean(ax[mask])),
                                        float(np.mean(ay[mask]))))
                        screen_pts.append((tx, ty))

                        canvas[:] = 0
                        cv2.circle(canvas,(tx,ty),30,(0,255,50),-1)
                        cv2.putText(canvas, f"Recorded! ({i+1}/{total})",
                                    (SCREEN_W//2-140, SCREEN_H//2),
                                    cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,50),2)
                        cv2.imshow(win, canvas); cv2.waitKey(700); break

            cv2.imshow(win, canvas)

    cv2.destroyWindow(win)
    cx, cy = _fit_calib(offsets, screen_pts)
    _calib_cx = cx; _calib_cy = cy; _is_calibrated = True; _nose_init = False
    _kalman.reset(SCREEN_W//2, SCREEN_H//2)
    _save_calib(cx, cy, _nose_rest_x, _nose_rest_y, _iod_rest)
    print(f"[pupil] Calibration done — {total} points.")
    return True


def _run_full_setup():
    print("[pupil] ===== SETUP =====")
    if not _step1_face_distance(): return False
    if not _step2_measure_nose():  return False
    if not _step3_calibrate():     return False
    return True


# =============================================================================
# PUBLIC API
# =============================================================================

def init_camera():
    global cap, face_mesh, dot_x, dot_y, _overlay_canvas

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65,
    )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    dot_x = dot_y = float(SCREEN_W//2)
    _overlay_canvas = np.zeros((SCREEN_H, SCREEN_W, 3), np.uint8)

    cv2.namedWindow("GazeDot", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("GazeDot", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    if not _load_calib():
        print("[pupil] No calibration — running setup.")
        while not _run_full_setup():
            print("[pupil] Setup incomplete — retrying...")
    else:
        print("[pupil] Calibration loaded. Press R to redo.")

    _kalman.reset(SCREEN_W//2, SCREEN_H//2)
    print(f"[pupil] v14 (nose mode) ready | {SCREEN_W}x{SCREEN_H} | pywin32={_HAS_WIN32}")


def get_ear():
    """
    Main tracking loop.
    - Reads nose tip position → computes offset from rest → maps to screen via calibration.
    - Also computes EAR for blink detection.
    - Returns EAR value for the caller.
    """
    global dot_x, dot_y, _last_ear, _smooth_ox, _smooth_oy, _nose_init

    raw, display, lm, h, w = _grab_full()
    if display is None:
        return float(_last_ear)

    ear = _last_ear

    if lm:
        # --- EAR for blink ---
        l_e = _calc_ear(lm, MP_LEFT_EYE,  w, h)
        r_e = _calc_ear(lm, MP_RIGHT_EYE, w, h)
        ear = (l_e + r_e) / 2.0
        _last_ear = ear

        # --- Nose offset ---
        ox, oy = _nose_offset(lm, w, h)

        if not _nose_init:
            _smooth_ox = ox; _smooth_oy = oy; _nose_init = True
        else:
            _smooth_ox += NOSE_SMOOTH * (ox - _smooth_ox)
            _smooth_oy += NOSE_SMOOTH * (oy - _smooth_oy)

        if _is_calibrated:
            raw_sx, raw_sy = _apply_calib(_smooth_ox, _smooth_oy)
        else:
            raw_sx, raw_sy = SCREEN_W//2, SCREEN_H//2

        kx, ky = _kalman.update(raw_sx, raw_sy)
        dot_x  = float(np.clip(kx, 0, SCREEN_W-1))
        dot_y  = float(np.clip(ky, 0, SCREEN_H-1))
        _set_cursor(dot_x, dot_y)

        # --- Debug overlay ---
        for idx in MP_LEFT_EYE + MP_RIGHT_EYE:
            cv2.circle(display, _dp(lm,idx,w,h), 2, (0,180,0), -1)
        # Nose tip — big orange dot
        cv2.circle(display, _dp(lm, NOSE, w, h), 8, (0,165,255), -1)
        cv2.circle(display, _dp(lm, NOSE, w, h), 3, (255,255,255), -1)

        status  = "CLOSED" if ear < EAR_THRESHOLD else "OPEN"
        col     = (0,165,255) if _blink_state=="WAIT" else (255,255,255)
        cv2.putText(display,
                    f"v14-NOSE | EAR:{ear:.3f}[{status}] "
                    f"nose_off=({_smooth_ox:+.2f},{_smooth_oy:+.2f}) {_blink_state}",
                    (6,24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 2)
        cv2.putText(display, "R=recalibrate  Q=quit",
                    (6,h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (90,90,90), 1)
    else:
        cv2.putText(display, "No face — improve lighting or move closer",
                    (8,28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,80,255), 2)

    cv2.imshow("Nose Control Preview", display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        print("[pupil] Recalibrating..."); _run_full_setup()

    _overlay_canvas.fill(0)
    cv2.circle(_overlay_canvas, (int(dot_x),int(dot_y)), 16, (0,255,255), -1)
    cv2.circle(_overlay_canvas, (int(dot_x),int(dot_y)), 22, (0,200,200),  2)
    cv2.imshow("GazeDot", _overlay_canvas)
    cv2.waitKey(1)

    return float(ear)


def update_gaze():
    pass


def detect_blink(ear):
    global _closed_count, _blink_state, _first_blink_time
    result="NONE"; now=time.time(); ear=float(ear)
    if _blink_state=="WAIT":
        if now-_first_blink_time>DOUBLE_BLINK_GAP:
            result="SINGLE_BLINK"; _blink_state="IDLE"
    if ear<EAR_THRESHOLD:
        _closed_count+=1
    else:
        if _closed_count>=CONSEC_FRAMES:
            if _blink_state=="IDLE":
                _first_blink_time=now; _blink_state="WAIT"
            elif _blink_state=="WAIT":
                result="DOUBLE_BLINK"; _blink_state="IDLE"
        _closed_count=0
    return str(result)


def do_click():
    pyautogui.click(int(dot_x), int(dot_y))

def do_scroll():
    _scroll_down()

def release_camera():
    global cap, face_mesh
    if face_mesh: face_mesh.close()
    if cap:       cap.release()
    cv2.destroyAllWindows()
    print("[pupil] Camera released.")


# =============================================================================
# STANDALONE
# =============================================================================
if __name__ == "__main__":
    import subprocess

    if os.path.exists(CALIB_FILE):
        os.remove(CALIB_FILE)
        print("[pupil] Old calibration deleted — fresh setup will run.")

    subprocess.Popen([
        "C:/Program Files/Google/Chrome/Application/chrome.exe",
        "--start-maximized", "https://www.instagram.com/reels/"
    ])
    time.sleep(3)

    init_camera()
    print("\n[v14-NOSE] Running — nose-tip cursor control.")
    print("  Tilt head     ->  cursor moves")
    print("  1 blink       ->  click")
    print("  2 blinks      ->  scroll down")
    print("  R (preview)   ->  recalibrate")
    print("  Q             ->  quit\n")

    while True:
        ear    = get_ear()
        result = detect_blink(ear)
        if result == "SINGLE_BLINK":
            do_click()
            print(f"  [CLICK]  ({int(dot_x)}, {int(dot_y)})")
        elif result == "DOUBLE_BLINK":
            do_scroll()
            print(f"  [SCROLL] ({int(dot_x)}, {int(dot_y)})")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    release_camera()
