import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 2
PREDICTOR_PATH = "C:/Users/Akram Ali Faridi/Documents/UiPath/EyeScrollingControl/script/shape_predictor_68_face_landmarks.dat"

cap = None
detector = None
predictor = None
COUNTER = 0

LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

def eye_aspect_ratio(eye_pts):
    A = dist.euclidean(eye_pts[1], eye_pts[5])
    B = dist.euclidean(eye_pts[2], eye_pts[4])
    C = dist.euclidean(eye_pts[0], eye_pts[3])
    return (A + B) / (2.0 * C)

def init_camera():
    global cap, detector, predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    cap = cv2.VideoCapture(0)

def get_ear():
    global cap, detector, predictor
    ret, frame = cap.read()
    if not ret:
        return 0.35
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    if len(faces) == 0:
        return 0.35
    shape = predictor(gray, faces[0])
    coords = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
    left_ear = eye_aspect_ratio(coords[LEFT_EYE])
    right_ear = eye_aspect_ratio(coords[RIGHT_EYE])
    return (left_ear + right_ear) / 2.0

def detect_blink(ear):
    global COUNTER
    if ear < EAR_THRESHOLD:
        COUNTER += 1
        if COUNTER >= CONSEC_FRAMES:
            COUNTER = 0
            return "BLINK"
    else:
        COUNTER = 0
    return "OPEN"

def release_camera():
    global cap
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    init_camera()
    print("Camera started. Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get the EAR and check for blinks
        ear = get_ear()
        status = detect_blink(ear)

        # Draw the status on the screen
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Status: {status}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show the camera window
        cv2.imshow("Eye Control", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    release_camera()