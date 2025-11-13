import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

# --- Config ---
SHOW_VIDEO = True            # set False to run headless (no window)
IMG_W, IMG_H = 640, 480
VIS_THRESH = 0.7
DOWN_ANGLE = 70              # knee angle threshold to consider 'down'
UP_ANGLE = 160               # knee angle threshold to consider 'up'
BOTTOM_HOLD_FRAMES = 3       # require being below DOWN_ANGLE this many frames
SMOOTH_WIN = 5               # angle moving-average window

# --- Helpers ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ang = np.degrees(np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0]))
    ang = abs(ang)
    if ang > 180:
        ang = 360 - ang
    return ang

def visible(landmarks, *idxs):
    return all(landmarks[i].visibility >= VIS_THRESH for i in idxs)

# --- MediaPipe ---
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- State ---
counter = 0
stage = "idle"               # idle -> going_down -> bottom -> going_up
bottom_frames = 0
angles_hist = deque(maxlen=SMOOTH_WIN)
last_print = time.time()

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

print("Starting squat counter. Tips:")
print("- Ensure full body is visible (hips, knees, ankles).")
print("- Tidy the background to avoid ghost poses.")
print("- Terminal will show live angles and events. Press 'q' to quit.\n")

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        print("Camera frame not captured.")
        break

    frame = cv2.resize(frame, (IMG_W, IMG_H))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark

        # Require both legs visible for robustness
        if visible(lm, mp_pose.PoseLandmark.LEFT_HIP.value,
                      mp_pose.PoseLandmark.LEFT_KNEE.value,
                      mp_pose.PoseLandmark.LEFT_ANKLE.value,
                      mp_pose.PoseLandmark.RIGHT_HIP.value,
                      mp_pose.PoseLandmark.RIGHT_KNEE.value,
                      mp_pose.PoseLandmark.RIGHT_ANKLE.value):

            LHIP = (lm[mp_pose.PoseLandmark.LEFT_HIP.value].x,  lm[mp_pose.PoseLandmark.LEFT_HIP.value].y)
            LKNE = (lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y)
            LANK = (lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y)

            RHIP = (lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x,  lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y)
            RKNE = (lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y)
            RANK = (lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y)

            left_ang  = calculate_angle(LHIP, LKNE, LANK)
            right_ang = calculate_angle(RHIP, RKNE, RANK)
            angle = (left_ang + right_ang) / 2.0

            # Smooth angle
            angles_hist.append(angle)
            smooth_angle = sum(angles_hist) / len(angles_hist)

            # ---- Terminal live print (throttled) ----
            now = time.time()
            if now - last_print > 0.25:  # print ~4 times/sec to keep it readable
                print(f"Angle={smooth_angle:6.1f}°, Stage={stage:<11} | Reps={counter}")
                last_print = now

            # ---- State machine ----
            if stage in ("idle", "going_up"):
                if smooth_angle < DOWN_ANGLE:
                    stage = "going_down"

            if stage == "going_down":
                if smooth_angle < DOWN_ANGLE:
                    bottom_frames += 1
                    if bottom_frames >= BOTTOM_HOLD_FRAMES:
                        stage = "bottom"
                        print("⬇️  Bottom reached (below threshold). Hold detected.")
                else:
                    # aborted descent
                    stage = "idle"
                    bottom_frames = 0

            elif stage == "bottom":
                if smooth_angle > UP_ANGLE:
                    stage = "going_up"

            elif stage == "going_up":
                if smooth_angle > UP_ANGLE:
                    counter += 1
                    print(f"✅ Squat counted! Total reps = {counter}\n")
                    stage = "idle"
                    bottom_frames = 0

        else:
            # Not all leg landmarks are visible; guide user
            print("⚠️  Legs not fully visible (hip/knee/ankle). Step back or adjust camera.")
    else:
        print("⚠️  No pose detected. Make sure your whole body is in frame.")

    # ---- Optional video ----
    if SHOW_VIDEO:
        if res.pose_landmarks:
            mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                                   mp_draw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2))
        # small on-screen HUD (optional)
        cv2.putText(frame, f"Reps: {counter}  Stage: {stage}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Squat Counter (terminal-driven)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()