import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate joint angle
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

cap = cv2.VideoCapture(0)

counter = 0
stage = None
last_seen_time = time.time()

with mp_pose.Pose(min_detection_confidence=0.6,
                  min_tracking_confidence=0.6) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # ✅ Handle no detection
        if results.pose_landmarks is None:
            print("⚠️ No skeleton detected, waiting...")
            cv2.putText(frame, "No body detected - step back / adjust camera",
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if time.time() - last_seen_time > 5:
                counter = 0
                stage = None
                print("🔄 Counter reset (no skeleton for 5s)")

            cv2.imshow("Squat Counter", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            time.sleep(0.1)
            continue

        last_seen_time = time.time()

        # Extract landmarks
        landmarks = results.pose_landmarks.landmark
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        # Calculate knee angle
        angle = calculate_angle(hip, knee, ankle)

        # Convert angle to % progress (straight=100, squat=0)
        progress = np.interp(angle, (90, 170), (0, 100))  
        progress = np.clip(progress, 0, 100)

        print(f"Hip: {hip}, Knee: {knee}, Ankle: {ankle}, Angle: {angle:.2f}, Progress: {progress:.1f}%")

        # ✅ Squat logic based on progress bar
        if progress < 30 and stage != "down":  # Deep squat detected
            stage = "down"
        if progress > 90 and stage == "down":  # Fully back up
            stage = "up"
            counter += 1
            print(f"✅ Squat Count: {counter}")

        # Draw skeleton
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=1)
        )

        # Progress bar visualization
        bar_x, bar_y = 50, 400
        bar_height = 200
        filled = int((progress/100) * bar_height)
        cv2.rectangle(frame, (bar_x, bar_y-bar_height), (bar_x+30, bar_y), (255,255,255), 2)
        cv2.rectangle(frame, (bar_x, bar_y-filled), (bar_x+30, bar_y), (0,255,0), -1)
        cv2.putText(frame, f"{int(progress)}%", (bar_x, bar_y-bar_height-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Overlay counter
        cv2.putText(frame, f"Reps: {counter}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.imshow("Squat Counter", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()