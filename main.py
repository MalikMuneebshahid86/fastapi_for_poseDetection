import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI

app = FastAPI()

mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

count = 0
level = None


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
        np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


@app.get("/")
def read_root():
    return {"message": "Welcome to the Bicep Curls API"}


@app.get("/bicep-curls-right")
def biceps_curls_right():
    global count, level

    capture = cv2.VideoCapture(0)
    with mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5) as pose:
        while capture.isOpened():
            ret, frame = capture.read()

            # Changing image color
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Making detection
            res = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = res.pose_landmarks.landmark

                # Get coordinates
                shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow_R = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist_R = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # Calculate angle
                angle = calculate_angle(shoulder_r, elbow_R, wrist_R)

                # Visualize angle
                cv2.putText(image, str(angle),
                            tuple(np.multiply(
                                elbow_R, [780, 240]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Curl counter logic
                if angle > 160:
                    level = "down"
                if angle < 45 and level == 'down':
                    level = "up"
                    count += 1
                    print(count)

            except:
                pass

            cv2.putText(image, 'COUNT', (20, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(count),
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, 'LEVEL', (100, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, level,
                        (95, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            mp_draw.draw_landmarks(
                image, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            print(res)
            cv2.imshow('Right side Bicep Curls', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        capture.release()
        cv2.destroyAllWindows()
