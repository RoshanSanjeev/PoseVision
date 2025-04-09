import cv2
import mediapipe as mp
import time
import math
import os
import subprocess

class poseDetector:
    def __init__(self, mode=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.mode,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec_connections = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0), thickness=2, circle_radius=2)

    def findPose(self, frame, draw=True):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.result = self.pose.process(frame_rgb)
        if self.result.pose_landmarks and draw:
            self.mp_drawing.draw_landmarks(
                frame,
                self.result.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                connection_drawing_spec=self.drawing_spec_connections
            )
        return frame, self.result

    def getPosition(self, img, draw=True):
        lmList = []
        if self.result.pose_landmarks:
            for id, lm in enumerate(self.result.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 6, (0, 0, 0), thickness=2)
                    cv2.circle(img, (cx, cy), 4, (255, 255, 255), thickness=-1)
        return lmList

    def getAngle(self, p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        angle = math.degrees(
            math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
        )
        angle = abs(angle)
        if angle > 180:
            angle = 360 - angle
        return angle

def fix_video_with_ffmpeg(input_path, output_path):
    print("🔧 Fixing video container with ffmpeg...")
    fixed_path = output_path.replace(".mp4", "_fixed.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", output_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        fixed_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    os.replace(fixed_path, output_path)
    print("✅ Video container fixed!")

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    detector = poseDetector()
    pTime = 0
    frame_count = 0
    feedback_times = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success or frame is None:
            break

        frame, result = detector.findPose(frame)
        lmList = detector.getPosition(frame)
        left_angle = right_angle = None
        current_time = frame_count / 24.0  # assuming 24 FPS

        if lmList:
            lmDict = {id: (x, y) for id, x, y in lmList}

            if all(k in lmDict for k in [23, 25, 27]):
                left_angle = detector.getAngle(lmDict[23], lmDict[25], lmDict[27])
                cv2.putText(frame, f"{int(left_angle)}deg", lmDict[25],
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

            if all(k in lmDict for k in [24, 26, 28]):
                right_angle = detector.getAngle(lmDict[24], lmDict[26], lmDict[28])
                cv2.putText(frame, f"{int(right_angle)}deg", lmDict[26],
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

            if left_angle is not None and right_angle is not None:
                diff = abs(left_angle - right_angle)
                if diff >= 4.5:
                    detector.drawing_spec_connections = detector.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                    feedback_times.append(round(current_time, 2))
                else:
                    detector.drawing_spec_connections = detector.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

                detector.mp_drawing.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    detector.mp_pose.POSE_CONNECTIONS,
                    connection_drawing_spec=detector.drawing_spec_connections
                )

        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime

        cv2.putText(frame, f'FPS: {int(fps)}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if out is None:
            height, width = frame.shape[:2]
            out = cv2.VideoWriter(output_path, fourcc, 24.0, (width, height))

        out.write(frame)
        frame_count += 1

    cap.release()
    if out:
        out.release()
        fix_video_with_ffmpeg(input_path, output_path)

    # Group close feedback times into ranges
    grouped_feedback = []
    if feedback_times:
        start = feedback_times[0]
        end = start
        for i in range(1, len(feedback_times)):
            current = feedback_times[i]
            if current - end <= 1.0:
                end = current
            else:
                grouped_feedback.append({
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "feedback": "Squat appears uneven. Try incorporating Cossack squats with or without a kettlebell to improve flexibility and unilateral strength."
                })
                start = current
                end = current
        grouped_feedback.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "feedback": "Squat appears uneven. Try incorporating Cossack squats with or without a kettlebell to improve flexibility and unilateral strength."
        })

    return grouped_feedback
