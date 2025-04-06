import cv2
import mediapipe as mp
import time
import math

class poseDetector:
    def __init__(self, mode=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        # Initialize MediaPipe Pose model
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.mode,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Define drawing specs once in init
        self.drawing_spec_connections = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0),  # Green
            thickness=2,
            circle_radius=2
        )
#draws pipes connected by dots
    def findPose(self, frame, draw=True):
        # Convert BGR to RGB
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
#draws dots
    def getPosition(self, img, draw=True):
        lmList = []
        if self.result.pose_landmarks:
            for id, lm in enumerate(self.result.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    # Outer circle (border)
                    cv2.circle(img, (cx, cy), 6, (0, 0, 0), thickness=2)  # Black border

                    # Inner circle (fill)
                    cv2.circle(img, (cx, cy), 4, (255, 255, 255), thickness=-1)  # white fill
        return lmList
    def getAngle(self, p1, p2, p3):
        # p1, p2, p3 are tuples: (x, y)
        x1, y1 = p1
        x2, y2 = p2  # joint in the middle
        x3, y3 = p3

        angle = math.degrees(
            math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
        )
        angle = abs(angle)
        if angle > 180:
            angle = 360 - angle
        return angle


def main():
    cap = cv2.VideoCapture('/Users/roshansanjeev/Desktop/AdvancedComputerVision/Videos/FrontSquat.mp4')
    pTime = 0
    detector = poseDetector()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Detect pose and draw landmarks
        frame, result = detector.findPose(frame)

        # Get landmark positions
        lmList = detector.getPosition(frame)
        drawing_spec_connections = None  # Default placeholder

        if lmList:
            lmDict = {id: (x, y) for id, x, y in lmList}
            left_angle = right_angle = None

            if all(k in lmDict for k in [23, 25, 27]):
                left_angle = detector.getAngle(lmDict[23], lmDict[25], lmDict[27])
                cv2.putText(frame, f"{int(left_angle)}deg", lmDict[25],
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

            if all(k in lmDict for k in [24, 26, 28]):
                right_angle = detector.getAngle(lmDict[24], lmDict[26], lmDict[28])
                cv2.putText(frame, f"{int(right_angle)}deg", lmDict[26],
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

            # Now compare if both angles exist
            if left_angle is not None and right_angle is not None:
                if abs(left_angle - right_angle) < 5:
                    drawing_spec_connections = mp.solutions.drawing_utils.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=2)  # Green
                else:
                    drawing_spec_connections = mp.solutions.drawing_utils.DrawingSpec(
                        color=(0, 0, 255), thickness=2, circle_radius=2)  # Red

                # Redraw with updated spec
                detector.mp_drawing.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    detector.mp_pose.POSE_CONNECTIONS,
                    connection_drawing_spec=drawing_spec_connections
                )



                # FPS calculation
                cTime = time.time()
                fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
                pTime = cTime

                # Display FPS
                cv2.putText(frame, f'FPS: {int(fps)}', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Show frame
                cv2.imshow('MediaPipe Pose', frame)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
