import cv2 as cv
import mediapipe as mp
import time

# 1. Clear camera stream initialization
cap = cv.VideoCapture(0)

# 2. Instantiate configuration DIRECTLY to bypass the cache collision
# Explicitly omitting 'result_callback' ensures validation passes
options = mp.tasks.vision.HolisticLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path='./AI-SHA/holistic_landmarker.task'),
    running_mode=mp.tasks.vision.RunningMode.VIDEO, # MUST match detect_for_video
    min_face_detection_confidence=0.5,
    min_hand_landmarks_confidence=0.5
)

# Helper function to plot coordinates safely
def draw_group(frame, landmark_list):
    if landmark_list:
        for landmark in landmark_list:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv.circle(frame, (x, y), 3, (0, 255, 0), -1)

# 3. Open pipeline task instance
with mp.tasks.vision.HolisticLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video stream frame capture failed.")
            break

        # Process standard video color alignments
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Monotonically increasing milliseconds sequence
        frame_timestamp_ms = int(time.time() * 1000)

        # Run detection mapping synchronously
        landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # 4. Safely extract and draw holistic attributes
        if landmarker_result:
            # Draw Hand tracks
            draw_group(frame, landmarker_result.left_hand_landmarks)
            draw_group(frame, landmarker_result.right_hand_landmarks)
            
            # Draw Face mesh & Pose tracking maps
            draw_group(frame, landmarker_result.face_landmarks)
            draw_group(frame, landmarker_result.pose_landmarks)

        # Show Output Live Window
        cv.imshow('AI-SHA Holistic Pipeline', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
