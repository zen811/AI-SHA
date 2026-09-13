import time
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils



BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
POSE_GRAPH = vision.PoseLandmarksConnections.POSE_LANDMARKS
HAND_GRAPH = vision.HandLandmarksConnections.HAND_CONNECTIONS

model_path = './holistic_landmarker.task'        #the actual model which does the landmarking


options = HolisticLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    min_face_detection_confidence = 0.5,
    min_pose_detection_confidence=0.5,
    min_hand_landmarks_confidence=0.5,
)


cap = cv.VideoCapture(0)
ret, frame = cap.read()
height, width, channels = frame.shape


def draw_landmarks_group(frame, landmarks_list):
    end_pt=None
    if landmarks_list:
        for landmark in landmarks_list:
            x = int(landmark.x * frame.shape[1])            # Convert normalized relative coordinates to absolute pixel values
            y = int(landmark.y * frame.shape[0])
            start_pt = (x, y)
            landmark_coordinates=[landmark.x,landmark.y,landmark.z]
            text=[round(_,3) for _ in landmark_coordinates]
            cv.putText(frame,str(text),(x,y),cv.FONT_HERSHEY_SIMPLEX,0.25,(0, 255, 0),2)
            cv.circle(frame, (x, y), 3, (0, 255, 0), -1)      #Draw a tracking node dot
            drawing_utils.draw_landmarks(frame, landmarker_result.pose_landmarks, POSE_GRAPH)
            drawing_utils.draw_landmarks(frame,landmarker_result.left_hand_landmarks,HAND_GRAPH)
            drawing_utils.draw_landmarks(frame,landmarker_result.right_hand_landmarks,HAND_GRAPH)



with HolisticLandmarker.create_from_options(options) as landmarker:
    while True:

        ret, frame = cap.read()  #reading the actual frame
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)        #conversion of the frame into RGB to be used by mediapipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame) #
        frame_timestamp_ms = int(time.time() * 1000)        #time tracking

        landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms) #landmarker detection

        if landmarker_result:
            draw_landmarks_group(frame, landmarker_result.left_hand_landmarks)
            draw_landmarks_group(frame, landmarker_result.right_hand_landmarks)            # Draw Hands tracking targets
            
            draw_landmarks_group(frame, landmarker_result.pose_landmarks)            # Draw Pose skeletal target tracks
            
            #draw_landmarks_group(frame, landmarker_result.face_landmarks)             # Draw Face structure map
        cv.imshow('MediaPipe Tasks Tracking', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv.destroyAllWindows()


    