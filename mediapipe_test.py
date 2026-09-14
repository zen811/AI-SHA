import sys
import time
import pickle
import pyttsx3
import cv2 as cv
import subprocess
import pandas as pd
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils

# engine = pyttsx3.init()


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

with open('./Gesture_model.pkl', 'rb') as file:
    model_pack = pickle.load(file) 

xgb_classifier = model_pack['model']
scalar = model_pack['scaler']
label_enc = model_pack['label_encoder']

# def speak(text):
#     engine.say(text)
#     engine.runAndWait()

def Gesture_checker(raw_data):
    
    scaled_data = scalar.transform(raw_data)
    
    numeric_prediction = xgb_classifier.predict(scaled_data)
    
    text_prediction = label_enc.inverse_transform(numeric_prediction)
    
    return text_prediction[0]

def draw_landmarks_group(frame, landmarks_list):
    end_pt=None
    if landmarks_list:
        for landmark in landmarks_list:
            x = int(landmark.x * frame.shape[1])            # Convert normalized relative coordinates to absolute pixel values
            y = int(landmark.y * frame.shape[0])
            start_pt = (x, y)
            landmark_coordinates=[landmark.x,landmark.y,landmark.z]
            text=[round(_,3) for _ in landmark_coordinates]
            cv.putText(frame,str(text),(x,y),cv.FONT_HERSHEY_SIMPLEX,0.25,(0, 225, 0),2)
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
        landmarkdata = []
        if landmarker_result:
            draw_landmarks_group(frame, landmarker_result.left_hand_landmarks)
            draw_landmarks_group(frame, landmarker_result.right_hand_landmarks)            # Draw Hands tracking targets
            draw_landmarks_group(frame, landmarker_result.pose_landmarks)            # Draw Pose skeletal target tracks
            
        if landmarker_result.pose_landmarks and len(landmarker_result.pose_landmarks) > 0:
            pose_points = (landmarker_result.pose_landmarks[0]
                            if isinstance(landmarker_result.pose_landmarks[0], list)
                            else landmarker_result.pose_landmarks
                            )
            for landmark in pose_points:
                landmarkdata.extend([landmark.x, landmark.y, landmark.z])
        else:
            landmarkdata.extend([0.0] * (33 * 3))


        if landmarker_result.left_hand_landmarks and len(landmarker_result.left_hand_landmarks)>0:
            left_hand_points=(landmarker_result.left_hand_landmarks[0]
                              if isinstance(landmarker_result.left_hand_landmarks[0],list)
                              else landmarker_result.left_hand_landmarks)
            for landmark in left_hand_points:
                landmarkdata.extend([landmark.x,landmark.y,landmark.z])
        else:
            landmarkdata.extend([0.0] * (21 * 3))


        if landmarker_result.right_hand_landmarks and len(landmarker_result.right_hand_landmarks)>0:
            right_hand_points=(landmarker_result.right_hand_landmarks[0]
                              if isinstance(landmarker_result.right_hand_landmarks[0],list)
                              else landmarker_result.right_hand_landmarks)
            for landmark in right_hand_points:
                landmarkdata.extend([landmark.x,landmark.y,landmark.z])
        else:
            landmarkdata.extend([0.0] * (21 * 3))
        
        Image_frame_data = pd.DataFrame([landmarkdata],columns=range(225))
        result = Gesture_checker(Image_frame_data)
        text_res_prev = ""

        if result==0:
            text_res="Hello"
        if result==1:
            text_res="Thank You"
        # if result==2:
        #     text_res="Sorry"
        # if result ==3:
        #     text_res="Bye"

        cv.putText(frame, str(text_res), (50, 80),
           cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        # if text_res != text_res_prev:
        #     threading.Thread(
        #     target=speak,
        #     args=(text_res,),
        #     daemon=True
        # ).start()
        cv.imshow('MediaPipe Tasks Tracking', frame)
        print(text_res)
        text_res_prev = text_res

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv.destroyAllWindows()


    
