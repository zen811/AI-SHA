from pathlib import Path
import mediapipe as mp
import pandas as pd
import cv2 as cv

BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


model_path = str(Path('./model_training/holistic_landmarker.task').resolve())
folder_path = Path("./model_training/gestures/hello").resolve()

file_list=[]
for file in folder_path.iterdir():
    if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        file_list.append(str(file.resolve()))


options = HolisticLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    min_face_detection_confidence=0.5,
    min_pose_detection_confidence=0.5,
    min_hand_landmarks_confidence=0.5)




landmark_hello=[]
with HolisticLandmarker.create_from_options(options) as landmarker:
    for id,file in enumerate(file_list):
        mp_image = mp.Image.create_from_file(str(file))
        result = landmarker.detect(mp_image)
        landmarkdata=[]
        for idx, landmark in enumerate(result.pose_landmarks):
            print(f"image{id:02d} Poses\nIndex {idx:02d} -> X: {landmark.x:.4f}, Y: {landmark.y:.4f}, Z: {landmark.z:.4f}")
            landmarkdata.append(idx)
            landmarkdata.append(landmark.x)
            landmarkdata.append(landmark.y)
            landmarkdata.append(landmark.z)
        for idx,landmark in enumerate(result.left_hand_landmarks):
            print(f"image{id:02d} left hand \nIndex {idx:02d} -> X: {landmark.x:.4f}, Y: {landmark.y:.4f}, Z: {landmark.z:.4f}")
            landmarkdata.append(idx)
            landmarkdata.append(landmark.x)
            landmarkdata.append(landmark.y)
            landmarkdata.append(landmark.z)
        for idx,landmark in enumerate(result.right_hand_landmarks):
            print(f"image{id:02d} right hand\nIndex {idx:02d} -> X: {landmark.x:.4f}, Y: {landmark.y:.4f}, Z: {landmark.z:.4f}")
            landmarkdata.append(idx)
            landmarkdata.append(landmark.x)
            landmarkdata.append(landmark.y)
            landmarkdata.append(landmark.z)
        landmarkdata.append(0)
        landmark_hello.append(landmarkdata)

df = pd.DataFrame(landmark_hello)
df.to_csv("Hello.csv", index=False)





