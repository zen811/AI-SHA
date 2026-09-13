from pathlib import Path
import mediapipe as mp
import pandas as pd
import cv2 as cv

BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


model_path = './model_training/holistic_landmarker.task'
folder_path = Path("./model_training/gestures/hello")
file_list=[]
for file in folder_path.iterdir():
    if file.is_file():
        file_list.append(file)

print(file_list)