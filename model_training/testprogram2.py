from pathlib import Path
import mediapipe as mp
import pandas as pd
import cv2 as cv

BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode



model_path = './AI-SHA/model_training/holistic_landmarker.task'
folder_path = Path("./AI-SHA/model_training/gestures/hello")
file_list=[]
for file in folder_path.iterdir():
    if file.is_file():
        file_list.append(str(Path(file)))

print(file_list)



from pathlib import Path

# Your original list of PosixPath objects
path_list = [Path('/user/docs/file1.txt'), Path('/user/docs/file2.txt')]

# Convert them to a list of strings
string_list = [str(path) for path in path_list]

print(string_list)
# Output: ['/user/docs/file1.txt', '/user/docs/file2.txt']
