from pathlib import Path
import cv2 as cv
import mediapipe as mp
import pandas as pd

BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = str(Path("./model_training/holistic_landmarker.task").resolve())
folder_path = Path("./model_training/gestures/Bye").resolve()

file_list = [
    str(file.resolve())
    for file in folder_path.iterdir()
    if file.is_file() and file.suffix.lower() in [".jpg", ".jpeg", ".png"]
]

options = HolisticLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    min_face_detection_confidence=0.5,
    min_pose_detection_confidence=0.5,
    min_hand_landmarks_confidence=0.5,
)

landmark_hello = []

with HolisticLandmarker.create_from_options(options) as landmarker:
  for id, file in enumerate(file_list):
    image = cv.imread(file)
    resized_image = cv.resize(image, (640, 480))
    rgb_image = cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    result = landmarker.detect(mp_image)
    landmarkdata = []

    if result.pose_landmarks and len(result.pose_landmarks) > 0:      #Pose Landmarks (33 landmarks x 3 coordinates = 99 values)
      pose_points = (                                                 #Outer index handles the pose list return wrapper
          result.pose_landmarks[0]
          if isinstance(result.pose_landmarks[0], list)
          else result.pose_landmarks
      )
      for landmark in pose_points:
        landmarkdata.extend([landmark.x, landmark.y, landmark.z])
        
    else:
      landmarkdata.extend([0.0] * (33 * 3))

    if result.left_hand_landmarks and len(result.left_hand_landmarks) > 0: #Left Hand Landmarks (21 landmarks x 3 coordinates = 63 values)
      left_hand_points = (
          result.left_hand_landmarks[0]
          if isinstance(result.left_hand_landmarks[0], list)
          else result.left_hand_landmarks
      )
      for landmark in left_hand_points:
        landmarkdata.extend([landmark.x, landmark.y, landmark.z])
    else:
      landmarkdata.extend([0.0] * (21 * 3))

    if result.right_hand_landmarks and len(result.right_hand_landmarks) > 0:    #Right Hand Landmarks (21 landmarks x 3 coordinates = 63 values)
      right_hand_points = (
          result.right_hand_landmarks[0]
          if isinstance(result.right_hand_landmarks[0], list)
          else result.right_hand_landmarks
      )
      for landmark in right_hand_points:
        landmarkdata.extend([landmark.x, landmark.y, landmark.z])
    else:
      landmarkdata.extend([0.0] * (21 * 3))

    landmarkdata.append(3)
    landmark_hello.append(landmarkdata)

df = pd.DataFrame(landmark_hello)
df.to_csv("./model_training/Bye.csv", index=False)