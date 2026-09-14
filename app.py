import time
import pickle
import av
import cv2 as cv
import pandas as pd
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils

st.set_page_config(page_title="Real-time Gesture Recognition", layout="wide")
st.title("Sign Language & Gesture Recognition")

# 1. Load ML Model & Mediapipe Options
@st.cache_resource
def load_models():
    BaseOptions = mp.tasks.BaseOptions
    HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    model_path = './holistic_landmarker.task'
    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    
    with open('./Gesture_model.pkl', 'rb') as file:
        model_pack = pickle.load(file)
        
    return options, model_pack

options, model_pack = load_models()
xgb_classifier = model_pack['model']
scaler = model_pack['scaler']
label_enc = model_pack['label_encoder']

POSE_GRAPH = vision.PoseLandmarksConnections.POSE_LANDMARKS
HAND_GRAPH = vision.HandLandmarksConnections.HAND_CONNECTIONS

def Gesture_checker(raw_data):
    scaled_data = scaler.transform(raw_data)
    numeric_prediction = xgb_classifier.predict(scaled_data)
    text_prediction = label_enc.inverse_transform(numeric_prediction)
    return text_prediction[0]

# 2. WebRTC Video Processor Class
class LandmarkProcessor:
    def __init__(self):
        self.landmarker = mp.tasks.vision.HolisticLandmarker.create_from_options(options)
        self.timestamp_ms = 0
        self.frame_count = 0
        self.last_result = "Unknown"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.frame_count += 1

        img = frame.to_ndarray(format="bgr24")

                # Only run MediaPipe on every 3rd frame
        if self.frame_count % 3 != 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        img = frame.to_ndarray(format="bgr24")

        rgb_frame = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        self.timestamp_ms += 33

        landmarker_result = self.landmarker.detect_for_video(
            mp_image,
         self.timestamp_ms
      )

        return av.VideoFrame.from_ndarray(img, format="bgr24")
# 3. WebRTC Streamer Setup
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="gesture-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=LandmarkProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=False,
)
