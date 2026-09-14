import time
import pickle
import av
import cv2 as cv
import pandas as pd
import mediapipe as mp
import streamlit as st

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
)

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils


# --------------------------------------------------
# STREAMLIT PAGE
# --------------------------------------------------

st.set_page_config(
    page_title="Real-time Gesture Recognition",
    layout="wide"
)

st.title("Sign Language & Gesture Recognition")


# --------------------------------------------------
# 1. LOAD ML MODEL & MEDIAPIPE OPTIONS
# --------------------------------------------------

@st.cache_resource
def load_models():

    BaseOptions = mp.tasks.BaseOptions
    HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    model_path = "./holistic_landmarker.task"

    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=model_path
        ),
        running_mode=VisionRunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )

    with open("./Gesture_model.pkl", "rb") as file:
        model_pack = pickle.load(file)

    return options, model_pack


options, model_pack = load_models()

xgb_classifier = model_pack["model"]
scaler = model_pack["scaler"]
label_enc = model_pack["label_encoder"]


# --------------------------------------------------
# MEDIAPIPE DRAWING CONNECTIONS
# --------------------------------------------------

POSE_GRAPH = vision.PoseLandmarksConnections.POSE_LANDMARKS
HAND_GRAPH = vision.HandLandmarksConnections.HAND_CONNECTIONS


# --------------------------------------------------
# 2. WEBRTC VIDEO PROCESSOR
# --------------------------------------------------

class LandmarkProcessor:

    def __init__(self):

        self.landmarker = (
            mp.tasks.vision.HolisticLandmarker.create_from_options(
                options
            )
        )

        self.timestamp_ms = 0
        self.frame_count = 0

        self.last_prediction = "Unknown"


    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:

        # Count incoming frames
        self.frame_count += 1

        # Convert frame to OpenCV image
        img = frame.to_ndarray(format="bgr24")


        # --------------------------------------------------
        # PROCESS ONLY EVERY 10TH FRAME
        # --------------------------------------------------

        if self.frame_count % 10 == 0:

            # BGR -> RGB
            rgb_frame = cv.cvtColor(
                img,
                cv.COLOR_BGR2RGB
            )

            # Create MediaPipe image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Increasing timestamp
            self.timestamp_ms += 33


            # --------------------------------------------------
            # MEDIAPIPE
            # --------------------------------------------------

            landmarker_result = (
                self.landmarker.detect_for_video(
                    mp_image,
                    self.timestamp_ms
                )
            )


            # --------------------------------------------------
            # CREATE 225 LANDMARK FEATURES
            # --------------------------------------------------

            landmarkdata = []


            # --------------------------------------------------
            # POSE
            # 33 landmarks × 3 = 99
            # --------------------------------------------------

            if (
                landmarker_result.pose_landmarks
                and len(landmarker_result.pose_landmarks) > 0
            ):

                pose_points = (
                    landmarker_result.pose_landmarks[0]
                    if isinstance(
                        landmarker_result.pose_landmarks[0],
                        list
                    )
                    else landmarker_result.pose_landmarks
                )

                for landmark in pose_points:

                    landmarkdata.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ])

            else:

                landmarkdata.extend([0.0] * 99)


            # --------------------------------------------------
            # LEFT HAND
            # 21 landmarks × 3 = 63
            # --------------------------------------------------

            if (
                landmarker_result.left_hand_landmarks
                and len(landmarker_result.left_hand_landmarks) > 0
            ):

                left_hand_points = (
                    landmarker_result.left_hand_landmarks[0]
                    if isinstance(
                        landmarker_result.left_hand_landmarks[0],
                        list
                    )
                    else landmarker_result.left_hand_landmarks
                )

                for landmark in left_hand_points:

                    landmarkdata.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ])

            else:

                landmarkdata.extend([0.0] * 63)


            # --------------------------------------------------
            # RIGHT HAND
            # 21 landmarks × 3 = 63
            # --------------------------------------------------

            if (
                landmarker_result.right_hand_landmarks
                and len(landmarker_result.right_hand_landmarks) > 0
            ):

                right_hand_points = (
                    landmarker_result.right_hand_landmarks[0]
                    if isinstance(
                        landmarker_result.right_hand_landmarks[0],
                        list
                    )
                    else landmarker_result.right_hand_landmarks
                )

                for landmark in right_hand_points:

                    landmarkdata.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ])

            else:

                landmarkdata.extend([0.0] * 63)


            # --------------------------------------------------
            # VERIFY 225 FEATURES
            # --------------------------------------------------

            cv.putText(
                img,
                f"Landmarks: {len(landmarkdata)}",
                (50, 80),
                cv.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )


            # --------------------------------------------------
            # TEST SCALER ONLY
            # --------------------------------------------------

            Image_frame_data = pd.DataFrame(
                [landmarkdata],
                columns=range(225)
            )


            # IMPORTANT:
            # We are testing ONLY scaler.transform()
            # XGBoost prediction is NOT being called yet.

            scaled_data = scaler.transform(
                Image_frame_data
            )


            # --------------------------------------------------
            # SCALER SUCCESS
            # --------------------------------------------------

            cv.putText(
                img,
                "SCALER OK",
                (50, 130),
                cv.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

        else:

            cv.putText(
                img,
                "Running...",
                (50, 80),
                cv.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )


        # --------------------------------------------------
        # RETURN FRAME
        # --------------------------------------------------

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


# --------------------------------------------------
# 3. WEBRTC CONFIGURATION
# --------------------------------------------------

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    }
)


# --------------------------------------------------
# 4. START CAMERA
# --------------------------------------------------

webrtc_streamer(

    key="gesture-detection",

    mode=WebRtcMode.SENDRECV,

    rtc_configuration=RTC_CONFIGURATION,

    video_processor_factory=LandmarkProcessor,

    media_stream_constraints={

        "video": {

            "width": {
                "ideal": 640
            },

            "height": {
                "ideal": 480
            },

            "frameRate": {
                "ideal": 15,
                "max": 20
            },
        },

        "audio": False,
    },

    async_processing=True,
)