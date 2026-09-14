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


st.set_page_config(
    page_title="Real-time Gesture Recognition",
    layout="wide"
)

st.title("Sign Language & Gesture Recognition")


# ============================================================
# LOAD MODELS
# ============================================================

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


# ============================================================
# EXTRACT SAVED MODELS
# ============================================================

xgb_classifier = model_pack["model"]
scaler = model_pack["scaler"]
label_enc = model_pack["label_encoder"]


# ============================================================
# XGBOOST COMPATIBILITY SETTINGS
# ============================================================

# Limit XGBoost to one CPU thread.
try:
    xgb_classifier.set_params(
        n_jobs=1
    )
except Exception:
    pass


# Compatibility with models created using older XGBoost versions.
try:
    if hasattr(
        xgb_classifier,
        "use_label_encoder"
    ):
        xgb_classifier.use_label_encoder = False

except Exception:
    pass


# ============================================================
# LANDMARK PROCESSOR
# ============================================================

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


    def recv(
        self,
        frame: av.VideoFrame
    ) -> av.VideoFrame:

        # ----------------------------------------------------
        # FRAME COUNT
        # ----------------------------------------------------

        self.frame_count += 1

        img = frame.to_ndarray(
            format="bgr24"
        )


        # ----------------------------------------------------
        # RUN INFERENCE EVERY 10TH FRAME
        # ----------------------------------------------------

        if self.frame_count % 10 == 0:

            # Convert BGR -> RGB
            rgb_frame = cv.cvtColor(
                img,
                cv.COLOR_BGR2RGB
            )


            # Create MediaPipe image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )


            # MediaPipe requires increasing timestamps
            self.timestamp_ms += 33


            # ------------------------------------------------
            # MEDIAPIPE HOLISTIC
            # ------------------------------------------------

            landmarker_result = (
                self.landmarker.detect_for_video(
                    mp_image,
                    self.timestamp_ms
                )
            )


            # ------------------------------------------------
            # EXTRACT LANDMARKS
            # ------------------------------------------------

            landmarkdata = []


            # =================================================
            # POSE
            # =================================================

            if (
                landmarker_result.pose_landmarks
                and len(
                    landmarker_result.pose_landmarks
                ) > 0
            ):

                pose_points = (
                    landmarker_result.pose_landmarks[0]
                    if isinstance(
                        landmarker_result.pose_landmarks[0],
                        list
                    )
                    else
                    landmarker_result.pose_landmarks
                )

                for landmark in pose_points:

                    landmarkdata.extend(
                        [
                            landmark.x,
                            landmark.y,
                            landmark.z
                        ]
                    )

            else:

                landmarkdata.extend(
                    [0.0] * 99
                )


            # =================================================
            # LEFT HAND
            # =================================================

            if (
                landmarker_result.left_hand_landmarks
                and len(
                    landmarker_result.left_hand_landmarks
                ) > 0
            ):

                left_hand_points = (
                    landmarker_result.left_hand_landmarks[0]
                    if isinstance(
                        landmarker_result.left_hand_landmarks[0],
                        list
                    )
                    else
                    landmarker_result.left_hand_landmarks
                )

                for landmark in left_hand_points:

                    landmarkdata.extend(
                        [
                            landmark.x,
                            landmark.y,
                            landmark.z
                        ]
                    )

            else:

                landmarkdata.extend(
                    [0.0] * 63
                )


            # =================================================
            # RIGHT HAND
            # =================================================

            if (
                landmarker_result.right_hand_landmarks
                and len(
                    landmarker_result.right_hand_landmarks
                ) > 0
            ):

                right_hand_points = (
                    landmarker_result.right_hand_landmarks[0]
                    if isinstance(
                        landmarker_result.right_hand_landmarks[0],
                        list
                    )
                    else
                    landmarker_result.right_hand_landmarks
                )

                for landmark in right_hand_points:

                    landmarkdata.extend(
                        [
                            landmark.x,
                            landmark.y,
                            landmark.z
                        ]
                    )

            else:

                landmarkdata.extend(
                    [0.0] * 63
                )


            # ------------------------------------------------
            # VERIFY LANDMARK COUNT
            # ------------------------------------------------

            cv.putText(
                img,
                f"Landmarks: {len(landmarkdata)}",
                (50, 70),
                cv.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )


            # =================================================
            # CREATE DATAFRAME
            # =================================================

            Image_frame_data = pd.DataFrame(
                [landmarkdata],
                columns=range(225)
            )


            # =================================================
            # SCALE DATA
            # =================================================

            scaled_data = scaler.transform(
                Image_frame_data
            )


            cv.putText(
                img,
                "SCALER OK",
                (50, 110),
                cv.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )


            # =================================================
            # XGBOOST PREDICTION
            # =================================================

            try:

                cv.putText(
                    img,
                    "PREDICTING...",
                    (50, 150),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2
                )


                # Run prediction
                numeric_prediction = (
                    xgb_classifier.predict(
                        scaled_data
                    )
                )


                # Prediction succeeded
                cv.putText(
                    img,
                    "XGBOOST PREDICT OK",
                    (50, 190),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )


                # Convert numeric prediction
                # into the original class label
                text_prediction = (
                    label_enc.inverse_transform(
                        numeric_prediction
                    )
                )


                self.last_prediction = str(
                    text_prediction[0]
                )


            except Exception as e:

                self.last_prediction = (
                    "XGBoost ERROR"
                )


                cv.putText(
                    img,
                    "XGBOOST ERROR",
                    (50, 150),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2
                )


                print(
                    "XGBoost error:",
                    repr(e)
                )


        else:

            cv.putText(
                img,
                "Running...",
                (50, 70),
                cv.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )


        # =====================================================
        # DISPLAY PREDICTION
        # =====================================================

        cv.putText(
            img,
            f"Prediction: {self.last_prediction}",
            (50, 250),
            cv.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )


        # =====================================================
        # RETURN FRAME
        # =====================================================

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


# ============================================================
# WEBRTC CONFIGURATION
# ============================================================

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


# ============================================================
# START WEBRTC
# ============================================================

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