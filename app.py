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


# ============================================================
# STREAMLIT PAGE
# ============================================================

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

    HolisticLandmarkerOptions = (
        mp.tasks.vision.HolisticLandmarkerOptions
    )

    VisionRunningMode = (
        mp.tasks.vision.RunningMode
    )

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
# LANDMARK PROCESSOR
# ============================================================

class LandmarkProcessor:

    def __init__(self):

        self.landmarker = (
            mp.tasks.vision.HolisticLandmarker
            .create_from_options(options)
        )

        self.timestamp_ms = 0
        self.frame_count = 0

        self.last_prediction = "Unknown"

        # Store the error permanently
        self.last_error = ""

        # Store the stage where error happened
        self.error_stage = ""


    def recv(
        self,
        frame: av.VideoFrame
    ) -> av.VideoFrame:

        self.frame_count += 1

        img = frame.to_ndarray(
            format="bgr24"
        )


        # ====================================================
        # PROCESS EVERY 10TH FRAME
        # ====================================================

        if self.frame_count % 10 == 0:

            try:

                # =================================================
                # MEDIAPIPE
                # =================================================

                self.error_stage = "MediaPipe"

                rgb_frame = cv.cvtColor(
                    img,
                    cv.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                self.timestamp_ms += 33

                landmarker_result = (
                    self.landmarker.detect_for_video(
                        mp_image,
                        self.timestamp_ms
                    )
                )


                # =================================================
                # LANDMARK EXTRACTION
                # =================================================

                self.error_stage = "Landmark extraction"

                landmarkdata = []


                # -------------------------------------------------
                # POSE
                # -------------------------------------------------

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


                # -------------------------------------------------
                # LEFT HAND
                # -------------------------------------------------

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


                # -------------------------------------------------
                # RIGHT HAND
                # -------------------------------------------------

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


                # =================================================
                # CHECK FEATURE COUNT
                # =================================================

                if len(landmarkdata) != 225:

                    raise ValueError(
                        f"Expected 225 features, "
                        f"got {len(landmarkdata)}"
                    )


                # =================================================
                # DATAFRAME
                # =================================================

                self.error_stage = "DataFrame"

                Image_frame_data = pd.DataFrame(
                    [landmarkdata],
                    columns=range(225)
                )


                # =================================================
                # SCALER
                # =================================================

                self.error_stage = "Scaler"

                scaled_data = scaler.transform(
                    Image_frame_data
                )


                # =================================================
                # SCALED DATAFRAME
                # =================================================

                scaled_dataframe = pd.DataFrame(
                    scaled_data,
                    columns=Image_frame_data.columns
                )


                # =================================================
                # XGBOOST
                # =================================================

                self.error_stage = "XGBoost prediction"

                numeric_prediction = (
                    xgb_classifier.predict(
                        scaled_dataframe
                    )
                )


                # =================================================
                # LABEL DECODER
                # =================================================

                self.error_stage = "Label decoding"

                text_prediction = (
                    label_enc.inverse_transform(
                        numeric_prediction
                    )
                )


                # =================================================
                # SUCCESS
                # =================================================

                self.last_prediction = str(
                    text_prediction[0]
                )

                self.last_error = ""
                self.error_stage = ""


            except Exception as e:

                # =================================================
                # SAVE ERROR PERMANENTLY
                # =================================================

                error_type = type(e).__name__
                error_message = str(e)

                self.last_error = (
                    f"{error_type}: {error_message}"
                )


                self.last_prediction = (
                    "XGBoost ERROR"
                )


                # Print full error to server logs
                print(
                    "\n==============================",
                    flush=True
                )

                print(
                    "GESTURE RECOGNITION ERROR",
                    flush=True
                )

                print(
                    "Stage:",
                    self.error_stage,
                    flush=True
                )

                print(
                    "Error type:",
                    error_type,
                    flush=True
                )

                print(
                    "Error:",
                    error_message,
                    flush=True
                )

                print(
                    "==============================\n",
                    flush=True
                )


        # ====================================================
        # DISPLAY NORMAL STATUS
        # ====================================================

        if self.last_error == "":

            cv.putText(
                img,
                "Running...",
                (50, 60),
                cv.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        else:

            # =================================================
            # DISPLAY ERROR PERMANENTLY
            # =================================================

            cv.putText(
                img,
                "ERROR - CHECK BELOW",
                (30, 60),
                cv.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

            cv.putText(
                img,
                f"Stage: {self.error_stage}",
                (30, 100),
                cv.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )

            # Display first part of error
            error_display = self.last_error[:85]

            cv.putText(
                img,
                error_display,
                (30, 140),
                cv.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2
            )


        # ====================================================
        # LANDMARK COUNT
        # ====================================================

        if self.last_error == "":

            cv.putText(
                img,
                "Landmarks: 225",
                (30, 100),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


        # ====================================================
        # PREDICTION
        # ====================================================

        cv.putText(
            img,
            f"Prediction: {self.last_prediction}",
            (30, 210),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            3
        )


        # ====================================================
        # RETURN FRAME
        # ====================================================

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