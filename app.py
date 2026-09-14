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
    page_title="AI-SHA",
    layout="wide"
)

st.title("AI-SHA")


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
# XGBOOST COMPATIBILITY FIX
# ============================================================

try:

    # Old XGBoost attributes
    xgb_classifier.use_label_encoder = False

    xgb_classifier.gpu_id = -1

    xgb_classifier.predictor = "auto"

except Exception as e:

    st.error(
        f"Could not add XGBoost compatibility attributes: {e}"
    )


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

        self.last_error = ""
        self.error_stage = ""

        self.xgb_attribute_status = (
            hasattr(
                xgb_classifier,
                "use_label_encoder"
            )
        )

        self.gpu_attribute_status = (
            hasattr(
                xgb_classifier,
                "gpu_id"
            )
        )

        self.predictor_attribute_status = (
            hasattr(
                xgb_classifier,
                "predictor"
            )
        )


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

                self.error_stage = (
                    "Landmark extraction"
                )

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
                # FEATURE COUNT
                # =================================================

                self.error_stage = (
                    "Feature count"
                )

                if len(landmarkdata) != 225:

                    raise ValueError(
                        f"Expected 225 features, "
                        f"got {len(landmarkdata)}"
                    )


                # =================================================
                # DATAFRAME
                # =================================================

                self.error_stage = (
                    "DataFrame"
                )

                Image_frame_data = pd.DataFrame(
                    [landmarkdata],
                    columns=range(225)
                )


                # =================================================
                # SCALER
                # =================================================

                self.error_stage = (
                    "Scaler"
                )

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
                # XGBOOST ATTRIBUTE CHECK
                # =================================================

                self.error_stage = (
                    "XGBoost attribute check"
                )

                # Force all known legacy attributes.

                xgb_classifier.use_label_encoder = False
                xgb_classifier.gpu_id = -1
                xgb_classifier.predictor = "auto"


                self.xgb_attribute_status = (
                    hasattr(
                        xgb_classifier,
                        "use_label_encoder"
                    )
                )

                self.gpu_attribute_status = (
                    hasattr(
                        xgb_classifier,
                        "gpu_id"
                    )
                )

                self.predictor_attribute_status = (
                    hasattr(
                        xgb_classifier,
                        "predictor"
                    )
                )


                # =================================================
                # XGBOOST PREDICTION
                # =================================================

                self.error_stage = (
                    "XGBoost prediction"
                )

                numeric_prediction = (
                    xgb_classifier.predict(
                        scaled_dataframe
                    )
                )


                # =================================================
                # GESTURE WORD MAPPING
                # =================================================

                self.error_stage = (
                    "Gesture word mapping"
                )

                numeric_prediction = int(
                    numeric_prediction[0]
                )

                gesture_words = {

                    0: "Hello",

                    1: "Thank You",

                    2: "Sorry",

                    3: "Bye",

                }

                text_prediction = gesture_words.get(
                    numeric_prediction,
                    "Unknown"
                )


                # =================================================
                # SUCCESS
                # =================================================

                self.last_prediction = (
                    text_prediction
                )

                self.last_error = ""

                self.error_stage = ""


            except Exception as e:

                # =================================================
                # SAVE ERROR
                # =================================================

                error_type = type(e).__name__

                error_message = str(e)

                self.last_error = (
                    f"{error_type}: {error_message}"
                )

                self.last_prediction = (
                    "XGBoost ERROR"
                )


                # =================================================
                # SERVER LOG
                # =================================================

                print(
                    "\n================================",
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
                    "use_label_encoder exists:",
                    hasattr(
                        xgb_classifier,
                        "use_label_encoder"
                    ),
                    flush=True
                )

                print(
                    "gpu_id exists:",
                    hasattr(
                        xgb_classifier,
                        "gpu_id"
                    ),
                    flush=True
                )

                print(
                    "predictor exists:",
                    hasattr(
                        xgb_classifier,
                        "predictor"
                    ),
                    flush=True
                )

                print(
                    "================================\n",
                    flush=True
                )


        # ====================================================
        # VIDEO STATUS
        # ====================================================

        if self.last_error == "":

            cv.putText(
                img,
                "Running...",
                (30, 60),
                cv.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        else:

            # =================================================
            # PERMANENT ERROR DISPLAY
            # =================================================

            cv.putText(
                img,
                "XGBOOST ERROR",
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


            # -------------------------------------------------
            # ERROR LINE
            # -------------------------------------------------

            error_line = self.last_error[:90]

            cv.putText(
                img,
                error_line,
                (30, 140),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )


            # -------------------------------------------------
            # ATTRIBUTE STATUS
            # -------------------------------------------------

            cv.putText(
                img,
                f"use_label_encoder: "
                f"{self.xgb_attribute_status}",
                (30, 180),
                cv.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2
            )

            cv.putText(
                img,
                f"gpu_id: "
                f"{self.gpu_attribute_status}",
                (30, 205),
                cv.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2
            )

            cv.putText(
                img,
                f"predictor: "
                f"{self.predictor_attribute_status}",
                (30, 230),
                cv.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2
            )


        # ====================================================
        # PREDICTION DISPLAY
        # ====================================================

        cv.putText(
            img,
            f"Prediction: {self.last_prediction}",
            (30, 275),
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