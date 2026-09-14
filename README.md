Artificial Intelligence Sophisticated for Help and Aid (AI-SHA)

An end-to-end computer vision and machine learning pipeline that extracts body and hand landmarks using MediaPipe and OpenCV, trains a custom XGBoost classifier, and performs real-time Indian Sign Language (ISL) gesture prediction via live webcam feed.

 Tech Stack:
 ->OpenCV (cv2): Image preprocessing (resizing, color space conversions) and live video capture.
 ->MediaPipe: Holistic landmarker extraction (Pose, Left Hand, Right Hand).
 ->XGBoost: Gradient-boosted decision trees for multi-class gesture classification.
 ->Pandas & NumPy: Data structuring, dataset padding, and array manipulation.
 ->Pickle: Model serialization for loading the trained pipeline in real-time execution.
 
  Pipeline Overview
  ->Preprocessing: Images are loaded, resized to standard dimensions ($640 \times 480$), and converted to RGB.
  ->Feature Extraction: MediaPipe Holistic Landmarker detects 33 Pose and 42 Hand (21 per hand) landmarks ($X, Y, Z$ coordinates = 225 features). Missing hands are padded with zeros to ensure a consistent 226-column feature vector per frame.
  ->Model Training: Landmarks and gesture labels are saved to CSV, cleaned, and used to train an XGBoost Classifier. The model is serialized via pickle.
  ->Real-time Inference: The main program processes a live webcam stream, extracts landmark feature vectors per frame, and outputs ISL predictions in real-time.