import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from PIL import Image
import time

# Set page config
st.set_page_config(
    page_title="Sign Language Detection",
    page_icon="👋",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f5;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .title {
        text-align: center;
        color: #2c3e50;
    }
    .subtitle {
        text-align: center;
        color: #7f8c8d;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.markdown("<h1 class='title'>Sign Language Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Real-time sign language detection using computer vision</p>", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    try:
        model_dict = pickle.load(open('./model.p', 'rb'))
        return model_dict['model']
    except FileNotFoundError:
        st.error("model.p file not found. Please run train_classifier.py first.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define labels
labels_dict = {
    0: 'hello', 1: 'Bombardilo crocodilo', 2: 'capuchino assassino', 3: 'its never really over', 4: 'E',
    5: 'F', 6: 'G', 7: 'MERA KHEL KHTM HAIII', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
    25: 'Z'
}

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Camera Feed")
    # Create a placeholder for the video feed
    video_placeholder = st.empty()

with col2:
    st.markdown("### Detected Sign")
    # Create a placeholder for the detected sign
    sign_placeholder = st.empty()
    st.markdown("### Instructions")
    st.markdown("""
    1. Make sure your camera is properly connected
    2. Position your hand clearly in front of the camera
    3. Make the sign you want to detect
    4. The detected sign will appear in the box above
    """)

# Initialize camera
cap = None
camera_index = 0
while camera_index < 4:
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        st.success(f"Camera initialized at index {camera_index}")
        break
    camera_index += 1

if not cap or not cap.isOpened():
    st.error("Could not initialize camera. Please check your camera connection.")
    st.stop()

# Main loop
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error reading frame from camera")
            continue

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        detected_sign = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            data_aux = []
            x_ = []
            y_ = []

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            if len(data_aux) == 42:
                try:
                    prediction = model.predict([np.asarray(data_aux)])
                    detected_sign = labels_dict[int(prediction[0])]

                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) + 10
                    y2 = int(max(y_) * H) + 10

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                    cv2.putText(frame, detected_sign, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                              cv2.LINE_AA)
                except Exception as e:
                    st.error(f"Error making prediction: {e}")

        # Display the video feed
        video_placeholder.image(frame, channels="BGR", use_column_width=True)
        
        # Display the detected sign
        if detected_sign:
            sign_placeholder.markdown(f"<h2 style='text-align: center; color: #2c3e50;'>{detected_sign}</h2>", unsafe_allow_html=True)
        else:
            sign_placeholder.markdown("<h2 style='text-align: center; color: #7f8c8d;'>No sign detected</h2>", unsafe_allow_html=True)

        time.sleep(0.01)  # Small delay to prevent high CPU usage

except KeyboardInterrupt:
    st.info("Stopping sign language detection...")
finally:
    cap.release()
    cv2.destroyAllWindows() 