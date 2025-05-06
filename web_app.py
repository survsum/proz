from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    print("model.p file not found. Please run train_classifier.py first.")
    exit(1)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define labels
labels_dict = {
    0: 'A', 1: 'B ', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'MERA KHEL KHTM HAIII', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
    25: 'Z'
}

def generate_frames():
    camera_index = 0
    cap = None
    while camera_index < 4:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"Camera initialized at index {camera_index}")
            break
        camera_index += 1

    if not cap or not cap.isOpened():
        print("Could not initialize camera. Please check your camera connection.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        # Add gothic border to the frame
        cv2.rectangle(frame, (0, 0), (W, H), (139, 0, 0), 10)
        
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
                    predicted_character = labels_dict[int(prediction[0])]

                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) + 10
                    y2 = int(max(y_) * H) + 10

                    # Draw gothic-style box and text
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (139, 0, 0), 4)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.3, (139, 0, 0), 3,
                              cv2.LINE_AA)
                except Exception as e:
                    print(f"Error making prediction: {e}")

        # Add gothic title
        cv2.putText(frame, "Sign Language Detection", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (139, 0, 0), 3,
                   cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 