import pickle
import cv2
import mediapipe as mp
import numpy as np
import os

# Load the model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    raise Exception("model.p file not found. Please run train_classifier.py first.")
except Exception as e:
    raise Exception(f"Error loading model: {e}")

# Initialize camera
camera_index = 0
cap = None
while camera_index < 4:  # Try first 4 camera indices
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        print(f"Camera initialized at index {camera_index}")
        break
    camera_index += 1

if not cap or not cap.isOpened():
    raise Exception("Could not initialize camera. Please check your camera connection.")

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

print("Starting sign language detection. Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame from camera")
            continue

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            # Process landmarks for prediction
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

            if len(data_aux) == 42:  # 21 landmarks * 2 coordinates
                try:
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict[int(prediction[0])]

                    # Draw bounding box and label
                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) + 10
                    y2 = int(max(y_) * H) + 10

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                              cv2.LINE_AA)
                except Exception as e:
                    print(f"Error making prediction: {e}")

        cv2.imshow('Sign Language Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nStopping sign language detection...")
finally:
    cap.release()
    cv2.destroyAllWindows()
