import os
import pickle
import numpy as np
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    raise Exception(f"Data directory {DATA_DIR} does not exist. Please run collect_imgs.py first.")

data = []
labels = []
min_samples = float('inf')

for dir_ in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(class_dir):
        continue
        
    class_samples = 0
    for img_path in os.listdir(class_dir):
        if not img_path.endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        try:
            img = cv2.imread(os.path.join(class_dir, img_path))
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
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
                    data.append(data_aux)
                    labels.append(dir_)
                    class_samples += 1
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            
    min_samples = min(min_samples, class_samples)
    print(f"Processed {class_samples} samples for class {dir_}")

# Balance the dataset by taking the minimum number of samples from each class
balanced_data = []
balanced_labels = []
class_counts = {}

for i, label in enumerate(labels):
    if label not in class_counts:
        class_counts[label] = 0
    if class_counts[label] < min_samples:
        balanced_data.append(data[i])
        balanced_labels.append(label)
        class_counts[label] += 1

print(f"Final balanced dataset size: {len(balanced_data)} samples")

try:
    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': balanced_data, 'labels': balanced_labels}, f)
    print("Dataset successfully saved to data.pickle")
except Exception as e:
    print(f"Error saving dataset: {e}")
