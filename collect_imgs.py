import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 1
dataset_size = 100


camera_index = 0
cap = None
while camera_index < 4: 
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        break
    camera_index += 1

if not cap or not cap.isOpened():
    raise Exception("Could not initialize camera. Please check your camera connection.")

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    done = False
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame from camera")
            continue
            
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame from camera")
            continue
            
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        try:
            cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
            counter += 1
        except Exception as e:
            print(f"Error saving image: {e}")

cap.release()
cv2.destroyAllWindows()
