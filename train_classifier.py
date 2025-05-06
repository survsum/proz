import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

try:
    data_dict = pickle.load(open('./data.pickle', 'rb'))
except FileNotFoundError:
    raise Exception("data.pickle file not found. Please run create_dataset.py first.")
except Exception as e:
    raise Exception(f"Error loading data.pickle: {e}")

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

if len(data) == 0 or len(labels) == 0:
    raise Exception("No data found in data.pickle. Please check your dataset.")

print(f"Training on {len(data)} samples")

# Split the data
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

print(f"Training set size: {len(x_train)}")
print(f"Test set size: {len(x_test)}")

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

try:
    model.fit(x_train, y_train)
except Exception as e:
    raise Exception(f"Error training model: {e}")

# Evaluate the model
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print('\nModel Evaluation:')
print('-' * 50)
print(f'Accuracy: {score * 100:.2f}%')
print('\nClassification Report:')
print(classification_report(y_test, y_predict))
print('-' * 50)

# Save the model
try:
    with open('model.p', 'wb') as f:
        pickle.dump({'model': model}, f)
    print("\nModel successfully saved to model.p")
except Exception as e:
    raise Exception(f"Error saving model: {e}")
