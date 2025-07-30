import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

# Path to dataset folder
data_dir = 'dataset'
categories = os.listdir(data_dir)

data = []
labels = []

print("✅ Loading images...")

# Loop through each class folder
for idx, category in enumerate(categories):
    category_path = os.path.join(data_dir, category)
    if not os.path.isdir(category_path):
        continue
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Couldn't read image: {img_path}")
            continue
        img = cv2.resize(img, (64, 64))
        data.append(img.flatten())
        labels.append(idx)

# Check if data was loaded
if len(data) == 0:
    print("❌ No images found! Make sure your dataset folders are not empty.")
    exit()

print(f"✅ Loaded {len(data)} images from {len(categories)} classes.")

# Convert to numpy arrays
X = np.array(data)
y = np.array(labels)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
print("✅ Training model...")
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"🎯 Model accuracy: {accuracy*100:.2f}%")

# Save model and category labels
os.makedirs('model', exist_ok=True)
with open('model/classifier.pkl', 'wb') as f:
    pickle.dump((model, categories), f)

print("✅ Model saved to model/classifier.pkl")
