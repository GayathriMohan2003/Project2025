from flask import Flask, render_template, request
import cv2
import numpy as np
import pickle
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
try:
    model, categories = pickle.load(open('model/classifier.pkl', 'rb'))
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Failed to load model:", e)
    exit()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        print("📥 Received POST request.")
        file = request.files['image']
        if not file:
            print("❌ No file received.")
            return render_template('index.html', result="No file received")

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        print(f"✅ Saved file to {filepath}")

        # Read and predict
        img = cv2.imread(filepath)
        if img is None:
            print("❌ Could not read uploaded image.")
            return render_template('index.html', result="Failed to read image")

        img = cv2.resize(img, (64, 64)).flatten().reshape(1, -1)
        prediction = model.predict(img)[0]
        result = categories[prediction]
        print(f"🎯 Predicted: {result}")

        return render_template('index.html', result=result, img_path=filepath)

    print("📡 Serving GET request.")
    return render_template('index.html')

if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    app.run(debug=True)
