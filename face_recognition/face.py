import cv2
import os
import numpy as np

# === Step 1: Capture and save your face image ===
def capture_face():
    cam = cv2.VideoCapture(0)
    print("📸 Press 's' to save your face")

    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Capture Your Face", gray)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite("my_face.jpg", gray)
            print("✅ Face saved as 'my_face.jpg'")
            break
        elif cv2.waitKey(1) & 0xFF == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

# === Step 2: Compare new face with saved face using histogram ===
def compare_faces():
    if not os.path.exists("my_face.jpg"):
        print("❗ Please run capture first.")
        return

    saved_face = cv2.imread("my_face.jpg", cv2.IMREAD_GRAYSCALE)
    saved_hist = cv2.calcHist([saved_face], [0], None, [256], [0, 256])
    cv2.normalize(saved_hist, saved_hist)

    cam = cv2.VideoCapture(0)
    print("👀 Comparing faces... Press ESC to exit.")

    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        cv2.normalize(current_hist, current_hist)

        similarity = cv2.compareHist(saved_hist, current_hist, cv2.HISTCMP_CORREL)
        label = "Your face" if similarity > 0.9 else "Unknown"

        cv2.putText(frame, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if label == "Your face" else (0, 0, 255), 2)
        cv2.imshow("Face Check", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

# === Run ===
if not os.path.exists("my_face.jpg"):
    capture_face()

compare_faces()
