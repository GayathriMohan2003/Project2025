import cv2
import os

# Ask user for class name
class_name = input("Enter class name (e.g., class1): ")

# Define where to save captured images
DATASET_PATH = 'dataset'
save_path = os.path.join(DATASET_PATH, class_name)
os.makedirs(save_path, exist_ok=True)

# Use default webcam (Mac usually uses index 0)
cap = cv2.VideoCapture(0)

# Check if webcam is accessible
if not cap.isOpened():
    print("❌ Error: Could not access the webcam.")
    exit()

print("✅ Webcam is open.")
print("Press 'c' to capture an image, 'q' to quit.")

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame.")
        break

    # Show the video feed
    cv2.imshow("Press 'c' to capture, 'q' to quit", frame)

    # Wait for key press
    key = cv2.waitKey(1)

    # 'c' to capture image
    if key & 0xFF == ord('c'):
        filename = f"{class_name}_{count}.jpg"
        filepath = os.path.join(save_path, filename)
        cv2.imwrite(filepath, frame)
        print(f"✅ Saved: {filepath}")
        count += 1

    # 'q' to quit
    elif key & 0xFF == ord('q'):
        print("📷 Closing webcam.")
        break

cap.release()
cv2.destroyAllWindows()
