import cv2

# Load the Haarcascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press SPACE to capture image...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    cv2.imshow("Webcam - Press SPACE to capture", frame)

    key = cv2.waitKey(1)
    if key % 256 == 32:  # Space key
        # Save captured frame
        img = frame.copy()
        break

# Release the webcam
cap.release()
cv2.destroyAllWindows()

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Show the result
cv2.imshow("Detected Face", img)
cv2.waitKey(0)
cv2.destroyAllWindows()