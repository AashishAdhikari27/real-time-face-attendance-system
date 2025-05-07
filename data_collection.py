import cv2
import os

person_name = input("Enter your name: ").strip()
save_path = os.path.join("data", "raw", person_name)
os.makedirs(save_path, exist_ok=True)



face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
img_count = 0
max_images = 100

print("[INFO] Collecting data. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        img_path = os.path.join(save_path, f"{img_count}.jpg")
        cv2.imwrite(img_path, face)
        img_count += 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Collecting Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or img_count >= max_images:
        break

cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Collected {img_count} face images in {save_path}")