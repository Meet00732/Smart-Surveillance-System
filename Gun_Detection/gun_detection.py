import cv2
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('mask_detection_4.model')

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

color = {0: (0,255,0), 1: (0,0,255)}

while True:
    tf, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        face_img = gray[y:y+w, x:x+w]
        resized = cv2.resize(face_img, (200, 200))
        normalized = resized/255.0
        reshaped = np.reshape(normalized, (1, 200, 200, 1))
        result = model.predict(reshaped)

        label = np.argmax(result, axis=1)[0]
        print(categories[label])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, categories[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255))

    cv2.imshow('LIVE', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()