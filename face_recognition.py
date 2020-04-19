import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('face.yml')
cascadePath = "cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
# Тип шрифта
font = cv2.FONT_HERSHEY_SIMPLEX

# iniciate id counter
id = 0

# Список имен для id
names = ['None', 'Nadya','unknow', 'unknow']

cam = cv2.VideoCapture(0)
cam.set(3, 400)  # set video width
cam.set(4, 300)  # set video height
cap = cv2.VideoCapture('SV.avi')
cap.set(3, 400)  # set video width
cap.set(4, 300)
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret1, img1 = cap.read()
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(10, 10),
    )

    faces1 = faceCascade.detectMultiScale(
        gray1,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(10, 10),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Проверяем что лицо распознано
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))

        else:
            id = "unknow"
            confidence = "  {0}%".format(round(100 - confidence))
        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
    for (x, y, w, h) in faces1:
        cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray1[y:y + h, x:x + w])

        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))

        else:
            id = "unknow"
            confidence = "  {0}%".format(round(100 - confidence))
        cv2.putText(img1, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img1, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
    cv2.namedWindow('video', cv2.WINDOW_NORMAL)
    cv2.imshow('video', img1)
    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff  # 'ESC' для Выхода
    if k == 27:
        break

cam.release()
cap.release()
cv2.destroyAllWindows()