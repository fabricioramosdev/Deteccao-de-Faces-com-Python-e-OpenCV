import cv2

classify_face = cv2.CascadeClassifier('cascades\\haarcascade_frontalface_default.xml')
classify_eye = cv2.CascadeClassifier('cascades\\haarcascade_eye.xml')

image = cv2.imread('pessoas\\beatles.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces_detected = classify_face.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=9, minSize=(30, 30))

for (x, y, w, h) in faces_detected:
    image = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

    face = image[y:y+h, x:x+w]
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    eye_detected = classify_eye.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=2)

    for(ox, oy, ow, oh) in eye_detected:
        cv2.rectangle(face, (ox, oy), (ox + ow, oy + oh), (0, 190, 0), 2)


# cv2.imshow('Faces e olhos detectados', face_gray)
cv2.imshow('Faces e olhos detectados', image)
cv2.waitKey()

