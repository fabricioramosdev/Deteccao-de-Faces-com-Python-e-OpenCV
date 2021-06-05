import cv2

classify_face = cv2.CascadeClassifier('cascades\\haarcascade_frontalface_default.xml')
image = cv2.imread('pessoas\\pessoas2.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces_detected = classify_face.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=9, minSize=(30, 30))
#  print(len(faces_detected))
#  print(faces_detected)

for (x, y, w, h) in faces_detected:
    # print(x, y, w, h)
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

cv2.imshow('Faces encontradas', image)
cv2.waitKey()


