import cv2
classify_clock = cv2.CascadeClassifier('cascades\\relogios.xml')

image = cv2.imread('outros\\relogio2.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detected = classify_clock.detectMultiScale(image_gray, scaleFactor=1.01, minNeighbors=5)

for (x, y, w, h) in detected:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)



cv2.imshow('Clock image', image)
cv2.waitKey()
