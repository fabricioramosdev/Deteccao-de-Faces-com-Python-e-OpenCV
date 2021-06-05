import cv2
classify_clock = cv2.CascadeClassifier('cascades\\relogios.xml')

image = cv2.imread('objetos\\relogio1.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detected = classify_clock.detectMultiScale(image_gray)

for (x, y, w, h) in detected:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)



cv2.imshow('Clock image', image)
cv2.waitKey()
