import cv2
classify_car = cv2.CascadeClassifier('cascades\\cars.xml')

image = cv2.imread('outros\\carro3.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detected = classify_car.detectMultiScale(image_gray, scaleFactor=1.01)

for (x, y, w, h) in detected:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)



cv2.imshow('Car image', image)
cv2.waitKey()
