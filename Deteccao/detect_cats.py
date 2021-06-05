import cv2
classify_cat = cv2.CascadeClassifier('cascades\\haarcascade_frontalcatface.xml')

cat_image =  cv2.imread('animais\\gato2.jpg')
cat_image_gray = cv2.cvtColor(cat_image, cv2.COLOR_BGR2GRAY)

cat_detected = classify_cat.detectMultiScale(cat_image_gray, scaleFactor=1.08)

for (x, y, w, h) in cat_detected:
    cv2.rectangle(cat_image, (x, y), (x+w, y+h), (0, 255, 0), 2)



cv2.imshow('Cat image gray', cat_image)
cv2.waitKey()
