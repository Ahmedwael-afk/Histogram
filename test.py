import cv2 as cv
import numpy as np

img = cv.imread("Images_and_Videos/girl.jpg")
cv.imshow("img",img)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#gray_2 = cv.cvtColor(gray,cv.COLOR_BGR2GRAY)
cv.waitKey(0)

#print(np.shape(img),(np.shape(gray)))
# print(img.shape[1:2])
# print(gray.shape[:2])
x = np.shape(img)
y= np.shape(gray)
print(y[-1])
print(x[-1])