import cv2 
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread("Images_and_Videos/girl.jpg")
b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
hist_b = cv2.calcHist([b],[0],None,[256],[0,256])
hist_g = cv2.calcHist([g],[0],None,[256],[0,256])
hist_r = cv2.calcHist([r],[0],None,[256],[0,256])
plt.plot(hist_r, color='r', label="r")
plt.plot(hist_g, color='g', label="g")
plt.plot(hist_b, color='b', label="b")
plt.legend()
plt.show() 
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# h, s, v = img2[:,:,0], img2[:,:,1], img2[:,:,2]
h,s,v = cv2.split(img2)
hist_h = cv2.calcHist([h],[0],None,[256],[0,256])
hist_s = cv2.calcHist([s],[0],None,[256],[0,256])
hist_v = cv2.calcHist([v],[0],None,[256],[0,256])
#plt.plot(hist_h, color='r', label="h")
#plt.plot(hist_s, color='g', label="s")
plt.plot(hist_v, color='b', label="v")
plt.legend()
plt.show()