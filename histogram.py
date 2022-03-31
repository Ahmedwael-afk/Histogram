import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread("Images_and_Videos/land.jpg")
Values = np.array(img.flatten())
histogram = np.zeros(256)
print(Values,histogram,len(Values))
for i in range(len(Values)):
	histogram[Values[i]] += 1

plt.plot(histogram)
plt.show()

histo = np.array (histogram/len(Values))
print (histo)