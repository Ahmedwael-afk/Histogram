import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread("Images_and_Videos/land.jpg", 0)
w , h = img.shape[:2]
print(w,h)
Values = np.array(img.flatten())
histogram = np.zeros(256)

for i in range(len(Values)):
	histogram[Values[i]] += 1

plt.plot(histogram)
plt.show()
print(len(histogram))

histo = np.array (histogram)/(len(Values)*40)
Sum = np.zeros(len(histo))
print(len(Sum))
for i in range(256):
	Sum[i] = sum(histo[:i+1])
	
Sum_2 = 0
for i in range(len(histo)):
	Sum_2 += Sum[i] 



New_intensities = np.uint8(255*Sum)
print(New_intensities)
plt.plot(New_intensities)
plt.show()



print(Sum_2)
print(Sum,type(Sum))