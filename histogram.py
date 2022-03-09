import cv2
import numpy as np
from matplotlib import pyplot as plt 


img = cv2.imread("contact-based_fingerprints\\first_session\\1_1.jpg",0)

equ = cv2.equalizeHist(img)
#cv2.imwrite("hist.png", equ)
plt.hist(img.flat, bins=100, range=(0, 255))
plt.show()
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
plt.hist(equ.flat, bins=100, range=(0, 255))
plt.show()
plt.imshow(cv2.cvtColor(equ, cv2.COLOR_BGR2RGB))
plt.show()



img = cv2.imread("contact-based_fingerprints\\first_session\\1_1.jpg",0)
clahe = cv2.createCLAHE(clipLimit =6.0, tileGridSize=(8,8))
cl_img = clahe.apply(img)
plt.hist(cl_img.flat, bins=100, range=(100, 255))
plt.show()
ret, thresh3 = cv2.threshold(cl_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.imshow(cv2.cvtColor(thresh3, cv2.COLOR_BGR2RGB))
plt.show()
cv2.imwrite("hist.jpg",thresh3)

