from skimage.filters import gabor
from skimage import data, io
from matplotlib import pyplot as plt  
import cv2

image = cv2.imread('hist.jpg')


image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
filt_real, filt_imag = gabor(image, frequency=.7)
plt.figure()            
plt.imshow(cv2.cvtColor(filt_imag, cv2.COLOR_BGR2RGB))    
plt.show()               
cv2.imwrite("gobar.jpg",filt_real)

