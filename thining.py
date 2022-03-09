
import matplotlib.pyplot as plt
from skimage.util import invert
import cv2

import numpy as np
from matplotlib import pyplot as plt 

img = cv2.imread("temp2.jpg")
plt.imshow(img)
plt.show()
p=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.imshow(cv2.cvtColor(p, cv2.COLOR_BGR2RGB))
plt.show()
# perform thin
thinned = cv2.ximgproc.thinning(p)
# display results
plt.imshow(cv2.cvtColor(thinned, cv2.COLOR_BGR2RGB))    
plt.show()  
cv2.imwrite("thin.jpg",thinned)
  