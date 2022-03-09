import os
import cv2
import numpy as np
from skimage.filters import gabor
from tqdm import tqdm
from PIL import Image
from skimage import filters, morphology
from sklearn import metrics
import matplotlib.pyplot as plt


path=os.listdir("first")

person=[]
dataset=[]
# global variables
EXPAND_WIDTH = 250
EXPAND_HEIGHT = 250

CropWidth = 220
CropHeight = 220
ExpWidth = 250
ExpHeight = 250

def get_fp_region(img_path, crop_width=250, crop_height=250):

    CropWidth = crop_width
    CropHeight = crop_height
    ExpWidth = EXPAND_WIDTH
    ExpHeight = EXPAND_HEIGHT

    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    thresh = filters.threshold_otsu(image)

    picBW = image > thresh
    bw = morphology.closing(picBW, morphology.square(3))
    # plot_image(bw, "B&W")
    
    cleared = bw.copy()

    img_width = image.shape[1]
    img_height = image.shape[0]
    #print(img_width, img_height)

    crop_l = img_width
    crop_r = 0
    crop_t = img_height
    crop_b = 0
    for i in range(img_height):
        for j in range(img_width):
            if cleared[i, j] == False:
                if (crop_l > j):
                    crop_l = j
                if (crop_r < j):
                    crop_r = j
                if (crop_t > i):
                    crop_t = i
                if (crop_b < i):
                    crop_b = i

    if ((crop_r - crop_l) < CropWidth):
        diff = CropWidth - (crop_r - crop_l)
        if (crop_r + crop_l > CropWidth): # right
            if (img_width - crop_r > diff / 2):
                crop_r += diff / 2
                crop_l -= diff / 2
            else:
                crop_r = img_width - 1
                crop_l = crop_r - (CropWidth + 2)
        else: # left
            if (crop_l > diff / 2):
                crop_l -= diff / 2
                crop_r += diff / 2
            else:
                crop_l = 1
                crop_r = crop_l + (CropWidth + 2)
    if ((crop_b - crop_t) < CropHeight):
        diff = CropHeight - (crop_b - crop_t)
        if (crop_b + crop_t > CropHeight): # bottom
            if (img_height - crop_b > diff / 2):
                crop_b += diff / 2
                crop_t -= diff / 2
            else:
                crop_b = img_height - 1
                crop_t = crop_b - (CropHeight + 2)
        else: # top
            if (crop_t > diff / 2):
                crop_t -= diff / 2
                crop_b += diff / 2
            else:
                crop_t = 1
                crop_b = crop_t + (CropHeight + 2)

    # expand region for rotation
    crop_l = (crop_r + crop_l - CropWidth) / 2
    crop_r = crop_l + CropWidth
    crop_t = (crop_t + crop_b - CropHeight) / 2
    crop_b = crop_t + CropHeight
    crop_l = (int)(crop_l - ((ExpWidth - CropWidth) / 2))
    crop_r = (int)(crop_r + ((ExpWidth - CropWidth) / 2))
    crop_t = (int)(crop_t - ((ExpHeight - CropHeight) / 2))
    crop_b = (int)(crop_b + ((ExpHeight - CropHeight) / 2))

    # check expanded region
    diff = 0
    if (crop_l < 0):
        diff = 0 - crop_l
        crop_l = crop_l + diff
        crop_r = crop_r + diff
    if (crop_r >= img_width):
        diff = crop_r - (img_width - 1)
        crop_l = crop_l - diff
        crop_r = crop_r - diff

    diff = 0
    if (crop_t < 0):
        diff = 0 - crop_t
        crop_t = crop_t + diff
        crop_b = crop_b + diff
    if (crop_b >= img_height):
        diff = crop_b - (img_height - 1)
        crop_t = crop_t - diff
        crop_b = crop_b - diff
    # crop for process image
    crop_x = (ExpWidth - CropWidth) / 2
    crop_y = (ExpHeight - CropHeight) / 2
    img = Image.open(img_path)
    image = img.crop([crop_l, crop_t, crop_r, crop_b])

    img_c = image.crop([crop_x, crop_y, crop_x + CropWidth, crop_y + CropHeight])
    
    return img_c

for i in tqdm(range(len(path))):
    m,_ = path[i].split(".")
    x,y=m.split("_")
    person.append(int(x))
    img = cv2.imread("first\\"+path[i])
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit =7.0, tileGridSize=(8,8))
    cl_img = clahe.apply(img)
    
    ret, thresh3 = cv2.threshold(cl_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    filt_real, filt_imag = gabor(thresh3, frequency=0.9)
    
    cv2.imwrite("temp.jpg",filt_real)
    
    thinned=get_fp_region("temp.jpg")
    thinned.save("temp2.jpg")
    
    temp=cv2.imread("temp2.jpg")
    dataset.append(temp)

datasets=np.asarray(dataset)
datasets=datasets/255

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(datasets, person, test_size=0.15,random_state=24)


import time
from keras.applications.vgg19 import VGG19

VGG_model = VGG19(weights='imagenet', include_top=False, input_shape=(250,250, 3))
for layer in VGG_model.layers:
	layer.trainable = False
    
VGG_model.summary()

feature_extractor=VGG_model.predict(X_train)
feature=feature_extractor.reshape(feature_extractor.shape[0],-1)

feature_extractor_test=VGG_model.predict(X_test)
feature_test=feature_extractor_test.reshape(feature_extractor_test.shape[0],-1)

times=[]
accuracy_train=[]
accuracy_test=[]


from sklearn.ensemble import RandomForestClassifier
model1 = RandomForestClassifier(n_estimators = 200, random_state =42)
start_time = time.time()

model1.fit(feature,y_train)

times.append(time.time()-start_time)

y_pred = model1.predict(feature_test)
accuracy_test.append(metrics.accuracy_score(y_test, y_pred))

print ("Random forest test set Accuracy = ", metrics.accuracy_score(y_test, y_pred))
y_pred = model1.predict(feature)
accuracy_train.append(metrics.accuracy_score(y_train, y_pred))
print ("Random forest train set Accuracy = ", metrics.accuracy_score(y_train, y_pred))


from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=3)

start_time = time.time()
model2.fit(feature,y_train)
times.append(time.time()-start_time)
y_pred = model2.predict(feature_test)
accuracy_test.append(metrics.accuracy_score(y_test, y_pred))
print ("KNN test setAccuracy = ", metrics.accuracy_score(y_test, y_pred))
y_pred = model2.predict(feature)
accuracy_train.append(metrics.accuracy_score(y_train, y_pred))
print ("KNN train setAccuracy = ", metrics.accuracy_score(y_train, y_pred))


from sklearn.naive_bayes import GaussianNB
#Create a Gaussian Classifier
model3 = GaussianNB()
# Train the model using the training sets

start_time = time.time()
model3.fit(feature,y_train)
times.append(time.time()-start_time)
y_pred = model3.predict(feature_test)
accuracy_test.append(metrics.accuracy_score(y_test, y_pred))
print ("NB test set Accuracy = ", metrics.accuracy_score(y_test, y_pred))
y_pred = model3.predict(feature)
accuracy_train.append(metrics.accuracy_score(y_train, y_pred))
print ("NB train set Accuracy = ", metrics.accuracy_score(y_train, y_pred))



from sklearn import svm
#Create a svm Classifier
model4 = svm.SVC(kernel='linear') # Linear Kernel
start_time = time.time()
model4.fit(feature,y_train)
times.append(time.time()-start_time)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
y_pred = model4.predict(feature_test)
accuracy_test.append(metrics.accuracy_score(y_test, y_pred))
print ("SVM test set Accuracy = ", metrics.accuracy_score(y_test, y_pred))
y_pred = model4.predict(feature)
accuracy_train.append(metrics.accuracy_score(y_train, y_pred))

print ("SVM train set Accuracy = ", metrics.accuracy_score(y_train, y_pred))



data = {"Algorithms": ["Random forest", "KNN", "Naive bayes","SVM"],
        "time_taken": times}
df = pd.DataFrame(data, columns=['Algorithms', 'time_taken'])
plots = sns.barplot(x="Algorithms", y="time_taken", data=df)

for p in plots.patches:
    plots.annotate(format(p.get_height(), '.4f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   size=12,
                   xytext = (0,7), 
                   textcoords = 'offset points')
plt.xlabel("Algorithms", size=14)
plt.ylabel("Time Taken(sec)", size=14)
plt.title("Time taken by algorithms to train,with preprocessing")
plt.show()



data = {"Algorithms": ["Random forest", "KNN", "Naive bayes","SVM"],
        "accuracy": accuracy_test}
df = pd.DataFrame(data, columns=['Algorithms', 'accuracy'])
plots = sns.barplot(x="Algorithms", y="accuracy", data=df)
for p in plots.patches:
    plots.annotate(format(p.get_height()*100, '.2f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   size=12,
                   xytext = (0,7), 
                   textcoords = 'offset points')
plt.xlabel("Algorithms", size=14)
plt.ylabel("accuracy", size=14)
plt.title("Accuracy on test set,with preprocessing")
plt.show()



data = {"Algorithms": ["Random forest", "KNN", "Naive bayes","SVM"],
        "accuracy": accuracy_train}
df = pd.DataFrame(data, columns=['Algorithms', 'accuracy'])
plots = sns.barplot(x="Algorithms", y="accuracy", data=df)
for p in plots.patches:
    plots.annotate(format(p.get_height()*100, '.2f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   size=12,
                   xytext = (0,7), 
                   textcoords = 'offset points')
plt.xlabel("Algorithms", size=14)
plt.ylabel("accuracy", size=14)
plt.title("Accuracy on training set,with preprocessing")
plt.show()


